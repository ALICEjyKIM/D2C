"""Gurobi formulation for D2C assortment and channel allocation."""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from src.instance import validate_instance, validate_state
from src.types import Instance, MILPSolution, SolverConfig, State


@dataclass(slots=True)
class _Variables:
    y: gp.tupledict
    q: gp.tupledict
    x: gp.tupledict
    g: gp.tupledict
    e: gp.tupledict


def solve_milp(
    *,
    instance: Instance,
    state: State,
    start_period: int,
    horizon: int,
    solver_config: SolverConfig | None = None,
) -> MILPSolution:
    """Solve one myopic or look-ahead MILP from the supplied state."""
    config = solver_config or SolverConfig()
    active_periods = _validate_solve_arguments(
        instance, state, start_period, horizon, config
    )

    model = gp.Model("d2c_assortment_allocation")
    _configure_solver(model, config)
    variables = _create_variables(model, instance, active_periods)
    _set_objective(model, instance, variables, active_periods, start_period)
    _add_constraints(model, instance, state, variables, active_periods)

    model.optimize()
    status = _status_name(model.Status)
    if model.SolCount == 0:
        raise RuntimeError(
            f"Gurobi finished with status {status} and no feasible solution"
        )

    return _extract_solution(
        model=model,
        instance=instance,
        variables=variables,
        active_periods=active_periods,
        start_period=start_period,
        status=status,
    )


def _validate_solve_arguments(
    instance: Instance,
    state: State,
    start_period: int,
    horizon: int,
    config: SolverConfig,
) -> tuple[int, ...]:
    validate_instance(instance)
    validate_state(instance, state)
    if isinstance(start_period, bool) or not isinstance(start_period, int):
        raise ValueError("start_period must be an integer")
    if not 1 <= start_period <= instance.periods:
        raise ValueError("start_period must be within the instance periods")
    if state.period != start_period:
        raise ValueError("state.period must equal start_period")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(config.output_flag, bool):
        raise ValueError("output_flag must be boolean")
    if config.time_limit is not None and config.time_limit <= 0:
        raise ValueError("time_limit must be positive when provided")
    if config.mip_gap is not None and config.mip_gap < 0:
        raise ValueError("mip_gap must be nonnegative when provided")

    end_period = min(start_period + horizon - 1, instance.periods)
    return tuple(range(start_period, end_period + 1))


def _configure_solver(model: gp.Model, config: SolverConfig) -> None:
    model.Params.OutputFlag = int(config.output_flag)
    if config.time_limit is not None:
        model.Params.TimeLimit = config.time_limit
    if config.mip_gap is not None:
        model.Params.MIPGap = config.mip_gap


def _create_variables(
    model: gp.Model,
    instance: Instance,
    active_periods: tuple[int, ...],
) -> _Variables:
    y = model.addVars(
        instance.sku_ids,
        active_periods,
        vtype=GRB.BINARY,
        name="y",
    )
    q = model.addVars(instance.sku_ids, active_periods, lb=0.0, name="q")
    feasible_x_keys = tuple(
        (retailer_id, sku_id, period)
        for retailer_id, sku_id in instance.feasible_retailer_sku_pairs
        for period in active_periods
    )
    x = model.addVars(feasible_x_keys, lb=0.0, name="x")
    g = model.addVars(
        instance.retailer_ids,
        active_periods,
        lb=0.0,
        ub=1.0,
        name="g",
    )
    e = model.addVars(
        instance.retailer_ids,
        active_periods,
        lb=0.0,
        ub=1.0,
        name="e",
    )
    return _Variables(y=y, q=q, x=x, g=g, e=e)


def _set_objective(
    model: gp.Model,
    instance: Instance,
    variables: _Variables,
    active_periods: tuple[int, ...],
    start_period: int,
) -> None:
    discounted_profit = gp.quicksum(
        instance.gamma ** (period - start_period)
        * (
            gp.quicksum(
                instance.skus[sku_id].d2c_margin * variables.q[sku_id, period]
                for sku_id in instance.sku_ids
            )
            + gp.quicksum(
                instance.retailers[retailer_id].wholesale_margins[sku_id]
                * variables.x[retailer_id, sku_id, period]
                for retailer_id, sku_id in instance.feasible_retailer_sku_pairs
            )
        )
        for period in active_periods
    )
    model.setObjective(discounted_profit, GRB.MAXIMIZE)


def _add_constraints(
    model: gp.Model,
    instance: Instance,
    state: State,
    variables: _Variables,
    active_periods: tuple[int, ...],
) -> None:
    feasible_pairs = instance.feasible_retailer_sku_pairs
    pairs_by_sku = {
        sku_id: tuple(pair for pair in feasible_pairs if pair[1] == sku_id)
        for sku_id in instance.sku_ids
    }

    for period in active_periods:
        # C1: D2C quantity is available only for listed SKUs.
        for sku_id, sku in instance.skus.items():
            model.addConstr(
                variables.q[sku_id, period]
                <= sku.d2c_demand * variables.y[sku_id, period],
                name=f"d2c_link[{sku_id},{period}]",
            )

        # C2: D2C assortment cardinality.
        model.addConstr(
            gp.quicksum(
                variables.y[sku_id, period] for sku_id in instance.sku_ids
            )
            <= instance.max_d2c_skus,
            name=f"assortment_limit[{period}]",
        )

        # C3: Retailer allocation cannot exceed retained baseline orders.
        for retailer_id, sku_id in feasible_pairs:
            baseline_order = instance.retailers[retailer_id].base_orders[sku_id]
            model.addConstr(
                variables.x[retailer_id, sku_id, period]
                <= baseline_order * variables.g[retailer_id, period],
                name=f"retailer_order[{retailer_id},{sku_id},{period}]",
            )

        # C4: Total channel allocation respects each SKU supply limit.
        for sku_id, sku in instance.skus.items():
            retailer_total = gp.quicksum(
                variables.x[retailer_id, pair_sku_id, period]
                for retailer_id, pair_sku_id in pairs_by_sku[sku_id]
            )
            model.addConstr(
                variables.q[sku_id, period] + retailer_total <= sku.supply_limit,
                name=f"sku_supply[{sku_id},{period}]",
            )

        # C5: Shared production capacity across both channels.
        model.addConstr(
            gp.quicksum(
                instance.skus[sku_id].capacity_use
                * (
                    variables.q[sku_id, period]
                    + gp.quicksum(
                        variables.x[retailer_id, pair_sku_id, period]
                        for retailer_id, pair_sku_id in pairs_by_sku[sku_id]
                    )
                )
                for sku_id in instance.sku_ids
            )
            <= instance.capacity,
            name=f"shared_capacity[{period}]",
        )

        # C6: Exposure combines listing and normalized D2C quantity.
        for retailer_id, retailer in instance.retailers.items():
            exposure_expression = (1.0 / len(retailer.sku_ids)) * gp.quicksum(
                instance.beta * variables.y[sku_id, period]
                + (1.0 - instance.beta)
                * variables.q[sku_id, period]
                / instance.skus[sku_id].d2c_demand
                for sku_id in retailer.sku_ids
            )
            model.addConstr(
                variables.e[retailer_id, period] == exposure_expression,
                name=f"exposure[{retailer_id},{period}]",
            )

    for retailer_id in instance.retailer_ids:
        model.addConstr(
            variables.g[retailer_id, active_periods[0]]
            == state.order_retention[retailer_id],
            name=f"initial_retention[{retailer_id},{active_periods[0]}]",
        )

    # C7: Exposure in one active period changes next-period order retention.
    for period, next_period in zip(active_periods, active_periods[1:]):
        for retailer_id in instance.retailer_ids:
            model.addConstr(
                variables.g[retailer_id, next_period]
                == instance.rho * variables.g[retailer_id, period]
                + (1.0 - instance.rho)
                * (1.0 - instance.kappa * variables.e[retailer_id, period]),
                name=f"retention_transition[{retailer_id},{period}]",
            )


def _extract_solution(
    *,
    model: gp.Model,
    instance: Instance,
    variables: _Variables,
    active_periods: tuple[int, ...],
    start_period: int,
    status: str,
) -> MILPSolution:
    selected_d2c_skus = {
        period: tuple(
            sku_id
            for sku_id in instance.sku_ids
            if variables.y[sku_id, period].X > 0.5
        )
        for period in active_periods
    }
    d2c_quantity = {
        (sku_id, period): _clean_value(variables.q[sku_id, period].X)
        for sku_id in instance.sku_ids
        for period in active_periods
    }
    retailer_quantity = {
        (retailer_id, sku_id, period): _clean_value(
            variables.x[retailer_id, sku_id, period].X
        )
        for retailer_id, sku_id in instance.feasible_retailer_sku_pairs
        for period in active_periods
    }
    exposure = {
        (retailer_id, period): _clean_value(variables.e[retailer_id, period].X)
        for retailer_id in instance.retailer_ids
        for period in active_periods
    }
    order_retention = {
        (retailer_id, period): _clean_value(variables.g[retailer_id, period].X)
        for retailer_id in instance.retailer_ids
        for period in active_periods
    }

    return MILPSolution(
        status=status,
        objective_value=float(model.ObjVal),
        start_period=start_period,
        horizon=len(active_periods),
        selected_d2c_skus=selected_d2c_skus,
        d2c_quantity=d2c_quantity,
        retailer_quantity=retailer_quantity,
        exposure=exposure,
        order_retention=order_retention,
        runtime=float(model.Runtime),
        num_variables=model.NumVars,
        num_constraints=model.NumConstrs,
    )


def _clean_value(value: float, tolerance: float = 1e-9) -> float:
    return 0.0 if abs(value) <= tolerance else float(value)


def _status_name(status_code: int) -> str:
    status_names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.WORK_LIMIT: "WORK_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }
    return status_names.get(status_code, f"STATUS_{status_code}")
