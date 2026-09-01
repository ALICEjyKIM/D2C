"""Gurobi model for D2C assortment and channel allocation."""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from src.instance import validate_instance, validate_state
from src.types import Instance, MILPSolution, SolverConfig, State


def solve_milp(
    *,
    instance: Instance,
    state: State,
    start_period: int,
    horizon: int,
    solver_config: SolverConfig | None = None,
) -> MILPSolution:
    """Solve the myopic or look-ahead model from an observed state."""
    config = solver_config or SolverConfig()
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
    periods = tuple(range(start_period, end_period + 1))
    sku_ids = instance.sku_ids
    retailer_ids = instance.retailer_ids
    pairs = instance.feasible_retailer_sku_pairs
    retailers_by_sku = {
        i: tuple(r for r, j in pairs if j == i) for i in sku_ids
    }
    x_index = [(r, i, t) for r, i in pairs for t in periods]

    model = gp.Model("d2c_assortment_allocation")
    model.Params.OutputFlag = int(config.output_flag)
    if config.time_limit is not None:
        model.Params.TimeLimit = config.time_limit
    if config.mip_gap is not None:
        model.Params.MIPGap = config.mip_gap

    y = model.addVars(sku_ids, periods, vtype=GRB.BINARY, name="y")
    q = model.addVars(sku_ids, periods, lb=0.0, name="q")
    # Only create x for valid retailer-SKU pairs.
    x = model.addVars(x_index, lb=0.0, name="x")
    g = model.addVars(retailer_ids, periods, lb=0.0, ub=1.0, name="g")
    e = model.addVars(retailer_ids, periods, lb=0.0, ub=1.0, name="e")

    model.setObjective(
        gp.quicksum(
            instance.gamma ** (t - start_period)
            * (
                gp.quicksum(instance.skus[i].d2c_margin * q[i, t] for i in sku_ids)
                + gp.quicksum(
                    instance.retailers[r].wholesale_margins[i] * x[r, i, t]
                    for r, i in pairs
                )
            )
            for t in periods
        ),
        GRB.MAXIMIZE,
    )

    for t in periods:
        # C1: listing-demand link
        for i in sku_ids:
            model.addConstr(
                q[i, t] <= instance.skus[i].d2c_demand * y[i, t],
                name=f"d2c_link[{i},{t}]",
            )

        # C2: assortment cardinality
        model.addConstr(
            gp.quicksum(y[i, t] for i in sku_ids) <= instance.max_d2c_skus,
            name=f"assortment_limit[{t}]",
        )

        # C3: retailer order upper bound
        for r, i in pairs:
            model.addConstr(
                x[r, i, t] <= instance.retailers[r].base_orders[i] * g[r, t],
                name=f"retailer_order[{r},{i},{t}]",
            )

        # C4: SKU supply limit
        for i in sku_ids:
            model.addConstr(
                q[i, t] + gp.quicksum(x[r, i, t] for r in retailers_by_sku[i])
                <= instance.skus[i].supply_limit,
                name=f"sku_supply[{i},{t}]",
            )

        # C5: shared production capacity
        model.addConstr(
            gp.quicksum(
                instance.skus[i].capacity_use
                * (
                    q[i, t]
                    + gp.quicksum(x[r, i, t] for r in retailers_by_sku[i])
                )
                for i in sku_ids
            )
            <= instance.capacity,
            name=f"shared_capacity[{t}]",
        )

        # C6: retailer exposure
        for r in retailer_ids:
            carried_skus = instance.retailers[r].sku_ids
            model.addConstr(
                e[r, t]
                == gp.quicksum(
                    instance.beta * y[i, t]
                    + (1.0 - instance.beta)
                    * q[i, t]
                    / instance.skus[i].d2c_demand
                    for i in carried_skus
                )
                / len(carried_skus),
                name=f"exposure[{r},{t}]",
            )

    # Fix the observed state at the start of the horizon.
    for r in retailer_ids:
        model.addConstr(
            g[r, periods[0]] == state.order_retention[r],
            name=f"initial_retention[{r},{periods[0]}]",
        )

    # C7: retailer order-retention transition
    for t, next_t in zip(periods, periods[1:]):
        for r in retailer_ids:
            model.addConstr(
                g[r, next_t]
                == instance.rho * g[r, t]
                + (1.0 - instance.rho) * (1.0 - instance.kappa * e[r, t]),
                name=f"retention_transition[{r},{t}]",
            )

    # C8 is enforced by the variable domains and bounds above.
    model.optimize()

    status = _status_name(model.Status)
    if model.SolCount == 0:
        raise RuntimeError(
            f"Gurobi finished with status {status} and no feasible solution"
        )

    def value(var: gp.Var) -> float:
        return 0.0 if abs(var.X) <= 1e-9 else float(var.X)

    selected = {
        t: tuple(i for i in sku_ids if y[i, t].X > 0.5) for t in periods
    }
    q_value = {(i, t): value(q[i, t]) for i in sku_ids for t in periods}
    x_value = {(r, i, t): value(x[r, i, t]) for r, i in pairs for t in periods}
    e_value = {(r, t): value(e[r, t]) for r in retailer_ids for t in periods}
    g_value = {(r, t): value(g[r, t]) for r in retailer_ids for t in periods}

    return MILPSolution(
        status=status,
        objective_value=float(model.ObjVal),
        start_period=start_period,
        horizon=len(periods),
        selected_d2c_skus=selected,
        d2c_quantity=q_value,
        retailer_quantity=x_value,
        exposure=e_value,
        order_retention=g_value,
        runtime=float(model.Runtime),
        num_variables=model.NumVars,
        num_constraints=model.NumConstrs,
    )


def _status_name(code: int) -> str:
    names = {
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
    return names.get(code, f"STATUS_{code}")
