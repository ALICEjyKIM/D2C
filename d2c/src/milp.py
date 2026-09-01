"""Gurobi formulation of the D2C assortment and allocation MILP (see docs/model.md)."""

import gurobipy as gp
from gurobipy import GRB

from src.types import Instance, MILPSolution, SolverConfig, State

STATUS_NAME = {getattr(GRB.Status, n): n for n in dir(GRB.Status) if n.isupper()}


def solve_milp(
    instance: Instance,
    state: State,
    horizon: int,
    config: SolverConfig = SolverConfig(),
) -> MILPSolution:
    """Solve the myopic (horizon=1) or look-ahead model from an observed state."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not 1 <= state.period <= instance.periods:
        raise ValueError("state.period lies outside the instance periods")
    if state.order_retention.keys() != instance.retailers.keys():
        raise ValueError("state.order_retention must cover exactly all retailers")

    skus, retailers = instance.skus, instance.retailers
    start = state.period
    # The look-ahead window is truncated at the end of the instance.
    periods = range(start, min(start + horizon - 1, instance.periods) + 1)
    pairs = instance.retailer_sku_pairs
    retailers_of = {i: [r for r, j in pairs if j == i] for i in skus}  # R_i

    model = gp.Model("d2c_assortment_allocation")
    model.Params.OutputFlag = int(config.output_flag)
    if config.time_limit is not None:
        model.Params.TimeLimit = config.time_limit
    if config.mip_gap is not None:
        model.Params.MIPGap = config.mip_gap

    y = model.addVars(skus, periods, vtype=GRB.BINARY, name="y")
    q = model.addVars(skus, periods, lb=0.0, name="q")
    # x exists only for the feasible retailer-SKU pairs.
    x = model.addVars([(r, i, t) for r, i in pairs for t in periods], lb=0.0, name="x")
    g = model.addVars(retailers, periods, lb=0.0, ub=1.0, name="g")
    e = model.addVars(retailers, periods, lb=0.0, ub=1.0, name="e")

    model.setObjective(
        gp.quicksum(
            instance.gamma ** (t - start)
            * (
                gp.quicksum(skus[i].d2c_margin * q[i, t] for i in skus)
                + gp.quicksum(
                    retailers[r].wholesale_margins[i] * x[r, i, t] for r, i in pairs
                )
            )
            for t in periods
        ),
        GRB.MAXIMIZE,
    )

    for t in periods:
        # C1: D2C sales require the SKU to be listed
        model.addConstrs(
            (q[i, t] <= skus[i].d2c_demand * y[i, t] for i in skus), name=f"C1[{t}]"
        )

        # C2: assortment cardinality
        model.addConstr(
            gp.quicksum(y[i, t] for i in skus) <= instance.max_d2c_skus, name=f"C2[{t}]"
        )

        # C3: retailer orders scale with their current retention
        model.addConstrs(
            (x[r, i, t] <= retailers[r].base_orders[i] * g[r, t] for r, i in pairs),
            name=f"C3[{t}]",
        )

        # C4: per-SKU supply limit across both channels
        model.addConstrs(
            (
                q[i, t] + gp.quicksum(x[r, i, t] for r in retailers_of[i])
                <= skus[i].supply_limit
                for i in skus
            ),
            name=f"C4[{t}]",
        )

        # C5: shared production capacity
        model.addConstr(
            gp.quicksum(
                skus[i].capacity_use
                * (q[i, t] + gp.quicksum(x[r, i, t] for r in retailers_of[i]))
                for i in skus
            )
            <= instance.capacity,
            name=f"C5[{t}]",
        )

        # C6: exposure of retailer r, averaged over the SKUs it carries
        model.addConstrs(
            (
                e[r, t]
                == gp.quicksum(
                    instance.beta * y[i, t]
                    + (1.0 - instance.beta) * q[i, t] / skus[i].d2c_demand
                    for i in retailers[r].base_orders
                )
                / len(retailers[r].base_orders)
                for r in retailers
            ),
            name=f"C6[{t}]",
        )

    # Retention is observed at the start of the horizon and follows C7 afterwards.
    model.addConstrs(
        (g[r, start] == state.order_retention[r] for r in retailers), name="g_observed"
    )
    model.addConstrs(
        (
            g[r, t + 1]
            == instance.rho * g[r, t]
            + (1.0 - instance.rho) * (1.0 - instance.kappa * e[r, t])
            for t in periods[:-1]
            for r in retailers
        ),
        name="C7",
    )

    # C8 is carried by the variable domains declared above.
    model.optimize()

    status = STATUS_NAME.get(model.Status, str(model.Status))
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi finished with status {status} and no solution")

    def val(var):
        return 0.0 if abs(var.X) <= 1e-9 else var.X

    return MILPSolution(
        status=status,
        objective_value=model.ObjVal,
        start_period=start,
        horizon=len(periods),
        selected_d2c_skus={
            t: tuple(i for i in skus if y[i, t].X > 0.5) for t in periods
        },
        d2c_quantity={(i, t): val(q[i, t]) for i in skus for t in periods},
        retailer_quantity={
            (r, i, t): val(x[r, i, t]) for r, i in pairs for t in periods
        },
        exposure={(r, t): val(e[r, t]) for r in retailers for t in periods},
        order_retention={(r, t): val(g[r, t]) for r in retailers for t in periods},
        runtime=model.Runtime,
        num_variables=model.NumVars,
        num_constraints=model.NumConstrs,
    )
