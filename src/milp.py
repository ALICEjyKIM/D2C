"""D2C assortment와 채널 배분을 위한 Gurobi MILP."""

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
    """현재 state에서 myopic 또는 look-ahead 문제를 푼다."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not 1 <= state.period <= instance.periods:
        raise ValueError("state.period lies outside the instance periods")
    if state.order_retention.keys() != instance.retailers.keys():
        raise ValueError("state.order_retention must cover exactly all retailers")

    skus, retailers = instance.skus, instance.retailers
    start = state.period
    # 남은 기간보다 길면 horizon을 자동으로 줄인다.
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
    # Retailer가 실제로 취급하는 SKU 조합에만 x를 만든다.
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
        # C1: D2C에 올린 SKU만 판매할 수 있다.
        model.addConstrs(
            (q[i, t] <= skus[i].d2c_demand * y[i, t] for i in skus), name=f"C1[{t}]"
        )

        # C2: 한 기간에 올릴 수 있는 SKU 수를 제한한다.
        model.addConstr(
            gp.quicksum(y[i, t] for i in skus) <= instance.max_d2c_skus, name=f"C2[{t}]"
        )

        # C3: Retailer 공급량은 현재 retention이 반영된 주문량을 넘지 않는다.
        model.addConstrs(
            (x[r, i, t] <= retailers[r].base_orders[i] * g[r, t] for r, i in pairs),
            name=f"C3[{t}]",
        )

        # C4: D2C와 Retailer를 합쳐 SKU별 공급 한도를 지킨다.
        model.addConstrs(
            (
                q[i, t] + gp.quicksum(x[r, i, t] for r in retailers_of[i])
                <= skus[i].supply_limit
                for i in skus
            ),
            name=f"C4[{t}]",
        )

        # C5: 두 채널이 같은 생산 capacity를 나눠 쓴다.
        model.addConstr(
            gp.quicksum(
                skus[i].capacity_use
                * (q[i, t] + gp.quicksum(x[r, i, t] for r in retailers_of[i]))
                for i in skus
            )
            <= instance.capacity,
            name=f"C5[{t}]",
        )

        # C6: 각 Retailer가 취급하는 SKU 기준으로 D2C 노출을 계산한다.
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

    # 시작 시점 retention은 관측값으로 고정하고 이후 기간은 C7로 연결한다.
    model.addConstrs(
        (g[r, start] == state.order_retention[r] for r in retailers), name="g_observed"
    )
    model.addConstrs(
        (
            g[r, t + 1]
            == instance.rho * g[r, t]
            + (1.0 - instance.rho)
            * (1.0 - instance.response_for(r) * e[r, t])
            for t in periods[:-1]
            for r in retailers
        ),
        name="C7",
    )

    # C8의 범위 조건은 변수 생성 시점에 반영했다.
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
