"""D2C 배분을 위한 결정론적 MILP와 minimax regret MILP."""

import math

import gurobipy as gp
from gurobipy import GRB

from src.scenarios import validate_scenarios
from src.types import (
    FirstStageDecision,
    Instance,
    MILPSolution,
    MinimaxRegretResult,
    ScenarioBestResult,
    ScenarioRegretResult,
    SolverConfig,
    State,
)


STATUS_NAME = {getattr(GRB.Status, n): n for n in dir(GRB.Status) if n.isupper()}


def solve_milp(
    instance: Instance,
    state: State,
    horizon: int,
    config: SolverConfig = SolverConfig(),
    scenario: dict | None = None,
) -> MILPSolution:
    """현재 state에서 결정론적 계획을 푼다."""
    _validate_state(instance, state)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    start = state.period
    end = min(start + horizon - 1, instance.periods)
    periods = tuple(range(start, end + 1))
    rows = _scenario_rows(instance, scenario, periods)

    model = _new_model("d2c_assortment_allocation", config)
    plan = _add_scenario_plan(model, instance, state, periods, rows)
    model.setObjective(plan["profit"], GRB.MAXIMIZE)
    model.optimize()

    status = _solution_status(model)
    return _extract_solution(
        instance,
        periods,
        plan,
        status,
        model.ObjVal,
        model.Runtime,
        model.NumVars,
        model.NumConstrs,
    )


def calculate_scenario_best_profits(
    instance: Instance,
    state: State,
    scenarios: list[dict],
    config: SolverConfig = SolverConfig(),
) -> dict[str, ScenarioBestResult]:
    """각 시나리오를 완전히 아는 경우의 최고이익을 계산한다."""
    periods = _validate_scenario_set(instance, state, scenarios)
    results = {}
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        solution = solve_milp(
            instance,
            state,
            len(periods),
            config,
            scenario=scenario,
        )
        if solution.status != "OPTIMAL":
            raise RuntimeError(
                f"scenario {scenario_id} benchmark must be optimal, "
                f"got {solution.status}"
            )
        results[scenario_id] = ScenarioBestResult(
            scenario_id=scenario_id,
            best_profit=solution.objective_value,
            solution=solution,
        )
    return results


def solve_minimax_relative_regret(
    instance: Instance,
    state: State,
    scenarios: list[dict],
    epsilon: float = 1e-6,
    config: SolverConfig = SolverConfig(),
    best_results: dict[str, ScenarioBestResult] | None = None,
    fixed_decision: FirstStageDecision | None = None,
) -> MinimaxRegretResult:
    """공통 당기결정을 갖는 정확한 minimax relative regret 문제를 푼다."""
    periods = _validate_scenario_set(instance, state, scenarios)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    if best_results is None:
        best_results = calculate_scenario_best_profits(
            instance, state, scenarios, config
        )
    scenario_ids = [scenario["scenario_id"] for scenario in scenarios]
    if set(best_results) != set(scenario_ids):
        raise ValueError("best_results must cover exactly all scenarios")

    model, plans, theta = _build_regret_model(
        instance,
        state,
        scenarios,
        periods,
        epsilon,
        config,
        best_results,
        fixed_decision,
    )

    model.setObjective(theta, GRB.MINIMIZE)
    model.optimize()
    status = _solution_status(model)
    if status != "OPTIMAL":
        raise RuntimeError(f"minimax regret model must be optimal, got {status}")

    theta_star = _clean_value(theta.X)
    # 같은 theta 안에서는 각 시나리오 계획의 총이익을 최대화한다
    theta.LB = theta_star
    theta.UB = theta_star
    model.setObjective(
        gp.quicksum(plans[scenario_id]["profit"] for scenario_id in scenario_ids),
        GRB.MAXIMIZE,
    )
    model.optimize()
    status = _solution_status(model)
    if status != "OPTIMAL":
        raise RuntimeError(f"minimax regret tie-break must be optimal, got {status}")

    return _extract_minimax_result(
        instance,
        periods,
        scenario_ids,
        plans,
        best_results,
        epsilon,
        model,
        status,
        theta_star,
    )


def calculate_scenario_profit(
    instance: Instance,
    scenario: dict,
    solution: MILPSolution,
) -> float:
    """반환된 해의 시나리오별 계획이익을 직접 다시 계산한다."""
    periods = tuple(
        range(solution.start_period, solution.start_period + solution.horizon)
    )
    rows = _scenario_rows(instance, scenario, periods)
    start = solution.start_period
    return sum(
        instance.gamma ** (t - start)
        * (
            sum(
                rows[t]["d2c_margin"][i] * solution.d2c_quantity[i, t]
                for i in instance.skus
            )
            + sum(
                rows[t]["retailer_margin"][r][i]
                * solution.retailer_quantity[r, i, t]
                for r, i in instance.retailer_sku_pairs
            )
        )
        for t in periods
    )


def first_stage_decision(
    instance: Instance,
    solution: MILPSolution,
) -> FirstStageDecision:
    """MILP 해에서 공통 당기결정을 꺼낸다."""
    start = solution.start_period
    return FirstStageDecision(
        selected_d2c_skus=solution.selected_d2c_skus[start],
        d2c_quantity={
            i: solution.d2c_quantity[i, start] for i in instance.skus
        },
        retailer_quantity={
            (r, i): solution.retailer_quantity[r, i, start]
            for r, i in instance.retailer_sku_pairs
        },
    )


def _build_regret_model(
    instance: Instance,
    state: State,
    scenarios: list[dict],
    periods: tuple[int, ...],
    epsilon: float,
    config: SolverConfig,
    best_results: dict[str, ScenarioBestResult],
    fixed_decision: FirstStageDecision | None,
) -> tuple[gp.Model, dict[str, dict], gp.Var]:
    model = _new_model("minimax_relative_regret", config)
    plans = {}
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario["scenario_id"]
        rows = _scenario_rows(instance, scenario, periods)
        plans[scenario_id] = _add_scenario_plan(
            model,
            instance,
            state,
            periods,
            rows,
            prefix=f"s{index}",
        )

    start = state.period
    scenario_ids = list(plans)
    reference = plans[scenario_ids[0]]
    for scenario_id in scenario_ids[1:]:
        plan = plans[scenario_id]
        # 현재 기간의 세 결정만 모든 시나리오에서 동일하게 묶는다
        model.addConstrs(
            (plan["y"][i, start] == reference["y"][i, start] for i in instance.skus),
            name=f"nonanticipative_y[{scenario_id}]",
        )
        model.addConstrs(
            (plan["q"][i, start] == reference["q"][i, start] for i in instance.skus),
            name=f"nonanticipative_q[{scenario_id}]",
        )
        model.addConstrs(
            (
                plan["x"][r, i, start] == reference["x"][r, i, start]
                for r, i in instance.retailer_sku_pairs
            ),
            name=f"nonanticipative_x[{scenario_id}]",
        )

    if fixed_decision is not None:
        _fix_first_stage(model, instance, start, reference, fixed_decision)

    theta = model.addVar(lb=0.0, name="theta")
    for scenario_id in scenario_ids:
        best_profit = best_results[scenario_id].best_profit
        if not math.isfinite(best_profit):
            raise ValueError(f"best profit for {scenario_id} must be finite")
        denominator = max(epsilon, abs(best_profit))
        # z_s >= z_s* - theta |z_s*| 를 선형제약으로 넣는다
        model.addConstr(
            plans[scenario_id]["profit"]
            >= best_profit - theta * denominator,
            name=f"relative_regret[{scenario_id}]",
        )
    return model, plans, theta


def _extract_minimax_result(
    instance: Instance,
    periods: tuple[int, ...],
    scenario_ids: list[str],
    plans: dict[str, dict],
    best_results: dict[str, ScenarioBestResult],
    epsilon: float,
    model: gp.Model,
    status: str,
    theta: float,
) -> MinimaxRegretResult:
    scenario_results = {}
    for scenario_id in scenario_ids:
        plan = plans[scenario_id]
        policy_profit = float(plan["profit"].getValue())
        best_profit = best_results[scenario_id].best_profit
        absolute_regret = best_profit - policy_profit
        if abs(absolute_regret) <= 1e-7:
            absolute_regret = 0.0
        elif absolute_regret < 0:
            raise RuntimeError(
                f"policy profit exceeds benchmark for {scenario_id}"
            )
        relative_regret = absolute_regret / max(epsilon, abs(best_profit))
        solution = _extract_solution(
            instance,
            periods,
            plan,
            status,
            policy_profit,
            model.Runtime,
            model.NumVars,
            model.NumConstrs,
        )
        scenario_results[scenario_id] = ScenarioRegretResult(
            scenario_id=scenario_id,
            best_profit=best_profit,
            policy_profit=policy_profit,
            absolute_regret=absolute_regret,
            relative_regret=relative_regret,
            solution=solution,
        )

    reference_id = scenario_ids[0]
    common_decision = first_stage_decision(
        instance, scenario_results[reference_id].solution
    )
    worst_scenario = max(
        scenario_ids,
        key=lambda scenario_id: scenario_results[scenario_id].relative_regret,
    )
    return MinimaxRegretResult(
        status=status,
        theta=theta,
        start_period=periods[0],
        horizon=len(periods),
        common_decision=common_decision,
        scenario_results=scenario_results,
        worst_scenario=worst_scenario,
        runtime=model.Runtime,
        num_variables=model.NumVars,
        num_constraints=model.NumConstrs,
    )


def _add_scenario_plan(
    model: gp.Model,
    instance: Instance,
    state: State,
    periods: tuple[int, ...],
    rows: dict[int, dict],
    prefix: str = "",
) -> dict:
    skus, retailers = instance.skus, instance.retailers
    pairs = instance.retailer_sku_pairs
    retailers_of = {i: [r for r, j in pairs if j == i] for i in skus}
    name = f"{prefix}_" if prefix else ""

    y = model.addVars(skus, periods, vtype=GRB.BINARY, name=f"{name}y")
    q = model.addVars(skus, periods, lb=0.0, name=f"{name}q")
    x = model.addVars(
        [(r, i, t) for r, i in pairs for t in periods],
        lb=0.0,
        name=f"{name}x",
    )
    g = model.addVars(retailers, periods, lb=0.0, ub=1.0, name=f"{name}g")
    e = model.addVars(retailers, periods, lb=0.0, ub=1.0, name=f"{name}e")

    start = state.period
    profit = gp.quicksum(
        instance.gamma ** (t - start)
        * (
            gp.quicksum(rows[t]["d2c_margin"][i] * q[i, t] for i in skus)
            + gp.quicksum(
                rows[t]["retailer_margin"][r][i] * x[r, i, t]
                for r, i in pairs
            )
        )
        for t in periods
    )

    for t in periods:
        model.addConstrs(
            (
                q[i, t] <= rows[t]["d2c_demand"][i] * y[i, t]
                for i in skus
            ),
            name=f"{name}C1[{t}]",
        )
        model.addConstr(
            gp.quicksum(y[i, t] for i in skus) <= instance.max_d2c_skus,
            name=f"{name}C2[{t}]",
        )
        model.addConstrs(
            (
                x[r, i, t]
                <= rows[t]["retailer_base_demand"][r][i] * g[r, t]
                for r, i in pairs
            ),
            name=f"{name}C3[{t}]",
        )
        model.addConstrs(
            (
                q[i, t] + gp.quicksum(x[r, i, t] for r in retailers_of[i])
                <= rows[t]["supply_limit"][i]
                for i in skus
            ),
            name=f"{name}C4[{t}]",
        )
        model.addConstr(
            gp.quicksum(
                skus[i].capacity_use
                * (q[i, t] + gp.quicksum(x[r, i, t] for r in retailers_of[i]))
                for i in skus
            )
            <= rows[t]["capacity"],
            name=f"{name}C5[{t}]",
        )
        model.addConstrs(
            (
                e[r, t]
                == gp.quicksum(
                    instance.beta * y[i, t]
                    + (1.0 - instance.beta)
                    * q[i, t]
                    / rows[t]["d2c_demand"][i]
                    for i in retailers[r].base_orders
                )
                / len(retailers[r].base_orders)
                for r in retailers
            ),
            name=f"{name}C6[{t}]",
        )

    model.addConstrs(
        (g[r, start] == state.order_retention[r] for r in retailers),
        name=f"{name}g_observed",
    )
    model.addConstrs(
        (
            g[r, t + 1]
            == rows[t]["persistence"][r] * g[r, t]
            + (1.0 - rows[t]["persistence"][r])
            * (1.0 - rows[t]["response"][r] * e[r, t])
            for t in periods[:-1]
            for r in retailers
        ),
        name=f"{name}C7",
    )
    return {"y": y, "q": q, "x": x, "g": g, "e": e, "profit": profit}


def _scenario_rows(
    instance: Instance,
    scenario: dict | None,
    periods: tuple[int, ...],
) -> dict[int, dict]:
    if scenario is None:
        return {
            t: {
                "d2c_demand": {
                    i: sku.d2c_demand for i, sku in instance.skus.items()
                },
                "retailer_base_demand": {
                    r: dict(retailer.base_orders)
                    for r, retailer in instance.retailers.items()
                },
                "d2c_margin": {
                    i: sku.d2c_margin for i, sku in instance.skus.items()
                },
                "retailer_margin": {
                    r: dict(retailer.wholesale_margins)
                    for r, retailer in instance.retailers.items()
                },
                "d2c_fixed_cost": {i: 0.0 for i in instance.skus},
                "supply_limit": {
                    i: sku.supply_limit for i, sku in instance.skus.items()
                },
                "capacity": instance.capacity,
                "response": {
                    r: instance.response_for(r) for r in instance.retailers
                },
                "persistence": {r: instance.rho for r in instance.retailers},
            }
            for t in periods
        }

    validate_scenarios([scenario])
    available = {row["period"]: row for row in scenario["periods"]}
    if any(t not in available for t in periods):
        raise ValueError("scenario does not cover every active period")

    rows = {}
    for t in periods:
        row = available[t]
        _require_keys(row["d2c_demand"], instance.skus, "d2c_demand")
        _require_keys(row["d2c_margin"], instance.skus, "d2c_margin")
        _require_keys(row["d2c_fixed_cost"], instance.skus, "d2c_fixed_cost")
        _require_keys(
            row["retailer_base_demand"], instance.retailers, "retailer_base_demand"
        )
        _require_keys(row["retailer_margin"], instance.retailers, "retailer_margin")

        d2c_demand = {i: float(row["d2c_demand"][i]) for i in instance.skus}
        if any(value <= 0 for value in d2c_demand.values()):
            raise ValueError("d2c_demand values must be positive")
        fixed_cost = {i: float(row["d2c_fixed_cost"][i]) for i in instance.skus}
        if any(abs(value) > 1e-12 for value in fixed_cost.values()):
            raise ValueError("d2c_fixed_cost must remain zero")
        supply_limit = row.get(
            "supply_limit",
            {i: sku.supply_limit for i, sku in instance.skus.items()},
        )
        _require_keys(supply_limit, instance.skus, "supply_limit")
        supply_limit = {i: float(supply_limit[i]) for i in instance.skus}
        if any(
            not math.isfinite(value) or value < 0
            for value in supply_limit.values()
        ):
            raise ValueError("supply_limit values must be finite and nonnegative")
        capacity = float(row.get("capacity", instance.capacity))
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("capacity must be finite and positive")

        base_demand = {}
        retailer_margin = {}
        for r, retailer in instance.retailers.items():
            _require_keys(
                row["retailer_base_demand"][r],
                retailer.base_orders,
                f"retailer_base_demand[{r}]",
            )
            _require_keys(
                row["retailer_margin"][r],
                retailer.wholesale_margins,
                f"retailer_margin[{r}]",
            )
            base_demand[r] = {
                i: float(row["retailer_base_demand"][r][i])
                for i in retailer.base_orders
            }
            retailer_margin[r] = {
                i: float(row["retailer_margin"][r][i])
                for i in retailer.wholesale_margins
            }

        rows[t] = {
            "d2c_demand": d2c_demand,
            "retailer_base_demand": base_demand,
            "d2c_margin": {
                i: float(row["d2c_margin"][i]) for i in instance.skus
            },
            "retailer_margin": retailer_margin,
            "d2c_fixed_cost": fixed_cost,
            "supply_limit": supply_limit,
            "capacity": capacity,
            "response": _retailer_values(
                instance, row["response"], "response"
            ),
            "persistence": _retailer_values(
                instance, row["persistence"], "persistence"
            ),
        }
    return rows


def _validate_scenario_set(
    instance: Instance,
    state: State,
    scenarios: list[dict],
) -> tuple[int, ...]:
    _validate_state(instance, state)
    validate_scenarios(scenarios)
    starts = {scenario["start_period"] for scenario in scenarios}
    ends = {scenario["end_period"] for scenario in scenarios}
    if starts != {state.period} or len(ends) != 1:
        raise ValueError("all scenarios must share the current start and end period")
    end = ends.pop()
    if end > instance.periods:
        raise ValueError("scenario end lies outside the instance periods")
    periods = tuple(range(state.period, end + 1))
    observed = None
    observed_fields = (
        "d2c_demand",
        "retailer_base_demand",
        "d2c_margin",
        "retailer_margin",
        "d2c_fixed_cost",
        "supply_limit",
        "capacity",
    )
    for scenario in scenarios:
        rows = _scenario_rows(instance, scenario, periods)
        current = rows[state.period]
        if observed is None:
            observed = current
        elif any(current[field] != observed[field] for field in observed_fields):
            raise ValueError(
                "current-period market inputs must be common across scenarios"
            )
        retention = scenario.get("current_order_retention")
        if retention is not None:
            _require_keys(retention, instance.retailers, "current_order_retention")
            if any(
                not math.isclose(
                    float(retention[r]), state.order_retention[r], abs_tol=1e-9
                )
                for r in instance.retailers
            ):
                raise ValueError(
                    "current order retention must match the observed state"
                )
    return periods


def _retailer_values(instance: Instance, values, name: str) -> dict[str, float]:
    if isinstance(values, dict):
        _require_keys(values, instance.retailers, name)
        return {r: float(values[r]) for r in instance.retailers}
    value = float(values)
    return {r: value for r in instance.retailers}


def _require_keys(values: dict, expected: dict, name: str):
    if values.keys() != expected.keys():
        raise ValueError(f"{name} must cover exactly the expected keys")


def _validate_state(instance: Instance, state: State):
    if not 1 <= state.period <= instance.periods:
        raise ValueError("state.period lies outside the instance periods")
    if state.order_retention.keys() != instance.retailers.keys():
        raise ValueError("state.order_retention must cover exactly all retailers")


def _new_model(name: str, config: SolverConfig) -> gp.Model:
    model = gp.Model(name)
    model.Params.OutputFlag = int(config.output_flag)
    if config.time_limit is not None:
        model.Params.TimeLimit = config.time_limit
    if config.mip_gap is not None:
        model.Params.MIPGap = config.mip_gap
    return model


def _solution_status(model: gp.Model) -> str:
    status = STATUS_NAME.get(model.Status, str(model.Status))
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi finished with status {status} and no solution")
    return status


def _extract_solution(
    instance: Instance,
    periods: tuple[int, ...],
    plan: dict,
    status: str,
    objective_value: float,
    runtime: float,
    num_variables: int,
    num_constraints: int,
) -> MILPSolution:
    y, q, x, g, e = (
        plan["y"],
        plan["q"],
        plan["x"],
        plan["g"],
        plan["e"],
    )
    return MILPSolution(
        status=status,
        objective_value=float(objective_value),
        start_period=periods[0],
        horizon=len(periods),
        selected_d2c_skus={
            t: tuple(i for i in instance.skus if y[i, t].X > 0.5)
            for t in periods
        },
        d2c_quantity={
            (i, t): _clean_value(q[i, t].X)
            for i in instance.skus
            for t in periods
        },
        retailer_quantity={
            (r, i, t): _clean_value(x[r, i, t].X)
            for r, i in instance.retailer_sku_pairs
            for t in periods
        },
        exposure={
            (r, t): _clean_value(e[r, t].X)
            for r in instance.retailers
            for t in periods
        },
        order_retention={
            (r, t): _clean_value(g[r, t].X)
            for r in instance.retailers
            for t in periods
        },
        runtime=float(runtime),
        num_variables=int(num_variables),
        num_constraints=int(num_constraints),
    )


def _fix_first_stage(
    model: gp.Model,
    instance: Instance,
    start: int,
    plan: dict,
    decision: FirstStageDecision,
):
    selected = set(decision.selected_d2c_skus)
    if not selected <= instance.skus.keys():
        raise ValueError("fixed decision contains an unknown SKU")
    if len(selected) > instance.max_d2c_skus:
        raise ValueError("fixed decision selects too many D2C SKUs")
    _require_keys(decision.d2c_quantity, instance.skus, "fixed d2c_quantity")
    expected_pairs = dict.fromkeys(instance.retailer_sku_pairs)
    _require_keys(
        decision.retailer_quantity,
        expected_pairs,
        "fixed retailer_quantity",
    )
    if any(
        not math.isfinite(value) or value < 0
        for value in (
            list(decision.d2c_quantity.values())
            + list(decision.retailer_quantity.values())
        )
    ):
        raise ValueError("fixed quantities must be finite and nonnegative")

    model.addConstrs(
        (plan["y"][i, start] == int(i in selected) for i in instance.skus),
        name="fixed_y",
    )
    model.addConstrs(
        (
            plan["q"][i, start] == decision.d2c_quantity[i]
            for i in instance.skus
        ),
        name="fixed_q",
    )
    model.addConstrs(
        (
            plan["x"][r, i, start] == decision.retailer_quantity[r, i]
            for r, i in instance.retailer_sku_pairs
        ),
        name="fixed_x",
    )


def _clean_value(value: float) -> float:
    return 0.0 if abs(value) <= 1e-9 else float(value)
