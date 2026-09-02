"""Myopic과 look-ahead 정책을 rolling-horizon으로 실행한다."""

from src.instance import make_initial_state
from src.milp import solve_milp
from src.transition import next_state
from src.types import (
    Instance,
    MILPSolution,
    PeriodResult,
    SimulationResult,
    SolverConfig,
)


def simulate(
    instance: Instance,
    planning_horizon: int,
    config: SolverConfig = SolverConfig(),
) -> SimulationResult:
    """기존 planning_horizon 호출 방식을 유지한다."""
    policy = "myopic" if planning_horizon == 1 else "lookahead"
    return _run(instance, policy, instance.periods, planning_horizon, config)


def run_simulation(
    instance: Instance,
    policy: str,
    periods: int | None = None,
    config: SolverConfig = SolverConfig(),
) -> SimulationResult:
    """Myopic 또는 look-ahead 정책을 rolling-horizon으로 실행한다."""
    if policy not in {"myopic", "lookahead"}:
        raise ValueError("policy must be 'myopic' or 'lookahead'")

    periods = instance.periods if periods is None else periods
    if not 1 <= periods <= instance.periods:
        raise ValueError("periods must be between 1 and instance.periods")

    planning_horizon = 1 if policy == "myopic" else periods
    return _run(instance, policy, periods, planning_horizon, config)


def _run(
    instance: Instance,
    policy: str,
    periods: int,
    planning_horizon: int,
    config: SolverConfig,
) -> SimulationResult:
    state = make_initial_state(instance)
    results = []
    while state.period <= periods:
        t = state.period
        horizon = min(planning_horizon, periods - t + 1)
        solution = solve_milp(instance, state, horizon, config)

        # 이번 기간에 실행할 배분만 가져온다
        results.append(period_result(instance, solution, t))
        exposure = {r: solution.exposure[r, t] for r in instance.retailers}

        # D2C 노출을 반영해 다음 기간 주문상태를 갱신한다
        state = next_state(instance, state, exposure)

    gamma = instance.gamma
    return SimulationResult(
        instance_id=instance.instance_id,
        policy=policy,
        planning_horizon=planning_horizon,
        periods=results,
        cumulative_profit=sum(p.total_profit for p in results),
        discounted_profit=sum(gamma ** (p.period - 1) * p.total_profit for p in results),
    )


def period_result(instance: Instance, solution: MILPSolution, t: int) -> PeriodResult:
    """계획에서 기간 t의 실행 결과와 이익을 따로 정리한다."""
    skus, retailers = instance.skus, instance.retailers
    pairs = instance.retailer_sku_pairs
    retailers_of = {i: [r for r, j in pairs if j == i] for i in skus}
    q, x = solution.d2c_quantity, solution.retailer_quantity

    d2c_profit = sum(skus[i].d2c_margin * q[i, t] for i in skus)
    wholesale_profit = sum(retailers[r].wholesale_margins[i] * x[r, i, t] for r, i in pairs)
    shipped = {i: q[i, t] + sum(x[r, i, t] for r in retailers_of[i]) for i in skus}
    capacity_used = sum(skus[i].capacity_use * shipped[i] for i in skus)

    return PeriodResult(
        period=t,
        selected_d2c_skus=solution.selected_d2c_skus[t],
        d2c_quantity={i: q[i, t] for i in skus},
        retailer_quantity={(r, i): x[r, i, t] for r, i in pairs},
        d2c_profit=d2c_profit,
        wholesale_profit=wholesale_profit,
        total_profit=d2c_profit + wholesale_profit,
        supply_slack={i: skus[i].supply_limit - shipped[i] for i in skus},
        capacity_used=capacity_used,
        capacity_utilization=capacity_used / instance.capacity,
        exposure={r: solution.exposure[r, t] for r in retailers},
        order_retention={r: solution.order_retention[r, t] for r in retailers},
    )
