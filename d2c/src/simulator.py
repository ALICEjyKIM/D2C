"""Rolling-horizon simulation of a myopic or look-ahead planning policy."""

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
    """Run all instance.periods, re-planning each period and executing its first."""
    state = make_initial_state(instance)
    periods = []
    while state.period <= instance.periods:
        t = state.period
        # solve_milp truncates the look-ahead window at the end of the instance.
        solution = solve_milp(instance, state, planning_horizon, config)
        periods.append(period_result(instance, solution, t))
        exposure = {r: solution.exposure[r, t] for r in instance.retailers}
        state = next_state(instance, state, exposure)

    gamma = instance.gamma
    return SimulationResult(
        instance_id=instance.instance_id,
        planning_horizon=planning_horizon,
        periods=periods,
        cumulative_profit=sum(p.total_profit for p in periods),
        discounted_profit=sum(gamma ** (p.period - 1) * p.total_profit for p in periods),
    )


def period_result(instance: Instance, solution: MILPSolution, t: int) -> PeriodResult:
    """Profit and slack accounting for period t of a solved (look-ahead) plan."""
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
