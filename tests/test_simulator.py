import pytest


gp = pytest.importorskip("gurobipy")

from src.instance import load_instance, make_initial_state  # noqa: E402
from src.milp import solve_milp  # noqa: E402
from src.simulator import simulate  # noqa: E402


TOY_PATH = "configs/toy.json"
TOL = 1e-6


@pytest.fixture
def toy():
    return load_instance(TOY_PATH)


def run(instance, horizon):
    try:
        return simulate(instance, horizon)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi license or runtime is unavailable: {exc}")


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_simulation_runs_every_period_once(toy, horizon):
    result = run(toy, horizon)

    assert [p.period for p in result.periods] == list(range(1, toy.periods + 1))


def test_last_period_look_ahead_is_truncated(toy):
    # H = 6 starting at period 6 must collapse to a single active period.
    result = run(toy, horizon=6)
    last = result.periods[-1]

    assert last.period == toy.periods
    assert set(last.selected_d2c_skus) <= set(toy.skus)


def test_first_executed_period_equals_the_myopic_plan(toy):
    result = run(toy, horizon=1)
    plan = solve_milp(toy, make_initial_state(toy), horizon=1)
    first = result.periods[0]

    assert set(first.selected_d2c_skus) == set(plan.selected_d2c_skus[1])
    # gamma^0 = 1, so the single-period plan objective is the realized profit.
    assert first.total_profit == pytest.approx(plan.objective_value, abs=1e-4)


def test_exposure_and_retention_stay_in_unit_interval(toy):
    for p in run(toy, horizon=3).periods:
        for r in toy.retailers:
            assert -TOL <= p.exposure[r] <= 1.0 + TOL
            assert -TOL <= p.order_retention[r] <= 1.0 + TOL


def test_realized_retention_follows_c7_between_periods(toy):
    periods = run(toy, horizon=3).periods
    for prev, cur in zip(periods, periods[1:]):
        for r in toy.retailers:
            expected = toy.rho * prev.order_retention[r] + (1.0 - toy.rho) * (
                1.0 - toy.kappa * prev.exposure[r]
            )
            assert cur.order_retention[r] == pytest.approx(expected, abs=TOL)


def test_realized_decisions_respect_period_constraints(toy):
    pairs = toy.retailer_sku_pairs
    retailers_of = {i: [r for r, j in pairs if j == i] for i in toy.skus}

    for p in run(toy, horizon=3).periods:
        selected = set(p.selected_d2c_skus)
        assert len(selected) <= toy.max_d2c_skus

        for i, sku in toy.skus.items():
            shipped = p.d2c_quantity[i] + sum(p.retailer_quantity[r, i] for r in retailers_of[i])
            assert -TOL <= p.d2c_quantity[i] <= sku.d2c_demand * (i in selected) + TOL
            assert shipped <= sku.supply_limit + TOL
            assert p.supply_slack[i] == pytest.approx(sku.supply_limit - shipped, abs=TOL)

        for r, i in pairs:
            bound = toy.retailers[r].base_orders[i] * p.order_retention[r]
            assert -TOL <= p.retailer_quantity[r, i] <= bound + TOL

        assert p.capacity_used <= toy.capacity + TOL


def test_cumulative_and_discounted_profit_aggregation(toy):
    result = run(toy, horizon=3)

    undiscounted = sum(p.total_profit for p in result.periods)
    discounted = sum(
        toy.gamma ** (p.period - 1) * p.total_profit for p in result.periods
    )
    assert result.cumulative_profit == pytest.approx(undiscounted, abs=1e-4)
    assert result.discounted_profit == pytest.approx(discounted, abs=1e-4)
    # Discounting with gamma < 1 can only shrink a stream of positive profits.
    assert result.discounted_profit < result.cumulative_profit
    for p in result.periods:
        assert p.total_profit == pytest.approx(p.d2c_profit + p.wholesale_profit, abs=1e-4)
