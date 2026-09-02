import pytest


gp = pytest.importorskip("gurobipy")

from src.instance import load_instance, make_initial_state  # noqa: E402
from src.milp import solve_milp  # noqa: E402
from src.simulator import run_simulation  # noqa: E402


TOY_PATH = "configs/toy.json"
TOL = 1e-6


@pytest.fixture
def toy():
    return load_instance(TOY_PATH)


def run(instance, policy):
    try:
        return run_simulation(instance, policy, instance.periods)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi license or runtime is unavailable: {exc}")


@pytest.mark.parametrize("policy", ["myopic", "lookahead"])
def test_simulation_runs_exactly_three_periods(toy, policy):
    result = run(toy, policy)

    assert toy.periods == 3
    assert result.policy == policy
    assert [p.period for p in result.periods] == [1, 2, 3]


@pytest.mark.parametrize("policy", ["myopic", "lookahead"])
def test_initial_order_state_is_one(toy, policy):
    first = run(toy, policy).periods[0]

    assert first.order_retention == {r: 1.0 for r in toy.retailers}


@pytest.mark.parametrize("policy", ["myopic", "lookahead"])
def test_exposure_and_order_state_stay_in_unit_interval(toy, policy):
    for period in run(toy, policy).periods:
        for r in toy.retailers:
            assert -TOL <= period.exposure[r] <= 1.0 + TOL
            assert -TOL <= period.order_retention[r] <= 1.0 + TOL


@pytest.mark.parametrize("policy", ["myopic", "lookahead"])
def test_total_profit_is_sum_of_period_profits(toy, policy):
    result = run(toy, policy)

    expected = sum(period.total_profit for period in result.periods)
    assert result.total_profit == pytest.approx(expected, abs=1e-4)
    assert result.cumulative_profit == pytest.approx(expected, abs=1e-4)


def test_only_current_period_decision_is_executed(toy, monkeypatch):
    import src.simulator as simulator

    calls = []
    original_solve = simulator.solve_milp

    def tracked_solve(instance, state, horizon, config):
        calls.append((state.period, horizon))
        return original_solve(instance, state, horizon, config)

    monkeypatch.setattr(simulator, "solve_milp", tracked_solve)
    result = run(toy, "lookahead")

    assert calls == [(1, 3), (2, 2), (3, 1)]
    assert len(result.periods) == 3
    for period in result.periods:
        assert set(period.d2c_quantity) == set(toy.skus)
        assert set(period.retailer_quantity) == set(toy.retailer_sku_pairs)


@pytest.mark.parametrize("policy", ["myopic", "lookahead"])
def test_exposure_updates_next_period_order_state(toy, policy):
    periods = run(toy, policy).periods

    assert any(value > TOL for value in periods[0].exposure.values())
    for current, following in zip(periods, periods[1:]):
        for r in toy.retailers:
            expected = toy.rho * current.order_retention[r] + (1.0 - toy.rho) * (
                1.0 - toy.kappa * current.exposure[r]
            )
            assert following.order_retention[r] == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("policy", ["myopic", "lookahead"])
def test_same_input_produces_same_result(toy, policy):
    assert run(toy, policy) == run(toy, policy)


def test_first_myopic_decision_matches_single_period_milp(toy):
    result = run(toy, "myopic")
    plan = solve_milp(toy, make_initial_state(toy), horizon=1)
    first = result.periods[0]

    assert first.selected_d2c_skus == plan.selected_d2c_skus[1]
    assert first.d2c_quantity == {i: plan.d2c_quantity[i, 1] for i in toy.skus}
    assert first.retailer_quantity == {
        (r, i): plan.retailer_quantity[r, i, 1]
        for r, i in toy.retailer_sku_pairs
    }


def test_realized_decisions_respect_period_constraints(toy):
    retailers_of = {
        i: [r for r, j in toy.retailer_sku_pairs if j == i]
        for i in toy.skus
    }

    for period in run(toy, "lookahead").periods:
        selected = set(period.selected_d2c_skus)
        assert len(selected) <= toy.max_d2c_skus

        for i, sku in toy.skus.items():
            shipped = period.d2c_quantity[i] + sum(
                period.retailer_quantity[r, i] for r in retailers_of[i]
            )
            assert -TOL <= period.d2c_quantity[i]
            assert period.d2c_quantity[i] <= sku.d2c_demand * (i in selected) + TOL
            assert shipped <= sku.supply_limit + TOL

        for r, i in toy.retailer_sku_pairs:
            order = toy.retailers[r].base_orders[i] * period.order_retention[r]
            assert -TOL <= period.retailer_quantity[r, i] <= order + TOL

        assert period.capacity_used <= toy.capacity + TOL


def test_discounted_profit_is_preserved(toy):
    result = run(toy, "lookahead")
    expected = sum(
        toy.gamma ** (period.period - 1) * period.total_profit
        for period in result.periods
    )

    assert result.discounted_profit == pytest.approx(expected, abs=1e-4)


def test_toy_uses_same_cumulative_and_planning_profit_basis(toy):
    result = run(toy, "lookahead")

    assert toy.gamma == 1.0
    assert result.discounted_profit == pytest.approx(
        result.cumulative_profit, abs=1e-4
    )


def test_invalid_policy_is_rejected(toy):
    with pytest.raises(ValueError, match="policy"):
        run_simulation(toy, "unknown", toy.periods)
