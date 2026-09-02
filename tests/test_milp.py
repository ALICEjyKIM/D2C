import math

import pytest


gp = pytest.importorskip("gurobipy")

from src.instance import load_instance, make_initial_state  # noqa: E402
from src.milp import solve_milp  # noqa: E402


TOY_PATH = "configs/toy.json"
TOL = 1e-6


@pytest.fixture
def toy():
    return load_instance(TOY_PATH)


def solve(instance, horizon):
    try:
        return solve_milp(instance, make_initial_state(instance), horizon)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi license or runtime is unavailable: {exc}")


def test_horizon_one_solution_is_feasible_and_intuitive(toy):
    solution = solve(toy, horizon=1)

    assert solution.status == "OPTIMAL"
    assert math.isfinite(solution.objective_value)
    assert solution.horizon == 1
    # A와 B는 순기여이익이 가장 높고 현재 공급 조건에서도 선택 가능하다.
    assert set(solution.selected_d2c_skus[1]) == {"A", "B"}
    assert_feasible(toy, solution)


def test_horizon_three_transitions_and_feasibility(toy):
    solution = solve(toy, horizon=3)

    assert solution.status == "OPTIMAL"
    assert solution.horizon == 3
    assert solution.selected_d2c_skus.keys() == {1, 2, 3}
    assert_feasible(toy, solution)

    for t in (1, 2):
        for r in toy.retailers:
            expected = toy.rho * solution.order_retention[r, t] + (1.0 - toy.rho) * (
                1.0 - toy.kappa * solution.exposure[r, t]
            )
            got = solution.order_retention[r, t + 1]
            assert got == pytest.approx(expected, abs=TOL)


def assert_feasible(instance, solution):
    """반환된 해로 C1~C8을 다시 계산한다."""
    periods = range(solution.start_period, solution.start_period + solution.horizon)
    pairs = instance.retailer_sku_pairs
    retailers_of = {i: [r for r, j in pairs if j == i] for i in instance.skus}
    q, x, g, e = (
        solution.d2c_quantity,
        solution.retailer_quantity,
        solution.order_retention,
        solution.exposure,
    )

    for t in periods:
        selected = set(solution.selected_d2c_skus[t])
        assert len(selected) <= instance.max_d2c_skus

        for i, sku in instance.skus.items():
            shipped = q[i, t] + sum(x[r, i, t] for r in retailers_of[i])
            assert q[i, t] >= -TOL
            assert q[i, t] <= sku.d2c_demand * (i in selected) + TOL
            assert shipped <= sku.supply_limit + TOL

        for r, i in pairs:
            order = instance.retailers[r].base_orders[i] * g[r, t]
            assert -TOL <= x[r, i, t] <= order + TOL

        capacity_used = sum(
            sku.capacity_use * (q[i, t] + sum(x[r, i, t] for r in retailers_of[i]))
            for i, sku in instance.skus.items()
        )
        assert capacity_used <= instance.capacity + TOL

        for r in instance.retailers:
            assert -TOL <= e[r, t] <= 1.0
            assert -TOL <= g[r, t] <= 1.0 + TOL
