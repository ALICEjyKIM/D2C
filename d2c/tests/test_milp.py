"""Feasibility and dynamics tests for the Gurobi model."""

import math
from pathlib import Path

import pytest


gp = pytest.importorskip("gurobipy")

from src.instance import load_instance, make_initial_state  # noqa: E402
from src.milp import solve_milp  # noqa: E402
from src.types import Instance, MILPSolution, SolverConfig  # noqa: E402


TOY_PATH = Path(__file__).resolve().parents[1] / "configs" / "toy.json"
TOLERANCE = 1e-6


def test_horizon_one_solution_is_feasible_and_intuitive() -> None:
    instance = load_instance(TOY_PATH)
    solution = _solve_or_skip(instance, horizon=1)

    assert solution.status == "OPTIMAL"
    assert math.isfinite(solution.objective_value)
    assert solution.horizon == 1
    assert set(solution.selected_d2c_skus[1]) == {"A", "B"}
    _assert_solution_feasible(instance, solution)


def test_horizon_three_transitions_and_feasibility() -> None:
    instance = load_instance(TOY_PATH)
    solution = _solve_or_skip(instance, horizon=3)

    assert solution.status == "OPTIMAL"
    assert solution.horizon == 3
    assert solution.selected_d2c_skus.keys() == {1, 2, 3}
    _assert_solution_feasible(instance, solution)

    for period in (1, 2):
        for r in instance.retailer_ids:
            expected_retention = (
                instance.rho * solution.order_retention[r, period]
                + (1.0 - instance.rho)
                * (1.0 - instance.kappa * solution.exposure[r, period])
            )
            assert solution.order_retention[r, period + 1] == pytest.approx(
                expected_retention, abs=TOLERANCE
            )


def _solve_or_skip(instance: Instance, horizon: int) -> MILPSolution:
    try:
        return solve_milp(
            instance=instance,
            state=make_initial_state(instance),
            start_period=1,
            horizon=horizon,
            solver_config=SolverConfig(output_flag=False),
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi license or runtime is unavailable: {exc}")


def _assert_solution_feasible(
    instance: Instance,
    solution: MILPSolution,
) -> None:
    periods = range(
        solution.start_period,
        solution.start_period + solution.horizon,
    )
    pairs = instance.feasible_retailer_sku_pairs
    retailers_by_sku = {
        i: tuple(r for r, j in pairs if j == i) for i in instance.sku_ids
    }
    q = solution.d2c_quantity
    x = solution.retailer_quantity
    g = solution.order_retention
    e = solution.exposure

    for t in periods:
        selected = set(solution.selected_d2c_skus[t])
        assert len(selected) <= instance.max_d2c_skus

        for i, sku in instance.skus.items():
            y_it = float(i in selected)
            assert q[i, t] >= -TOLERANCE
            assert q[i, t] <= sku.d2c_demand * y_it + TOLERANCE
            assert (
                q[i, t] + sum(x[r, i, t] for r in retailers_by_sku[i])
                <= sku.supply_limit + TOLERANCE
            )

        for r, i in pairs:
            actual_order = instance.retailers[r].base_orders[i] * g[r, t]
            assert x[r, i, t] >= -TOLERANCE
            assert x[r, i, t] <= actual_order + TOLERANCE

        capacity_used = sum(
            sku.capacity_use
            * (
                q[i, t] + sum(x[r, i, t] for r in retailers_by_sku[i])
            )
            for i, sku in instance.skus.items()
        )
        assert capacity_used <= instance.capacity + TOLERANCE

        for r in instance.retailer_ids:
            assert -TOLERANCE <= e[r, t] <= 1.0
            assert -TOLERANCE <= g[r, t] <= 1.0 + TOLERANCE
