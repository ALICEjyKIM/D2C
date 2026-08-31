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
        for retailer_id in instance.retailer_ids:
            expected_retention = (
                instance.rho * solution.order_retention[retailer_id, period]
                + (1.0 - instance.rho)
                * (1.0 - instance.kappa * solution.exposure[retailer_id, period])
            )
            assert solution.order_retention[
                retailer_id, period + 1
            ] == pytest.approx(expected_retention, abs=TOLERANCE)


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
    active_periods = range(
        solution.start_period,
        solution.start_period + solution.horizon,
    )
    for period in active_periods:
        selected = set(solution.selected_d2c_skus[period])
        assert len(selected) <= instance.max_d2c_skus

        for sku_id, sku in instance.skus.items():
            d2c_quantity = solution.d2c_quantity[sku_id, period]
            listing = float(sku_id in selected)
            assert d2c_quantity >= -TOLERANCE
            assert d2c_quantity <= sku.d2c_demand * listing + TOLERANCE

            retailer_total = sum(
                solution.retailer_quantity[retailer_id, pair_sku_id, period]
                for retailer_id, pair_sku_id
                in instance.feasible_retailer_sku_pairs
                if pair_sku_id == sku_id
            )
            assert d2c_quantity + retailer_total <= sku.supply_limit + TOLERANCE

        for retailer_id, sku_id in instance.feasible_retailer_sku_pairs:
            allocation = solution.retailer_quantity[retailer_id, sku_id, period]
            actual_order = (
                instance.retailers[retailer_id].base_orders[sku_id]
                * solution.order_retention[retailer_id, period]
            )
            assert allocation >= -TOLERANCE
            assert allocation <= actual_order + TOLERANCE

        capacity_used = sum(
            sku.capacity_use
            * (
                solution.d2c_quantity[sku_id, period]
                + sum(
                    solution.retailer_quantity[retailer_id, pair_sku_id, period]
                    for retailer_id, pair_sku_id
                    in instance.feasible_retailer_sku_pairs
                    if pair_sku_id == sku_id
                )
            )
            for sku_id, sku in instance.skus.items()
        )
        assert capacity_used <= instance.capacity + TOLERANCE

        for retailer_id in instance.retailer_ids:
            assert -TOLERANCE <= solution.exposure[retailer_id, period] <= 1.0
            assert (
                -TOLERANCE
                <= solution.order_retention[retailer_id, period]
                <= 1.0 + TOLERANCE
            )
