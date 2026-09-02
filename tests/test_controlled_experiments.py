import copy
from collections import Counter

import pytest

import src.controlled_experiments as controlled
from src.controlled_experiments import (
    build_controlled_cases,
    load_experiment_levels,
    run_controlled_experiments,
    run_heterogeneity_experiments,
    save_experiment_results,
)
from src.instance import load_instance
from src.types import PeriodResult, SimulationResult


TOY_PATH = "configs/toy.json"
TOL = 1e-6


@pytest.fixture
def toy():
    return load_instance(TOY_PATH)


@pytest.fixture
def levels():
    return load_experiment_levels()


def test_builds_exactly_27_controlled_cases_without_mutating_source(toy, levels):
    original = copy.deepcopy(toy)
    cases = build_controlled_cases(toy, levels)

    assert len(cases) == 27
    assert len({case["case_id"] for case in cases}) == 27
    assert {
        (case["margin_level"], case["response"], case["persistence"])
        for case in cases
    } == {
        (margin, response, persistence)
        for margin in levels["margin_levels"]
        for response in levels["response_levels"]
        for persistence in levels["persistence_levels"]
    }
    assert toy == original


def test_margin_levels_use_average_retailer_margin(toy, levels):
    cases = build_controlled_cases(toy, levels)

    for case in cases:
        level = case["margin_level"]
        for i, sku in toy.skus.items():
            retailer_margins = [
                retailer.wholesale_margins[i]
                for retailer in toy.retailers.values()
                if i in retailer.wholesale_margins
            ]
            average = sum(retailer_margins) / len(retailer_margins)
            expected = average + level * (sku.d2c_margin - average)
            assert case["instance"].skus[i].d2c_margin == pytest.approx(expected)


def test_each_case_runs_each_policy_once_with_consistent_parameters(
    toy, levels, monkeypatch
):
    cases = build_controlled_cases(toy, levels)
    calls = []

    def fake_run(instance, policy, periods):
        calls.append((instance.instance_id, policy, periods, instance.kappa, instance.rho))
        return fake_result(instance, policy)

    monkeypatch.setattr(controlled, "run_simulation", fake_run)
    summary_rows, period_rows = run_controlled_experiments(cases)

    assert len(summary_rows) == 54
    assert len(period_rows) == 162
    assert Counter((row["case_id"], row["policy"]) for row in summary_rows) == {
        (case["case_id"], policy): 1
        for case in cases
        for policy in controlled.POLICIES
    }
    for instance_id, _, periods, response, persistence in calls:
        case = next(case for case in cases if case["instance"].instance_id == instance_id)
        assert periods == 3
        assert response == case["response"]
        assert persistence == case["persistence"]


def test_heterogeneity_keeps_average_response_at_point_four(
    toy, levels, monkeypatch
):
    seen = []

    def fake_run(instance, policy, periods):
        seen.append((instance.retailer_responses, policy))
        return fake_result(instance, policy)

    monkeypatch.setattr(controlled, "run_simulation", fake_run)
    rows = run_heterogeneity_experiments(toy, levels["heterogeneity_levels"])

    assert len(rows) == 6
    for responses, _ in seen:
        assert sum(responses.values()) / len(responses) == pytest.approx(0.4)
    for row in rows:
        assert row["average_response"] == pytest.approx(0.4)
        for r in toy.retailers:
            assert f"retailer_quantity_{r}" in row
            assert f"average_exposure_{r}" in row
            assert f"final_order_state_{r}" in row


def test_experiment_results_are_saved_as_separate_csv_files(
    toy, levels, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        controlled,
        "run_simulation",
        lambda instance, policy, periods: fake_result(instance, policy),
    )
    cases = build_controlled_cases(toy, levels)
    summary_rows, period_rows = run_controlled_experiments(cases[:1])
    heterogeneity_rows = run_heterogeneity_experiments(
        toy, levels["heterogeneity_levels"][:1]
    )

    paths = save_experiment_results(
        summary_rows, period_rows, heterogeneity_rows, tmp_path
    )

    assert {path.name for path in paths.values()} == {
        "controlled_summary.csv",
        "controlled_periods.csv",
        "heterogeneity_summary.csv",
    }
    assert all(path.read_text(encoding="utf-8").count("\n") > 1 for path in paths.values())


def test_real_controlled_results_satisfy_requested_invariants(toy, levels):
    gp = pytest.importorskip("gurobipy")
    cases = build_controlled_cases(toy, levels)
    try:
        summary_rows, period_rows = run_controlled_experiments(cases)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi license or runtime is unavailable: {exc}")

    assert len(summary_rows) == 54
    assert len(period_rows) == 162
    summaries = {(row["case_id"], row["policy"]): row for row in summary_rows}
    for case in cases:
        for policy in controlled.POLICIES:
            key = (case["case_id"], policy)
            rows = [
                row
                for row in period_rows
                if (row["case_id"], row["policy"]) == key
            ]
            assert len(rows) == 3
            assert sum(row["period_profit"] for row in rows) == pytest.approx(
                summaries[key]["total_profit"], abs=1e-4
            )
            for row in rows:
                assert float(row["response"]) == case["response"]
                assert float(row["persistence"]) == case["persistence"]
                assert row["capacity_used"] <= case["instance"].capacity + TOL
                for r in toy.retailers:
                    assert -TOL <= row[f"exposure_{r}"] <= 1.0 + TOL
                    assert -TOL <= row[f"order_state_{r}"] <= 1.0 + TOL


def fake_result(instance, policy):
    periods = []
    for period in range(1, instance.periods + 1):
        period_profit = float(period)
        periods.append(
            PeriodResult(
                period=period,
                selected_d2c_skus=(),
                d2c_quantity={i: 0.0 for i in instance.skus},
                retailer_quantity={pair: 0.0 for pair in instance.retailer_sku_pairs},
                d2c_profit=0.0,
                wholesale_profit=period_profit,
                total_profit=period_profit,
                supply_slack={i: sku.supply_limit for i, sku in instance.skus.items()},
                capacity_used=0.0,
                capacity_utilization=0.0,
                exposure={r: 0.0 for r in instance.retailers},
                order_retention={r: 1.0 for r in instance.retailers},
            )
        )
    total = sum(row.total_profit for row in periods)
    return SimulationResult(
        instance_id=instance.instance_id,
        policy=policy,
        planning_horizon=1 if policy == "myopic" else instance.periods,
        periods=periods,
        cumulative_profit=total,
        discounted_profit=total,
    )
