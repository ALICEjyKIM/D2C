import copy

import numpy as np
import pytest

import src.scenarios as scenario_module
from src.instance import load_instance
from src.scenarios import (
    build_regret_scenarios,
    generate_evaluation_paths,
    generate_planning_scenarios,
    sample_range,
    validate_scenarios,
)


TOY_PATH = "configs/toy.json"
TOL = 1e-12


@pytest.fixture
def toy():
    return load_instance(TOY_PATH)


@pytest.mark.parametrize("num_scenarios", [10, 30])
def test_requested_scenario_sizes_are_generated(toy, num_scenarios):
    scenarios = generate_planning_scenarios(toy, 1, 3, num_scenarios, 0.2, 17)

    assert len(scenarios) == num_scenarios
    assert validate_scenarios(scenarios)
    for scenario in scenarios:
        assert [row["period"] for row in scenario["periods"]] == [1, 2, 3]


def test_generated_values_stay_inside_requested_ranges(toy):
    uncertainty = 0.3
    scenarios = generate_planning_scenarios(toy, 1, 3, 10, uncertainty, 7)

    for scenario in scenarios:
        for row in scenario["periods"]:
            for i, sku in toy.skus.items():
                assert_in_range(row["d2c_demand"][i], sku.d2c_demand, uncertainty)
                assert_in_range(row["d2c_margin"][i], sku.d2c_margin, uncertainty)
                assert row["d2c_fixed_cost"][i] == 0.0
            for r, retailer in toy.retailers.items():
                assert 0.0 <= row["response"][r] <= 1.0
                assert_in_range(row["response"][r], toy.response_for(r), uncertainty)
                for i in retailer.base_orders:
                    assert_in_range(
                        row["retailer_base_demand"][r][i],
                        retailer.base_orders[i],
                        uncertainty,
                    )
                    assert_in_range(
                        row["retailer_margin"][r][i],
                        retailer.wholesale_margins[i],
                        uncertainty,
                    )
            for value in row["persistence"].values():
                assert 0.0 <= value <= 1.0
                assert_in_range(value, toy.rho, uncertainty)


def test_same_seed_repeats_and_different_seed_changes_paths(toy):
    first = generate_planning_scenarios(toy, 1, 3, 10, 0.2, 101)
    repeated = generate_planning_scenarios(toy, 1, 3, 10, 0.2, 101)
    changed = generate_planning_scenarios(toy, 1, 3, 10, 0.2, 102)

    assert first == repeated
    assert first != changed


def test_planning_and_evaluation_paths_are_separate(toy):
    planning = generate_planning_scenarios(toy, 1, 3, 10, 0.2, 11)
    evaluation = generate_evaluation_paths(toy, 1, 3, 10, 0.2, 29)

    assert all(scenario["kind"] == "planning" for scenario in planning)
    assert all(scenario["kind"] == "evaluation" for scenario in evaluation)
    assert planning[0]["scenario_id"] == "S001"
    assert evaluation[0]["scenario_id"] == "E001"
    assert planning != evaluation


def test_zero_uncertainty_preserves_base_values_and_instance(toy):
    original = copy.deepcopy(toy)
    scenarios = generate_planning_scenarios(toy, 2, 3, 10, 0.0, 3)

    assert toy == original
    for scenario in scenarios:
        assert [row["period"] for row in scenario["periods"]] == [2, 3]
        for row in scenario["periods"]:
            assert row["d2c_demand"] == {
                i: sku.d2c_demand for i, sku in toy.skus.items()
            }
            assert row["d2c_margin"] == {
                i: sku.d2c_margin for i, sku in toy.skus.items()
            }
            assert row["response"] == {
                r: toy.response_for(r) for r in toy.retailers
            }
            assert row["persistence"] == {r: toy.rho for r in toy.retailers}


def test_builds_three_explicit_regret_scenarios(toy):
    scenarios = build_regret_scenarios(toy)

    assert [scenario["scenario_id"] for scenario in scenarios] == [
        "D2C-friendly",
        "Balanced",
        "Relationship-sensitive",
    ]
    assert validate_scenarios(scenarios)
    assert scenarios[0]["periods"][0]["d2c_demand"]["A"] > toy.skus["A"].d2c_demand
    assert scenarios[2]["periods"][0]["response"] == {
        r: 0.6 for r in toy.retailers
    }


def test_sample_range_uses_each_latin_hypercube_stratum_once():
    samples = sample_range(2.0, 6.0, 10, np.random.default_rng(5))
    strata = np.floor((samples - 2.0) / 4.0 * 10).astype(int)

    assert sorted(strata) == list(range(10))


def test_numpy_fallback_is_reproducible(toy, monkeypatch):
    monkeypatch.setattr(scenario_module, "qmc", None)

    first = generate_planning_scenarios(toy, 1, 3, 10, 0.1, 23)
    repeated = generate_planning_scenarios(toy, 1, 3, 10, 0.1, 23)

    assert first == repeated
    assert validate_scenarios(first)


def test_invalid_generation_arguments_are_rejected(toy):
    with pytest.raises(ValueError, match="period range"):
        generate_planning_scenarios(toy, 0, 3, 10, 0.2, 1)
    with pytest.raises(ValueError, match="num_scenarios"):
        generate_planning_scenarios(toy, 1, 3, 0, 0.2, 1)
    with pytest.raises(ValueError, match="uncertainty"):
        generate_planning_scenarios(toy, 1, 3, 10, 1.1, 1)


def assert_in_range(value, base, uncertainty):
    assert base * (1.0 - uncertainty) - TOL <= value
    assert value <= base * (1.0 + uncertainty) + TOL
