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
from src.types import State


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
    assert planning[0]["periods"][0]["d2c_demand"] == {
        i: sku.d2c_demand for i, sku in toy.skus.items()
    }
    assert evaluation[0]["periods"][0]["d2c_demand"] != planning[0]["periods"][0][
        "d2c_demand"
    ]


def test_zero_uncertainty_preserves_base_values_and_instance(toy):
    original = copy.deepcopy(toy)
    state = State(
        period=2,
        order_retention={"R1": 0.91, "R2": 0.87, "R3": 0.95},
    )
    scenarios = generate_planning_scenarios(
        toy, 2, 3, 10, 0.0, 3, current_state=state
    )

    assert toy == original
    for scenario in scenarios:
        assert scenario["current_order_retention"] == state.order_retention
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
    state = State(
        period=1,
        order_retention={r: 1.0 for r in toy.retailers},
    )
    scenarios = build_regret_scenarios(toy, current_state=state)

    assert [scenario["scenario_id"] for scenario in scenarios] == [
        "D2C-friendly",
        "Balanced",
        "Relationship-sensitive",
    ]
    assert validate_scenarios(scenarios)
    current_rows = [scenario["periods"][0] for scenario in scenarios]
    observed_fields = (
        "supply_limit",
        "capacity",
        "d2c_demand",
        "retailer_base_demand",
        "d2c_margin",
        "retailer_margin",
        "d2c_fixed_cost",
    )
    assert all(
        all(row[field] == current_rows[0][field] for field in observed_fields)
        for row in current_rows[1:]
    )
    assert current_rows[0]["d2c_demand"]["A"] == toy.skus["A"].d2c_demand
    assert scenarios[0]["periods"][1]["d2c_demand"]["A"] > toy.skus["A"].d2c_demand
    assert scenarios[2]["periods"][0]["response"] == {
        r: 0.6 for r in toy.retailers
    }


def test_planning_scenarios_fix_observed_current_inputs(toy):
    state = State(
        period=1,
        order_retention={"R1": 0.92, "R2": 0.88, "R3": 0.96},
    )
    scenarios = generate_planning_scenarios(
        toy, 1, 3, 10, 0.3, 41, current_state=state
    )

    for scenario in scenarios:
        current = scenario["periods"][0]
        assert current["supply_limit"] == {
            i: sku.supply_limit for i, sku in toy.skus.items()
        }
        assert current["capacity"] == toy.capacity
        assert current["d2c_demand"] == {
            i: sku.d2c_demand for i, sku in toy.skus.items()
        }
        assert current["retailer_base_demand"] == {
            r: retailer.base_orders for r, retailer in toy.retailers.items()
        }
        assert current["d2c_margin"] == {
            i: sku.d2c_margin for i, sku in toy.skus.items()
        }
        assert current["retailer_margin"] == {
            r: retailer.wholesale_margins
            for r, retailer in toy.retailers.items()
        }
        assert scenario["current_order_retention"] == state.order_retention

    assert any(
        scenario["periods"][1]["d2c_demand"]
        != scenarios[0]["periods"][1]["d2c_demand"]
        for scenario in scenarios[1:]
    )


def test_planning_scenarios_copy_an_evaluation_path_observation(toy):
    evaluation = generate_evaluation_paths(toy, 2, 3, 1, 0.2, 19)[0]
    observation = evaluation["periods"][0]
    state = State(
        period=2,
        order_retention={"R1": 0.93, "R2": 0.89, "R3": 0.91},
    )
    scenarios = build_regret_scenarios(
        toy,
        start_period=2,
        end_period=3,
        current_state=state,
        current_observation=observation,
    )
    lhs_scenarios = generate_planning_scenarios(
        toy,
        2,
        3,
        3,
        0.2,
        23,
        current_state=state,
        current_observation=observation,
    )
    observed_fields = (
        "supply_limit",
        "capacity",
        "d2c_demand",
        "retailer_base_demand",
        "d2c_margin",
        "retailer_margin",
        "d2c_fixed_cost",
    )

    for scenario in scenarios + lhs_scenarios:
        current = scenario["periods"][0]
        assert all(current[field] == observation[field] for field in observed_fields)
        assert scenario["current_order_retention"] == state.order_retention

    assert scenarios[0]["periods"][0]["response"] != observation["response"]


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
