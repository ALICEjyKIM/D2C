import pytest


gp = pytest.importorskip("gurobipy")

from src.instance import load_instance, make_initial_state  # noqa: E402
from src.milp import (  # noqa: E402
    calculate_scenario_best_profits,
    calculate_scenario_profit,
    first_stage_decision,
    solve_minimax_relative_regret,
)
from src.scenarios import (  # noqa: E402
    build_regret_scenarios,
    generate_evaluation_paths,
)


TOY_PATH = "configs/toy.json"
TOL = 1e-6


@pytest.fixture(scope="module")
def solved():
    instance = load_instance(TOY_PATH)
    state = make_initial_state(instance)
    scenarios = build_regret_scenarios(instance, current_state=state)
    try:
        best_results = calculate_scenario_best_profits(instance, state, scenarios)
        result = solve_minimax_relative_regret(
            instance,
            state,
            scenarios,
            best_results=best_results,
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi license or runtime is unavailable: {exc}")
    return instance, state, scenarios, best_results, result


def test_scenario_values_change_objective_and_constraints(solved):
    instance, _, scenarios, best_results, _ = solved
    friendly = best_results["D2C-friendly"].solution
    balanced = best_results["Balanced"].solution

    assert friendly.objective_value > balanced.objective_value
    assert sum(friendly.d2c_quantity[i, 2] for i in instance.skus) > sum(
        balanced.d2c_quantity[i, 2] for i in instance.skus
    )
    for scenario in scenarios:
        rows = {row["period"]: row for row in scenario["periods"]}
        solution = best_results[scenario["scenario_id"]].solution
        for t in rows:
            for i in instance.skus:
                assert solution.d2c_quantity[i, t] <= rows[t]["d2c_demand"][i] + TOL


def test_single_scenario_has_zero_relative_regret(solved):
    instance, state, scenarios, _, _ = solved
    balanced = [scenario for scenario in scenarios if scenario["scenario_id"] == "Balanced"]
    best = calculate_scenario_best_profits(instance, state, balanced)
    result = solve_minimax_relative_regret(
        instance, state, balanced, best_results=best
    )

    assert result.theta == pytest.approx(0.0, abs=TOL)
    assert result.scenario_results["Balanced"].relative_regret == pytest.approx(
        0.0, abs=TOL
    )


def test_solver_accepts_a_common_realized_current_observation(solved):
    instance, state, _, _, _ = solved
    observation = generate_evaluation_paths(instance, 1, 3, 1, 0.1, 37)[0][
        "periods"
    ][0]
    scenarios = build_regret_scenarios(
        instance,
        current_state=state,
        current_observation=observation,
    )
    balanced = [
        scenario for scenario in scenarios
        if scenario["scenario_id"] == "Balanced"
    ]
    best = calculate_scenario_best_profits(instance, state, balanced)

    assert best["Balanced"].solution.status == "OPTIMAL"
    assert scenarios[0]["periods"][0]["d2c_demand"] != {
        i: sku.d2c_demand for i, sku in instance.skus.items()
    }


def test_current_decisions_are_common_across_scenarios(solved):
    instance, _, _, _, result = solved
    decisions = [
        first_stage_decision(instance, row.solution)
        for row in result.scenario_results.values()
    ]

    assert all(decision == decisions[0] for decision in decisions[1:])
    assert result.common_decision == decisions[0]


def test_current_inputs_and_order_retention_are_common(solved):
    instance, state, scenarios, _, _ = solved
    current_rows = [scenario["periods"][0] for scenario in scenarios]

    for current in current_rows:
        assert current["supply_limit"] == {
            i: sku.supply_limit for i, sku in instance.skus.items()
        }
        assert current["capacity"] == instance.capacity
        assert current["d2c_demand"] == {
            i: sku.d2c_demand for i, sku in instance.skus.items()
        }
        assert current["retailer_base_demand"] == {
            r: retailer.base_orders for r, retailer in instance.retailers.items()
        }
        assert current["d2c_margin"] == {
            i: sku.d2c_margin for i, sku in instance.skus.items()
        }
        assert current["retailer_margin"] == {
            r: retailer.wholesale_margins
            for r, retailer in instance.retailers.items()
        }
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
        all(current[field] == current_rows[0][field] for field in observed_fields)
        for current in current_rows[1:]
    )
    assert all(
        scenario["current_order_retention"] == state.order_retention
        for scenario in scenarios
    )


def test_future_inputs_diverge_and_response_changes_next_state(solved):
    instance, _, scenarios, _, result = solved
    future_rows = {
        scenario["scenario_id"]: scenario["periods"][1]
        for scenario in scenarios
    }

    assert future_rows["D2C-friendly"]["d2c_demand"] != future_rows["Balanced"][
        "d2c_demand"
    ]
    assert future_rows["D2C-friendly"]["d2c_margin"] != future_rows["Balanced"][
        "d2c_margin"
    ]
    next_states = {
        scenario_id: tuple(
            row.solution.order_retention[r, 2] for r in instance.retailers
        )
        for scenario_id, row in result.scenario_results.items()
    }
    assert len(set(next_states.values())) == 3


def test_future_plans_can_differ_by_scenario(solved):
    instance, _, _, _, result = solved
    future_quantities = {
        scenario_id: tuple(
            row.solution.d2c_quantity[i, t]
            for t in (2, 3)
            for i in instance.skus
        )
        for scenario_id, row in result.scenario_results.items()
    }

    assert len(set(future_quantities.values())) > 1


def test_reported_scenario_profits_match_direct_calculation(solved):
    instance, _, scenarios, _, result = solved

    assert instance.gamma == 1.0
    for scenario in scenarios:
        row = result.scenario_results[scenario["scenario_id"]]
        calculated = calculate_scenario_profit(instance, scenario, row.solution)
        periods = {period["period"]: period for period in scenario["periods"]}
        cumulative = sum(
            sum(
                periods[t]["d2c_margin"][i]
                * row.solution.d2c_quantity[i, t]
                for i in instance.skus
            )
            + sum(
                periods[t]["retailer_margin"][r][i]
                * row.solution.retailer_quantity[r, i, t]
                for r, i in instance.retailer_sku_pairs
            )
            for t in periods
        )
        assert row.policy_profit == pytest.approx(calculated, abs=1e-5)
        assert row.policy_profit == pytest.approx(cumulative, abs=1e-5)
        assert row.solution.objective_value == pytest.approx(calculated, abs=1e-5)
        benchmark = calculate_scenario_profit(
            instance,
            scenario,
            solved[3][scenario["scenario_id"]].solution,
        )
        assert solved[3][scenario["scenario_id"]].best_profit == pytest.approx(
            benchmark, abs=1e-5
        )


def test_theta_matches_direct_maximum_relative_regret(solved):
    _, _, _, _, result = solved
    direct = max(row.relative_regret for row in result.scenario_results.values())

    assert result.theta == pytest.approx(direct, abs=TOL)
    assert result.worst_scenario == max(
        result.scenario_results,
        key=lambda scenario_id: result.scenario_results[scenario_id].relative_regret,
    )


def test_minimax_regret_is_no_worse_than_balanced_nominal_policy(solved):
    instance, state, scenarios, best_results, result = solved
    nominal_decision = first_stage_decision(
        instance, best_results["Balanced"].solution
    )
    nominal = solve_minimax_relative_regret(
        instance,
        state,
        scenarios,
        best_results=best_results,
        fixed_decision=nominal_decision,
    )

    assert result.theta <= nominal.theta + TOL


def test_all_scenario_plans_satisfy_model_constraints(solved):
    instance, state, scenarios, _, result = solved
    retailers_of = {
        i: [r for r, j in instance.retailer_sku_pairs if j == i]
        for i in instance.skus
    }

    for scenario in scenarios:
        rows = {row["period"]: row for row in scenario["periods"]}
        solution = result.scenario_results[scenario["scenario_id"]].solution
        for t, values in rows.items():
            selected = set(solution.selected_d2c_skus[t])
            assert len(selected) <= instance.max_d2c_skus
            for i, sku in instance.skus.items():
                q = solution.d2c_quantity[i, t]
                shipped = q + sum(
                    solution.retailer_quantity[r, i, t]
                    for r in retailers_of[i]
                )
                assert -TOL <= q <= values["d2c_demand"][i] * (i in selected) + TOL
                assert shipped <= values["supply_limit"][i] + TOL

            for r, i in instance.retailer_sku_pairs:
                limit = (
                    values["retailer_base_demand"][r][i]
                    * solution.order_retention[r, t]
                )
                assert -TOL <= solution.retailer_quantity[r, i, t] <= limit + TOL

            capacity_used = sum(
                sku.capacity_use
                * (
                    solution.d2c_quantity[i, t]
                    + sum(
                        solution.retailer_quantity[r, i, t]
                        for r in retailers_of[i]
                    )
                )
                for i, sku in instance.skus.items()
            )
            assert capacity_used <= values["capacity"] + TOL

            for r, retailer in instance.retailers.items():
                expected_exposure = sum(
                    instance.beta * (i in selected)
                    + (1.0 - instance.beta)
                    * solution.d2c_quantity[i, t]
                    / values["d2c_demand"][i]
                    for i in retailer.base_orders
                ) / len(retailer.base_orders)
                assert solution.exposure[r, t] == pytest.approx(
                    expected_exposure, abs=TOL
                )
                assert -TOL <= solution.exposure[r, t] <= 1.0 + TOL
                assert -TOL <= solution.order_retention[r, t] <= 1.0 + TOL

        for t in (1, 2):
            values = rows[t]
            for r in instance.retailers:
                expected_state = (
                    values["persistence"][r] * solution.order_retention[r, t]
                    + (1.0 - values["persistence"][r])
                    * (1.0 - values["response"][r] * solution.exposure[r, t])
                )
                assert solution.order_retention[r, t + 1] == pytest.approx(
                    expected_state, abs=TOL
                )

        for r in instance.retailers:
            assert solution.order_retention[r, 1] == pytest.approx(
                state.order_retention[r], abs=TOL
            )


def test_same_inputs_reproduce_regret_values(solved):
    instance, state, scenarios, best_results, first = solved
    repeated = solve_minimax_relative_regret(
        instance,
        state,
        scenarios,
        best_results=best_results,
    )

    assert repeated.theta == pytest.approx(first.theta, abs=TOL)
    assert repeated.common_decision == first.common_decision
    assert {
        scenario_id: row.policy_profit
        for scenario_id, row in repeated.scenario_results.items()
    } == pytest.approx(
        {
            scenario_id: row.policy_profit
            for scenario_id, row in first.scenario_results.items()
        },
        abs=1e-5,
    )
