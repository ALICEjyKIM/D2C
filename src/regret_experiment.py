"""세 시나리오의 minimax relative regret 결과를 출력한다."""

from pathlib import Path

from src.instance import load_instance, make_initial_state
from src.milp import (
    calculate_scenario_best_profits,
    solve_minimax_relative_regret,
)
from src.scenarios import build_regret_scenarios


ROOT = Path(__file__).resolve().parents[1]
TOY_PATH = ROOT / "configs" / "toy.json"
PREVIOUS_THETA = 0.008808737166289179
PREVIOUS_PROFITS = {
    "D2C-friendly": (6806.184572, 6746.230681),
    "Balanced": (6333.542032, 6333.542032),
    "Relationship-sensitive": (6147.811331, 6143.502000),
}


def main():
    instance = load_instance(TOY_PATH)
    state = make_initial_state(instance)
    scenarios = build_regret_scenarios(instance, current_state=state)
    best_results = calculate_scenario_best_profits(instance, state, scenarios)
    result = solve_minimax_relative_regret(
        instance,
        state,
        scenarios,
        best_results=best_results,
    )

    print(
        "Scenario | Best profit z* | Policy profit z | "
        "Absolute regret | Relative regret"
    )
    for scenario in scenarios:
        row = result.scenario_results[scenario["scenario_id"]]
        print(
            f"{row.scenario_id} | {row.best_profit:.6f} | "
            f"{row.policy_profit:.6f} | {row.absolute_regret:.6f} | "
            f"{row.relative_regret:.6%}"
        )

    decision = result.common_decision
    print(f"\nCommon period-1 D2C assortment: {list(decision.selected_d2c_skus)}")
    print(
        "Common period-1 D2C quantities: "
        + ", ".join(f"{i}={value:.3f}" for i, value in decision.d2c_quantity.items())
    )
    print(
        "Common period-1 Retailer quantities: "
        + ", ".join(
            f"{r}-{i}={value:.3f}"
            for (r, i), value in decision.retailer_quantity.items()
        )
    )
    print(f"Maximum relative regret theta: {result.theta:.6%}")
    print(f"Worst-regret scenario: {result.worst_scenario}")
    print("Future D2C quantities:")
    for scenario_id, row in result.scenario_results.items():
        future = []
        for period in range(
            result.start_period + 1,
            result.start_period + result.horizon,
        ):
            quantities = ", ".join(
                f"{i}={row.solution.d2c_quantity[i, period]:.3f}"
                for i in instance.skus
            )
            future.append(f"period {period}: {quantities}")
        print(f"  {scenario_id}: {'; '.join(future)}")

    print("Change from previous result (old timing, gamma=0.98):")
    for scenario_id, row in result.scenario_results.items():
        previous_best, previous_policy = PREVIOUS_PROFITS[scenario_id]
        print(
            f"  {scenario_id}: delta z*={row.best_profit - previous_best:+.6f}, "
            f"delta z={row.policy_profit - previous_policy:+.6f}"
        )
    print(f"  delta theta={result.theta - PREVIOUS_THETA:+.6%}")


if __name__ == "__main__":
    main()
