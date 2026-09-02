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


def main():
    instance = load_instance(TOY_PATH)
    state = make_initial_state(instance)
    scenarios = build_regret_scenarios(instance)
    best_results = calculate_scenario_best_profits(instance, state, scenarios)
    result = solve_minimax_relative_regret(
        instance,
        state,
        scenarios,
        best_results=best_results,
    )

    print("Scenario | Best profit z* | Policy profit z | Relative regret")
    for scenario in scenarios:
        row = result.scenario_results[scenario["scenario_id"]]
        print(
            f"{row.scenario_id} | {row.best_profit:.6f} | "
            f"{row.policy_profit:.6f} | {row.relative_regret:.6%}"
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


if __name__ == "__main__":
    main()
