"""Solve and summarize the controlled toy instance."""

from pathlib import Path

from src.instance import load_instance, make_initial_state
from src.milp import solve_milp
from src.types import Instance, MILPSolution, SolverConfig


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    instance = load_instance(PROJECT_ROOT / "configs" / "toy.json")
    state = make_initial_state(instance)
    solution = solve_milp(
        instance=instance,
        state=state,
        start_period=state.period,
        horizon=instance.default_horizon,
        solver_config=SolverConfig(output_flag=False),
    )
    _print_solution(instance, solution)


def _print_solution(instance: Instance, solution: MILPSolution) -> None:
    print(f"Instance: {instance.instance_id}")
    print(f"Status: {solution.status}")
    print(f"Objective: {solution.objective_value:.2f}")

    for period in range(
        solution.start_period,
        solution.start_period + solution.horizon,
    ):
        print(f"\nPeriod {period}")
        print(f"D2C assortment: {list(solution.selected_d2c_skus[period])}")
        print("D2C quantities:")
        for sku_id in instance.sku_ids:
            quantity = solution.d2c_quantity[sku_id, period]
            print(f"  {sku_id}: {quantity:.2f}")

        print("Retailer allocations:")
        for retailer_id, sku_id in instance.feasible_retailer_sku_pairs:
            quantity = solution.retailer_quantity[retailer_id, sku_id, period]
            print(f"  {retailer_id}-{sku_id}: {quantity:.2f}")

        capacity_used = sum(
            sku.capacity_use
            * (
                solution.d2c_quantity[sku_id, period]
                + sum(
                    solution.retailer_quantity[r, i, period]
                    for r, i in instance.feasible_retailer_sku_pairs
                    if i == sku_id
                )
            )
            for sku_id, sku in instance.skus.items()
        )
        print(f"Capacity used: {capacity_used:.2f} / {instance.capacity:.2f}")


if __name__ == "__main__":
    main()
