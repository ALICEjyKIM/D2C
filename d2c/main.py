"""Solve and summarize the controlled toy instance."""

from src.instance import load_instance, make_initial_state
from src.milp import solve_milp
from src.types import Instance, MILPSolution

INSTANCE_PATH = "configs/toy.json"


def main():
    instance = load_instance(INSTANCE_PATH)
    state = make_initial_state(instance)
    report(instance, solve_milp(instance, state, instance.default_horizon))


def report(instance: Instance, solution: MILPSolution):
    print(f"Instance: {instance.instance_id}")
    print(f"Status: {solution.status}")
    print(f"Objective: {solution.objective_value:.2f}")

    pairs = instance.retailer_sku_pairs
    for t in range(solution.start_period, solution.start_period + solution.horizon):
        print(f"\nPeriod {t}")
        print(f"D2C assortment: {list(solution.selected_d2c_skus[t])}")

        print("D2C quantities:")
        for i in instance.skus:
            print(f"  {i}: {solution.d2c_quantity[i, t]:.2f}")

        print("Retailer allocations:")
        for r, i in pairs:
            print(f"  {r}-{i}: {solution.retailer_quantity[r, i, t]:.2f}")

        capacity_used = sum(
            instance.skus[i].capacity_use * solution.d2c_quantity[i, t]
            for i in instance.skus
        ) + sum(
            instance.skus[i].capacity_use * solution.retailer_quantity[r, i, t]
            for r, i in pairs
        )
        print(f"Capacity used: {capacity_used:.2f} / {instance.capacity:.2f}")


if __name__ == "__main__":
    main()
