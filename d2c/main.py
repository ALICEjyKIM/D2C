"""Solve the controlled toy instance and run rolling-horizon policy comparisons."""

from src.instance import load_instance, make_initial_state
from src.milp import solve_milp
from src.simulator import period_result, simulate
from src.types import Instance, MILPSolution, SimulationResult

INSTANCE_PATH = "configs/toy.json"


def main():
    instance = load_instance(INSTANCE_PATH)
    state = make_initial_state(instance)

    print("### Baseline plan")
    report(instance, solve_milp(instance, state, instance.default_horizon))

    print("\n### Rolling-horizon policies")
    for horizon in (1, 3, 6):
        report_simulation(instance, simulate(instance, horizon))


def report(instance: Instance, solution: MILPSolution):
    print(f"Instance: {instance.instance_id}")
    print(f"Status: {solution.status}")
    print(f"Objective: {solution.objective_value:.2f}")

    pairs = instance.retailer_sku_pairs
    for t in range(solution.start_period, solution.start_period + solution.horizon):
        p = period_result(instance, solution, t)
        print(f"\nPeriod {t}")
        print(f"D2C assortment: {list(p.selected_d2c_skus)}")

        print("D2C quantities:")
        for i in instance.skus:
            print(f"  {i}: {p.d2c_quantity[i]:.2f}")

        print("Retailer allocations:")
        for r, i in pairs:
            print(f"  {r}-{i}: {p.retailer_quantity[r, i]:.2f}")

        print(
            f"Profit: d2c {p.d2c_profit:.2f} + wholesale {p.wholesale_profit:.2f} "
            f"= {p.total_profit:.2f}"
        )
        print(
            "Supply slack: "
            + ", ".join(f"{i} {s:.2f}" for i, s in p.supply_slack.items())
        )
        print(
            f"Capacity: {p.capacity_used:.2f} / {instance.capacity:.2f} "
            f"({p.capacity_utilization:.1%})"
        )
        for r in instance.retailers:
            print(f"  {r}: exposure {p.exposure[r]:.3f}  retention {p.order_retention[r]:.3f}")


def report_simulation(instance: Instance, result: SimulationResult):
    label = {1: "Myopic", 3: "Dynamic"}.get(result.planning_horizon, "Look-ahead")
    print(f"\n{label} (planning_horizon={result.planning_horizon})")
    print(
        f"Cumulative profit {result.cumulative_profit:.2f}  "
        f"discounted {result.discounted_profit:.2f}"
    )
    for p in result.periods:
        retention = "  ".join(f"{r} {p.order_retention[r]:.3f}" for r in instance.retailers)
        print(
            f"  t={p.period}  assortment {{{', '.join(p.selected_d2c_skus)}}}  "
            f"profit {p.total_profit:.2f}  cap {p.capacity_utilization:.0%}  "
            f"retention [{retention}]"
        )


if __name__ == "__main__":
    main()
