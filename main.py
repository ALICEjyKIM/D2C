"""Toy instance를 풀고 rolling-horizon 정책 결과를 비교한다."""

from src.instance import load_instance, make_initial_state
from src.milp import solve_milp
from src.simulator import period_result, run_simulation
from src.types import Instance, MILPSolution, SimulationResult

INSTANCE_PATH = "configs/toy.json"


def main():
    instance = load_instance(INSTANCE_PATH)
    state = make_initial_state(instance)

    print("### Baseline plan")
    report(instance, solve_milp(instance, state, instance.default_horizon))

    print("\n### Rolling-horizon policies")
    for policy in ("myopic", "lookahead"):
        report_simulation(instance, run_simulation(instance, policy, instance.periods))


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
    label = "Myopic" if result.policy == "myopic" else "Look-ahead"
    print(f"\n{label} 누적이익: {result.cumulative_profit:.2f}")
    for p in result.periods:
        print(f"\n기간 {p.period}")
        print(f"  D2C listing: {list(p.selected_d2c_skus)}")
        print(
            "  D2C 공급량: "
            + ", ".join(f"{i}={p.d2c_quantity[i]:.2f}" for i in instance.skus)
        )
        print(
            "  Retailer 공급량: "
            + ", ".join(
                f"{r}-{i}={p.retailer_quantity[r, i]:.2f}"
                for r, i in instance.retailer_sku_pairs
            )
        )
        print(
            "  D2C 노출: "
            + ", ".join(f"{r}={p.exposure[r]:.3f}" for r in instance.retailers)
        )
        print(
            "  주문상태: "
            + ", ".join(
                f"{r}={p.order_retention[r]:.3f}" for r in instance.retailers
            )
        )
        print(f"  기간이익: {p.total_profit:.2f}")


if __name__ == "__main__":
    main()
