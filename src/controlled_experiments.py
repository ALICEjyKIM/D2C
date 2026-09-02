"""통제조건에서 Myopic과 Look-ahead 정책을 비교한다."""

import csv
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import fmean

from src.instance import load_instance, validate_instance
from src.types import Instance, SKU


ROOT = Path(__file__).resolve().parents[1]
TOY_PATH = ROOT / "configs" / "toy.json"
CONFIG_PATH = ROOT / "configs" / "controlled_experiments.json"
RESULTS_PATH = ROOT / "results"
POLICIES = ("myopic", "lookahead")


def run_simulation(*args, **kwargs):
    """실험을 실행할 때만 solver 의존 모듈을 불러온다."""
    from src.simulator import run_simulation as simulate

    return simulate(*args, **kwargs)


def load_experiment_levels(path: str | Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_controlled_cases(instance: Instance, levels: dict) -> list[dict]:
    """27개 통제조건을 만든다."""
    cases = []
    combinations = product(
        levels["margin_levels"],
        levels["response_levels"],
        levels["persistence_levels"],
    )
    for number, (margin_level, response, persistence) in enumerate(
        combinations, start=1
    ):
        case_instance = replace(
            instance,
            instance_id=f"{instance.instance_id}_controlled_{number:03d}",
            skus=_adjust_d2c_margins(instance, float(margin_level)),
            kappa=float(response),
            rho=float(persistence),
            retailer_responses=None,
        )
        validate_instance(case_instance)
        cases.append(
            {
                "case_id": f"C{number:03d}",
                "margin_level": float(margin_level),
                "response": float(response),
                "persistence": float(persistence),
                "instance": case_instance,
            }
        )
    return cases


def run_controlled_experiments(
    cases: list[dict],
) -> tuple[list[dict], list[dict]]:
    """각 통제조건에서 두 정책을 실행한다."""
    summary_rows = []
    period_rows = []
    for case in cases:
        instance = case["instance"]
        for policy in POLICIES:
            started = time.perf_counter()
            result = run_simulation(instance, policy, instance.periods)
            runtime = time.perf_counter() - started
            summary_rows.append(_summary_row(case, result, runtime))
            period_rows.extend(_period_rows(case, result))
    return summary_rows, period_rows


def run_heterogeneity_experiments(
    instance: Instance,
    heterogeneity_levels: list[list[float]],
) -> list[dict]:
    """평균 반응은 같고 Retailer별 차이만 바꿔 실행한다."""
    rows = []
    names = ("homogeneous", "medium", "high")
    for number, values in enumerate(heterogeneity_levels, start=1):
        responses = _retailer_responses(instance, values)
        case_instance = replace(
            instance,
            instance_id=f"{instance.instance_id}_heterogeneity_{number:03d}",
            retailer_responses=responses,
        )
        validate_instance(case_instance)
        case = {
            "case_id": f"H{number:03d}",
            "heterogeneity": (
                names[number - 1]
                if number <= len(names)
                else f"level_{number}"
            ),
            "response": fmean(responses.values()),
            "persistence": case_instance.rho,
            "instance": case_instance,
        }
        for policy in POLICIES:
            started = time.perf_counter()
            result = run_simulation(case_instance, policy, case_instance.periods)
            runtime = time.perf_counter() - started
            rows.append(_heterogeneity_row(case, result, runtime))
    return rows


def save_experiment_results(
    summary_rows: list[dict],
    period_rows: list[dict],
    heterogeneity_rows: list[dict],
    output_dir: str | Path = RESULTS_PATH,
) -> dict[str, Path]:
    """실험 결과를 분석하기 쉬운 CSV로 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "controlled_summary": output_dir / "controlled_summary.csv",
        "controlled_periods": output_dir / "controlled_periods.csv",
        "heterogeneity_summary": output_dir / "heterogeneity_summary.csv",
    }
    _write_csv(paths["controlled_summary"], summary_rows)
    _write_csv(paths["controlled_periods"], period_rows)
    _write_csv(paths["heterogeneity_summary"], heterogeneity_rows)
    return paths


def _adjust_d2c_margins(instance: Instance, level: float) -> dict[str, SKU]:
    margins = {i: [] for i in instance.skus}
    for retailer in instance.retailers.values():
        for i, margin in retailer.wholesale_margins.items():
            margins[i].append(margin)

    adjusted = {}
    for i, sku in instance.skus.items():
        retailer_margin = fmean(margins[i]) if margins[i] else sku.d2c_margin
        d2c_margin = retailer_margin + level * (
            sku.d2c_margin - retailer_margin
        )
        adjusted[i] = replace(sku, d2c_margin=d2c_margin)
    return adjusted


def _retailer_responses(
    instance: Instance, configured: list[float]
) -> dict[str, float]:
    if not configured:
        raise ValueError("heterogeneity level must not be empty")
    target = fmean(configured)
    if not math.isclose(target, 0.4, abs_tol=1e-9):
        raise ValueError("heterogeneity response average must be 0.4")

    retailers = list(instance.retailers)
    if len(retailers) == len(configured):
        values = [float(value) for value in configured]
    elif len(retailers) == 1:
        values = [target]
    else:
        spread = max(target - min(configured), max(configured) - target)
        values = [
            target - spread + 2.0 * spread * index / (len(retailers) - 1)
            for index in range(len(retailers))
        ]
    return dict(zip(retailers, values))


def _summary_row(case: dict, result, runtime: float) -> dict:
    all_exposure = [
        value for period in result.periods for value in period.exposure.values()
    ]
    return {
        "case_id": case["case_id"],
        "margin_level": case["margin_level"],
        "response": case["response"],
        "persistence": case["persistence"],
        "policy": result.policy,
        "total_profit": result.total_profit,
        "total_d2c_quantity": sum(
            sum(period.d2c_quantity.values()) for period in result.periods
        ),
        "total_retailer_quantity": sum(
            sum(period.retailer_quantity.values()) for period in result.periods
        ),
        "final_average_order_state": fmean(
            result.periods[-1].order_retention.values()
        ),
        "average_exposure": fmean(all_exposure),
        "runtime": runtime,
    }


def _period_rows(case: dict, result) -> list[dict]:
    instance = case["instance"]
    rows = []
    for period in result.periods:
        row = {
            "case_id": case["case_id"],
            "margin_level": case["margin_level"],
            "response": case["response"],
            "persistence": case["persistence"],
            "policy": result.policy,
            "period": period.period,
            "period_profit": period.total_profit,
            "total_d2c_quantity": sum(period.d2c_quantity.values()),
            "total_retailer_quantity": sum(period.retailer_quantity.values()),
            "average_exposure": fmean(period.exposure.values()),
            "average_order_state": fmean(period.order_retention.values()),
            "capacity_used": period.capacity_used,
        }
        row.update(
            {
                f"listed_{i}": int(i in period.selected_d2c_skus)
                for i in instance.skus
            }
        )
        row.update(
            {
                f"d2c_quantity_{i}": period.d2c_quantity[i]
                for i in instance.skus
            }
        )
        row.update(
            {
                f"retailer_quantity_{r}_{i}": period.retailer_quantity[r, i]
                for r, i in instance.retailer_sku_pairs
            }
        )
        row.update(
            {f"exposure_{r}": period.exposure[r] for r in instance.retailers}
        )
        row.update(
            {
                f"order_state_{r}": period.order_retention[r]
                for r in instance.retailers
            }
        )
        rows.append(row)
    return rows


def _heterogeneity_row(case: dict, result, runtime: float) -> dict:
    instance = case["instance"]
    row = {
        "case_id": case["case_id"],
        "heterogeneity": case["heterogeneity"],
        "average_response": case["response"],
        "persistence": case["persistence"],
        "policy": result.policy,
        "total_profit": result.total_profit,
        "total_d2c_quantity": sum(
            sum(period.d2c_quantity.values()) for period in result.periods
        ),
        "total_retailer_quantity": sum(
            sum(period.retailer_quantity.values()) for period in result.periods
        ),
        "runtime": runtime,
    }
    for r in instance.retailers:
        row[f"response_{r}"] = instance.response_for(r)
        row[f"retailer_quantity_{r}"] = sum(
            period.retailer_quantity[r, i]
            for period in result.periods
            for retailer, i in instance.retailer_sku_pairs
            if retailer == r
        )
        row[f"average_exposure_{r}"] = fmean(
            period.exposure[r] for period in result.periods
        )
        row[f"final_order_state_{r}"] = result.periods[-1].order_retention[r]
    return row


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"no rows to save in {path.name}")

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    instance = load_instance(TOY_PATH)
    levels = load_experiment_levels()
    cases = build_controlled_cases(instance, levels)
    summary_rows, period_rows = run_controlled_experiments(cases)
    heterogeneity_rows = run_heterogeneity_experiments(
        instance, levels["heterogeneity_levels"]
    )
    paths = save_experiment_results(
        summary_rows, period_rows, heterogeneity_rows
    )

    print(f"통제조건: {len(cases)}개, 정책 실행: {len(summary_rows)}회")
    print(f"이질성 조건: {len(heterogeneity_rows) // len(POLICIES)}개")
    for path in paths.values():
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
