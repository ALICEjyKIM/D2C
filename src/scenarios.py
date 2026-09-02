"""계획 시나리오와 평가경로를 생성한다."""

import math

import numpy as np

try:
    from scipy.stats import qmc
except ImportError:  # pragma: no cover - SciPy가 없는 환경에서 사용한다
    qmc = None

from src.types import Instance


def sample_range(low, high, samples, rng):
    """한 구간을 Latin hypercube 방식으로 표본화한다."""
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("sample range must be finite and ordered")

    unit = (np.arange(samples) + rng.random(samples)) / samples
    rng.shuffle(unit)
    return low + (high - low) * unit


def generate_planning_scenarios(
    instance,
    start_period,
    end_period,
    num_scenarios,
    uncertainty,
    seed,
):
    """현재 기간부터 마지막 기간까지 계획 시나리오를 만든다."""
    return _generate_paths(
        instance,
        start_period,
        end_period,
        num_scenarios,
        uncertainty,
        seed,
        kind="planning",
    )


def generate_evaluation_paths(
    instance,
    start_period,
    end_period,
    num_scenarios,
    uncertainty,
    seed,
):
    """정책 성능을 외부에서 확인할 평가경로를 만든다."""
    return _generate_paths(
        instance,
        start_period,
        end_period,
        num_scenarios,
        uncertainty,
        seed,
        kind="evaluation",
    )


def validate_scenarios(scenarios):
    """생성된 경로의 구조와 기본 범위를 확인한다."""
    if not scenarios:
        raise ValueError("scenarios must not be empty")

    identifiers = set()
    required = {
        "period",
        "d2c_demand",
        "retailer_base_demand",
        "d2c_margin",
        "retailer_margin",
        "d2c_fixed_cost",
        "response",
        "persistence",
    }
    for scenario in scenarios:
        identifier = scenario.get("scenario_id")
        if not identifier or identifier in identifiers:
            raise ValueError("scenario_id must be present and unique")
        identifiers.add(identifier)

        periods = scenario.get("periods")
        if not isinstance(periods, list) or not periods:
            raise ValueError("each scenario needs at least one period")
        period_numbers = [row.get("period") for row in periods]
        expected = list(range(scenario["start_period"], scenario["end_period"] + 1))
        if period_numbers != expected:
            raise ValueError("scenario periods must be consecutive")

        for row in periods:
            if not required <= row.keys():
                raise ValueError("scenario period is missing required inputs")
            for field in (
                "d2c_demand",
                "retailer_base_demand",
                "d2c_margin",
                "retailer_margin",
                "d2c_fixed_cost",
            ):
                values = list(_nested_values(row[field]))
                if not values or any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"{field} values must be finite and nonnegative")

            responses = list(_nested_values(row["response"]))
            if not responses or any(not 0 <= value <= 1 for value in responses):
                raise ValueError("response values must be in [0, 1]")
            if not 0 <= row["persistence"] <= 1:
                raise ValueError("persistence must be in [0, 1]")
    return True


def _generate_paths(
    instance: Instance,
    start_period: int,
    end_period: int,
    num_scenarios: int,
    uncertainty: float,
    seed: int,
    kind: str,
) -> list[dict]:
    if not 1 <= start_period <= end_period <= instance.periods:
        raise ValueError("period range lies outside the instance periods")
    if num_scenarios < 1:
        raise ValueError("num_scenarios must be at least 1")
    if not math.isfinite(uncertainty) or not 0 <= uncertainty <= 1:
        raise ValueError("uncertainty must be in [0, 1]")

    periods = range(start_period, end_period + 1)
    dimensions_per_period = (
        3 * len(instance.skus)
        + 2 * len(instance.retailer_sku_pairs)
        + len(instance.retailers)
        + 1
    )
    dimensions = len(periods) * dimensions_per_period
    unit_samples = _latin_hypercube(num_scenarios, dimensions, seed)

    scenarios = []
    prefix = "S" if kind == "planning" else "E"
    for scenario_index, sample in enumerate(unit_samples, start=1):
        column = 0
        period_rows = []

        def draw(base, unit_interval=False):
            nonlocal column
            low = base * (1.0 - uncertainty)
            high = base * (1.0 + uncertainty)
            value = low + (high - low) * sample[column]
            column += 1
            if unit_interval:
                return min(1.0, max(0.0, float(value)))
            return max(0.0, float(value))

        for period in periods:
            # 이번 기간에 필요한 미래만 생성한다
            d2c_demand = {
                i: draw(sku.d2c_demand) for i, sku in instance.skus.items()
            }
            retailer_base_demand = {
                r: {
                    i: draw(retailer.base_orders[i])
                    for i in retailer.base_orders
                }
                for r, retailer in instance.retailers.items()
            }
            d2c_margin = {
                i: draw(sku.d2c_margin) for i, sku in instance.skus.items()
            }
            retailer_margin = {
                r: {
                    i: draw(retailer.wholesale_margins[i])
                    for i in retailer.wholesale_margins
                }
                for r, retailer in instance.retailers.items()
            }
            d2c_fixed_cost = {
                i: draw(getattr(sku, "d2c_fixed_cost", 0.0))
                for i, sku in instance.skus.items()
            }
            response = {
                r: draw(instance.response_for(r), unit_interval=True)
                for r in instance.retailers
            }
            # 반응계수는 0과 1 사이로 제한한다
            persistence = draw(instance.rho, unit_interval=True)
            period_rows.append(
                {
                    "period": period,
                    "d2c_demand": d2c_demand,
                    "retailer_base_demand": retailer_base_demand,
                    "d2c_margin": d2c_margin,
                    "retailer_margin": retailer_margin,
                    "d2c_fixed_cost": d2c_fixed_cost,
                    "response": response,
                    "persistence": persistence,
                }
            )

        scenarios.append(
            {
                "scenario_id": f"{prefix}{scenario_index:03d}",
                "kind": kind,
                "start_period": start_period,
                "end_period": end_period,
                "periods": period_rows,
            }
        )

    validate_scenarios(scenarios)
    return scenarios


def _latin_hypercube(samples: int, dimensions: int, seed: int) -> np.ndarray:
    if qmc is not None:
        return qmc.LatinHypercube(d=dimensions, seed=seed).random(n=samples)

    rng = np.random.default_rng(seed)
    return np.column_stack(
        [sample_range(0.0, 1.0, samples, rng) for _ in range(dimensions)]
    )


def _nested_values(values):
    if isinstance(values, dict):
        for value in values.values():
            yield from _nested_values(value)
    else:
        yield float(values)
