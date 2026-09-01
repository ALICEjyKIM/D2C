"""Tests for instance loading and validation."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.instance import load_instance, make_initial_state, validate_instance
from src.types import Retailer


TOY_PATH = Path(__file__).resolve().parents[1] / "configs" / "toy.json"


def test_load_toy_instance() -> None:
    instance = load_instance(TOY_PATH)

    assert len(instance.skus) == 4
    assert len(instance.retailers) == 3
    assert instance.max_d2c_skus == 2
    assert instance.feasible_retailer_sku_pairs == (
        ("R1", "A"),
        ("R1", "C"),
        ("R2", "A"),
        ("R2", "D"),
        ("R3", "B"),
        ("R3", "C"),
        ("R3", "D"),
    )


def test_initial_retention_is_one_for_all_retailers() -> None:
    instance = load_instance(TOY_PATH)
    state = make_initial_state(instance)

    assert state.period == 1
    assert state.order_retention == {"R1": 1.0, "R2": 1.0, "R3": 1.0}


def test_validation_rejects_beta_outside_unit_interval() -> None:
    instance = load_instance(TOY_PATH)

    with pytest.raises(ValueError, match="beta"):
        validate_instance(replace(instance, beta=1.01))


def test_validation_rejects_mismatched_retailer_sku_keys() -> None:
    instance = load_instance(TOY_PATH)
    bad_r1 = Retailer(
        retailer_id="R1",
        base_orders=instance.retailers["R1"].base_orders,
        wholesale_margins={"A": 8.0},
    )

    with pytest.raises(ValueError, match="identical SKU keys"):
        validate_instance(
            replace(instance, retailers={**instance.retailers, "R1": bad_r1})
        )
