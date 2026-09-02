from dataclasses import replace

import pytest

from src.instance import load_instance, make_initial_state, validate_instance


TOY_PATH = "configs/toy.json"


@pytest.fixture
def toy():
    return load_instance(TOY_PATH)


def test_load_toy_instance(toy):
    assert toy.periods == 3
    assert len(toy.skus) == 4
    assert len(toy.retailers) == 3
    assert toy.max_d2c_skus == 2
    assert toy.retailer_sku_pairs == (
        ("R1", "A"),
        ("R1", "C"),
        ("R2", "A"),
        ("R2", "D"),
        ("R3", "B"),
        ("R3", "C"),
        ("R3", "D"),
    )
    assert {i: sku.d2c_margin for i, sku in toy.skus.items()} == {
        "A": 10.0,
        "B": 9.0,
        "C": 8.25,
        "D": 7.5,
    }


def test_initial_retention_is_one_for_all_retailers(toy):
    state = make_initial_state(toy)

    assert state.period == 1
    assert state.order_retention == {"R1": 1.0, "R2": 1.0, "R3": 1.0}


def test_validation_rejects_beta_outside_unit_interval(toy):
    with pytest.raises(ValueError, match="beta"):
        validate_instance(replace(toy, beta=1.01))


def test_validation_rejects_mismatched_retailer_sku_keys(toy):
    r1 = replace(toy.retailers["R1"], wholesale_margins={"A": 8.0})

    with pytest.raises(ValueError, match="identical SKU keys"):
        validate_instance(replace(toy, retailers={**toy.retailers, "R1": r1}))
