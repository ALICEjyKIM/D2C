from dataclasses import replace

import pytest

from src.instance import load_instance, make_initial_state
from src.transition import next_state
from src.types import State


TOY_PATH = "configs/toy.json"


@pytest.fixture
def toy():
    return load_instance(TOY_PATH)


def test_transition_matches_worked_example(toy):
    # rho = 0.6, kappa = 0.2 in the toy instance; g = 1, e = 0.5 -> 0.96.
    state = State(period=1, order_retention={r: 1.0 for r in toy.retailers})
    nxt = next_state(toy, state, {r: 0.5 for r in toy.retailers})

    assert nxt.period == 2
    for r in toy.retailers:
        assert nxt.order_retention[r] == pytest.approx(0.96)


def test_transition_reproduces_c7_for_arbitrary_state(toy):
    inst = replace(toy, rho=0.7, kappa=0.35)
    state = State(period=3, order_retention={"R1": 0.8, "R2": 0.4, "R3": 0.55})
    exposure = {"R1": 0.2, "R2": 0.9, "R3": 0.5}

    nxt = next_state(inst, state, exposure)

    assert nxt.period == 4
    for r in inst.retailers:
        expected = inst.rho * state.order_retention[r] + (1.0 - inst.rho) * (
            1.0 - inst.kappa * exposure[r]
        )
        assert nxt.order_retention[r] == pytest.approx(expected)
        assert 0.0 <= nxt.order_retention[r] <= 1.0


def test_transition_rejects_incomplete_exposure(toy):
    state = make_initial_state(toy)
    with pytest.raises(ValueError, match="exposure"):
        next_state(toy, state, {"R1": 0.5})
