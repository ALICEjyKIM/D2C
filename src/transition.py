"""MILP의 C7과 같은 식으로 order retention을 갱신한다."""

from src.types import Instance, State


def next_state(instance: Instance, state: State, exposure: dict[str, float]) -> State:
    """현재 exposure를 반영해 다음 기간 state를 만든다."""
    if exposure.keys() != instance.retailers.keys():
        raise ValueError("exposure must cover exactly all retailers")

    rho = instance.rho
    return State(
        period=state.period + 1,
        order_retention={
            r: rho * state.order_retention[r]
            + (1.0 - rho) * (1.0 - instance.response_for(r) * exposure[r])
            for r in instance.retailers
        },
    )
