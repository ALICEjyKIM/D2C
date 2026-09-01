"""Deterministic order-retention transition, matching MILP constraint C7."""

from src.types import Instance, State


def next_state(instance: Instance, state: State, exposure: dict[str, float]) -> State:
    """Advance one period: g[r,t+1] = rho g[r,t] + (1-rho)(1 - kappa e[r,t])."""
    if exposure.keys() != instance.retailers.keys():
        raise ValueError("exposure must cover exactly all retailers")

    rho, kappa = instance.rho, instance.kappa
    return State(
        period=state.period + 1,
        order_retention={
            r: rho * state.order_retention[r]
            + (1.0 - rho) * (1.0 - kappa * exposure[r])
            for r in instance.retailers
        },
    )
