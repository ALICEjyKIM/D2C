"""Domain data and solution containers for the D2C model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SKU:
    d2c_margin: float
    d2c_demand: float
    supply_limit: float
    capacity_use: float


@dataclass(frozen=True, slots=True)
class Retailer:
    base_orders: dict[str, float]
    wholesale_margins: dict[str, float]


@dataclass(frozen=True, slots=True)
class Instance:
    instance_id: str
    periods: int
    default_horizon: int
    max_d2c_skus: int
    capacity: float
    beta: float
    rho: float
    kappa: float
    gamma: float
    initial_order_retention: float
    skus: dict[str, SKU]
    retailers: dict[str, Retailer]

    @property
    def retailer_sku_pairs(self) -> tuple[tuple[str, str], ...]:
        """The (r, i) pairs with i in I_r, in retailer then SKU order."""
        return tuple(
            (r, i)
            for r, retailer in self.retailers.items()
            for i in retailer.base_orders
        )


@dataclass(frozen=True, slots=True)
class State:
    period: int
    order_retention: dict[str, float]


@dataclass(frozen=True, slots=True)
class SolverConfig:
    output_flag: bool = False
    time_limit: float | None = None
    mip_gap: float | None = None


@dataclass(frozen=True, slots=True)
class MILPSolution:
    status: str
    objective_value: float
    start_period: int
    horizon: int
    selected_d2c_skus: dict[int, tuple[str, ...]]  # t -> listed SKUs
    d2c_quantity: dict[tuple[str, int], float]  # (i, t)
    retailer_quantity: dict[tuple[str, str, int], float]  # (r, i, t)
    exposure: dict[tuple[str, int], float]  # (r, t)
    order_retention: dict[tuple[str, int], float]  # (r, t)
    runtime: float
    num_variables: int
    num_constraints: int
