"""Domain data structures for D2C optimization."""

from dataclasses import dataclass


SKUPeriod = tuple[str, int]
RetailerSKU = tuple[str, str]
RetailerSKUPeriod = tuple[str, str, int]
RetailerPeriod = tuple[str, int]


@dataclass(frozen=True, slots=True)
class SKU:
    sku_id: str
    d2c_margin: float
    d2c_demand: float
    supply_limit: float
    capacity_use: float


@dataclass(frozen=True, slots=True)
class Retailer:
    retailer_id: str
    base_orders: dict[str, float]
    wholesale_margins: dict[str, float]

    @property
    def sku_ids(self) -> tuple[str, ...]:
        return tuple(self.base_orders)


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
    def sku_ids(self) -> tuple[str, ...]:
        return tuple(self.skus)

    @property
    def retailer_ids(self) -> tuple[str, ...]:
        return tuple(self.retailers)

    @property
    def feasible_retailer_sku_pairs(self) -> tuple[RetailerSKU, ...]:
        return tuple(
            (retailer_id, sku_id)
            for retailer_id, retailer in self.retailers.items()
            for sku_id in retailer.sku_ids
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
    selected_d2c_skus: dict[int, tuple[str, ...]]
    d2c_quantity: dict[SKUPeriod, float]
    retailer_quantity: dict[RetailerSKUPeriod, float]
    exposure: dict[RetailerPeriod, float]
    order_retention: dict[RetailerPeriod, float]
    runtime: float
    num_variables: int
    num_constraints: int
