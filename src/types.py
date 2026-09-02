"""D2C 모델에서 쓰는 데이터와 결과 타입."""

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
    retailer_responses: dict[str, float] | None = None

    @property
    def retailer_sku_pairs(self) -> tuple[tuple[str, str], ...]:
        """Retailer가 실제로 취급하는 (r, i) 조합만 반환한다."""
        return tuple(
            (r, i)
            for r, retailer in self.retailers.items()
            for i in retailer.base_orders
        )

    def response_for(self, retailer: str) -> float:
        if self.retailer_responses is None:
            return self.kappa
        return self.retailer_responses[retailer]


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
    selected_d2c_skus: dict[int, tuple[str, ...]]  # 기간별 D2C assortment
    d2c_quantity: dict[tuple[str, int], float]  # (SKU, 기간)
    retailer_quantity: dict[tuple[str, str, int], float]  # (Retailer, SKU, 기간)
    exposure: dict[tuple[str, int], float]  # (Retailer, 기간)
    order_retention: dict[tuple[str, int], float]  # (Retailer, 기간)
    runtime: float
    num_variables: int
    num_constraints: int


@dataclass(frozen=True, slots=True)
class PeriodResult:
    """Rolling-horizon에서 실제 실행한 한 기간의 결과."""

    period: int
    selected_d2c_skus: tuple[str, ...]
    d2c_quantity: dict[str, float]  # SKU별 q[i, t]
    retailer_quantity: dict[tuple[str, str], float]  # (r, i)별 x[r, i, t]
    d2c_profit: float
    wholesale_profit: float
    total_profit: float
    supply_slack: dict[str, float]  # SKU별 남은 공급 여력
    capacity_used: float
    capacity_utilization: float  # 사용량 / 전체 capacity
    exposure: dict[str, float]  # Retailer별 e[r, t]
    order_retention: dict[str, float]  # 기간 시작 시점의 g[r, t]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    instance_id: str
    policy: str
    planning_horizon: int
    periods: list[PeriodResult]
    cumulative_profit: float  # 기간별 이익의 단순 합
    discounted_profit: float  # gamma를 적용한 기간별 이익의 합

    @property
    def total_profit(self) -> float:
        return self.cumulative_profit
