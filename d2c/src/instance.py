"""Instance loading and validation."""

import json
import math
from pathlib import Path

from src.types import Instance, Retailer, SKU, State


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc

    try:
        instance = Instance(
            instance_id=data["instance_id"],
            periods=int(data["periods"]),
            default_horizon=int(data["default_horizon"]),
            max_d2c_skus=int(data["max_d2c_skus"]),
            capacity=float(data["capacity"]),
            beta=float(data["beta"]),
            rho=float(data["rho"]),
            kappa=float(data["kappa"]),
            gamma=float(data["gamma"]),
            initial_order_retention=float(data["initial_order_retention"]),
            skus={
                i: SKU(
                    d2c_margin=float(row["d2c_margin"]),
                    d2c_demand=float(row["d2c_demand"]),
                    supply_limit=float(row["supply_limit"]),
                    capacity_use=float(row["capacity_use"]),
                )
                for i, row in data["skus"].items()
            },
            retailers={
                r: Retailer(
                    base_orders={i: float(v) for i, v in row["base_orders"].items()},
                    wholesale_margins={
                        i: float(v) for i, v in row["wholesale_margins"].items()
                    },
                )
                for r, row in data["retailers"].items()
            },
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed instance in {path}: {exc}") from exc

    validate_instance(instance)
    return instance


def validate_instance(instance: Instance):
    if not instance.skus or not instance.retailers:
        raise ValueError("instance needs at least one SKU and one retailer")
    if instance.periods < 1:
        raise ValueError("periods must be at least 1")
    if not 1 <= instance.default_horizon <= instance.periods:
        raise ValueError("default_horizon must be between 1 and periods")
    if not 1 <= instance.max_d2c_skus <= len(instance.skus):
        raise ValueError("max_d2c_skus must be between 1 and the number of SKUs")

    _positive(instance.capacity, "capacity")
    _unit_interval(instance.beta, "beta")
    _unit_interval(instance.rho, "rho")
    _unit_interval(instance.kappa, "kappa")
    _unit_interval(instance.initial_order_retention, "initial_order_retention")
    if not 0 < instance.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")

    for i, sku in instance.skus.items():
        _nonnegative(sku.d2c_margin, f"SKU {i} d2c_margin")
        _nonnegative(sku.supply_limit, f"SKU {i} supply_limit")
        # C1 and C6 divide by d2c_demand, C5 scales by capacity_use.
        _positive(sku.d2c_demand, f"SKU {i} d2c_demand")
        _positive(sku.capacity_use, f"SKU {i} capacity_use")

    for r, retailer in instance.retailers.items():
        if not retailer.base_orders:
            raise ValueError(f"retailer {r} carries no SKUs")
        if retailer.base_orders.keys() != retailer.wholesale_margins.keys():
            raise ValueError(
                f"retailer {r}: base_orders and wholesale_margins "
                "must have identical SKU keys"
            )
        unknown = retailer.base_orders.keys() - instance.skus.keys()
        if unknown:
            raise ValueError(f"retailer {r} references unknown SKUs: {sorted(unknown)}")
        for i, order in retailer.base_orders.items():
            _nonnegative(order, f"retailer {r} base order for {i}")
            _nonnegative(
                retailer.wholesale_margins[i], f"retailer {r} wholesale margin for {i}"
            )


def make_initial_state(instance: Instance) -> State:
    return State(
        period=1,
        order_retention={
            r: instance.initial_order_retention for r in instance.retailers
        },
    )


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _unit_interval(value, name):
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
