"""Instance loading and validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.types import Instance, Retailer, SKU, State


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ValueError("root must be a JSON object")

    try:
        raw_skus = data["skus"]
        raw_retailers = data["retailers"]
    except KeyError as exc:
        raise ValueError(f"missing required field: {exc.args[0]}") from exc

    if not isinstance(raw_skus, dict) or not isinstance(raw_retailers, dict):
        raise ValueError("skus and retailers must be JSON objects")

    skus: dict[str, SKU] = {}
    retailers: dict[str, Retailer] = {}
    try:
        for i, row in raw_skus.items():
            if not isinstance(row, dict):
                raise ValueError(f"skus.{i} must be a JSON object")
            skus[i] = SKU(
                sku_id=i,
                d2c_margin=_number(row["d2c_margin"], f"{i}.d2c_margin"),
                d2c_demand=_number(row["d2c_demand"], f"{i}.d2c_demand"),
                supply_limit=_number(row["supply_limit"], f"{i}.supply_limit"),
                capacity_use=_number(row["capacity_use"], f"{i}.capacity_use"),
            )

        for r, row in raw_retailers.items():
            if not isinstance(row, dict):
                raise ValueError(f"retailers.{r} must be a JSON object")
            orders = row["base_orders"]
            margins = row["wholesale_margins"]
            if not isinstance(orders, dict) or not isinstance(margins, dict):
                raise ValueError(f"retailer {r} orders and margins must be objects")
            retailers[r] = Retailer(
                retailer_id=r,
                base_orders={
                    i: _number(value, f"{r}.{i}.base_order")
                    for i, value in orders.items()
                },
                wholesale_margins={
                    i: _number(value, f"{r}.{i}.wholesale_margin")
                    for i, value in margins.items()
                },
            )

        instance = Instance(
            instance_id=data["instance_id"],
            periods=data["periods"],
            default_horizon=data["default_horizon"],
            max_d2c_skus=data["max_d2c_skus"],
            capacity=_number(data["capacity"], "capacity"),
            beta=_number(data["beta"], "beta"),
            rho=_number(data["rho"], "rho"),
            kappa=_number(data["kappa"], "kappa"),
            gamma=_number(data["gamma"], "gamma"),
            initial_order_retention=_number(
                data["initial_order_retention"], "initial_order_retention"
            ),
            skus=skus,
            retailers=retailers,
        )
    except KeyError as exc:
        raise ValueError(f"missing required field: {exc.args[0]}") from exc

    validate_instance(instance)
    return instance


def validate_instance(instance: Instance) -> None:
    if not isinstance(instance.instance_id, str) or not instance.instance_id.strip():
        raise ValueError("instance_id must be non-empty")
    if not instance.skus:
        raise ValueError("at least one SKU is required")
    if not instance.retailers:
        raise ValueError("at least one retailer is required")
    if isinstance(instance.periods, bool) or not isinstance(instance.periods, int):
        raise ValueError("periods must be an integer")
    if instance.periods < 1:
        raise ValueError("periods must be at least 1")
    if (
        isinstance(instance.default_horizon, bool)
        or not isinstance(instance.default_horizon, int)
        or not 1 <= instance.default_horizon <= instance.periods
    ):
        raise ValueError("default_horizon must be between 1 and periods")
    if (
        isinstance(instance.max_d2c_skus, bool)
        or not isinstance(instance.max_d2c_skus, int)
        or not 1 <= instance.max_d2c_skus <= len(instance.skus)
    ):
        raise ValueError("max_d2c_skus must be between 1 and the number of SKUs")

    _positive(instance.capacity, "capacity")
    _unit_interval(instance.beta, "beta")
    _unit_interval(instance.rho, "rho")
    _unit_interval(instance.kappa, "kappa")
    if not math.isfinite(instance.gamma) or not 0 < instance.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    _unit_interval(instance.initial_order_retention, "initial_order_retention")

    for i, sku in instance.skus.items():
        if i != sku.sku_id or not i.strip():
            raise ValueError(f"invalid SKU mapping key: {i!r}")
        _nonnegative(sku.d2c_margin, f"SKU {i} d2c_margin")
        _positive(sku.d2c_demand, f"SKU {i} d2c_demand")
        _nonnegative(sku.supply_limit, f"SKU {i} supply_limit")
        _positive(sku.capacity_use, f"SKU {i} capacity_use")

    for r, retailer in instance.retailers.items():
        if r != retailer.retailer_id or not r.strip():
            raise ValueError(f"invalid retailer mapping key: {r!r}")
        if not retailer.base_orders:
            raise ValueError(f"retailer {r} must carry at least one SKU")
        if retailer.base_orders.keys() != retailer.wholesale_margins.keys():
            raise ValueError(
                f"retailer {r} base_orders and wholesale_margins "
                "must have identical SKU keys"
            )

        unknown = retailer.base_orders.keys() - instance.skus.keys()
        if unknown:
            raise ValueError(f"retailer {r} references unknown SKUs: {sorted(unknown)}")
        for i, order in retailer.base_orders.items():
            _nonnegative(order, f"retailer {r} SKU {i} base order")
            _nonnegative(
                retailer.wholesale_margins[i],
                f"retailer {r} SKU {i} wholesale margin",
            )


def validate_state(instance: Instance, state: State) -> None:
    if not 1 <= state.period <= instance.periods:
        raise ValueError("state period must be within the instance periods")
    if state.order_retention.keys() != instance.retailers.keys():
        raise ValueError("state order_retention must cover exactly all retailers")
    for r, retention in state.order_retention.items():
        _unit_interval(retention, f"retailer {r} retention")


def make_initial_state(instance: Instance) -> State:
    state = State(
        period=1,
        order_retention={
            r: instance.initial_order_retention for r in instance.retailer_ids
        },
    )
    validate_state(instance, state)
    return state


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
