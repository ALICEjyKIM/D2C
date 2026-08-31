"""Load and validate optimization instances."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.types import Instance, Retailer, SKU, State


def load_instance(path: str | Path) -> Instance:
    """Load, parse, and validate an instance from JSON."""
    instance_path = Path(path)
    try:
        with instance_path.open(encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {instance_path}: {exc.msg}") from exc

    instance = _parse_instance(_as_mapping(raw_data, "root"))
    validate_instance(instance)
    return instance


def validate_instance(instance: Instance) -> None:
    """Raise ValueError when an instance violates model assumptions."""
    if not instance.instance_id.strip():
        raise ValueError("instance_id must be non-empty")
    if not instance.skus:
        raise ValueError("at least one SKU is required")
    if not instance.retailers:
        raise ValueError("at least one retailer is required")
    if instance.periods < 1:
        raise ValueError("periods must be at least 1")
    if not 1 <= instance.default_horizon <= instance.periods:
        raise ValueError("default_horizon must be between 1 and periods")
    if not 1 <= instance.max_d2c_skus <= len(instance.skus):
        raise ValueError("max_d2c_skus must be between 1 and the number of SKUs")
    _require_positive(instance.capacity, "capacity")
    _require_unit_interval(instance.beta, "beta")
    _require_unit_interval(instance.rho, "rho")
    _require_unit_interval(instance.kappa, "kappa")
    if not math.isfinite(instance.gamma) or not 0 < instance.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    _require_unit_interval(
        instance.initial_order_retention, "initial_order_retention"
    )

    for sku_id, sku in instance.skus.items():
        if sku_id != sku.sku_id or not sku_id.strip():
            raise ValueError(f"invalid SKU mapping key: {sku_id!r}")
        _require_nonnegative(sku.d2c_margin, f"SKU {sku_id} d2c_margin")
        _require_positive(sku.d2c_demand, f"SKU {sku_id} d2c_demand")
        _require_nonnegative(sku.supply_limit, f"SKU {sku_id} supply_limit")
        _require_positive(sku.capacity_use, f"SKU {sku_id} capacity_use")

    for retailer_id, retailer in instance.retailers.items():
        if retailer_id != retailer.retailer_id or not retailer_id.strip():
            raise ValueError(f"invalid retailer mapping key: {retailer_id!r}")
        if not retailer.base_orders:
            raise ValueError(f"retailer {retailer_id} must carry at least one SKU")
        if retailer.base_orders.keys() != retailer.wholesale_margins.keys():
            raise ValueError(
                f"retailer {retailer_id} base_orders and wholesale_margins "
                "must have identical SKU keys"
            )
        unknown_skus = retailer.base_orders.keys() - instance.skus.keys()
        if unknown_skus:
            raise ValueError(
                f"retailer {retailer_id} references unknown SKUs: "
                f"{sorted(unknown_skus)}"
            )
        for sku_id, base_order in retailer.base_orders.items():
            _require_nonnegative(
                base_order, f"retailer {retailer_id} SKU {sku_id} base order"
            )
            _require_nonnegative(
                retailer.wholesale_margins[sku_id],
                f"retailer {retailer_id} SKU {sku_id} wholesale margin",
            )


def validate_state(instance: Instance, state: State) -> None:
    """Validate state coverage, period, and retention bounds."""
    if not 1 <= state.period <= instance.periods:
        raise ValueError("state period must be within the instance periods")
    if state.order_retention.keys() != instance.retailers.keys():
        raise ValueError("state order_retention must cover exactly all retailers")
    for retailer_id, retention in state.order_retention.items():
        _require_unit_interval(retention, f"retailer {retailer_id} retention")


def make_initial_state(instance: Instance) -> State:
    """Create the period-one state specified by the instance."""
    state = State(
        period=1,
        order_retention={
            retailer_id: instance.initial_order_retention
            for retailer_id in instance.retailer_ids
        },
    )
    validate_state(instance, state)
    return state


def _parse_instance(data: Mapping[str, Any]) -> Instance:
    skus_data = _as_mapping(_required(data, "skus"), "skus")
    retailers_data = _as_mapping(_required(data, "retailers"), "retailers")

    skus = {
        sku_id: _parse_sku(sku_id, _as_mapping(values, f"skus.{sku_id}"))
        for sku_id, values in skus_data.items()
    }
    retailers = {
        retailer_id: _parse_retailer(
            retailer_id,
            _as_mapping(values, f"retailers.{retailer_id}"),
        )
        for retailer_id, values in retailers_data.items()
    }

    return Instance(
        instance_id=_as_string(_required(data, "instance_id"), "instance_id"),
        periods=_as_integer(_required(data, "periods"), "periods"),
        default_horizon=_as_integer(
            _required(data, "default_horizon"), "default_horizon"
        ),
        max_d2c_skus=_as_integer(
            _required(data, "max_d2c_skus"), "max_d2c_skus"
        ),
        capacity=_as_number(_required(data, "capacity"), "capacity"),
        beta=_as_number(_required(data, "beta"), "beta"),
        rho=_as_number(_required(data, "rho"), "rho"),
        kappa=_as_number(_required(data, "kappa"), "kappa"),
        gamma=_as_number(_required(data, "gamma"), "gamma"),
        initial_order_retention=_as_number(
            _required(data, "initial_order_retention"),
            "initial_order_retention",
        ),
        skus=skus,
        retailers=retailers,
    )


def _parse_sku(sku_id: str, data: Mapping[str, Any]) -> SKU:
    return SKU(
        sku_id=sku_id,
        d2c_margin=_as_number(
            _required(data, "d2c_margin"), f"skus.{sku_id}.d2c_margin"
        ),
        d2c_demand=_as_number(
            _required(data, "d2c_demand"), f"skus.{sku_id}.d2c_demand"
        ),
        supply_limit=_as_number(
            _required(data, "supply_limit"), f"skus.{sku_id}.supply_limit"
        ),
        capacity_use=_as_number(
            _required(data, "capacity_use"), f"skus.{sku_id}.capacity_use"
        ),
    )


def _parse_retailer(retailer_id: str, data: Mapping[str, Any]) -> Retailer:
    orders_data = _as_mapping(
        _required(data, "base_orders"), f"retailers.{retailer_id}.base_orders"
    )
    margins_data = _as_mapping(
        _required(data, "wholesale_margins"),
        f"retailers.{retailer_id}.wholesale_margins",
    )
    return Retailer(
        retailer_id=retailer_id,
        base_orders={
            sku_id: _as_number(value, f"{retailer_id}.{sku_id}.base_order")
            for sku_id, value in orders_data.items()
        },
        wholesale_margins={
            sku_id: _as_number(value, f"{retailer_id}.{sku_id}.wholesale_margin")
            for sku_id, value in margins_data.items()
        },
    )


def _required(data: Mapping[str, Any], key: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:
        raise ValueError(f"missing required field: {key}") from exc


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _as_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _as_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _as_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _require_nonnegative(value: float, field: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")


def _require_positive(value: float, field: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive")


def _require_unit_interval(value: float, field: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field} must be in [0, 1]")
