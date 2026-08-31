"""Serialize Dev netlists for persistence."""
from __future__ import annotations

from env.netlist_graph import Dev


def dev_to_dict(d: Dev) -> dict:
    return {
        "dtype": d.dtype,
        "nets": list(d.nets),
        "sizes": d.sizes,
        "switch_param": d.switch_param,
        "on_in_states": list(d.on_in_states),
        "aux_sizes": list(d.aux_sizes),
        "is_control_pins": list(d.is_control_pins),
    }


def dev_from_dict(d: dict) -> Dev:
    return Dev(
        dtype=d["dtype"],
        nets=tuple(d["nets"]),
        sizes=d.get("sizes"),
        switch_param=d.get("switch_param"),
        on_in_states=tuple(d.get("on_in_states") or ()),
        aux_sizes=tuple(d.get("aux_sizes") or ()),
        is_control_pins=tuple(d.get("is_control_pins") or ()),
    )


def netlist_to_dict(nl: dict[str, Dev]) -> dict[str, dict]:
    return {k: dev_to_dict(v) for k, v in nl.items()}


def netlist_from_dict(raw: dict[str, dict]) -> dict[str, Dev]:
    return {k: dev_from_dict(v) for k, v in raw.items()}
