"""Runtime registration of composed topologies into the global name-keyed tables.

Generalizes the temporary injection pattern from ``tools/shunt_isolation_probe.py``
into a public API so ``mna_scorer``, ``compute_envelope``, and ``build_circuit_graph``
work on newly composed circuits without refactoring their signatures.
"""
from __future__ import annotations

import copy
from typing import Optional

from env import graph_utils
from env.netlist_graph import (
    Dev,
    N_STATES,
    PORT_NETS,
    TOPOLOGY_NETLIST,
    _normalize_name,
)
from sim import mna_scorer


# Snapshots for ``unregister`` / test isolation.
_BACKUP: dict[str, dict] = {}


def _snapshot(name: str) -> dict:
    return {
        "netlist": copy.deepcopy(TOPOLOGY_NETLIST.get(name)),
        "port_nets": PORT_NETS.get(name),
        "n_states": N_STATES.get(name),
        "param_keys": list(graph_utils.TOPOLOGY_PARAMS.get(name, [])),
        "ideal_step": mna_scorer._IDEAL_STEP.get(name),
    }


def register_topology(
    name: str,
    netlist: dict[str, Dev],
    port_nets: tuple[str, str] = ("in", "out"),
    n_states: int = 2,
    param_keys: Optional[list[str]] = None,
    ideal_step: float = -90.0,
    *,
    overwrite: bool = False,
) -> None:
    """Register a topology into all five name-keyed tables."""
    norm = _normalize_name(name) if name in TOPOLOGY_NETLIST else name
    if norm in TOPOLOGY_NETLIST and not overwrite:
        raise ValueError(f"topology {norm!r} already registered; pass overwrite=True")
    if norm in TOPOLOGY_NETLIST and norm not in _BACKUP:
        _BACKUP[norm] = _snapshot(norm)

    keys = list(param_keys or [])
    # Collect sized keys from the netlist if not supplied.
    if not keys:
        seen = set()
        for dev in netlist.values():
            if dev.sizes and dev.sizes not in seen:
                seen.add(dev.sizes)
                keys.append(dev.sizes)
            for aux in dev.aux_sizes:
                if aux not in seen:
                    seen.add(aux)
                    keys.append(aux)

    TOPOLOGY_NETLIST[norm] = copy.deepcopy(netlist)
    PORT_NETS[norm] = port_nets
    N_STATES[norm] = int(n_states)
    graph_utils.TOPOLOGY_PARAMS[norm] = keys
    mna_scorer._IDEAL_STEP[norm] = float(ideal_step)


def unregister_topology(name: str) -> None:
    """Remove a topology or restore a pre-registration snapshot."""
    norm = _normalize_name(name) if name in TOPOLOGY_NETLIST else name
    snap = _BACKUP.pop(norm, None)
    if not snap:
        TOPOLOGY_NETLIST.pop(norm, None)
        PORT_NETS.pop(norm, None)
        N_STATES.pop(norm, None)
        graph_utils.TOPOLOGY_PARAMS.pop(norm, None)
        mna_scorer._IDEAL_STEP.pop(norm, None)
        return
    if snap.get("netlist") is None:
        TOPOLOGY_NETLIST.pop(norm, None)
        PORT_NETS.pop(norm, None)
        N_STATES.pop(norm, None)
        graph_utils.TOPOLOGY_PARAMS.pop(norm, None)
        mna_scorer._IDEAL_STEP.pop(norm, None)
    else:
        TOPOLOGY_NETLIST[norm] = snap["netlist"]
        PORT_NETS[norm] = snap["port_nets"]
        N_STATES[norm] = snap["n_states"]
        graph_utils.TOPOLOGY_PARAMS[norm] = snap["param_keys"]
        if snap["ideal_step"] is not None:
            mna_scorer._IDEAL_STEP[norm] = snap["ideal_step"]


def registered_topologies() -> list[str]:
    return list(graph_utils.TOPOLOGY_PARAMS.keys())


def original_six() -> frozenset[str]:
    return frozenset({
        "Loaded_Line", "Switched_Line", "Reflection_Type",
        "Switched_Filter", "Vector_Modulator", "All_Pass",
    })
