"""Load and register the open-topology pool."""
from __future__ import annotations

import json
import os

from env.topology_registry import original_six, register_topology
from topology.serialize import netlist_from_dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POOL = os.path.join(REPO_ROOT, "results", "open_topo", "pool.json")


def load_pool(path: str = DEFAULT_POOL) -> dict:
    with open(path) as fh:
        return json.load(fh)


def all_topology_names(pool: dict | None = None) -> list[str]:
    pool = pool or load_pool()
    return sorted(original_six()) + [c["name"] for c in pool["composed"]]


def register_composed(pool: dict | None = None, path: str = DEFAULT_POOL) -> list[str]:
    """Register composed topologies from pool.json; return their names."""
    pool = pool or load_pool(path)
    names = []
    for entry in pool["composed"]:
        from sim import mna_scorer
        step = float(entry.get("ideal_step_deg", -90.0))
        register_topology(
            entry["name"],
            netlist_from_dict(entry["netlist"]),
            tuple(entry.get("port_nets", ("in", "out"))),
            2,
            entry["param_keys"],
            step,
            overwrite=True,
        )
        mna_scorer._IDEAL_STEP[entry["name"]] = step
        names.append(entry["name"])
    return names


def register_all(path: str = DEFAULT_POOL) -> list[str]:
    register_composed(path=path)
    return all_topology_names(load_pool(path))
