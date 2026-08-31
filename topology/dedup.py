"""Graph-isomorphism deduplication via 1-WL on the bipartite incidence graph."""
from __future__ import annotations

import hashlib
from collections import Counter

from env.netlist_graph import Dev, TOPOLOGY_NETLIST, PORT_NETS, build_circuit_graph
from env.topology_registry import register_topology, unregister_topology


def _net_role(net: str, port_in: str, port_out: str) -> str:
    if net in ("0", "GND", "gnd", "ground"):
        return "gnd"
    if net == port_in:
        return "port_in"
    if net == port_out:
        return "port_out"
    return "internal"


def _bipartite_wl_hash(netlist: dict[str, Dev],
                       port_nets: tuple[str, str] = ("in", "out"),
                       rounds: int = 4) -> str:
    """1-WL hash over (device dtype, net role) labels — values excluded."""
    port_in, port_out = port_nets
    dev_names = sorted(netlist.keys())
    net_set: set[str] = set()
    for dev in netlist.values():
        for n in dev.nets:
            if n not in ("0", "GND", "gnd", "ground"):
                net_set.add(n)
    net_names = sorted(net_set)

    dev_labels = {d: f"D:{netlist[d].dtype}" for d in dev_names}
    net_labels = {n: f"N:{_net_role(n, port_in, port_out)}" for n in net_names}

    for _ in range(rounds):
        new_dev: dict[str, str] = {}
        for d in dev_names:
            dev = netlist[d]
            nbrs: list[str] = []
            for pin, n in enumerate(dev.nets):
                if n in ("0", "GND", "gnd", "ground"):
                    nbrs.append(f"gnd:{pin}")
                else:
                    nbrs.append(f"{net_labels[n]}:{pin}")
            nbrs.sort()
            new_dev[d] = dev_labels[d] + "@" + "|".join(nbrs)
        new_net: dict[str, str] = {}
        for n in net_names:
            nbrs = []
            for d in dev_names:
                dev = netlist[d]
                if n in dev.nets:
                    pin = dev.nets.index(n)
                    nbrs.append(f"{dev_labels[d]}:{pin}")
            nbrs.sort()
            new_net[n] = net_labels[n] + "@" + "|".join(nbrs)
        dev_labels = new_dev
        net_labels = new_net

    sig = sorted([("D", d, dev_labels[d]) for d in dev_names] +
                 [("N", n, net_labels[n]) for n in net_names])
    return hashlib.sha256(repr(sig).encode()).hexdigest()[:16]


def hash_topology(name: str) -> str:
    nl = TOPOLOGY_NETLIST[name]
    ports = PORT_NETS.get(name, ("in", "out"))
    return _bipartite_wl_hash(nl, ports)


def hash_composed(netlist: dict[str, Dev],
                  port_nets: tuple[str, str] = ("in", "out")) -> str:
    return _bipartite_wl_hash(netlist, port_nets)


def seed_original_hashes() -> dict[str, str]:
    return {name: hash_topology(name) for name in original_six_names()}


def original_six_names() -> tuple[str, ...]:
    from env.topology_registry import original_six
    return tuple(sorted(original_six()))


def deduplicate(candidates: list, seeded: dict[str, str] | None = None) -> list:
    """Drop WL-hash collisions against seeded originals and within candidates."""
    seen = set(seeded.values()) if seeded else set()
    out = []
    for ct in candidates:
        h = hash_composed(ct.netlist, ct.port_nets)
        if h in seen:
            continue
        seen.add(h)
        ct.wl_hash = h  # type: ignore[attr-defined]
        out.append(ct)
    return out


def register_for_graph(name: str, ct) -> None:
    register_topology(
        name, ct.netlist, ct.port_nets, ct.n_states, ct.param_keys,
        ct.ideal_step, overwrite=True,
    )
