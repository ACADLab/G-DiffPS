"""Compose phase-shifter topologies from typed 2-port sections.

Parameter names carry unit suffixes (_mm, _pf, _nh) and a section infix (_s{k})
so ``action_to_params(bounds='electrical')`` routes bounds by element type.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional

import networkx as nx
import numpy as np

from env.netlist_graph import Dev, TOPOLOGY_NETLIST, build_circuit_graph
from env.topology_registry import original_six, register_topology
from sim.mna_scorer import solve_sparams
from train_diffusion import action_to_params


class SecKind(Enum):
    SER_L = auto()
    SER_C = auto()
    SER_TL = auto()
    SHUNT_L = auto()
    SHUNT_C = auto()
    SHUNT_STUB = auto()
    SW_SHUNT_C = auto()
    SW_SHUNT_L = auto()
    SW_SER_BYPASS = auto()
    BRIDGED_T = auto()
    BRANCH2 = auto()


# Passive arm vocabulary (no switches inside an arm).
ARM_KINDS = (
    SecKind.SER_L, SecKind.SER_C, SecKind.SER_TL,
    SecKind.SHUNT_L, SecKind.SHUNT_C, SecKind.SHUNT_STUB,
    SecKind.BRIDGED_T,
)

# Top-level section kinds for K=1,2 cascades.
TOP_KINDS = ARM_KINDS + (
    SecKind.SW_SHUNT_C, SecKind.SW_SHUNT_L, SecKind.SW_SER_BYPASS, SecKind.BRANCH2,
)

SWITCH_PARAM = "R_sw"


@dataclass
class Section:
    kind: SecKind
    polarity: int = 0  # on_in_states for switch-bearing sections
    arm_a: tuple["Section", ...] = field(default_factory=tuple)
    arm_b: tuple["Section", ...] = field(default_factory=tuple)


@dataclass
class ComposedTopology:
    name: str
    sections: tuple[Section, ...]
    netlist: dict[str, Dev]
    param_keys: list[str]
    port_nets: tuple[str, str] = ("in", "out")
    n_states: int = 2
    ideal_step: float = -90.0
    device_count: int = 0
    has_switch: bool = False


class _Builder:
    """Accumulate devices for one composition."""

    def __init__(self) -> None:
        self.devices: dict[str, Dev] = {}
        self.param_keys: list[str] = []
        self._seen_params: set[str] = set()
        self._dev_idx = 0
        self.has_switch = False

    def _uid(self, prefix: str) -> str:
        self._dev_idx += 1
        return f"{prefix}{self._dev_idx}"

    def _add_param(self, key: str) -> None:
        if key not in self._seen_params:
            self._seen_params.add(key)
            self.param_keys.append(key)

    def _add(self, dname: str, dev: Dev) -> None:
        self.devices[dname] = dev
        if dev.sizes:
            self._add_param(dev.sizes)
        for aux in dev.aux_sizes:
            self._add_param(aux)

    def _sw_states(self, polarity: int) -> tuple[int, ...]:
        return (0,) if polarity == 0 else (1,)

    def add_section(self, sec: Section, net_in: str, net_out: str,
                    prefix: str, sec_idx: int) -> None:
        p = f"{prefix}s{sec_idx}_"
        pol = sec.polarity
        on = self._sw_states(pol)

        if sec.kind == SecKind.SER_L:
            key = f"L_{prefix}s{sec_idx}_nh"
            self._add(f"L{p}", Dev("L", (net_in, net_out), sizes=key))

        elif sec.kind == SecKind.SER_C:
            key = f"C_{prefix}s{sec_idx}_pf"
            self._add(f"C{p}", Dev("C", (net_in, net_out), sizes=key))

        elif sec.kind == SecKind.SER_TL:
            zkey = f"Z0_{prefix}s{sec_idx}"
            lkey = f"L_{prefix}s{sec_idx}_mm"
            self._add(f"T{p}", Dev("TLine", (net_in, net_out), sizes=lkey,
                                    aux_sizes=(zkey,)))

        elif sec.kind == SecKind.SHUNT_L:
            key = f"L_{prefix}s{sec_idx}_nh"
            self._add(f"L{p}sh", Dev("L", (net_in, "0"), sizes=key))
            self._add(f"R{p}th", Dev("R_fixed", (net_in, net_out)))

        elif sec.kind == SecKind.SHUNT_C:
            key = f"C_{prefix}s{sec_idx}_pf"
            self._add(f"C{p}sh", Dev("C", (net_in, "0"), sizes=key))
            self._add(f"R{p}th", Dev("R_fixed", (net_in, net_out)))

        elif sec.kind == SecKind.SHUNT_STUB:
            zkey = f"Z0_{prefix}s{sec_idx}"
            lkey = f"L_{prefix}s{sec_idx}_mm"
            self._add(f"T{p}stub", Dev("TLine", (net_in, "0"), sizes=lkey,
                                       aux_sizes=(zkey,)))
            self._add(f"R{p}th", Dev("R_fixed", (net_in, net_out)))

        elif sec.kind == SecKind.SW_SHUNT_C:
            self.has_switch = True
            nld = f"{p}nld"
            key = f"C_{prefix}s{sec_idx}_pf"
            self._add(f"R{p}sw", Dev("R_switch", (net_in, nld),
                                     switch_param=SWITCH_PARAM, on_in_states=on))
            self._add(f"C{p}", Dev("C", (nld, "0"), sizes=key))
            self._add(f"R{p}th", Dev("R_fixed", (net_in, net_out)))

        elif sec.kind == SecKind.SW_SHUNT_L:
            self.has_switch = True
            nld = f"{p}nld"
            key = f"L_{prefix}s{sec_idx}_nh"
            self._add(f"R{p}sw", Dev("R_switch", (net_in, nld),
                                     switch_param=SWITCH_PARAM, on_in_states=on))
            self._add(f"L{p}", Dev("L", (nld, "0"), sizes=key))
            self._add(f"R{p}th", Dev("R_fixed", (net_in, net_out)))

        elif sec.kind == SecKind.SW_SER_BYPASS:
            self.has_switch = True
            key = f"L_{prefix}s{sec_idx}_nh"
            self._add(f"L{p}", Dev("L", (net_in, net_out), sizes=key))
            self._add(f"R{p}bp", Dev("R_switch", (net_in, net_out),
                                     switch_param=SWITCH_PARAM, on_in_states=on))

        elif sec.kind == SecKind.BRIDGED_T:
            m = f"{p}m"
            lkey = f"L_{prefix}s{sec_idx}_nh"
            cbr = f"C_{prefix}s{sec_idx}_br_pf"
            cc = f"C_{prefix}s{sec_idx}_c_pf"
            self._add(f"L{p}1", Dev("L", (net_in, m), sizes=lkey))
            self._add(f"L{p}2", Dev("L", (m, net_out), sizes=lkey))
            self._add(f"C{p}br", Dev("C", (net_in, net_out), sizes=cbr))
            self._add(f"C{p}c", Dev("C", (m, "0"), sizes=cc))

        else:
            raise ValueError(f"unknown section kind {sec.kind}")


def _build_arm_sections(
    arm: tuple[Section, ...],
    net_in: str,
    net_out: str,
    prefix: str,
    b: _Builder,
) -> None:
    if not arm:
        b._add(f"R{prefix}_wire", Dev("R_fixed", (net_in, net_out)))
        return
    nets = [net_in] + [f"{prefix}_n{i}" for i in range(len(arm) - 1)] + [net_out]
    for i, sec in enumerate(arm):
        b.add_section(sec, nets[i], nets[i + 1], prefix, i)


def compose_cascade(sections: tuple[Section, ...], name: str) -> ComposedTopology:
    b = _Builder()
    nets = ["in"] + [f"n{i}" for i in range(len(sections) - 1)] + ["out"]
    for i, sec in enumerate(sections):
        if sec.kind == SecKind.BRANCH2:
            b.has_switch = True
            p = f"c{i}_"
            a_in, a_out = f"{p}a_in", f"{p}a_out"
            b_in, b_out = f"{p}b_in", f"{p}b_out"
            b._add(f"R{p}inA", Dev("R_switch", (nets[i], a_in),
                                    switch_param=SWITCH_PARAM, on_in_states=(0,)))
            b._add(f"R{p}inB", Dev("R_switch", (nets[i], b_in),
                                    switch_param=SWITCH_PARAM, on_in_states=(1,)))
            _build_arm_sections(sec.arm_a, a_in, a_out, f"{p}A", b)
            _build_arm_sections(sec.arm_b, b_in, b_out, f"{p}B", b)
            b._add(f"R{p}outA", Dev("R_switch", (a_out, nets[i + 1]),
                                    switch_param=SWITCH_PARAM, on_in_states=(0,)))
            b._add(f"R{p}outB", Dev("R_switch", (b_out, nets[i + 1]),
                                    switch_param=SWITCH_PARAM, on_in_states=(1,)))
        else:
            b.add_section(sec, nets[i], nets[i + 1], "c", i)
    return ComposedTopology(
        name=name,
        sections=sections,
        netlist=b.devices,
        param_keys=b.param_keys,
        device_count=len(b.devices),
        has_switch=b.has_switch,
    )


def _is_gnd(n: str) -> bool:
    return n in ("0", "GND", "gnd", "ground")


def _adjacency_graph(netlist: dict[str, Dev], state: int) -> nx.Graph:
    g = nx.Graph()
    for dname, dev in netlist.items():
        if dev.dtype == "R_switch":
            on = state in dev.on_in_states
            if not on:
                continue
            pins = [n for n in dev.nets if not _is_gnd(n)]
            for i in range(len(pins)):
                for j in range(i + 1, len(pins)):
                    g.add_edge(pins[i], pins[j])
        elif dev.dtype in ("R_fixed", "C", "L", "TLine"):
            pins = [n for n in dev.nets if not _is_gnd(n)]
            if len(pins) >= 2:
                g.add_edge(pins[0], pins[1])
            elif len(pins) == 1:
                g.add_node(pins[0])
    return g


def has_path_both_states(netlist: dict[str, Dev], port_in: str, port_out: str) -> bool:
    for state in (0, 1):
        g = _adjacency_graph(netlist, state)
        if port_in not in g or port_out not in g:
            return False
        if not nx.has_path(g, port_in, port_out):
            return False
    return True


def count_switch_groups(netlist: dict[str, Dev]) -> int:
    params = {dev.switch_param for dev in netlist.values()
              if dev.switch_param}
    return len(params)


def validate_composed(ct: ComposedTopology, fc: float = 28.0) -> tuple[bool, str]:
    if not ct.has_switch:
        return False, "no_switch"
    if ct.device_count > 14:
        return False, "too_many_devices"
    if count_switch_groups(ct.netlist) > 1:
        return False, "multi_switch_group"
    if not has_path_both_states(ct.netlist, *ct.port_nets):
        return False, "no_path"
    # Register temporarily for MNA probe
    from env.topology_registry import register_topology, unregister_topology
    register_topology(
        ct.name, ct.netlist, ct.port_nets, ct.n_states, ct.param_keys,
        ct.ideal_step, overwrite=True,
    )
    try:
        from env.graph_utils import TOPOLOGY_PARAMS
        spec = {"fc_ghz": fc, "tech": 0}
        action = np.full(len(TOPOLOGY_PARAMS[ct.name]), 0.5)
        params = action_to_params(action, ct.name, spec, bounds="electrical",
                                  switch_model="ideal")
        for state in (0, 1):
            solve_sparams(ct.name, params, fc, state=state)
    except Exception as exc:
        return False, f"mna_fail:{exc}"
    finally:
        unregister_topology(ct.name)
    return True, "ok"


def _section(kind: SecKind, polarity: int = 0,
             arm_a: tuple[Section, ...] = (), arm_b: tuple[Section, ...] = ()) -> Section:
    return Section(kind=kind, polarity=polarity, arm_a=arm_a, arm_b=arm_b)


def _arm_section(kind: SecKind) -> Section:
    return Section(kind=kind, polarity=0)


def enumerate_candidates(max_k: int = 2) -> Iterator[ComposedTopology]:
    """Enumerate K=1 and K=2 cascades; validity applied by caller."""
    idx = 0
    polarities = (0, 1)

    def top_sections() -> Iterator[Section]:
        for kind in TOP_KINDS:
            if kind == SecKind.BRANCH2:
                for ak in ARM_KINDS:
                    for bk in ARM_KINDS:
                        if ak == bk:
                            continue
                        yield _section(SecKind.BRANCH2,
                                       arm_a=(_arm_section(ak),),
                                       arm_b=(_arm_section(bk),))
                        # Two-section arms for richer branch contrasts
                        for ak2 in ARM_KINDS:
                            if ak2 == ak:
                                continue
                            yield _section(
                                SecKind.BRANCH2,
                                arm_a=(_arm_section(ak), _arm_section(ak2)),
                                arm_b=(_arm_section(bk),),
                            )
            elif kind in (SecKind.SW_SHUNT_C, SecKind.SW_SHUNT_L, SecKind.SW_SER_BYPASS):
                for pol in polarities:
                    yield _section(kind, polarity=pol)
            else:
                yield _section(kind)

    tops = list(top_sections())

    for k in range(1, max_k + 1):
        if k == 1:
            combos = [(s,) for s in tops]
        else:
            combos = list(itertools.product(tops, repeat=2))
        for combo in combos:
            # Require at least one switch-bearing section in the cascade
            if not any(s.kind in (SecKind.SW_SHUNT_C, SecKind.SW_SHUNT_L,
                                  SecKind.SW_SER_BYPASS, SecKind.BRANCH2)
                       for s in combo):
                continue
            idx += 1
            name = f"Gen_{idx:03d}"
            ct = compose_cascade(combo, name)
            ok, _ = validate_composed(ct)
            if ok:
                yield ct


def stratified_subsample(candidates: list[ComposedTopology], n: int = 34,
                         seed: int = 42) -> list[ComposedTopology]:
    """Pick n topologies stratified by device count and section multiset."""
    if len(candidates) <= n:
        return candidates
    rng = np.random.default_rng(seed)

    def bucket(ct: ComposedTopology) -> tuple:
        kinds = tuple(sorted(s.kind.name for s in ct.sections))
        dc_bin = min(ct.device_count // 3, 4)
        return (dc_bin, kinds)

    buckets: dict[tuple, list[ComposedTopology]] = {}
    for ct in candidates:
        buckets.setdefault(bucket(ct), []).append(ct)

    picked: list[ComposedTopology] = []
    keys = list(buckets.keys())
    rng.shuffle(keys)
    per = max(1, n // len(keys))
    for key in keys:
        pool = buckets[key]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    if len(picked) < n:
        rest = [c for c in candidates if c not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]
