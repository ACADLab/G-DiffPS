"""Bipartite device/net incidence graphs for phase-shifter topologies.

The incidence matrix A (rows=nets, columns=devices) is the KCL/KVL
structure of each netlist: KCL is A i = 0, KVL is v = A^T e.

`TOPOLOGY_NETLIST` is transcribed verbatim from specset/templates/*.sp.
`build_circuit_graph` returns a HeteroData with net nodes, device nodes,
and typed incidence edges. Spec frequency and switch state enter as
node features so the embedding is continuous in the design request.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

# Device type vocabulary (order is the one-hot index).
DEVICE_TYPES = ["TLine", "R_switch", "R_fixed", "C", "L", "VCVS", "Port"]
DEVICE_TYPE_IDX = {t: i for i, t in enumerate(DEVICE_TYPES)}
N_DEVICE_TYPES = len(DEVICE_TYPES)

# Velocity of propagation matching the SPICE templates (eps_eff=2.5).
VP = 3e8 / math.sqrt(2.5)  # 1.897e8 m/s
Z0_REF = 50.0


@dataclass
class Dev:
    """One device in a topology netlist."""

    dtype: str
    nets: tuple  # ordered pin -> net name; "0" / "GND" are ground
    sizes: Optional[str] = None  # TOPOLOGY_PARAMS key, or None if not policy-sized
    # For switches: which STATE_TABLE symbol controls on/off, and which
    # states have this switch closed (R_on). Empty => not a state switch.
    switch_param: Optional[str] = None
    on_in_states: tuple = field(default_factory=tuple)
    # Extra sized params that share this device (e.g. Z0 on a TLine).
    aux_sizes: tuple = field(default_factory=tuple)
    # For VCVS: pin roles — (out+, out-, ctrl+, ctrl-).
    is_control_pins: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Declarative netlists (device -> connectivity), transcribed from templates.
# Port convention: first signal net is the RF input, last is the RF output.
# Ground is always named "0".
# ---------------------------------------------------------------------------

TOPOLOGY_NETLIST: dict[str, dict[str, Dev]] = {
    "Loaded_Line": {
        "T_main": Dev("TLine", ("in", "out"), sizes="L_quarter_mm",
                      aux_sizes=("Z0_line",)),
        "R_in_path": Dev("R_switch", ("in", "n_in"),
                         switch_param="R_path_in", on_in_states=(1,)),
        "C_in_load": Dev("C", ("n_in", "0"), sizes="C_load_pf"),
        "R_out_path": Dev("R_switch", ("out", "n_out"),
                          switch_param="R_path_out", on_in_states=(1,)),
        "C_out_load": Dev("C", ("n_out", "0"), sizes="C_load_pf"),
    },
    "Switched_Line": {
        "R_in_short": Dev("R_switch", ("in", "a_short"),
                          switch_param="R_short_in", on_in_states=(0,)),
        "R_in_long": Dev("R_switch", ("in", "a_long"),
                         switch_param="R_long_in", on_in_states=(1,)),
        "T_short": Dev("TLine", ("a_short", "b_short"), sizes="L_short_mm",
                       aux_sizes=("Z0_line",)),
        "T_long": Dev("TLine", ("a_long", "b_long"), sizes="L_long_mm",
                      aux_sizes=("Z0_line",)),
        "R_out_short": Dev("R_switch", ("b_short", "out"),
                           switch_param="R_short_out", on_in_states=(0,)),
        "R_out_long": Dev("R_switch", ("b_long", "out"),
                          switch_param="R_long_out", on_in_states=(1,)),
    },
    "Reflection_Type": {
        "T_top": Dev("TLine", ("p1", "p2"), sizes="L_quarter_mm",
                     aux_sizes=("Z0_branch",)),
        "T_bottom": Dev("TLine", ("p3", "p4"), sizes="L_quarter_mm",
                        aux_sizes=("Z0_branch",)),
        "T_left": Dev("TLine", ("p1", "p3"), sizes="L_quarter_mm",
                      aux_sizes=("Z0_main",)),
        "T_right": Dev("TLine", ("p2", "p4"), sizes="L_quarter_mm",
                       aux_sizes=("Z0_main",)),
        "C_baseA": Dev("C", ("p2", "0"), sizes="C_base_pf"),
        "C_tuneA": Dev("C", ("p2", "nA"), sizes="C_tune_pf"),
        "R_pathA": Dev("R_switch", ("nA", "0"),
                       switch_param="R_path", on_in_states=(1,)),
        "C_baseB": Dev("C", ("p3", "0"), sizes="C_base_pf"),
        "C_tuneB": Dev("C", ("p3", "nB"), sizes="C_tune_pf"),
        "R_pathB": Dev("R_switch", ("nB", "0"),
                       switch_param="R_path", on_in_states=(1,)),
    },
    "Switched_Filter": {
        "R_in_hpf": Dev("R_switch", ("in", "a_hpf"),
                        switch_param="R_hpf_in", on_in_states=(0,)),
        "Lp_hpf_in": Dev("L", ("a_hpf", "0"), sizes="L_hpf_nh"),
        "C_hpf_ser": Dev("C", ("a_hpf", "b_hpf"), sizes="C_hpf_pf"),
        "Lp_hpf_out": Dev("L", ("b_hpf", "0"), sizes="L_hpf_nh"),
        "R_out_hpf": Dev("R_switch", ("b_hpf", "out"),
                         switch_param="R_hpf_out", on_in_states=(0,)),
        "R_in_lpf": Dev("R_switch", ("in", "a_lpf"),
                        switch_param="R_lpf_in", on_in_states=(1,)),
        "Cp_lpf_in": Dev("C", ("a_lpf", "0"), sizes="C_lpf_pf"),
        "L_lpf_ser": Dev("L", ("a_lpf", "b_lpf"), sizes="L_lpf_nh"),
        "Cp_lpf_out": Dev("C", ("b_lpf", "0"), sizes="C_lpf_pf"),
        "R_out_lpf": Dev("R_switch", ("b_lpf", "out"),
                         switch_param="R_lpf_out", on_in_states=(1,)),
    },
    "Vector_Modulator": {
        "T_quad": Dev("TLine", ("in", "q_in"), sizes="L_quarter_mm",
                      aux_sizes=("Z0_line",)),
        "R_q_term": Dev("R_fixed", ("q_in", "0")),
        # VCVS pins: (out+, out-, ctrl+, ctrl-)
        "E_I": Dev("VCVS", ("sum", "inter", "in", "0"),
                   sizes="G_I_scale",
                   is_control_pins=(False, False, True, True)),
        "E_Q": Dev("VCVS", ("inter", "0", "q_in", "0"),
                   sizes="G_Q_scale",
                   is_control_pins=(False, False, True, True)),
        "R_drv_out": Dev("R_fixed", ("sum", "out")),
    },
    "All_Pass": {
        "R_in_apA": Dev("R_switch", ("in", "a_in"),
                        switch_param="R_apA_in", on_in_states=(0,)),
        "L_apA_ser1": Dev("L", ("a_in", "m_A"), sizes="L_apA_nh"),
        "L_apA_ser2": Dev("L", ("m_A", "a_out"), sizes="L_apA_nh"),
        "C_brA_brg": Dev("C", ("a_in", "a_out"), sizes="C_brA_pf"),
        "C_cA_shnt": Dev("C", ("m_A", "0"), sizes="C_cA_pf"),
        "R_out_apA": Dev("R_switch", ("a_out", "out"),
                         switch_param="R_apA_out", on_in_states=(0,)),
        "R_in_apB": Dev("R_switch", ("in", "b_in"),
                        switch_param="R_apB_in", on_in_states=(1,)),
        "L_apB_ser1": Dev("L", ("b_in", "m_B"), sizes="L_apB_nh"),
        "L_apB_ser2": Dev("L", ("m_B", "b_out"), sizes="L_apB_nh"),
        "C_brB_brg": Dev("C", ("b_in", "b_out"), sizes="C_brB_pf"),
        "C_cB_shnt": Dev("C", ("m_B", "0"), sizes="C_cB_pf"),
        "R_out_apB": Dev("R_switch", ("b_out", "out"),
                         switch_param="R_apB_out", on_in_states=(1,)),
    },
}

# Canonical port nets per topology (input, output).
PORT_NETS = {
    "Loaded_Line": ("in", "out"),
    "Switched_Line": ("in", "out"),
    "Reflection_Type": ("p1", "p4"),
    "Switched_Filter": ("in", "out"),
    "Vector_Modulator": ("in", "out"),
    "All_Pass": ("in", "out"),
}

# Number of phase states used for encoder pooling (VM has 16; we pool
# endpoints 0 and N/2 by default to keep cost bounded; full sweep optional).
N_STATES = {
    "Loaded_Line": 2,
    "Switched_Line": 2,
    "Reflection_Type": 2,
    "Switched_Filter": 2,
    "Vector_Modulator": 16,
    "All_Pass": 2,
}


# Canonical Vector_Modulator state table (cos/sin of k*22.5 deg), matching the
# template. The VM selects its phase state through the VCVS gains rather than a
# switch, so this table is what makes its graph state-dependent at all.
VM_IQ = [
    (math.cos(k * math.pi / 8), math.sin(k * math.pi / 8))
    for k in range(16)
]


def _normalize_name(topology_name: str) -> str:
    for key in TOPOLOGY_NETLIST:
        if key.lower().replace("_", "") == topology_name.lower().replace("_", ""):
            return key
    raise ValueError(f"Unknown topology: {topology_name}")


def _is_gnd(net: str) -> bool:
    return net in ("0", "GND", "gnd", "ground")


def device_names(topology_name: str) -> list[str]:
    name = _normalize_name(topology_name)
    return list(TOPOLOGY_NETLIST[name].keys())


def sized_devices(topology_name: str) -> list[tuple[str, str]]:
    """Return [(device_name, param_key), ...] for policy-sized devices.

    Devices that share a param (e.g. both C_in_load and C_out_load size
    C_load_pf) appear once per unique param, preferring the first device.
    """
    name = _normalize_name(topology_name)
    seen = set()
    out = []
    for dname, dev in TOPOLOGY_NETLIST[name].items():
        if dev.sizes and dev.sizes not in seen:
            seen.add(dev.sizes)
            out.append((dname, dev.sizes))
        for aux in dev.aux_sizes:
            if aux not in seen:
                seen.add(aux)
                out.append((dname, aux))
    return out


def _electrical_size(dtype: str, fc_ghz: float, nominal: float | None = None) -> float:
    """Dimensionless electrical size at fc (log10 of a normalized reactance).

    `nominal` is the device's physical value in template units (mm for TLine,
    nH for L, pF for C, ohms for R, gain for VCVS). When None the device falls
    back to its resonant value at fc, which makes the feature a function of
    type and frequency only -- see `nominal_params` for the sized path.
    """
    omega = 2.0 * math.pi * fc_ghz * 1e9
    if dtype == "TLine":
        # beta * l with l = lam4 by default
        lam4_m = VP / (4.0 * fc_ghz * 1e9)
        length_m = (nominal * 1e-3) if nominal is not None else lam4_m
        beta_l = omega * length_m / VP
        return float(math.log10(max(beta_l, 1e-6)))
    if dtype == "L":
        L = (nominal * 1e-9) if nominal is not None else Z0_REF / omega
        return float(math.log10(max(omega * L / Z0_REF, 1e-6)))
    if dtype == "C":
        C = (nominal * 1e-12) if nominal is not None else 1.0 / (omega * Z0_REF)
        return float(math.log10(max(1.0 / (omega * C * Z0_REF), 1e-6)))
    if dtype in ("R_switch", "R_fixed", "Port"):
        R = nominal if nominal is not None else Z0_REF
        return float(math.log10(max(abs(R) / Z0_REF, 1e-6)))
    if dtype == "VCVS":
        # A controlled source has a signed gain, not a reactance. The sign is
        # the whole point for the Vector Modulator, whose states differ by the
        # sign and ratio of the I/Q drives, so it is kept rather than logged.
        g = nominal if nominal is not None else 1.0
        return float(max(-10.0, min(10.0, g)))
    return 0.0


# Nominal sizing cache: (topology, fc_bucket, bounds) -> params dict.
_NOMINAL_CACHE: dict = {}

# Frequency resolution of the topology embedding. Graph construction is cached
# per bucket, so two specs inside one bucket receive the same z_tau: a
# deliberate cost/precision trade, and the reason the number of distinct
# embeddings is capped near |T| x n_buckets rather than by the spec set. Ten
# buckets per decade is ~13 over the 2-40 GHz range the specs cover.
FC_BUCKETS_PER_DECADE = 10


def fc_bucket(fc_ghz: float) -> int:
    """Log-spaced frequency bucket, shared by sizing and the encoder cache."""
    return int(round(FC_BUCKETS_PER_DECADE * math.log10(max(fc_ghz, 1.0))))


def bucket_fc_ghz(fc_ghz: float) -> float:
    """Representative frequency of the bucket containing `fc_ghz`."""
    return float(10.0 ** (fc_bucket(fc_ghz) / FC_BUCKETS_PER_DECADE))


def nominal_params(
    topology_name: str,
    spec_dict: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
) -> dict:
    """Spec-derived nominal sizing: the midpoint of the actor's sampling bounds.

    The encoder runs *before* the actor picks sizes, so real chosen values are
    not available when the topology embedding is formed. The midpoint of the
    bounds the actor will sample within is the natural stand-in: it varies with
    the spec (through fc-centred bounds) and stays automatically in sync with
    changes to `action_to_params`.

    R_on/R_off come from `switch_model` + tech, not from the midpoint action.

    Falls back to an empty dict -- i.e. resonant defaults -- if the sizing
    tables cannot be imported.
    """
    name = _normalize_name(topology_name)
    fc_ghz = float((spec_dict or {}).get("fc_ghz", 28.0))
    tech = int((spec_dict or {}).get("tech", 0))
    # Quantised to the same bucket the encoder caches on, so a cache hit and a
    # freshly built graph cannot disagree about device values.
    fc_q = bucket_fc_ghz(fc_ghz)
    key = (name, fc_bucket(fc_ghz), bounds, switch_model, tech)
    if key in _NOMINAL_CACHE:
        return _NOMINAL_CACHE[key]

    params: dict = {}
    try:
        # Imported lazily: train_diffusion imports this module at load time.
        from train_diffusion import TOPOLOGY_PARAMS, action_to_params

        keys = TOPOLOGY_PARAMS[name]
        mid = np.full(len(keys), 0.5, dtype=np.float32)
        params = action_to_params(
            mid, name, {"fc_ghz": fc_q, "tech": tech},
            sizing="log", bounds=bounds, switch_model=switch_model,
        )
        params = {k: v for k, v in params.items()}
    except Exception:
        params = {}

    _NOMINAL_CACHE[key] = params
    return params


def _as_float(params: dict, key: str | None, default: float | None) -> float | None:
    """Read a numeric param, tolerating SPICE suffix strings like '10k'."""
    if not key or key not in params:
        return default
    v = params[key]
    if isinstance(v, (int, float)):
        return float(v)
    try:
        from sim.mna_scorer import parse_spice_number

        return float(parse_spice_number(v, default if default is not None else 0.0))
    except Exception:
        return default


def _vcvs_gain(
    topology: str, dname: str, dev: Dev, params: dict, state: int
) -> float:
    """Gain of a VCVS in a given state, mirroring `mna_scorer.solve_sparams`.

    The Vector Modulator has no switches: it selects its phase state entirely
    through the I/Q drive gains. Without this the state index does not reach
    the graph at all, and every VM state encodes identically.
    """
    if topology == "Vector_Modulator" and dname in ("E_I", "E_Q"):
        g_i, g_q = VM_IQ[int(state) % len(VM_IQ)]
        if dname == "E_I":
            return 2.0 * g_i * (_as_float(params, "G_I_scale", 1.0) or 1.0)
        return 2.0 * g_q * (_as_float(params, "G_Q_scale", 1.0) or 1.0)
    return _as_float(params, dev.sizes, 1.0) or 1.0


def graph_device_names(topology_name: str, include_ports: bool = True) -> list[str]:
    """Device rows of `build_circuit_graph`, in order.

    Real devices keep the indices returned by `device_names()`; the synthetic
    port terminations are appended last so downstream row lookups by name stay
    valid.
    """
    names = device_names(topology_name)
    if include_ports:
        names = names + ["P_in", "P_out"]
    return names


def _port_devices(topology_name: str) -> dict[str, Dev]:
    """The 50-ohm source / load terminations, as explicit devices.

    The SPICE templates contain `Rsrc` and `Rload`, and `solve_sparams` stamps
    both as 1/Z0 to ground at the port nets. They were previously invisible to
    the encoder, so the port impedance the design is matched against was not
    represented anywhere in the graph.
    """
    port_in, port_out = PORT_NETS[_normalize_name(topology_name)]
    return {
        "P_in": Dev("Port", (port_in, "0")),
        "P_out": Dev("Port", (port_out, "0")),
    }


def build_circuit_graph(
    topology_name: str,
    spec_dict: dict | None = None,
    state: int = 0,
    params: dict | None = None,
    bounds: str = "electrical",
    include_ports: bool = True,
    switch_model: str = "ideal",
) -> HeteroData:
    """Build a HeteroData bipartite device/net incidence graph.

    Node features
    -------------
    net.x   : [is_ground, is_port_in, is_port_out, is_internal, log_degree]
    device.x: [type one-hot (7), is_sized, is_shunt, n_terminals,
               switch_is_on, log10(fc_ghz), elec_size, z0_norm,
               d_in_min, d_in_max, d_out_min, d_out_max, d_gnd_min, d_gnd_max]

    Edge features (device <-> net)
    ------------------------------
    edge_attr: [pin_index / 3, is_control_pin, sign,
                d_in_pin, d_out_pin, d_gnd_pin]

    `params` supplies physical device values; when omitted the midpoint of the
    actor's sampling bounds is used (`nominal_params`). Distances are per-pin,
    aggregated to (min, max) over a device's pins so they do not depend on the
    order the pin tuple was written in.
    """
    name = _normalize_name(topology_name)
    real_devices = TOPOLOGY_NETLIST[name]
    port_in, port_out = PORT_NETS[name]
    fc_ghz = float((spec_dict or {}).get("fc_ghz", 28.0))
    log_fc = math.log10(max(fc_ghz, 1e-3))

    if params is None:
        params = nominal_params(
            name, spec_dict, bounds=bounds, switch_model=switch_model,
        )
    r_on = _as_float(params, "R_on", 3.0)
    r_off = _as_float(params, "R_off", 1e4)

    devices = dict(real_devices)
    if include_ports:
        devices.update(_port_devices(name))

    # Collect nets: ground always index 0.
    net_names: list[str] = ["0"]
    net_index = {"0": 0}
    for dev in devices.values():
        for n in dev.nets:
            key = "0" if _is_gnd(n) else n
            if key not in net_index:
                net_index[key] = len(net_names)
                net_names.append(key)

    # Shortest-path distances on the net graph induced by the *real* devices.
    # The port terminations are excluded: they short every port to ground, which
    # would collapse dist_to_gnd for the network actually being designed.
    G = nx.Graph()
    G.add_nodes_from(net_names)
    for dname, dev in real_devices.items():
        pins = ["0" if _is_gnd(n) else n for n in dev.nets]
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                if pins[i] != pins[j]:
                    G.add_edge(pins[i], pins[j])

    def _dist(src: str, dst: str) -> float:
        if src not in G or dst not in G:
            return 5.0
        try:
            return float(nx.shortest_path_length(G, src, dst))
        except nx.NetworkXNoPath:
            return 5.0

    # Net features (degree from the full graph, so ports register as connected)
    G_full = G.copy()
    for dev in devices.values():
        pins = ["0" if _is_gnd(n) else n for n in dev.nets]
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                if pins[i] != pins[j]:
                    G_full.add_edge(pins[i], pins[j])

    net_feats = []
    for n in net_names:
        is_gnd = 1.0 if n == "0" else 0.0
        is_in = 1.0 if n == port_in else 0.0
        is_out = 1.0 if n == port_out else 0.0
        is_internal = 1.0 - max(is_gnd, is_in, is_out)
        deg = float(G_full.degree(n)) if n in G_full else 0.0
        net_feats.append([
            is_gnd, is_in, is_out, is_internal, math.log1p(deg),
        ])
    net_x = torch.tensor(net_feats, dtype=torch.float)

    # Device features + incidence edges
    dev_names = list(devices.keys())
    dev_feats = []
    edge_src, edge_dst, edge_attr = [], [], []  # device -> net
    edge_src_r, edge_dst_r, edge_attr_r = [], [], []  # net -> device

    for di, dname in enumerate(dev_names):
        dev = devices[dname]
        type_oh = [0.0] * N_DEVICE_TYPES
        type_oh[DEVICE_TYPE_IDX[dev.dtype]] = 1.0
        pins = ["0" if _is_gnd(n) else n for n in dev.nets]
        is_shunt = 1.0 if "0" in pins else 0.0
        is_sized = 1.0 if (dev.sizes or dev.aux_sizes) else 0.0

        switch_on = 0.0
        if dev.switch_param is not None:
            switch_on = 1.0 if int(state) in dev.on_in_states else 0.0

        # Physical value driving the electrical feature.
        if dev.dtype == "R_switch":
            nominal = r_on if switch_on > 0.5 else r_off
        elif dev.dtype in ("R_fixed", "Port"):
            nominal = Z0_REF
        elif dev.dtype == "VCVS":
            nominal = _vcvs_gain(name, dname, dev, params, state)
        else:
            nominal = _as_float(params, dev.sizes, None)
        elec = _electrical_size(dev.dtype, fc_ghz, nominal)

        # Characteristic impedance, where the device has one.
        z0_dev = None
        if dev.aux_sizes:
            z0_dev = _as_float(params, dev.aux_sizes[0], None)
        elif dev.dtype == "Port":
            z0_dev = Z0_REF
        z0_norm = math.log10(max(z0_dev, 1e-6) / Z0_REF) if z0_dev else 0.0

        # Per-pin distances, aggregated order-invariantly.
        d_in_pins = [_dist(p, port_in) / 5.0 for p in pins]
        d_out_pins = [_dist(p, port_out) / 5.0 for p in pins]
        d_gnd_pins = [_dist(p, "0") / 5.0 for p in pins]

        feat = type_oh + [
            is_sized, is_shunt, float(len(pins)), switch_on,
            log_fc, elec, z0_norm,
            min(d_in_pins), max(d_in_pins),
            min(d_out_pins), max(d_out_pins),
            min(d_gnd_pins), max(d_gnd_pins),
        ]
        dev_feats.append(feat)

        ctrl = list(dev.is_control_pins) if dev.is_control_pins else [False] * len(pins)
        while len(ctrl) < len(pins):
            ctrl.append(False)
        for pin_i, net in enumerate(pins):
            ni = net_index[net]
            # sign: +1 for pin 0, -1 for pin 1, 0 for extras (KVL convention)
            if pin_i == 0:
                sign = 1.0
            elif pin_i == 1:
                sign = -1.0
            else:
                sign = 0.0
            attr = [
                pin_i / 3.0, 1.0 if ctrl[pin_i] else 0.0, sign,
                d_in_pins[pin_i], d_out_pins[pin_i], d_gnd_pins[pin_i],
            ]
            edge_src.append(di)
            edge_dst.append(ni)
            edge_attr.append(attr)
            edge_src_r.append(ni)
            edge_dst_r.append(di)
            edge_attr_r.append(attr)

    data = HeteroData()
    data["net"].x = net_x
    data["device"].x = torch.tensor(dev_feats, dtype=torch.float)
    data["device"].names = dev_names
    data["net"].names = net_names
    data["device", "connects", "net"].edge_index = torch.tensor(
        [edge_src, edge_dst], dtype=torch.long
    )
    data["device", "connects", "net"].edge_attr = torch.tensor(
        edge_attr, dtype=torch.float
    )
    data["net", "rev_connects", "device"].edge_index = torch.tensor(
        [edge_src_r, edge_dst_r], dtype=torch.long
    )
    data["net", "rev_connects", "device"].edge_attr = torch.tensor(
        edge_attr_r, dtype=torch.float
    )
    data.topology = name
    data.state = int(state)
    data.fc_ghz = fc_ghz
    return data


def connected_component_count(topology_name: str) -> int:
    """Number of connected components treating devices as edges between nets."""
    name = _normalize_name(topology_name)
    G = nx.Graph()
    for dev in TOPOLOGY_NETLIST[name].values():
        pins = ["0" if _is_gnd(n) else n for n in dev.nets]
        G.add_nodes_from(pins)
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                if pins[i] != pins[j]:
                    G.add_edge(pins[i], pins[j])
    # Exclude isolated ground-only consideration: count components of the
    # full graph (ground joins shunt devices).
    return nx.number_connected_components(G)


def incidence_from_spice(template_path: str) -> dict[str, tuple]:
    """Parse a SPICE template into {device_name: (nets...)} for validation.

    Handles R/C/L two-terminal, T* four-node TLines (signal pins only),
    and E* four-node VCVS. Skips .control blocks and excitation/load devices.
    """
    devices = {}
    in_control = False
    with open(template_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            low = line.lower()
            if low.startswith(".control"):
                in_control = True
                continue
            if low.startswith(".endc"):
                in_control = False
                continue
            if in_control or line.startswith("."):
                continue
            # Skip sources / loads we don't model as topology devices
            if line[0] in "VvIi":
                continue
            toks = line.split()
            if len(toks) < 3:
                continue
            name = toks[0]
            prefix = name[0].upper()
            if prefix == "R":
                # Skip port terminations / source resistors
                if name.upper() in ("RSRC", "RLOAD", "RL"):
                    continue
                devices[name] = (toks[1], toks[2])
            elif prefix == "C":
                devices[name] = (toks[1], toks[2])
            elif prefix == "L":
                devices[name] = (toks[1], toks[2])
            elif prefix == "T":
                # Txxx n1 n2 n3 n4 Z0=... TD=...  — signal pins n1, n3
                devices[name] = (toks[1], toks[3])
            elif prefix == "E":
                # Exxx n+ n- nc+ nc- gain
                devices[name] = (toks[1], toks[2], toks[3], toks[4])
    return devices


def assert_matches_template(topology_name: str, template_path: str) -> None:
    """Raise AssertionError if TOPOLOGY_NETLIST disagrees with the .sp file."""
    name = _normalize_name(topology_name)
    parsed = incidence_from_spice(template_path)
    declared = TOPOLOGY_NETLIST[name]

    # Map declared names -> nets (normalize ground)
    def norm_nets(nets):
        return tuple("0" if _is_gnd(n) else n for n in nets)

    declared_nets = {k: norm_nets(v.nets) for k, v in declared.items()}
    parsed_nets = {k: norm_nets(v) for k, v in parsed.items()}

    missing = set(declared_nets) - set(parsed_nets)
    extra = set(parsed_nets) - set(declared_nets)
    if missing or extra:
        raise AssertionError(
            f"{name}: name mismatch. missing_in_spice={missing} extra_in_spice={extra}"
        )
    for k in declared_nets:
        if declared_nets[k] != parsed_nets[k]:
            raise AssertionError(
                f"{name}.{k}: declared {declared_nets[k]} != spice {parsed_nets[k]}"
            )
    n_cc = connected_component_count(name)
    if n_cc != 1:
        raise AssertionError(f"{name}: expected 1 connected component, got {n_cc}")
