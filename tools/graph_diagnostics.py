"""Phase-1 synthetic graph diagnostics: G1 round-trip, G2 permutation,
G3 terminal-swap, G7 perturbation ranking.

Usage:
    python tools/graph_diagnostics.py --out results/graphs/phase1_diagnostics.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import (
    TERMINAL_ROLE_IDX,
    TOPOLOGY_NETLIST,
    assert_matches_template,
    build_circuit_graph,
    incidence_from_graph,
    incidence_from_spice,
    nominal_params,
)
from models.circuit_encoder import CircuitEncoder, CircuitTypedEncoder, make_encoder
from topology.emit_spice import emit_spice

try:
    from sim.sky130.realizable import loaded_line_sky130_graph
except Exception:  # pragma: no cover
    loaded_line_sky130_graph = None


TEMPLATES = {
    "Loaded_Line": "loaded_line.sp",
    "Switched_Line": "switched_line.sp",
    "Reflection_Type": "reflection_type.sp",
    "Switched_Filter": "switched_filter.sp",
    "Vector_Modulator": "vector_modulator.sp",
    "All_Pass": "all_pass.sp",
}

SPEC = {"fc_ghz": 28.0, "tech": 0}


def _norm_nets(nets) -> tuple:
    return tuple("0" if n in ("0", "GND", "gnd", "ground") else n for n in nets)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = float(a.norm() * b.norm()) + 1e-12
    return float(torch.dot(a, b) / denom)


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.reshape(-1) - b.reshape(-1)).norm())


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return _l2(a, b) / (float(a.reshape(-1).norm()) + 1e-12)


def _encode(enc, data) -> torch.Tensor:
    enc.eval()
    enc._cache_enabled = False
    device = next(enc.parameters()).device
    data = data.to(device)
    z, _ = enc.encode_state(data)
    return z.detach()


def _fresh_encoder(kind: str, seed: int = 0, use_pe: bool = True):
    torch.manual_seed(seed)
    if kind == "circuit-typed":
        enc = CircuitTypedEncoder(use_pe=use_pe)
    elif kind == "circuit":
        enc = CircuitEncoder()
    else:
        enc = make_encoder(kind)
    enc.eval()
    enc._cache_enabled = False
    return enc


# ---------------------------------------------------------------------------
# G1 — SPICE → graph → SPICE
# ---------------------------------------------------------------------------

_PARAM_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _declared_params(topo: str) -> dict[str, set[str]]:
    out = {}
    for dname, dev in TOPOLOGY_NETLIST[topo].items():
        keys = set()
        if dev.sizes:
            keys.add(dev.sizes)
        keys.update(dev.aux_sizes)
        if dev.switch_param:
            keys.add(dev.switch_param)
        out[dname] = keys
    return out


def _spice_param_keys(template_path: str) -> dict[str, set[str]]:
    """Map device name -> identifiers appearing in its SPICE value expression."""
    from env.netlist_graph import incidence_from_spice as _  # noqa: F401
    keys = {}
    in_control = False
    skip = {"RSRC", "RLOAD", "RL"}
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
            if in_control or line.startswith(".") or line[0] in "VvIi":
                continue
            toks = line.split()
            if len(toks) < 3:
                continue
            name = toks[0]
            if name.upper() in skip:
                continue
            prefix = name[0].upper()
            if prefix not in "RCLT E":
                continue
            expr = " ".join(toks[3:])
            found = set()
            for ident in _PARAM_IDENT.findall(expr):
                if ident.lower() in ("z0", "td", "dc", "ac"):
                    continue
                if ident in ("e8",):
                    continue
                found.add(ident)
            keys[name] = found
    return keys


def g1_roundtrip(topo: str) -> dict:
    fname = TEMPLATES[topo]
    path = os.path.join(REPO_ROOT, "specset", "templates", fname)
    template_ok = True
    template_err = None
    try:
        assert_matches_template(topo, path)
    except AssertionError as exc:
        template_ok = False
        template_err = str(exc)

    parsed = {k: _norm_nets(v) for k, v in incidence_from_spice(path).items()}
    declared = {
        k: _norm_nets(dev.nets) for k, dev in TOPOLOGY_NETLIST[topo].items()
    }
    if parsed != declared:
        template_ok = False
        template_err = template_err or "parsed template nets != TOPOLOGY_NETLIST"
    g = build_circuit_graph(topo, SPEC, state=0, include_ports=False)
    from_graph = {
        k: _norm_nets(v)
        for k, v in incidence_from_graph(g).items()
        if k in declared
    }
    graph_ok = from_graph == declared

    emitted = emit_spice(topo, TOPOLOGY_NETLIST[topo])
    tmp = os.path.join(REPO_ROOT, "results", "graphs", f"_g1_{topo}.sp")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w") as fh:
        fh.write(emitted)
    reparsed = {k: _norm_nets(v) for k, v in incidence_from_spice(tmp).items()}
    emit_ok = reparsed == declared
    try:
        os.remove(tmp)
    except OSError:
        pass

    decl_params = _declared_params(topo)
    spice_params = _spice_param_keys(path)
    param_missing = {}
    for dname, keys in decl_params.items():
        # Switch params are FRAMEWORK_CONTROLLED on the spice line as {R_path_in};
        # sized keys appear in the value expression. Require each sized/aux key.
        sized = {k for k in keys if not k.startswith("R_") or k in ("R_on", "R_off")}
        # Keep sizes/aux; switch_param is allowed to appear as the value itself.
        expect = set()
        dev = TOPOLOGY_NETLIST[topo][dname]
        if dev.sizes:
            expect.add(dev.sizes)
        expect.update(dev.aux_sizes)
        have = spice_params.get(dname, set())
        miss = expect - have
        if miss:
            param_missing[dname] = sorted(miss)
    params_ok = not param_missing

    return {
        "topology": topo,
        "template_ok": template_ok,
        "template_error": template_err,
        "graph_ok": graph_ok,
        "emit_ok": emit_ok,
        "params_ok": params_ok,
        "param_missing": param_missing,
        "n_devices": len(declared),
        "pass": bool(template_ok and graph_ok and emit_ok and params_ok),
    }


# ---------------------------------------------------------------------------
# G2 — permutation invariance
# ---------------------------------------------------------------------------

def permute_hetero(data, rng: np.random.Generator):
    data = data.clone()
    n_n = int(data["net"].x.size(0))
    n_d = int(data["device"].x.size(0))
    perm_n = torch.tensor(rng.permutation(n_n), dtype=torch.long)
    perm_d = torch.tensor(rng.permutation(n_d), dtype=torch.long)
    inv_n = torch.empty_like(perm_n)
    inv_n[perm_n] = torch.arange(n_n)
    inv_d = torch.empty_like(perm_d)
    inv_d[perm_d] = torch.arange(n_d)

    data["net"].x = data["net"].x[perm_n]
    data["device"].x = data["device"].x[perm_d]
    if hasattr(data["net"], "names"):
        names_n = list(data["net"].names)
        data["net"].names = [names_n[i] for i in perm_n.tolist()]
    if hasattr(data["device"], "names"):
        names_d = list(data["device"].names)
        data["device"].names = [names_d[i] for i in perm_d.tolist()]
    if getattr(data["net"], "role", None):
        roles = list(data["net"].role)
        data["net"].role = [roles[i] for i in perm_n.tolist()]

    def _remap(store, src_inv, dst_inv):
        ei = store.edge_index.clone()
        ei[0] = src_inv[ei[0]]
        ei[1] = dst_inv[ei[1]]
        store.edge_index = ei

    _remap(data["device", "connects", "net"], inv_d, inv_n)
    _remap(data["net", "rev_connects", "device"], inv_n, inv_d)
    if "motif" in getattr(data, "node_types", []):
        ei_m = data["motif", "contains", "device"].edge_index.clone()
        ei_m[1] = inv_d[ei_m[1]]
        data["motif", "contains", "device"].edge_index = ei_m
    return data


def g2_permutation(enc, topo: str, typed: bool, n_trials: int = 8) -> dict:
    rng = np.random.default_rng(0)
    base = build_circuit_graph(topo, SPEC, state=0, typed=typed)
    z0 = _encode(enc, base)
    cosines, rels = [], []
    for _ in range(n_trials):
        zp = _encode(enc, permute_hetero(base, rng))
        cosines.append(_cosine(z0, zp))
        rels.append(_rel_l2(z0, zp))
    return {
        "topology": topo,
        "min_cosine": min(cosines),
        "mean_cosine": float(np.mean(cosines)),
        "max_rel_l2": max(rels),
        "mean_rel_l2": float(np.mean(rels)),
        "pass": min(cosines) >= 0.999,
    }


# ---------------------------------------------------------------------------
# G3 — terminal-swap sensitivity
# ---------------------------------------------------------------------------

def _swap_edge_roles(data, dname: str, role_a: str, role_b: str):
    """Swap relation types of two pins on one device; incidence set unchanged."""
    data = data.clone()
    names = list(data["device"].names)
    di = names.index(dname)
    ia = TERMINAL_ROLE_IDX[role_a]
    ib = TERMINAL_ROLE_IDX[role_b]

    def _swap_store(store, src_is_device: bool):
        ei = store.edge_index
        et = store.edge_type.clone()
        src_idx = 0 if src_is_device else 1
        mask = ei[src_idx] == di
        types = et[mask]
        loc_a = (types == ia).nonzero(as_tuple=False).view(-1)
        loc_b = (types == ib).nonzero(as_tuple=False).view(-1)
        if loc_a.numel() == 0 or loc_b.numel() == 0:
            return False
        global_idx = mask.nonzero(as_tuple=False).view(-1)
        ea, eb = int(global_idx[int(loc_a[0])]), int(global_idx[int(loc_b[0])])
        et[ea], et[eb] = et[eb], et[ea]
        store.edge_type = et
        attr = store.edge_attr.clone()
        tmp = attr[ea].clone()
        attr[ea] = attr[eb]
        attr[eb] = tmp
        store.edge_attr = attr
        return True

    ok = _swap_store(data["device", "connects", "net"], True)
    ok = ok and _swap_store(data["net", "rev_connects", "device"], False)
    if not ok:
        raise RuntimeError(f"could not swap {role_a}/{role_b} on {dname}")
    return data


def _swap_pin_indices(data, dname: str, pin_a: int, pin_b: int):
    """Untyped swap: exchange pin-index / sign / distances of two pins."""
    data = data.clone()
    names = list(data["device"].names)
    di = names.index(dname)

    def _swap_store(store, src_is_device: bool):
        ei = store.edge_index
        src_idx = 0 if src_is_device else 1
        mask = ei[src_idx] == di
        global_idx = mask.nonzero(as_tuple=False).view(-1)
        attr = store.edge_attr
        pins = [int(round(float(attr[int(g), 0]) * 3.0)) for g in global_idx.tolist()]
        if pin_a not in pins or pin_b not in pins:
            return False
        ea = int(global_idx[pins.index(pin_a)])
        eb = int(global_idx[pins.index(pin_b)])
        new_attr = attr.clone()
        tmp = new_attr[ea].clone()
        new_attr[ea] = new_attr[eb]
        new_attr[eb] = tmp
        store.edge_attr = new_attr
        return True

    ok = _swap_store(data["device", "connects", "net"], True)
    ok = ok and _swap_store(data["net", "rev_connects", "device"], False)
    if not ok:
        raise RuntimeError(f"could not swap pins {pin_a}/{pin_b} on {dname}")
    return data


def g3_terminal_swap(enc, typed: bool) -> dict:
    cases = []

    # Polar: VCVS out+ ↔ ctrl+ on Vector_Modulator. Degree of the undirected
    # incidence graph is unchanged; only the relation (or pin ordinal) moves.
    g = build_circuit_graph("Vector_Modulator", SPEC, state=0, typed=typed)
    z0 = _encode(enc, g)
    if typed:
        gs = _swap_edge_roles(g, "E_I", "out_p", "ctrl_p")
    else:
        gs = _swap_pin_indices(g, "E_I", 0, 2)
    zs = _encode(enc, gs)
    cases.append({
        "name": "vcvs_out_ctrl",
        "expect": "move",
        "cosine": _cosine(z0, zs),
        "rel_l2": _rel_l2(z0, zs),
        "l2": _l2(z0, zs),
    })

    # Symmetric control: capacitor pin swap. Electrically a no-op.
    g = build_circuit_graph("Loaded_Line", SPEC, state=0, typed=typed)
    z0 = _encode(enc, g)
    gs = _swap_pin_indices(g, "C_in_load", 0, 1)
    zs = _encode(enc, gs)
    cases.append({
        "name": "passive_cap_pins",
        "expect": "stay",
        "cosine": _cosine(z0, zs),
        "rel_l2": _rel_l2(z0, zs),
        "l2": _l2(z0, zs),
    })

    # MOS G↔D. Untyped graphs do not put G on the incidence edge, so this
    # case is skipped unless we have a typed SKY130 netlist.
    mos = None
    if typed and loaded_line_sky130_graph is not None:
        netlist, params = loaded_line_sky130_graph()
        g = build_circuit_graph(
            "Loaded_Line", SPEC, state=0, typed=True,
            devices_override=netlist, params=params,
        )
        if "M_in_path" in list(g["device"].names):
            z0 = _encode(enc, g)
            gs = _swap_edge_roles(g, "M_in_path", "drain", "gate")
            zs = _encode(enc, gs)
            mos = {
                "name": "mos_gd",
                "expect": "move",
                "cosine": _cosine(z0, zs),
                "rel_l2": _rel_l2(z0, zs),
                "l2": _l2(z0, zs),
            }
            cases.append(mos)

    polar = next(c for c in cases if c["name"] == "vcvs_out_ctrl")
    control = next(c for c in cases if c["name"] == "passive_cap_pins")
    # Polar swap should move more than a symmetric pin swap.
    polar_moves = polar["rel_l2"] > control["rel_l2"]
    return {
        "cases": cases,
        "polar_vs_passive_ratio": (
            polar["rel_l2"] / (control["rel_l2"] + 1e-12)
        ),
        "pass_polar_moves": polar_moves,
        "mos_present": mos is not None,
    }


# ---------------------------------------------------------------------------
# G7 — perturbation ranking
# ---------------------------------------------------------------------------

def g7_perturbation(enc, typed: bool) -> dict:
    topo = "Vector_Modulator"
    base_nl = copy.deepcopy(TOPOLOGY_NETLIST[topo])
    params = dict(nominal_params(topo, SPEC))
    g0 = build_circuit_graph(
        topo, SPEC, state=0, typed=typed, params=params,
        devices_override=base_nl,
    )
    z0 = _encode(enc, g0)

    def dist_for(nl=None, p=None):
        g = build_circuit_graph(
            topo, SPEC, state=0, typed=typed,
            params=p if p is not None else params,
            devices_override=nl if nl is not None else base_nl,
        )
        return _rel_l2(z0, _encode(enc, g))

    rows = []
    p_small = dict(params)
    if "G_I_scale" in p_small:
        p_small["G_I_scale"] = float(p_small["G_I_scale"]) * 1.05
    else:
        key = next(iter(p_small))
        p_small[key] = float(p_small[key]) * 1.05
    rows.append(("small_param", 1, dist_for(p=p_small)))

    p_wrong = dict(params)
    p_wrong["G_I_scale"] = 50.0
    rows.append(("wrong_param_group", 2, dist_for(p=p_wrong)))

    nl_conn = copy.deepcopy(base_nl)
    e = nl_conn["E_I"]
    nets = list(e.nets)
    nets[0] = "q_in"
    nl_conn["E_I"] = copy.copy(e)
    nl_conn["E_I"].nets = tuple(nets)
    rows.append(("wrong_connection", 3, dist_for(nl=nl_conn)))

    nl_term = copy.deepcopy(base_nl)
    e = nl_term["E_I"]
    nets = list(e.nets)
    nets[0], nets[2] = nets[2], nets[0]
    nl_term["E_I"] = copy.copy(e)
    nl_term["E_I"].nets = tuple(nets)
    rows.append(("wrong_terminal", 4, dist_for(nl=nl_term)))

    nl_brk = copy.deepcopy(base_nl)
    del nl_brk["T_quad"]
    rows.append(("broken_topology", 5, dist_for(nl=nl_brk)))

    sevs = np.array([r[1] for r in rows], dtype=float)
    dists = np.array([r[2] for r in rows], dtype=float)
    # Spearman via rank correlation
    sev_rank = np.argsort(np.argsort(sevs))
    dist_rank = np.argsort(np.argsort(dists))
    spearman = float(np.corrcoef(sev_rank, dist_rank)[0, 1])
    return {
        "rows": [
            {"name": n, "severity": int(s), "rel_l2": float(d)}
            for n, s, d in rows
        ],
        "spearman": spearman,
        "monotonic": bool(np.all(np.diff(dists) >= -1e-6)),
        "pass": spearman >= 0.4,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(encoders: list[str], pe_ablation: bool = True) -> dict:
    report = {"g1": [], "encoders": {}}
    for topo in TEMPLATES:
        report["g1"].append(g1_roundtrip(topo))
    report["g1_all_pass"] = all(r["pass"] for r in report["g1"])

    for kind in encoders:
        variants = [("default", True)]
        if kind == "circuit-typed" and pe_ablation:
            variants = [("with_pe", True), ("no_pe", False)]
        block = {}
        for tag, use_pe in variants:
            enc = _fresh_encoder(kind, seed=0, use_pe=use_pe)
            typed = kind == "circuit-typed"
            g2 = [g2_permutation(enc, t, typed=typed) for t in TEMPLATES]
            g3 = g3_terminal_swap(enc, typed=typed)
            g7 = g7_perturbation(enc, typed=typed)
            block[tag] = {
                "g2": g2,
                "g2_all_pass": all(r["pass"] for r in g2),
                "g3": g3,
                "g7": g7,
            }
        report["encoders"][kind] = block
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoders", default="circuit,circuit-typed",
        help="Comma-separated encoder names",
    )
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "phase1_diagnostics.json",
    ))
    ap.add_argument("--no-pe-ablation", action="store_true")
    args = ap.parse_args()
    kinds = [s.strip() for s in args.encoders.split(",") if s.strip()]
    report = run_all(kinds, pe_ablation=not args.no_pe_ablation)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({
        "g1_all_pass": report["g1_all_pass"],
        "g1": {r["topology"]: r["pass"] for r in report["g1"]},
        "encoders": {
            k: {
                tag: {
                    "g2_all_pass": v["g2_all_pass"],
                    "g3_polar_moves": v["g3"]["pass_polar_moves"],
                    "g3_ratio": v["g3"]["polar_vs_passive_ratio"],
                    "g7_spearman": v["g7"]["spearman"],
                    "g7_monotonic": v["g7"]["monotonic"],
                }
                for tag, v in block.items()
            }
            for k, block in report["encoders"].items()
        },
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
