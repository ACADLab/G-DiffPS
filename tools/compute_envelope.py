"""Populate r_star = max_a r_sim(tau, s, a), separately per switch model.

Why not best-of-K random
------------------------
With K=32 uniform draws, the chance of landing within +-10% of the optimum in
every action dimension is 22.7% at d=3 and 0.2% at d=6. r_star is on the critical
path for T4 ranking, S3 margin stratification and the deferred T1 gate, so a
biased-low estimate would corrupt all three -- and biased-low by an amount that
*grows with action dimension*, which silently penalizes exactly the topologies
with the most design freedom (All_Pass d=6, Reflection_Type d=5).

Why not per-spec DE either
--------------------------
12k specs x 6 topologies x 2 switch models is ~144k optimizations, ~24 h.

The factorization that makes this cheap
---------------------------------------
r_sim(tau, s, a) couples the spec and the action only through thresholds:

    metrics = f(tau, a, fc, tech, switch_model)      <- no spec targets
    r_sim   = g(metrics, s.targets)                  <- no action

So the reachable metric set for a cell (tau, switch_model, fc-bin, C_off-class)
is a property of the circuit alone. Build a Pareto archive of that set ONCE per
cell with a real optimizer, then

    r_star(tau, s) = max over archive of g(metrics, s.targets)

is a few thousand arithmetic ops per spec. 144 cells instead of 144k.

Because `bounds='electrical'` scales every sizing bound with fc, the electrical
metrics are near fc-invariant within a bin; area is not, so area is recomputed at
each spec's exact fc and tech from the archived action rather than being read
from the archive.

This yields a *certified achievable lower bound* on r_star: every archive point
is a real sizing, so the bound is attainable by construction. `--mode validate`
runs DE directly against individual specs' own rewards and reports the gap.

Modes
-----
  archive   build the Pareto archives (the expensive step; parallel over cells)
  assign    write r_star into a specset from the archives
  validate  per-spec DE on a subsample; report archive-bound tightness

Running-max alternative
-----------------------
`RunningEnvelope` maintains a per-(spec_id, topology, switch_model) max over
rewards actually observed during training. It costs nothing and is monotonically
correct, but only covers specs the actor has visited, so it supplements the
archive rather than replacing it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.reward import WEIGHTS_AREA, compute_sim_reward
from sim.area_model import estimate_area_mm2
from sim.mna_scorer import _IDEAL_STEP, _states_for_topo, aggregate_mna_metrics, \
    solve_sparams, sparams_to_metrics
from sim.mna_scorer import topology_admits_spec
from sim.switch_model import TECH_SWITCH
from specset.schema import atomic_write_json, load_specset
from train_diffusion import action_to_params

SWITCH_MODELS = ("ideal", "realistic")

# Log-spaced fc bins over the sampled range. Bin ratio 40^(1/8) = 1.55, so
# electrical metrics vary little inside a bin; area is recomputed exactly anyway.
FC_EDGES = np.geomspace(1.0, 40.0, 9)
FC_REFS = [float(math.sqrt(FC_EDGES[i] * FC_EDGES[i + 1]))
           for i in range(len(FC_EDGES) - 1)]

# Under `realistic`, tech enters the circuit only through C_off. techs 0 and 1
# share 20 fF, so there are two distinct electrical classes, not three.
def c_off_class(tech: int) -> float:
    return float(TECH_SWITCH.get(int(tech), TECH_SWITCH[0])[2])


C_OFF_CLASSES = sorted({c_off_class(t) for t in TECH_SWITCH})

# Pareto objectives, with the direction that counts as better.
PARETO_KEYS = (
    ("rms_phase_err_deg", -1),
    ("il_db", -1),
    ("rl_db", +1),
    ("gain_err_db", -1),
    ("area_ref_mm2", -1),
)


# Phase-quality grid the IL/RL/gain anchors are conditioned on. Spans the
# reachable span of `rms_phase_err_deg` under the house convention.
PHASE_GRID_DEG = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0)

# Vector_Modulator is an active topology: its template is an "ideal linear model
# emulating an active I/Q vector modulator", summing through two VCVS with a 2x
# factor sized to give 0 dB nominal insertion loss through the 50 ohm divider.
# It therefore reaches *negative* insertion loss on 20.1% of its Pareto front
# (down to -1.475 dB), which no passive topology can do, and it holds return
# loss at the scorer clip. Letting it set the cross-topology anchor would put
# every IL/RL target beyond the reach of the five passive topologies, so the
# term would stop discriminating -- the same saturation failure in the other
# direction. Anchors describe the *passive* achievable frontier.
#
# This does not exempt Vector_Modulator from the specs; it only means its specs
# are, like `realistic` vs `ideal`, a measurable difficulty delta rather than a
# confound baked into the anchor.
ACTIVE_TOPOLOGIES = frozenset({"Vector_Modulator"})

# `rl_db` saturates at a 60 dB clip in the scorer; treating the clip as a
# frontier would make every return-loss target unreachable for the wrong reason.
RL_ANCHOR_CAP_DB = 40.0
# Floor on the gain-error anchor: exact zeros come from symmetric topologies
# whose two states are identical by construction, and would make any kappa
# multiple still zero.
GAIN_ANCHOR_FLOOR_DB = 0.02


def fc_bin(fc_ghz: float) -> int:
    return int(np.clip(np.searchsorted(FC_EDGES, float(fc_ghz), side="right") - 1,
                       0, len(FC_REFS) - 1))


# ---------------------------------------------------------------------------
# circuit evaluation
# ---------------------------------------------------------------------------

def _electrical_metrics(
    topo: str, action: np.ndarray, fc: float, tech: int, switch_model: str,
    r_off_override: float | None = None,
) -> dict | None:
    params = action_to_params(
        action, topo, {"fc_ghz": fc, "tech": tech},
        bounds="electrical", switch_model=switch_model,
    )
    if r_off_override is not None:
        params = dict(params)
        params["R_off"] = f"{r_off_override:.6e}"
    states = _states_for_topo(topo)
    per = []
    for s in states:
        try:
            s11, s21 = solve_sparams(topo, params, fc, state=s)
            m = sparams_to_metrics(s11, s21)
            m["state"] = s
            per.append(m)
        except Exception:
            continue
    if len(per) < max(1, len(states) // 2):
        return None
    agg = aggregate_mna_metrics(per, _IDEAL_STEP.get(topo, -22.5))
    if agg is None:
        return None
    try:
        agg["area_ref_mm2"] = estimate_area_mm2(topo, params, fc_ghz=fc, tech=tech)
    except Exception:
        agg["area_ref_mm2"] = None
    agg.pop("per_state", None)
    return agg


def _pareto_filter(rows: list[dict]) -> list[dict]:
    """Keep non-dominated rows over PARETO_KEYS."""
    usable = [
        r for r in rows
        if all(r["metrics"].get(k) is not None for k, _ in PARETO_KEYS)
    ]
    if not usable:
        return []
    mat = np.array([
        [sgn * float(r["metrics"][k]) for k, sgn in PARETO_KEYS] for r in usable
    ])
    keep = []
    for i in range(len(usable)):
        # dominated if some j is >= everywhere and > somewhere
        others = np.delete(mat, i, axis=0)
        if others.size == 0:
            keep.append(usable[i])
            continue
        ge = np.all(others >= mat[i] - 1e-12, axis=1)
        gt = np.any(others > mat[i] + 1e-12, axis=1)
        if not np.any(ge & gt):
            keep.append(usable[i])
    return keep


def build_cell_archive(
    topo: str,
    switch_model: str,
    bin_idx: int,
    c_off: float | None,
    *,
    n_sobol: int = 1024,
    de_maxiter: int = 18,
    seed: int = 0,
    return_all_rows: bool = False,
) -> dict:
    """Sobol sweep + DE on several scalarizations -> Pareto archive for one cell."""
    from scipy.optimize import differential_evolution
    from scipy.stats import qmc
    from sim.switch_model import r_off_eff

    fc = FC_REFS[bin_idx]
    d = len(TOPOLOGY_PARAMS[topo])
    tech = 0
    r_off = None
    if switch_model == "realistic" and c_off is not None:
        r_off = r_off_eff(c_off, fc)

    def ev(a):
        return _electrical_metrics(topo, np.asarray(a, dtype=float), fc, tech,
                                   switch_model, r_off_override=r_off)

    rows: list[dict] = []

    sob = qmc.Sobol(d, scramble=True, seed=seed)
    pts = sob.random(n_sobol)
    for a in pts:
        m = ev(a)
        if m is not None:
            rows.append({"action": [float(x) for x in a], "metrics": m})

    # Scalarizations chosen to push toward each face of the trade-off surface,
    # so DE populates the front rather than one interior optimum.
    scalarizations = (
        {"phase": 1.0}, {"il": 1.0}, {"rl": 1.0}, {"gain": 1.0},
        {"area": 1.0}, {"phase": 0.5, "il": 0.3, "rl": 0.2},
    )
    for sc in scalarizations:
        def neg_obj(a, sc=sc):
            m = ev(a)
            if m is None:
                return 1e6
            v = 0.0
            if "phase" in sc:
                v += sc["phase"] * abs(m["rms_phase_err_deg"])
            if "il" in sc:
                v += sc["il"] * max(m["il_db"], 0.0)
            if "rl" in sc:
                v -= sc["rl"] * abs(m["rl_db"])
            if "gain" in sc:
                v += sc["gain"] * abs(m["gain_err_db"])
            if "area" in sc and m.get("area_ref_mm2") is not None:
                v += sc["area"] * float(m["area_ref_mm2"])
            return float(v)

        try:
            res = differential_evolution(
                neg_obj, bounds=[(0.0, 1.0)] * d, maxiter=de_maxiter,
                popsize=12, tol=0.01, seed=seed, polish=True, init="sobol",
            )
        except Exception:
            continue
        m = ev(res.x)
        if m is not None:
            rows.append({"action": [float(x) for x in res.x], "metrics": m})

    front = _pareto_filter(rows)
    if return_all_rows:
        return {"front": front, "all_rows": rows, "fc_ref_ghz": fc}
    return {
        "topology": topo,
        "switch_model": switch_model,
        "fc_bin": bin_idx,
        "fc_ref_ghz": fc,
        "c_off_f": c_off,
        "n_evaluated": len(rows),
        "n_front": len(front),
        "front": [
            {"action": r["action"],
             "m": {k: (None if r["metrics"].get(k) is None
                       else float(r["metrics"][k]))
                   for k in ("rms_phase_err_deg", "il_db", "rl_db",
                             "gain_err_db", "area_ref_mm2")}}
            for r in front
        ],
    }


def _cell_key(topo: str, switch_model: str, bin_idx: int, c_off: float | None) -> str:
    c = "na" if c_off is None else f"{c_off * 1e15:.0f}fF"
    return f"{topo}|{switch_model}|{bin_idx}|{c}"


def _cells(topologies: list[str] | None = None) -> list[tuple[str, str, int, float | None]]:
    topos = topologies if topologies is not None else list(TOPOLOGY_PARAMS)
    out = []
    for topo in topos:
        for bi in range(len(FC_REFS)):
            out.append((topo, "ideal", bi, None))
            for c in C_OFF_CLASSES:
                out.append((topo, "realistic", bi, c))
    return out


def _build_one(args):
    topo, sm, bi, c = args
    try:
        return _cell_key(topo, sm, bi, c), build_cell_archive(topo, sm, bi, c)
    except Exception as e:  # keep the sweep alive; report at the end
        return _cell_key(topo, sm, bi, c), {"error": repr(e)}


def _pool_init() -> None:
    """Load composed topologies into worker processes."""
    try:
        from topology.load_pool import register_composed
        register_composed()
    except Exception:
        pass


def cmd_archive(args) -> int:
    try:
        from topology.load_pool import register_composed
        register_composed()
    except Exception:
        pass
    topologies = None
    if getattr(args, "only_topologies", None):
        topologies = [t.strip() for t in args.only_topologies.split(",") if t.strip()]
    cells = _cells(topologies)
    print(f"building {len(cells)} cell archives "
          f"({len(topologies or TOPOLOGY_PARAMS)} topologies x {len(FC_REFS)} fc bins x "
          f"[ideal + {len(C_OFF_CLASSES)} C_off classes])")
    archives: dict = {}
    if getattr(args, "merge", None) and os.path.exists(args.merge):
        with open(args.merge) as fh:
            archives = json.load(fh).get("archives", {})
        print(f"  merged {len(archives)} existing cells from {args.merge}")

    def _report(i, key, val):
        archives[key] = val
        if "error" in val:
            print(f"  [{i}/{len(cells)}] {key}  ERROR {val['error']}", flush=True)
        else:
            print(f"  [{i}/{len(cells)}] {key}  "
                  f"front={val['n_front']}/{val['n_evaluated']}", flush=True)

    results = None
    if args.workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_pool_init) as ex:
                results = list(ex.map(_build_one, cells, chunksize=1))
        except (PermissionError, OSError, NotImplementedError) as e:
            # Sandboxes commonly block the semaphore syscalls process pools need.
            print(f"  [warn] process pool unavailable ({e}); running serially",
                  flush=True)
            results = None
    if results is None:
        results = (_build_one(c) for c in cells)

    for i, (key, val) in enumerate(results, 1):
        _report(i, key, val)
    payload = {
        "fc_edges": [float(x) for x in FC_EDGES],
        "fc_refs": FC_REFS,
        "c_off_classes": C_OFF_CLASSES,
        "pareto_keys": [k for k, _ in PARETO_KEYS],
        "weights": "WEIGHTS_AREA",
        "archives": archives,
    }
    atomic_write_json(args.archive, payload)
    n_err = sum(1 for v in archives.values() if "error" in v)
    print(f"wrote {args.archive}  ({n_err} cell errors)")
    return 1 if n_err else 0


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------

def cmd_frontier(args) -> int:
    """Emit per-fc-bin achievable anchors for IL / RL / gain error.

    These are the floors the house convention (FRAMEWORK.md §9.10) samples
    against, so that `max_il_db`, `min_rl_db` and `rms_gain_err_db` bind by
    construction instead of saturating against a hand-chosen absolute box.

        max_il_db       = min_tau IL*(fc-bin)   * kappa_il
        min_rl_db       = max_tau RL*(fc-bin)   / kappa_rl
        rms_gain_err_db = min_tau GAIN*(fc-bin) * kappa_gain

    The min/max run *across topologies*: the anchor is the best any topology in
    the set can do in that bin, so every term is within reach of the best
    topology and out of reach of the worst. That is what makes the term
    discriminate between topologies rather than saturate.

    Anchored on switch_model='ideal' only. Generating a second pool against
    'realistic' would make the difficulty of the two models incomparable; with
    one pool, `realistic` is uniformly harder and that delta is a reportable
    quantity rather than a confound.
    """
    with open(args.archive) as fh:
        payload = json.load(fh)
    archives = payload.get("archives", payload)

    anchors: dict[str, dict] = {}
    for bi in range(len(FC_REFS)):
        per_phase: dict[str, dict] = {}
        for phi in PHASE_GRID_DEG:
            best = {"il_db": math.inf, "rl_db": -math.inf,
                    "gain_err_db": math.inf}
            contributors: dict[str, str] = {}
            for topo in TOPOLOGY_PARAMS:
                if topo in ACTIVE_TOPOLOGIES:
                    continue
                cell = archives.get(_cell_key(topo, "ideal", bi, None))
                if not cell or "front" not in cell:
                    continue
                for row in cell["front"]:
                    m = row["m"]
                    pe = m.get("rms_phase_err_deg")
                    # Only designs that are actually phase shifters at this
                    # quality may set the anchor. Without this the anchor is
                    # set by a through-line: perfect IL and RL, no phase shift.
                    if pe is None or float(pe) > phi:
                        continue
                    for k, better in (("il_db", min), ("rl_db", max),
                                      ("gain_err_db", min)):
                        v = m.get(k)
                        if v is None:
                            continue
                        nv = better(best[k], float(v))
                        if nv != best[k]:
                            best[k] = nv
                            contributors[k] = topo
            if any(not math.isfinite(v) for v in best.values()):
                continue
            per_phase[f"{phi:g}"] = {
                # A passive network cannot have gain; the Vector_Modulator's
                # G_I/G_Q scaling can drive the MNA insertion loss slightly
                # negative, which is a model artifact, not an achievable floor.
                "il_db": max(0.0, best["il_db"]),
                # rl_db saturates at a 60 dB clip in the scorer, which is "as
                # matched as we measure" rather than a real frontier value.
                "rl_db": min(RL_ANCHOR_CAP_DB, best["rl_db"]),
                "gain_err_db": max(GAIN_ANCHOR_FLOOR_DB, best["gain_err_db"]),
                "best_topology": contributors,
            }
        if per_phase:
            anchors[str(bi)] = {"fc_ref_ghz": FC_REFS[bi], "by_phase": per_phase}

    out = {
        "schema": "frontier_anchors/2",
        "switch_model": "ideal",
        "fc_edges_ghz": [float(x) for x in FC_EDGES],
        "phase_grid_deg": list(PHASE_GRID_DEG),
        "rl_anchor_cap_db": RL_ANCHOR_CAP_DB,
        "gain_anchor_floor_db": GAIN_ANCHOR_FLOOR_DB,
        "topology_set": sorted(TOPOLOGY_PARAMS),
        "anchor_topologies": sorted(set(TOPOLOGY_PARAMS) - ACTIVE_TOPOLOGIES),
        "excluded_active": sorted(ACTIVE_TOPOLOGIES),
        "anchors": anchors,
        "note": (
            "Anchors are the best IL/RL/gain-error any topology in "
            "`topology_set` attains in each fc bin *subject to* holding RMS "
            "phase error at or below the grid value -- an unconstrained "
            "extremum is attained by a through-line with no phase shift and is "
            "not a usable floor. Spec generation therefore depends on the "
            "topology set: adding a topology that pushes a frontier shifts the "
            "anchors and changes spec difficulty. This coupling is intended "
            "and must be declared when the set changes -- regenerate the pools "
            "and re-report."
        ),
    }
    atomic_write_json(args.out, out)

    print(f"{'bin':>4s} {'fc ref':>8s} {'phase<=':>8s} "
          f"{'IL*':>7s} {'RL*':>7s} {'GAIN*':>7s}   best-topology")
    for bi, a in anchors.items():
        for phi, v in a["by_phase"].items():
            c = v["best_topology"]
            print(f"{bi:>4s} {a['fc_ref_ghz']:8.2f} {phi:>8s} "
                  f"{v['il_db']:7.3f} {v['rl_db']:7.2f} {v['gain_err_db']:7.3f}"
                  f"   IL={c.get('il_db','-')} RL={c.get('rl_db','-')}")
    print(f"\nwrote {args.out}")
    return 0


def r_star_for_spec(
    spec: dict, archives: dict, topologies: Iterable[str],
) -> dict[str, dict[str, float]]:
    """{switch_model: {topology: r_star}} for one spec, area recomputed exactly."""
    fc = float(spec["fc_ghz"])
    tech = int(spec["tech"])
    bi = fc_bin(fc)
    out: dict[str, dict[str, float]] = {}
    for sm in SWITCH_MODELS:
        c = None if sm == "ideal" else c_off_class(tech)
        per_topo = {}
        for topo in topologies:
            # (topology, spec)-only rejects belong in the envelope: no sizing
            # makes an active stage fit a power budget below its floor, so the
            # ranker should see the topology lose on power rather than never be
            # offered the trade.
            if not topology_admits_spec(topo, spec):
                per_topo[topo] = None
                continue
            cell = archives.get(_cell_key(topo, sm, bi, c))
            if not cell or "front" not in cell:
                per_topo[topo] = None
                continue
            best = -1e9
            for pt in cell["front"]:
                m = dict(pt["m"])
                # Exact area at this spec's fc/tech from the archived action.
                try:
                    params = action_to_params(
                        np.asarray(pt["action"], dtype=float), topo,
                        {"fc_ghz": fc, "tech": tech},
                        bounds="electrical", switch_model=sm,
                    )
                    m["area_mm2"] = estimate_area_mm2(
                        topo, params, fc_ghz=fc, tech=tech)
                except Exception:
                    m["area_mm2"] = m.get("area_ref_mm2")
                r = compute_sim_reward(m, spec, weights=WEIGHTS_AREA)
                if r > best:
                    best = r
            per_topo[topo] = float(best)
        out[sm] = per_topo
    return out


def cmd_assign(args) -> int:
    with open(args.archive) as fh:
        payload = json.load(fh)
    archives = payload["archives"]
    if getattr(args, "topology_list", None):
        topologies = [t.strip() for t in args.topology_list.split(",") if t.strip()]
    else:
        topologies = list(TOPOLOGY_PARAMS)

    specs = load_specset(args.specset)
    print(f"assigning r_star to {len(specs)} specs from {args.specset} "
          f"over {len(topologies)} topologies")
    margins = {sm: [] for sm in SWITCH_MODELS}
    sidecar_entries = []
    for i, entry in enumerate(specs):
        rs = r_star_for_spec(entry["spec"], archives, topologies)
        admissible = {t: topology_admits_spec(t, entry["spec"])
                      for t in topologies}
        rstar_block = {
            sm: {
                "per_topology": rs[sm],
                "best": max((v for v in rs[sm].values() if v is not None),
                            default=None),
                "argmax": max(
                    (t for t in topologies if rs[sm][t] is not None),
                    key=lambda t: rs[sm][t], default=None),
                "n_admissible": sum(admissible.values()),
            }
            for sm in SWITCH_MODELS
        }
        for sm in SWITCH_MODELS:
            vals = sorted((v for v in rs[sm].values() if v is not None),
                          reverse=True)
            if len(vals) >= 2:
                m = vals[0] - vals[1]
                rstar_block[sm]["margin"] = float(m)
                margins[sm].append(m)

        sidecar_entries.append({
            "spec_id": entry.get("spec_id", f"spec_{i}"),
            "spec": entry["spec"],
            "admissible": admissible,
            "r_star": rstar_block,
            "area_rank": entry.get("area_rank"),
            "heuristic_scores": entry.get("heuristic_scores"),
        })

        if not getattr(args, "out", None):
            entry["admissible"] = admissible
            entry["r_star"] = rstar_block

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(specs)}")

    strata = {}
    for sm in SWITCH_MODELS:
        arr = np.asarray(margins[sm]) if margins[sm] else np.zeros(1)
        q1, q2 = float(np.percentile(arr, 33.3)), float(np.percentile(arr, 66.7))
        strata[sm] = {"boundary_max": q1, "medium_max": q2,
                      "mean": float(arr.mean())}
        if not getattr(args, "out", None):
            for entry in specs:
                m = entry["r_star"][sm].get("margin")
                if m is None:
                    continue
                entry["r_star"][sm]["stratum"] = (
                    "boundary" if m <= q1 else ("medium" if m <= q2 else "easy")
                )

    if getattr(args, "out", None):
        for sm in SWITCH_MODELS:
            q1 = strata[sm]["boundary_max"]
            q2 = strata[sm]["medium_max"]
            for ent in sidecar_entries:
                m = ent["r_star"][sm].get("margin")
                if m is None:
                    continue
                ent["r_star"][sm]["stratum"] = (
                    "boundary" if m <= q1 else ("medium" if m <= q2 else "easy")
                )
        out_doc = {
            "r_star_source": os.path.basename(args.archive),
            "topology_list": topologies,
            "r_star_margin_strata": strata,
            "specs": sidecar_entries,
        }
        atomic_write_json(args.out, out_doc)
        print(f"margin strata (terciles): {json.dumps(strata, indent=2)}")
        print(f"wrote sidecar {args.out}")
        return 0

    with open(args.specset) as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw["specs"] = specs
        raw["r_star_source"] = os.path.basename(args.archive)
        raw["r_star_margin_strata"] = strata
        atomic_write_json(args.specset, raw)
    else:
        atomic_write_json(args.specset, specs)

    print(f"margin strata (terciles): {json.dumps(strata, indent=2)}")
    print(f"updated {args.specset}")
    return 0


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def cmd_validate(args) -> int:
    """Direct per-spec DE vs the archive bound, on a subsample."""
    from scipy.optimize import differential_evolution

    with open(args.archive) as fh:
        archives = json.load(fh)["archives"]
    specs = load_specset(args.specset)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(specs), size=min(args.n, len(specs)), replace=False)

    rows = []
    for j, i in enumerate(idx, 1):
        spec = specs[int(i)]["spec"]
        fc, tech = float(spec["fc_ghz"]), int(spec["tech"])
        bound = r_star_for_spec(spec, archives, TOPOLOGY_PARAMS)
        for sm in SWITCH_MODELS:
            for topo in TOPOLOGY_PARAMS:
                d = len(TOPOLOGY_PARAMS[topo])

                def neg_r(a):
                    m = _electrical_metrics(
                        topo, np.asarray(a, float), fc, tech, sm)
                    if m is None:
                        return 5.0
                    m["area_mm2"] = m.get("area_ref_mm2")
                    return -compute_sim_reward(m, spec, weights=WEIGHTS_AREA)

                res = differential_evolution(
                    neg_r, bounds=[(0.0, 1.0)] * d, maxiter=args.de_maxiter,
                    popsize=15, tol=0.01, seed=0, polish=True, init="sobol",
                )
                direct = float(-res.fun)
                arch = bound[sm][topo]
                rows.append({
                    "spec_id": specs[int(i)]["id"], "topology": topo,
                    "switch_model": sm, "archive": arch, "direct_de": direct,
                    "gap": None if arch is None else direct - arch,
                })
        print(f"  validated {j}/{len(idx)}")

    gaps = np.array([r["gap"] for r in rows if r["gap"] is not None])
    summary = {
        "n_comparisons": len(gaps),
        "de_maxiter": args.de_maxiter,
        "gap_mean": float(gaps.mean()),
        "gap_p50": float(np.percentile(gaps, 50)),
        "gap_p95": float(np.percentile(gaps, 95)),
        "gap_max": float(gaps.max()),
        "frac_archive_within_0p01": float(np.mean(gaps <= 0.01)),
        "frac_archive_within_0p05": float(np.mean(gaps <= 0.05)),
        "frac_archive_beats_de": float(np.mean(gaps < 0.0)),
        "note": "gap = direct_DE - archive_bound. Negative means the archive "
                "found a better sizing than per-spec DE, which is possible "
                "because the archive pools far more circuit evaluations.",
    }
    print(json.dumps(summary, indent=2))
    atomic_write_json(args.out, {"summary": summary, "rows": rows})
    print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# running max over the training buffer
# ---------------------------------------------------------------------------

class RunningEnvelope:
    """Per-(spec_id, topology, switch_model) running max of observed rewards.

    Free to maintain and monotonically correct, but only covers visited specs,
    so it tightens the archive bound rather than substituting for it.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self._m: dict[str, float] = {}
        if path and os.path.exists(path):
            with open(path) as fh:
                self._m = json.load(fh).get("max", {})

    @staticmethod
    def _k(spec_id: str, topology: str, switch_model: str) -> str:
        return f"{spec_id}|{topology}|{switch_model}"

    def update(self, spec_id: str, topology: str, switch_model: str,
               reward: float) -> None:
        k = self._k(spec_id, topology, switch_model)
        if reward > self._m.get(k, -1e9):
            self._m[k] = float(reward)

    def get(self, spec_id: str, topology: str, switch_model: str) -> float | None:
        return self._m.get(self._k(spec_id, topology, switch_model))

    def save(self) -> None:
        if self.path:
            atomic_write_json(self.path, {"n": len(self._m), "max": self._m})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    default_archive = os.path.join(REPO_ROOT, "results", "joint",
                                   "envelope_archive.json")

    p = sub.add_parser("archive", help="build Pareto archives (expensive)")
    p.add_argument("--archive", default=default_archive)
    p.add_argument("--merge", default=None,
                   help="merge new cells into an existing archive JSON")
    p.add_argument("--only-topologies", default=None,
                   help="comma-separated topology subset to build")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.set_defaults(fn=cmd_archive)

    p = sub.add_parser(
        "frontier", help="emit achievable IL/RL/gain anchors for spec sampling")
    p.add_argument("--archive", default=default_archive)
    p.add_argument("--out", default=os.path.join(
        REPO_ROOT, "specset", "frontier_anchors.json"))
    p.set_defaults(fn=cmd_frontier)

    p = sub.add_parser("assign", help="write r_star into a specset")
    p.add_argument("--archive", default=default_archive)
    p.add_argument("--specset", default=os.path.join(
        REPO_ROOT, "specset", "specset_eval.json"))
    p.add_argument("--out", default=None,
                   help="write sidecar JSON instead of updating specset in place")
    p.add_argument("--topology-list", default=None,
                   help="comma-separated topology list (default: all registered)")
    p.set_defaults(fn=cmd_assign)

    p = sub.add_parser("validate", help="per-spec DE vs archive bound")
    p.add_argument("--archive", default=default_archive)
    p.add_argument("--specset", default=os.path.join(
        REPO_ROOT, "specset", "specset_eval.json"))
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--de-maxiter", type=int, default=40)
    p.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "envelope_validate.json"))
    p.set_defaults(fn=cmd_validate)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
