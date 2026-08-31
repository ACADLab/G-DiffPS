#!/usr/bin/env python3
"""Build the open-topology pool: enumerate, dedup, subsample, emit SPICE."""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import assert_matches_template
from env.topology_registry import original_six, register_topology
from sim.mna_scorer import mna_score
from topology.compose import enumerate_candidates, stratified_subsample
from topology.dedup import deduplicate, seed_original_hashes
from topology.serialize import netlist_to_dict
from topology.emit_spice import emit_spice


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-new", type=int, default=34)
    ap.add_argument("--out-dir", default="results/open_topo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    template_dir = os.path.join(REPO_ROOT, "specset", "templates", "composed")
    os.makedirs(template_dir, exist_ok=True)

    print("enumerating composed candidates …")
    raw = list(enumerate_candidates(max_k=2))
    print(f"  valid after filter: {len(raw)}")

    seeded = seed_original_hashes()
    deduped = deduplicate(raw, seeded=seeded)
    print(f"  after 1-WL dedup (seeded with 6 originals): {len(deduped)}")

    picked = stratified_subsample(deduped, n=args.n_new, seed=args.seed)
    print(f"  subsampled to {len(picked)} new topologies")

    # Assign stable names Gen_T07 …
    for i, ct in enumerate(picked, start=7):
        ct.name = f"Gen_T{i:02d}"

    pool_meta = {
        "original_six": sorted(original_six()),
        "composed": [],
        "abort_checkpoint": {},
    }

    fc_probe = [7.96, 28.0, 38.0]
    scoring_ok = 0
    emit_ok = 0

    for ct in picked:
        register_topology(
            ct.name, ct.netlist, ct.port_nets, ct.n_states,
            ct.param_keys, ct.ideal_step, overwrite=True,
        )
        sp_path = os.path.join(template_dir, f"{ct.name.lower()}.sp")
        emit_spice(ct.name, ct.netlist, ct.ideal_step, sp_path)
        try:
            assert_matches_template(ct.name, sp_path)
            emit_ok += 1
        except AssertionError as exc:
            print(f"  [warn] template mismatch {ct.name}: {exc}")
            continue

        scores = []
        for fc in fc_probe:
            spec = {"fc_ghz": fc, "tech": 0, "max_il_db": 3.0,
                    "min_rl_db": 10.0, "rms_phase_err_deg": 10.0,
                    "max_area_mm2": 500.0, "rms_gain_err_db": 1.0}
            try:
                from env.graph_utils import TOPOLOGY_PARAMS
                import numpy as np
                from train_diffusion import action_to_params
                act = np.full(len(TOPOLOGY_PARAMS[ct.name]), 0.5)
                params = action_to_params(act, ct.name, spec,
                                          bounds="electrical", switch_model="ideal")
                s = mna_score(ct.name, params, spec)
                scores.append(float(s))
            except Exception as exc:
                scores.append(None)
                print(f"  [warn] mna_score {ct.name} @ {fc} GHz: {exc}")
        if all(s is not None for s in scores):
            scoring_ok += 1

        pool_meta["composed"].append({
            "name": ct.name,
            "device_count": ct.device_count,
            "param_keys": ct.param_keys,
            "netlist": netlist_to_dict(ct.netlist),
            "port_nets": list(ct.port_nets),
            "sections": [s.kind.name for s in ct.sections],
            "wl_hash": getattr(ct, "wl_hash", None),
            "ideal_step_deg": ct.ideal_step,
        })

    pool_meta["abort_checkpoint"] = {
        "n_distinct_valid": scoring_ok,
        "n_emit_roundtrip": emit_ok,
        "fc_probe_ghz": fc_probe,
        "passes_25": scoring_ok >= 25,
    }
    pool_path = os.path.join(args.out_dir, "pool.json")
    with open(pool_path, "w") as fh:
        json.dump(pool_meta, fh, indent=2)

    print(f"\nabort checkpoint: {scoring_ok}/{len(picked)} score at 3 fc values "
          f"(need >=25 distinct valid netlists)")
    print(f"emit round-trip: {emit_ok}/{len(picked)}")
    print(f"wrote {pool_path}")

    if scoring_ok < 25:
        print("ABORT: fewer than 25 valid netlists — stop and write methodology paper.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
