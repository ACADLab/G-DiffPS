"""D0: known-good vs known-bad designs for every topology.

For each family: decoder, A/B/C/D prior bits, MNA, reward, strict metrics.

    python tools/env_consistency.py --out results/graphs/d0_env_consistency.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import params_numeric
from env.reward import compute_sim_reward, strict_compliance
from sim.mna_scorer import mna_evaluate
from sim.physics_priors import check_physics_priors, explain_physics_priors
from train_diffusion import action_to_params

TOPOS = list(TOPOLOGY_PARAMS.keys())

SPEC = {
    "fc_ghz": 28.0,
    "bw_pct": 20.0,
    "rms_phase_err_deg": 5.0,
    "rms_gain_err_db": 1.0,
    "max_il_db": 3.0,
    "min_rl_db": 12.0,
    "pmax_mw": 50.0,
    "max_area_mm2": 100.0,
    "tech": 0,
}

# Intentionally non-physical / off-resonance sizings. Decoder may still emit
# them; prior and/or reward should not treat them as the nominal.
BAD_ACTIONS = {
    "Loaded_Line": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "Switched_Line": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "Reflection_Type": np.array([0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "Switched_Filter": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "Vector_Modulator": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "All_Pass": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}


def _mid_action(topo: str) -> np.ndarray:
    return np.full(len(TOPOLOGY_PARAMS[topo]), 0.5, dtype=np.float32)


def _score(topo: str, action, spec: dict) -> dict:
    keys = TOPOLOGY_PARAMS[topo]
    try:
        params = action_to_params(
            action, topo, spec, sizing="log", bounds="electrical",
            switch_model="ideal",
        )
        decoder_ok = all(k in params for k in keys)
    except Exception as exc:
        return {
            "decoder_ok": False,
            "decoder_error": str(exc),
            "prior_pass": False,
            "sim_ok": False,
            "reward": None,
            "strict": False,
        }
    numeric = params_numeric(params)
    expl = explain_physics_priors(
        topo, params, spec["fc_ghz"], pmax_mw=float(spec["pmax_mw"]),
    )
    prior = bool(check_physics_priors(
        topo, params, spec["fc_ghz"], pmax_mw=float(spec["pmax_mw"]),
    ))
    score, metrics = mna_evaluate(topo, params, spec)
    sim_ok = metrics is not None
    reward = float(score) if sim_ok else None
    if sim_ok:
        reward = float(compute_sim_reward(metrics, spec))
    strict = bool(strict_compliance(metrics, spec)) if sim_ok else False
    return {
        "decoder_ok": decoder_ok,
        "params": {k: numeric.get(k) for k in keys},
        "prior_pass": prior,
        "prior_explain": {
            "A_structural": expl["A_structural"],
            "B_physics": expl["B_physics"],
            "C_operating": expl["C_operating"],
            "D_sim_sanity": expl["D_sim_sanity"],
            "failed": expl["failed"],
            "notes": expl["notes"],
            "pass": expl["pass"],
            "check_physics_priors": expl["check_physics_priors"],
        },
        "sim_ok": sim_ok,
        "reward": reward,
        "strict": strict,
        "metrics": None if metrics is None else {
            k: (None if v is None else float(v))
            for k, v in metrics.items()
            if k in (
                "il_db", "rl_db", "rms_phase_err_deg", "gain_err_db", "area_mm2",
            )
        },
    }


def evaluate_topology(topo: str, spec: dict) -> dict:
    good = _score(topo, _mid_action(topo), spec)
    bad = _score(topo, BAD_ACTIONS[topo], spec)
    reward_ok = (
        good["reward"] is not None
        and bad["reward"] is not None
        and good["reward"] > bad["reward"]
    )
    consistent = (
        good["decoder_ok"]
        and good["prior_pass"]
        and good["sim_ok"]
        and reward_ok
    )
    return {
        "good": good,
        "bad": bad,
        "reward_good_gt_bad": reward_ok,
        "nominal_consistent": consistent,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d0_env_consistency.json",
    ))
    args = ap.parse_args()
    results = {}
    for topo in TOPOS:
        results[topo] = evaluate_topology(topo, SPEC)
        # All-Pass also at 2.4 GHz — the historical contradiction site.
        if topo == "All_Pass":
            spec_24 = dict(SPEC)
            spec_24["fc_ghz"] = 2.4
            results["All_Pass_2p4"] = evaluate_topology(topo, spec_24)

    gate = all(
        results[t]["nominal_consistent"] for t in TOPOS
    ) and results["All_Pass_2p4"]["nominal_consistent"]
    report = {"spec": SPEC, "gate": gate, "topologies": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    summary = {
        t: {
            "nominal_consistent": results[t]["nominal_consistent"],
            "good_prior": results[t]["good"]["prior_pass"],
            "good_sim": results[t]["good"]["sim_ok"],
            "good_strict": results[t]["good"]["strict"],
            "good_reward": results[t]["good"]["reward"],
            "bad_prior": results[t]["bad"]["prior_pass"],
            "bad_reward": results[t]["bad"]["reward"],
            "good_B": results[t]["good"]["prior_explain"]["B_physics"],
            "good_failed": results[t]["good"]["prior_explain"]["failed"],
        }
        for t in list(TOPOS) + ["All_Pass_2p4"]
    }
    print(json.dumps({"gate": gate, "summary": summary, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
