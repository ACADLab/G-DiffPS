"""
Topology-selection accuracy: does the ValueNet head recommend the topology that
actually achieves the best SPICE reward?  (App C ground-truth validation.)

For each spec:
  1. ValueNet scores all 6 topologies -> ranked recommendation.
  2. CFM actor sizes each topology (k samples); SPICE-eval; take best reward/topo.
  3. Empirical-best topology = argmax_topo (best SPICE reward), among feasible.
  4. Record top-1 / top-2 agreement of ValueNet ranking with the empirical best,
     and heuristic_agreement with the deprecated stored heuristic label (not
     ground truth — S0 retired heuristic labels as accuracy targets).

Usage:
  python topo_selection_accuracy.py --run runs_diffusion/run_20260530_031117 \
      --n-specs 120 --k 8 --seed 42 --out results/topo_select/acc_seed42.json
"""
import argparse, json, os, sys, random
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph, SLOT_ACTION_DIM
from models.gnn_encoder import TopologyEncoder
from models.diffusion_policy import FlowMatchingPolicy, ValueNet
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker
from specset.schema import SPEC_DIM, load_specset

TOPOS = ["Loaded_Line", "Switched_Line", "Reflection_Type",
         "Switched_Filter", "Vector_Modulator", "All_Pass"]
FEASIBLE_REWARD = 0.5   # a topology counts as "achievable" for this spec above this

# S3 strata, from r_star margin terciles written by tools/compute_envelope.py.
STRATA = ("boundary", "medium", "easy")


def load_models(run_dir, device):
    gnn = TopologyEncoder(in_channels=5, hidden_channels=64, out_channels=64).to(device)
    actor = FlowMatchingPolicy(
        action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64
    ).to(device)
    value_net = ValueNet(spec_dim=SPEC_DIM, graph_dim=64).to(device)
    gnn.load_state_dict(torch.load(os.path.join(run_dir, "gnn_encoder.pt"), map_location=device))
    actor.load_state_dict(torch.load(os.path.join(run_dir, "actor.pt"), map_location=device))
    value_net.load_state_dict(torch.load(os.path.join(run_dir, "value_net.pt"), map_location=device))
    gnn.eval(); actor.eval(); value_net.eval()
    return gnn, actor, value_net


def best_spice_reward(actor, z, spec_norm, topo, spec, env, k):
    """Best SPICE reward over k CFM samples for one topology."""
    best = -999.0
    for _ in range(k):
        with torch.no_grad():
            a = actor.sample(spec_norm, z).squeeze(0).cpu().numpy()
        params = action_to_params(a, topo, spec)
        if not check_physics_priors(topo, params, spec["fc_ghz"]):
            continue
        eb = env.compute_expert_bonus(topo, spec)
        nl = make_spice_netlist(topo, params)
        r, _, _ = parallel_eval_worker((nl, spec, topo, eb, None, 0.0, params))
        best = max(best, float(r))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-specs", type=int, default=120)
    ap.add_argument("--k", type=int, default=8, help="CFM samples per topology")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/topo_select/acc.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--specset", default=os.path.join(
        REPO_ROOT, "specset", "specset_eval.json"),
        help="Held-out eval pool. Must carry r_star for S3 stratification.")
    ap.add_argument("--switch-model", default="ideal",
                    choices=("ideal", "realistic"),
                    help="Which r_star envelope to stratify against.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    gnn, actor, value_net = load_models(args.run, device)
    env = PhaseShifterEnv()

    specset = load_specset(args.specset)
    if isinstance(specset, dict) and "specs" in specset:
        specset = specset["specs"]
    n_strat = sum(1 for e in specset
                  if (e.get("r_star") or {}).get(args.switch_model, {}).get("stratum"))
    if n_strat == 0:
        print("[warn] no r_star strata in this specset — S3 stratified accuracy "
              "will be omitted. Run `tools/compute_envelope.py assign` first.")
    idxs = list(range(len(specset)))
    random.shuffle(idxs)
    idxs = idxs[:args.n_specs]

    # Pre-encode the six topology graphs once.
    z_cache = {}
    for t in TOPOS:
        g = get_topology_graph(t)
        with torch.no_grad():
            z_cache[t] = gnn(g.x.to(device), g.edge_index.to(device))

    top1 = top2 = heuristic_agree = feasible_specs = 0
    n_with_heuristic = 0
    records = []
    # S3: per-stratum tallies, so aggregate accuracy cannot hide behind a prior.
    strat = {s: {"n": 0, "top1": 0, "top2": 0, "emp_best": []} for s in STRATA}
    for n, i in enumerate(idxs):
        entry = specset[i]
        spec = entry["spec"]
        rs = (entry.get("r_star") or {}).get(args.switch_model, {})
        stratum = rs.get("stratum")
        margin = rs.get("margin")
        # S0: heuristic labels are deprecated; not ground truth for accuracy.
        heur_label = entry.get("heuristic_topology_deprecated")
        spec_norm = torch.tensor(env._normalize(spec), dtype=torch.float, device=device).unsqueeze(0)

        # ValueNet ranking (no SPICE)
        vscores = {}
        for t in TOPOS:
            with torch.no_grad():
                vscores[t] = value_net(spec_norm, z_cache[t]).item()
        ranked = sorted(TOPOS, key=lambda t: vscores[t], reverse=True)

        # SPICE ground truth: best reward per topology
        rewards = {t: best_spice_reward(actor, z_cache[t], spec_norm, t, spec, env, args.k)
                   for t in TOPOS}
        emp_best = max(TOPOS, key=lambda t: rewards[t])
        emp_best_r = rewards[emp_best]

        rec = {"spec_id": entry["id"], "heuristic_topology_deprecated": heur_label,
               "vn_top1": ranked[0],
               "vn_top2": ranked[:2], "emp_best": emp_best,
               "emp_best_reward": emp_best_r, "feasible": emp_best_r >= FEASIBLE_REWARD,
               "r_star_stratum": stratum, "r_star_margin": margin}
        records.append(rec)

        if emp_best_r >= FEASIBLE_REWARD:
            feasible_specs += 1
            hit1 = ranked[0] == emp_best
            hit2 = emp_best in ranked[:2]
            if hit1: top1 += 1
            if hit2: top2 += 1
            if stratum in strat:
                strat[stratum]["n"] += 1
                strat[stratum]["top1"] += int(hit1)
                strat[stratum]["top2"] += int(hit2)
                strat[stratum]["emp_best"].append(emp_best)
            if heur_label is not None:
                n_with_heuristic += 1
                if ranked[0] == heur_label:
                    heuristic_agree += 1

        if (n + 1) % 20 == 0:
            d = max(1, feasible_specs)
            h = max(1, n_with_heuristic)
            print(f"  [{n+1}/{len(idxs)}] feasible={feasible_specs}  "
                  f"top1={top1/d*100:.1f}%  top2={top2/d*100:.1f}%  "
                  f"heuristic_agreement={heuristic_agree/h*100:.1f}%", flush=True)

    d = max(1, feasible_specs)
    h = max(1, n_with_heuristic)

    # Majority-class baseline: a constant predictor's score on this pool. With one
    # class near 50% this is the number aggregate accuracy must be read against,
    # not 1/6.
    from collections import Counter
    emp_counts = Counter(r["emp_best"] for r in records
                         if r["feasible"])
    majority_top1 = (max(emp_counts.values()) / d) if emp_counts else None

    strata_out = {}
    for s in STRATA:
        b = strat[s]
        if b["n"] == 0:
            strata_out[s] = None
            continue
        c = Counter(b["emp_best"])
        strata_out[s] = {
            "n": b["n"],
            "top1_accuracy": b["top1"] / b["n"],
            "top2_accuracy": b["top2"] / b["n"],
            "majority_baseline_top1": max(c.values()) / b["n"],
            "emp_best_distribution": dict(c),
        }

    summary = {
        "run": args.run, "seed": args.seed, "n_specs": len(idxs), "k": args.k,
        "specset": args.specset, "switch_model": args.switch_model,
        "feasible_specs": feasible_specs,
        "top1_accuracy": top1 / d,
        "top2_accuracy": top2 / d,
        # Not ground-truth accuracy — agreement with retired heuristic labels (S0).
        "heuristic_agreement": heuristic_agree / h if n_with_heuristic else None,
        "n_with_heuristic_label": n_with_heuristic,
        "random_baseline_top1": 1.0 / len(TOPOS),
        "majority_baseline_top1": majority_top1,
        "s3_strata": strata_out,
    }
    print("\n=== TOPOLOGY SELECTION ACCURACY ===")
    for k_, v_ in summary.items():
        if k_ != "s3_strata":
            print(f"  {k_}: {v_}")
    print("\n  --- S3 margin strata "
          f"(r_star, switch_model={args.switch_model}) ---")
    print(f"  {'stratum':10s} {'n':>5s} {'top1':>7s} {'top2':>7s} {'majority':>9s}")
    for s in STRATA:
        v = strata_out[s]
        if v is None:
            print(f"  {s:10s}     -       -       -         -")
            continue
        print(f"  {s:10s} {v['n']:5d} {v['top1_accuracy']:7.1%} "
              f"{v['top2_accuracy']:7.1%} {v['majority_baseline_top1']:9.1%}")
    print("  boundary = smallest r_star top-2 gap (hardest decisions).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
