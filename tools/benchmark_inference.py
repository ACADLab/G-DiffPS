"""Bounded joint-inference benchmark: sizing-only, MNA, evaluate-all, ValueNet→one.

Official deployment path: size all six + MNA rank with prior/pmax filter.
Narrative baseline (separate): ValueNet topology pick + size winner once.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import get_topology_graph
from inference_joint import (
    TOPOLOGY_NAMES, _sample_one, evaluate_all, load_policy, _resolve_flag, _run_config,
)
from models.circuit_encoder import uses_param_nodes
from sim.mna_scorer import mna_evaluate
from specset.schema import load_specset, TRAIN_SPECSET_PATH
from inference_topology_select import normalize_spec, score_topologies, load_models as load_valuenet


def _get_sample_specs(n_specs: int = 50) -> list[dict]:
    dataset = load_specset(TRAIN_SPECSET_PATH)
    return [entry["spec"] for entry in dataset[:n_specs]]


def _stats(xs: list[float]) -> dict:
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def benchmark_joint_inference(
    run_dir: str = "results/joint_ckpt_circuit_device",
    encoder: str = "circuit-typed",
    action_space: str = "device",
    bounds: str | None = None,
    switch_model: str | None = None,
    coupled_actions: bool = False,
    n_specs: int = 50,
    n_warmup: int = 10,
    device_str: str = "cpu",
    output_json: str | None = None,
) -> dict:
    device = torch.device(device_str)
    if not os.path.exists(run_dir):
        alt = "results/smoke_defaults"
        if os.path.exists(alt):
            run_dir = alt

    run_cfg = _run_config(run_dir)
    if encoder == "circuit-typed" and "encoder" in run_cfg:
        encoder = run_cfg["encoder"]
    bounds = _resolve_flag(bounds, run_cfg, "bounds", "electrical")
    switch_model = _resolve_flag(switch_model, run_cfg, "switch_model", "ideal")
    if switch_model not in ("ideal", "realistic"):
        raise ValueError(
            f"switch_model must be ideal|realistic; got {switch_model!r} "
            f"(pass CLI or set run_config.json)"
        )
    coupled = (
        bool(coupled_actions)
        or bool(run_cfg.get("coupled_actions", False))
        or uses_param_nodes(encoder)
    )

    gnn, actor, warned = load_policy(
        run_dir, encoder, action_space, device,
        coupled_actions=coupled,
    )
    topo_graphs = {t: get_topology_graph(t) for t in TOPOLOGY_NAMES} if encoder == "gin" else None

    sample_specs = _get_sample_specs(max(n_specs, n_warmup, 2))
    base_spec = sample_specs[0]
    fc_vals = [float(s["fc_ghz"]) for s in sample_specs[:n_specs]]

    # Determinism under fixed seed (evaluate-all path)
    torch.manual_seed(42)
    np.random.seed(42)
    ranked_1 = evaluate_all(
        base_spec, gnn, actor,
        encoder=encoder, action_space=action_space,
        device=device, n_samples=1, bounds=bounds,
        switch_model=switch_model, coupled_actions=coupled,
    )
    torch.manual_seed(42)
    np.random.seed(42)
    ranked_2 = evaluate_all(
        base_spec, gnn, actor,
        encoder=encoder, action_space=action_space,
        device=device, n_samples=1, bounds=bounds,
        switch_model=switch_model, coupled_actions=coupled,
    )
    is_deterministic = all(
        r1["topology"] == r2["topology"]
        and abs(r1["score"] - r2["score"]) < 1e-9
        and r1["prior_ok"] == r2["prior_ok"]
        for r1, r2 in zip(ranked_1, ranked_2)
    )

    # Cold measurement (first evaluate_all after load, no warmup)
    torch.manual_seed(0)
    t_cold = time.perf_counter()
    evaluate_all(
        base_spec, gnn, actor,
        encoder=encoder, action_space=action_space,
        device=device, n_samples=1, bounds=bounds,
        switch_model=switch_model, coupled_actions=coupled,
    )
    cold_ms = (time.perf_counter() - t_cold) * 1000.0

    for _ in range(n_warmup):
        evaluate_all(
            base_spec, gnn, actor,
            encoder=encoder, action_space=action_space,
            device=device, n_samples=1, bounds=bounds,
            switch_model=switch_model, coupled_actions=coupled,
        )

    test_specs = sample_specs[:n_specs]
    sizing_times_per_topo = {t: [] for t in TOPOLOGY_NAMES}
    mna_times_per_topo = {t: [] for t in TOPOLOGY_NAMES}
    total_sizing_all_six = []
    total_mna_all_six = []
    total_pipeline_times = []
    evaluate_all_times = []
    prior_pass_counts = []

    for s in test_specs:
        t_start = time.perf_counter()
        sn = normalize_spec(s).to(device)

        t_size_start = time.perf_counter()
        params_by_topo = {}
        for topo in TOPOLOGY_NAMES:
            t_topo_start = time.perf_counter()
            p = _sample_one(
                actor, gnn, topo, s, sn,
                encoder, action_space, device, topo_graphs,
                bounds=bounds, switch_model=switch_model,
                coupled_actions=coupled,
            )
            sizing_times_per_topo[topo].append((time.perf_counter() - t_topo_start) * 1000.0)
            params_by_topo[topo] = p
        total_sizing_all_six.append((time.perf_counter() - t_size_start) * 1000.0)

        t_mna_start = time.perf_counter()
        for topo in TOPOLOGY_NAMES:
            t_mna_topo = time.perf_counter()
            mna_evaluate(topo, params_by_topo[topo], s)
            mna_times_per_topo[topo].append((time.perf_counter() - t_mna_topo) * 1000.0)
        total_mna_all_six.append((time.perf_counter() - t_mna_start) * 1000.0)
        total_pipeline_times.append((time.perf_counter() - t_start) * 1000.0)

        t_ea = time.perf_counter()
        ranked = evaluate_all(
            s, gnn, actor,
            encoder=encoder, action_space=action_space,
            device=device, n_samples=1, bounds=bounds,
            switch_model=switch_model, coupled_actions=coupled,
        )
        evaluate_all_times.append((time.perf_counter() - t_ea) * 1000.0)
        prior_pass_counts.append(sum(1 for r in ranked if r["prior_ok"]))

    # ValueNet → size-one narrative path (GIN TopologyEncoder + value_net.pt only)
    valueneet_report = None
    vn_path = os.path.join(run_dir, "value_net.pt")
    if encoder == "gin" and os.path.isfile(vn_path):
        try:
            vn_gnn, value_net = load_valuenet(run_dir, device)
            for _ in range(min(5, n_warmup)):
                ranked_v = score_topologies(base_spec, vn_gnn, value_net, device)
                winner = ranked_v[0][0]
                sn = normalize_spec(base_spec).to(device)
                _sample_one(
                    actor, gnn, winner, base_spec, sn,
                    encoder, action_space, device, topo_graphs,
                    bounds=bounds, switch_model=switch_model,
                    coupled_actions=coupled,
                )
            rank_ms, size_ms, total_ms = [], [], []
            for s in test_specs:
                t0 = time.perf_counter()
                ranked_v = score_topologies(s, vn_gnn, value_net, device)
                t1 = time.perf_counter()
                winner = ranked_v[0][0]
                sn = normalize_spec(s).to(device)
                _sample_one(
                    actor, gnn, winner, s, sn,
                    encoder, action_space, device, topo_graphs,
                    bounds=bounds, switch_model=switch_model,
                    coupled_actions=coupled,
                )
                t2 = time.perf_counter()
                rank_ms.append((t1 - t0) * 1000.0)
                size_ms.append((t2 - t1) * 1000.0)
                total_ms.append((t2 - t0) * 1000.0)
            valueneet_report = {
                "path": "valuenet_rank_all_six_then_size_winner_once",
                "note": "Narrative / legacy baseline; not the deployment evaluate-all path.",
                "valuenet_rank_all_six_ms": _stats(rank_ms),
                "sizing_winner_only_ms": _stats(size_ms),
                "total_valuenet_plus_one_size_ms": _stats(total_ms),
            }
        except Exception as exc:
            valueneet_report = {"error": str(exc), "path": "valuenet_unavailable"}
    elif os.path.isfile(vn_path):
        valueneet_report = {
            "skipped": True,
            "reason": "ValueNet loader expects encoder=gin (TopologyEncoder); "
                      "re-run with --encoder gin on a gin checkpoint for this baseline.",
        }

    report = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(device),
            "torch_num_threads": torch.get_num_threads(),
        },
        "benchmark_config": {
            "run_dir": run_dir,
            "encoder": encoder,
            "action_space": action_space,
            "bounds": bounds,
            "switch_model": switch_model,
            "coupled_actions": coupled,
            "num_diffusion_steps": getattr(actor, "num_steps", 10),
            "n_specs_benchmarked": n_specs,
            "n_warmup": n_warmup,
            "fc_ghz_range": [min(fc_vals), max(fc_vals)] if fc_vals else None,
            "determinism_pass": is_deterministic,
            "checkpoint_warned_missing_weights": warned,
            "official_path": "evaluate_all_size_six_mna_rank_prior_pmax",
        },
        "latency_ms": {
            "cold_evaluate_all": cold_ms,
            "sizing_only_single_topo": {
                t: _stats(sizing_times_per_topo[t]) for t in TOPOLOGY_NAMES
            },
            "sizing_only_all_six_topologies": _stats(total_sizing_all_six),
            "mna_simulator_all_six_topologies": _stats(total_mna_all_six),
            "manual_size_plus_mna_pipeline": _stats(total_pipeline_times),
            "evaluate_all_official": _stats(evaluate_all_times),
            "valuenet_plus_one_size": valueneet_report,
        },
        "compliance": {
            "mean_prior_ok_count_of_six": float(np.mean(prior_pass_counts)),
            "prior_ok_counts": prior_pass_counts,
        },
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="Joint inference latency and determinism benchmark.")
    ap.add_argument("--run", default="results/transfer_pilot/dry_circuit_device")
    ap.add_argument("--encoder", default="circuit")
    ap.add_argument("--action-space", default="device")
    ap.add_argument("--bounds", default=None, help="Default: run_config or electrical")
    ap.add_argument("--switch-model", default=None, help="Default: run_config or ideal")
    ap.add_argument("--n-specs", type=int, default=20)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--out", default="results/graphs/benchmark_latency.json")
    args = ap.parse_args()

    report = benchmark_joint_inference(
        run_dir=args.run,
        encoder=args.encoder,
        action_space=args.action_space,
        bounds=args.bounds,
        switch_model=args.switch_model,
        n_specs=args.n_specs,
        n_warmup=args.n_warmup,
        output_json=args.out,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
