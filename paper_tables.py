"""
J9 — Build E3 and E4 paper tables from existing trace files.

E3 (Sample Efficiency): for each (method, scenario), how many SPICE calls did the
    method need to first reach reward ≥ R* (default 1.5)?
E4 (Deployment under fixed budget): for each (method, scenario), what is the
    best reward attained within B ∈ {1, 10, 50, 200, 1000, ALL} SPICE calls?

Also: aggregates LOOCV results (old vs new GNN) → App C table.

Outputs (LaTeX + CSV):
    results/paper_tables/E3_sample_efficiency.{tex,csv}
    results/paper_tables/E4_budgeted_deployment.{tex,csv}
    results/paper_tables/AppC_gnn_fix.{tex,csv}
"""

import json
import glob
import os
import re
import csv
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "results", "paper_tables")
os.makedirs(OUT_DIR, exist_ok=True)

R_STAR = 1.5
BUDGETS = [1, 10, 50, 200, 1000]

# Scenario key = (topology, fc_ghz)
SCENARIOS = [
    ("Loaded_Line", 28.0), ("Loaded_Line", 38.0),
    ("Switched_Line", 14.0), ("Switched_Line", 24.0),
    ("Vector_Modulator", 5.0), ("Vector_Modulator", 8.0),
    ("Reflection_Type", 28.0), ("Reflection_Type", 10.0),
    ("All_Pass", 2.4),
    ("Switched_Filter", 18.0),
]

METHODS = {
    "RS": ("results/baselines/random_search",     "rs_{topo}_fc{fc}_seed{seed}_trace.jsonl"),
    "DE": ("results/baselines/diff_evolution",    "de_{topo}_fc{fc}_seed{seed}_trace.jsonl"),
    "SA": ("results/baselines/simulated_annealing","sa_{topo}_fc{fc}_seed{seed}_trace.jsonl"),
    "BO": ("results/baselines/bayesian_opt",      "bo_{topo}_fc{fc}_seed{seed}_trace.jsonl"),
}
SEEDS = [42, 1337, 2026]


def load_trace(path):
    """Return list of (spice_calls, reward) tuples; cumulative-best, monotonic."""
    if not os.path.exists(path):
        return None
    rows = []
    best = -999.0
    for line in open(path):
        d = json.loads(line)
        # only count SPICE-evaluated entries (passed_prior or unknown)
        if d.get("passed_prior") is False:
            continue
        r = d.get("reward", -999.0)
        n = d.get("spice_calls", len(rows) + 1)
        if r > best:
            best = r
        rows.append((n, best))
    return rows


def first_reach(rows, threshold):
    for n, r in rows:
        if r >= threshold:
            return n
    return None  # never reached


def best_at(rows, budget):
    """Best reward within first `budget` SPICE calls."""
    if not rows:
        return -999.0
    best = -999.0
    for n, r in rows:
        if n > budget:
            break
        if r > best:
            best = r
    return best


def build_e3():
    """Sample efficiency: SPICE calls to first reach R* (mean over 3 seeds)."""
    print(f"\n========= E3 — Sample Efficiency (calls to reward ≥ {R_STAR}) =========\n")
    rows = []
    for topo, fc in SCENARIOS:
        row = {"topology": topo, "fc_ghz": fc}
        for method, (dir_, fname_tpl) in METHODS.items():
            ns = []
            for seed in SEEDS:
                fc_str = f"{fc}"  # keep "28.0" matching filename convention
                path = os.path.join(REPO_ROOT, dir_, fname_tpl.format(topo=topo, fc=fc_str, seed=seed))
                trace = load_trace(path)
                if trace is None:
                    continue
                n = first_reach(trace, R_STAR)
                if n is not None:
                    ns.append(n)
            row[method] = (sum(ns) / len(ns)) if ns else None
            row[f"{method}_n"] = len(ns)
        rows.append(row)
        line_parts = [f"{topo:<18} fc={fc:>5.1f}"]
        for m in METHODS:
            v = row[m]
            line_parts.append(f"{m}={v if v is None else f'{v:>7.0f}'}")
        print("  " + "  ".join(line_parts))

    # CSV
    with open(os.path.join(OUT_DIR, "E3_sample_efficiency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topology", "fc_ghz"] + list(METHODS.keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["topology", "fc_ghz"] + list(METHODS.keys())})

    # LaTeX
    tex = ["\\begin{tabular}{ll" + "r" * len(METHODS) + "}",
           "\\toprule",
           "Topology & $f_c$ (GHz) & " + " & ".join(METHODS) + " \\\\",
           "\\midrule"]
    for r in rows:
        cells = [f"{r['topology'].replace('_', ' ')}", f"{r['fc_ghz']:.1f}"]
        for m in METHODS:
            v = r[m]
            cells.append("—" if v is None else f"{v:.0f}")
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT_DIR, "E3_sample_efficiency.tex"), "w") as f:
        f.write("\n".join(tex))


def build_e4():
    """Best reward attained within fixed SPICE budgets."""
    print(f"\n========= E4 — Deployment under budget (best reward at B SPICE calls) =========\n")
    rows = []
    for topo, fc in SCENARIOS:
        for method, (dir_, fname_tpl) in METHODS.items():
            # collect traces from 3 seeds
            traces = []
            for seed in SEEDS:
                fc_str = f"{fc}"
                path = os.path.join(REPO_ROOT, dir_, fname_tpl.format(topo=topo, fc=fc_str, seed=seed))
                t = load_trace(path)
                if t:
                    traces.append(t)
            if not traces:
                continue
            row = {"topology": topo, "fc_ghz": fc, "method": method, "n_seeds": len(traces)}
            for B in BUDGETS:
                vals = [best_at(t, B) for t in traces]
                row[f"B={B}"] = sum(vals) / len(vals)
            rows.append(row)

    # Print compact view
    print(f"  {'Scenario':<32} {'Method':<5} " + " ".join(f"B={B:>5}" for B in BUDGETS))
    for r in rows:
        sc = f"{r['topology']}_fc{r['fc_ghz']:g}"
        vals = " ".join(f"{r[f'B={B}']:>+6.2f}" for B in BUDGETS)
        print(f"  {sc:<32} {r['method']:<5} {vals}")

    # CSV
    with open(os.path.join(OUT_DIR, "E4_budgeted_deployment.csv"), "w", newline="") as f:
        fields = ["topology", "fc_ghz", "method"] + [f"B={B}" for B in BUDGETS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # LaTeX (one block per scenario)
    tex = ["\\begin{tabular}{lll" + "r" * len(BUDGETS) + "}",
           "\\toprule",
           "Scenario & Method & " + " & ".join(f"B={B}" for B in BUDGETS) + " \\\\",
           "\\midrule"]
    cur_sc = None
    for r in rows:
        sc = f"{r['topology'].replace('_', ' ')} ({r['fc_ghz']:.1f} GHz)"
        cells = [sc if sc != cur_sc else "", r["method"]]
        cur_sc = sc
        for B in BUDGETS:
            cells.append(f"{r[f'B={B}']:.2f}")
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT_DIR, "E4_budgeted_deployment.tex"), "w") as f:
        f.write("\n".join(tex))


# Encoder-diagnostic: pairwise topology-embedding L2 distance (App B probe).
# Measured by scripts/probe_gnn_embeds.py. The LOOCV metric is ~unchanged across
# encoders, but the embedding geometry is not — this is the crux of the appendix:
# the zero-shot ceiling is set by the CFM actor's parameter prior, not the encoder.
EMBED_L2 = {
    "dying_relu":   {"mean": 0.000, "note": "post-ReLU collapse (all-zero)"},
    "sage_ln":      {"mean": 0.062, "note": "SAGEConv+LN, hypersphere collapse"},
    "gin":          {"mean": 4.031, "note": "GINConv+sum-pool, structurally distinct"},
}


def build_appc_gnn_fix():
    """Encoder diagnostic (App B): dying-ReLU -> SAGEConv+LN -> GIN.

    Shows the LOOCV success rate is statistically unchanged across all three
    encoders even though embedding L2 diversity jumps 0.06 -> 4.03. All three
    rows are the SAME framework (CFM actor, log sizing, IQL); only the topology
    encoder differs. No GNN-free or pre-existing runs enter this table.
    """
    print(f"\n========= App B — Encoder diagnostic (LOOCV invariance) =========\n")
    topos = ["Loaded_Line", "Switched_Line", "Reflection_Type",
             "Switched_Filter", "Vector_Modulator", "All_Pass"]
    rows = []
    for topo in topos:
        for tag, src in [("dying_relu",  "results/exp3_loocv"),
                         ("sage_ln",     "results/exp3_loocv_fixedgnn"),
                         ("gin",         "results/exp3_loocv_gin")]:
            srs, brs = [], []
            for seed in SEEDS:
                p = os.path.join(REPO_ROOT, src, f"zeroshot_run_*_{topo}_seed{seed}.json")
                hits = glob.glob(p)
                if not hits:
                    # fall back to .out parsing
                    out_p = os.path.join(REPO_ROOT, src, f"zeroshot_{topo}_seed{seed}.out")
                    if os.path.exists(out_p):
                        text = open(out_p).read()
                        m_sr = re.search(r"Success rate:\s+([\d.]+)%", text)
                        m_br = re.search(r"Best reward:\s+([+-]?[\d.]+)", text)
                        if m_sr and m_br:
                            srs.append(float(m_sr.group(1)))
                            brs.append(float(m_br.group(1)))
                else:
                    d = json.load(open(hits[0]))
                    srs.append(d["success_rate"] * 100)
                    brs.append(d["best_reward"])
            if srs:
                rows.append({"topology": topo, "config": tag,
                             "success_rate_mean": sum(srs) / len(srs),
                             "best_reward_mean": sum(brs) / len(brs),
                             "n_seeds": len(srs)})
    for r in rows:
        print(f"  {r['topology']:<20}  {r['config']:<14}  "
              f"success={r['success_rate_mean']:>6.1f}%  best={r['best_reward_mean']:>+6.3f}  n={r['n_seeds']}")
    print(f"\n  Embedding L2 diversity (mean pairwise):")
    for k, v in EMBED_L2.items():
        print(f"    {k:<12}  L2={v['mean']:.3f}  ({v['note']})")

    with open(os.path.join(OUT_DIR, "AppC_gnn_fix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topology", "config",
                                          "success_rate_mean", "best_reward_mean", "n_seeds"])
        w.writeheader()
        w.writerows(rows)

    # TeX: side-by-side LOOCV success per topology for the three encoders, with an
    # embedding-L2 footer row. Only renders configs that actually have data.
    cfg_order = [("dying_relu", "Dying-ReLU"), ("sage_ln", "SAGE+LN"), ("gin", "GIN (ours)")]
    have = {c for c in {r["config"] for r in rows}}
    cfgs = [(c, lbl) for c, lbl in cfg_order if c in have]
    by = {(r["topology"], r["config"]): r for r in rows}
    with open(os.path.join(OUT_DIR, "AppC_gnn_fix.tex"), "w") as f:
        f.write("% Auto-generated by paper_tables.py — App B encoder diagnostic.\n")
        colspec = "l" + "c" * len(cfgs)
        f.write("\\begin{tabular}{%s}\n\\toprule\n" % colspec)
        f.write("Topology & " + " & ".join(lbl for _, lbl in cfgs) + " \\\\\n\\midrule\n")
        for topo in topos:
            cells = []
            for c, _ in cfgs:
                r = by.get((topo, c))
                cells.append(f"{r['success_rate_mean']:.1f}\\%" if r else "--")
            f.write(topo.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n")
        l2_cells = [f"{EMBED_L2[c]['mean']:.2f}" if c in EMBED_L2 else "--" for c, _ in cfgs]
        f.write("Embed.\\ $L_2$ & " + " & ".join(l2_cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main():
    build_e3()
    build_e4()
    build_appc_gnn_fix()
    print(f"\n[J9] Tables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
