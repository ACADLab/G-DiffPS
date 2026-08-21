"""
Fig 3: Best reward vs SPICE budget B for G-DiffPS vs iterative baselines.
Reads from results/paper_tables/table_e4_budget.csv when available.
Falls back to synthetic illustrative data if results not ready.
"""
import os, sys, json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "figures", "fig3_budget.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 10 paper scenarios used throughout
SCENARIOS = [
    ("LL",  28), ("LL",  38), ("SL",  14), ("SL", 24),
    ("VM",   5), ("VM",   8), ("RT",  28), ("RT", 10),
    ("AP", 2.4), ("SF",  18),
]
BUDGETS = [1, 10, 50, 200, 1000]
METHODS = {
    "DE":  {"color": "#1f77b4", "ls": "-",  "marker": "o"},
    "SA":  {"color": "#ff7f0e", "ls": "-",  "marker": "s"},
    "BO":  {"color": "#9467bd", "ls": "-",  "marker": "^"},
    "SAC": {"color": "#8c564b", "ls": "-",  "marker": "D"},
    "NM":  {"color": "#7f7f7f", "ls": "--", "marker": "x"},
}
GDIFFPS_COLOR = "#2ca02c"

def load_real_data():
    """Try to load from paper_tables CSV produced by paper_tables.py."""
    csv_path = os.path.join(REPO, "results", "paper_tables", "table_e4_budget.csv")
    if not os.path.exists(csv_path):
        return None
    import csv
    rows = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            budget = int(row["budget"])
            reward = float(row["mean_reward"]) if row["mean_reward"] != "None" else np.nan
            std    = float(row["std_reward"])  if row["std_reward"]  != "None" else 0.0
            rows.setdefault(method, {})[budget] = (reward, std)
    return rows

def make_synthetic_data():
    """Illustrative curves showing amortization narrative."""
    rng = np.random.default_rng(42)
    data = {}
    asymptotes = {"DE": 1.55, "SA": 1.52, "BO": 1.50, "SAC": 1.45, "NM": 1.42}
    for method, asym in asymptotes.items():
        data[method] = {}
        for B in BUDGETS:
            progress = 1 - np.exp(-B / 80)
            noise = rng.normal(0, 0.03)
            data[method][B] = (asym * progress + noise, 0.05)
    # G-DiffPS constant (zero budget)
    gdiffps_r = 1.51
    data["G-DiffPS"] = {B: (gdiffps_r, 0.02) for B in BUDGETS}
    data["G-DiffPS"][0] = (gdiffps_r, 0.02)
    return data

real = load_real_data()
data = real if real is not None else make_synthetic_data()
is_real = real is not None
print(f"Data source: {'real results' if is_real else 'synthetic placeholder'}")

fig, ax = plt.subplots(figsize=(3.5, 2.6))

for method, style in METHODS.items():
    if method not in data:
        continue
    xs = sorted(data[method].keys())
    ys  = [data[method][b][0] for b in xs]
    err = [data[method][b][1] for b in xs]
    ax.errorbar(xs, ys, yerr=err,
                label=method, color=style["color"], ls=style["ls"],
                marker=style["marker"], markersize=4, lw=1.2, capsize=2)

# G-DiffPS: horizontal dashed line at B=0
if "G-DiffPS" in data:
    gdiffps_r, gdiffps_e = data["G-DiffPS"][BUDGETS[0]]
    ax.axhline(gdiffps_r, color=GDIFFPS_COLOR, ls="--", lw=1.5,
               label="G-DiffPS (0 SPICE)")
    ax.fill_between([0.8, 1100],
                    gdiffps_r - gdiffps_e, gdiffps_r + gdiffps_e,
                    color=GDIFFPS_COLOR, alpha=0.12)

if not is_real:
    ax.text(0.5, 0.05, "Placeholder — real data pending J1/J3/J4",
            transform=ax.transAxes, fontsize=5, color="gray",
            ha="center", style="italic")

ax.set_xscale("log")
ax.set_xlabel("SPICE budget $B$", fontsize=7)
ax.set_ylabel("Best reward (mean $\\pm$ std)", fontsize=7)
ax.set_title("Deployment quality vs.\ per-spec SPICE cost", fontsize=7)
ax.legend(fontsize=5, ncol=2, loc="lower right")
ax.tick_params(labelsize=6)
ax.set_xlim(0.8, 1500)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
