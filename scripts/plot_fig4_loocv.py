"""
Fig 4: Zero-shot LOOCV per held-out topology (GIN encoder, 3 seeds).
Two bars per topology: full spec compliance (reward>0.5) and physics-prior
pass rate. Reads results/exp3_loocv_gin/*.json. Honest single-framework view.
"""
import os, sys, json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "figures", "fig4_loocv.pdf")
GDIR = os.path.join(REPO, "results", "exp3_loocv_gin")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

TOPOS  = ["Switched_Line", "Vector_Modulator", "Loaded_Line",
          "Reflection_Type", "Switched_Filter", "All_Pass"]
LABELS = ["SL", "VM", "LL", "RT", "SF", "AP"]

def agg(topo, key):
    vals = []
    for seed in [42, 1337, 2026]:
        hits = glob.glob(os.path.join(GDIR, f"zeroshot_run_*_{topo}_seed{seed}.json"))
        if hits:
            d = json.load(open(hits[0]))
            vals.append(d.get(key, 0.0))
    return (np.mean(vals) if vals else 0.0, np.std(vals) if vals else 0.0)

comp  = [agg(t, "success_rate")    for t in TOPOS]
prior = [agg(t, "prior_pass_rate") for t in TOPOS]

x = np.arange(len(TOPOS)); w = 0.38
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.bar(x - w/2, [c[0]*100 for c in comp], w, yerr=[c[1]*100 for c in comp],
       capsize=2, color="#2ca02c", label="Spec compliance")
ax.bar(x + w/2, [p[0]*100 for p in prior], w, yerr=[p[1]*100 for p in prior],
       capsize=2, color="#9ecae1", label="Physics-prior pass")
ax.axhline(50, color="gray", ls=":", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=7)
ax.set_ylabel("Zero-shot rate (\\%)", fontsize=7)
ax.set_title("LOOCV: held-out topology transfer (GIN)", fontsize=7)
ax.set_ylim(0, 105); ax.tick_params(labelsize=6)
ax.legend(fontsize=5.5, loc="upper right")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
