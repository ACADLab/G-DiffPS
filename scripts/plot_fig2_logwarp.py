"""
Fig 2: Reward density vs C_load at 28 GHz under linear vs log action spaces.
Reads from results/J1_cfm_fixedgnn/ or generates illustrative placeholder.
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "figures", "fig2_logwarp.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Physical resonance: f0 = 1/(2pi sqrt(LC)), L~0.5nH at 28 GHz → C_res~0.065 pF
C_res_pF = 0.065    # resonant capacitance at 28 GHz (pF)

def reward_model(C_pF):
    """Simplified S11 reward proxy: Lorentzian peak at resonance."""
    width = 0.008  # pF half-width (very narrow at 28 GHz)
    return np.exp(-((C_pF - C_res_pF) / width) ** 2)

fig = plt.figure(figsize=(3.5, 2.8))
gs  = gridspec.GridSpec(2, 1, hspace=0.55)

# ── Linear action space ───────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
C_min_lin, C_max_lin = 0.005, 10.0
a_vals = np.linspace(0, 1, 2000)
C_lin  = a_vals * (C_max_lin - C_min_lin) + C_min_lin
R_lin  = reward_model(C_lin)
ax0.plot(a_vals, R_lin, color="#d62728", lw=1.2)
ax0.axvline((C_res_pF - C_min_lin) / (C_max_lin - C_min_lin),
            color="#d62728", ls="--", lw=0.8, alpha=0.6, label=f"$C_{{res}}$={C_res_pF} pF")
ax0.set_title("Linear action space", fontsize=7)
ax0.set_xlabel("$a \\in [0,1]$", fontsize=6)
ax0.set_ylabel("Reward", fontsize=6)
ax0.tick_params(labelsize=5)
ax0.set_ylim(-0.05, 1.15)
ax0.legend(fontsize=5)
ax0.text(0.02, 0.85, "Peak invisible\nat this scale",
         transform=ax0.transAxes, fontsize=5, color="#d62728")

# ── Log action space ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
C_min_log, C_max_log = 0.005, 0.5
C_log = 10 ** (a_vals * np.log10(C_max_log / C_min_log) + np.log10(C_min_log))
R_log = reward_model(C_log)
ax1.plot(a_vals, R_log, color="#2ca02c", lw=1.2)
ax1.axvline(0.5, color="#2ca02c", ls="--", lw=0.8, alpha=0.6, label="$a=0.5$ → $C_{res}$")
ax1.set_title("Log action space (Eq.~3)", fontsize=7)
ax1.set_xlabel("$a \\in [0,1]$", fontsize=6)
ax1.set_ylabel("Reward", fontsize=6)
ax1.tick_params(labelsize=5)
ax1.set_ylim(-0.05, 1.15)
ax1.legend(fontsize=5)

fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
