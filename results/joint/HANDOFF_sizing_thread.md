# Handoff: sizing/diagnostics thread → specset-redesign thread

Written 2026-08-25, after reading `SEVENTH_TOPOLOGY_DECISION.md`. Trimmed to
what that thread does **not** already have. Independently-confirmed overlaps are
noted briefly rather than restated.

---

## 1. NEW — the bounds are centred on the textbook answer, systematically

This is the item with the largest consequence and it is not yet recorded
anywhere. `SEVENTH_TOPOLOGY_DECISION.md` found two instances of it (All_Pass
midpoint symmetry, and pre-T0.4 equal `L_short`/`L_long` arms) and treated them
as separate nominal-point artifacts. They are the same rule, and it is general.

Three lines in `action_to_params(..., bounds="electrical")` set every reactive
window:

```python
elif key.endswith("_pf"):
    physical_val = log_scale(val_01, C0_pf / 10.0, C0_pf * 10.0)
elif key.endswith("_nh"):
    physical_val = log_scale(val_01, L0_nh / 10.0, L0_nh * 10.0)
elif key.endswith("_mm"):
    physical_val = log_scale(val_01, 0.4 * lam4, 2.5 * lam4)
```

with C0 = 1/(2π·fc·Z0), L0 = Z0/(2π·fc), lam4 = 47.43/fc_ghz. Each window is
log-symmetric about the canonical 50 Ω resonant value, so **a = 0.5 maps to the
textbook design value for every reactive element, in every topology, at every
frequency.** Verified exactly:

```
Switched_Filter   sqrt(L/C)    = 50.00 ohm at midpoint, all fc
All_Pass          sqrt(L/C_br) = 50.00 ohm at midpoint, all fc
Loaded_Line       L_quarter_mm at a=0.5 = 1.00003 * lam4 (2.4 GHz), 0.99941 (28 GHz)
```

Switched_Filter is the sharpest case: because the HPF and LPF arms receive
*identical* values they become exact electrical duals, pinning Δφ = 180.000° at
fc = 10 (179.499 at 2.4, 177.162 at 28) — precisely its `_IDEAL_STEP`. Its D1
`g_phase` lead of 0.811 is that symmetry, not capability; at random sizings it is
the **worst** of the six (median RMS phase error 58.0° vs 13.6° for Loaded_Line).

So All_Pass's midpoint (Δφ = 0, worst possible) and Switched_Filter's midpoint
(Δφ = 180°, best possible) are the *same phenomenon* with opposite luck. Any
nominal-point comparison ranks bounds placement, not topologies.

**Why it matters beyond diagnostics:** the paper's §4.2 argument — that log
scaling maps a = 0.5 to 0.050 pF, "the resonant target," and that this drives the
465× first-pass-yield improvement — is one instance of this policy. First-pass
yield measured from an untrained policy's mean output is then measuring where the
bounds were placed. The log warp is doing two separable things: (a) making
exploration steps multiplicative, a legitimate inductive bias, and (b) relocating
the solution to the centre of the initial sampling distribution.

**The separating ablation** (specified, *not yet run*): keep log scaling, multiply
both window endpoints by K = 3. Steps stay multiplicative; a = 0.5 becomes
deliberately wrong. If yield survives, the warp earned it; if it collapses, the
centring did. Run on Loaded_Line (where the 465× number lives) and
Switched_Filter. Paused only to avoid concurrent edits to `train_diffusion.py`.

---

## 2. NEW — All_Pass can meet real specs, not just produce phase

`SEVENTH_TOPOLOGY_DECISION.md` independently reached the same All_Pass
degeneracy diagnosis (box-best 179.8°, median 85.4°, 71.5% of draws > 45°;
my run: 179.97°, 86.4°, 71.1% — consistent). One thing it does not report, which
strengthens the "healthy away from the midpoint" claim considerably: those large
Δφ values are *also well matched*. Constraining uniform random search to
|Δφ| ∈ [85°, 95°], K = 20000:

```
fc=2.4   4.8% of draws in window   best RL=28.78 dB  (dphi -90.42, IL 0.57 dB)
fc=10    4.8% of draws in window   best RL=30.22 dB  (dphi  92.26, IL 0.56 dB)
fc=28    4.8% of draws in window   best RL=30.37 dB  (dphi  93.41, IL 0.56 dB)
```

A 90° cell at ~30 dB return loss and 0.56 dB insertion loss, at every carrier,
reachable by one uniform draw in twenty. All_Pass is not merely non-degenerate
away from the midpoint — it is competitive. This raises the stakes on the open
item already tracked there (give All_Pass a non-degenerate nominal point), and it
means the 4.4% class share is very likely an artifact rather than a property.

Minor note for whoever re-derives these: the *signed* Δφ distribution is
symmetric about zero because A/B labelling is arbitrary, so a signed median near
5° means "symmetric," not "no phase shift." The magnitude median is ~86°.

---

## 3. NEW — verified negative, do not re-investigate

The MNA phase sign convention is **correct**. The hypothesis worth ruling out was
that every `_IDEAL_STEP` is negative while the solver returns positive deltas,
which would score a good +90° circuit as 171° of error while leaving a ±180°
target untouched — and would have manufactured exactly Switched_Filter's
apparent advantage. Tested by recomputing every error with the ideal grid's sign
flipped: `rms_flip` is worse than `rms_asis` for all six topologies, and deltas
are predominantly negative as the grid expects.

Switched_Filter's two columns are exactly equal (58.02 vs 58.02) — a ±180° target
*is* sign-degenerate — but since the sign is right, that degeneracy is inert.

---

## 4. Confirmations (already known there, no action)

- **T0.4 has landed.** Recovered from the window endpoints:
  `[0.4, 2.5]·lam4` log-scaled, a = 0.5 → 1.000·λ/4. So `max_area_mm2` anchors are
  not invalidated by T0.4 — they need regenerating only because the pool changed
  from v1 600 to v2 10k/2k.
- **Seventh topology.** For provenance only: the original removal in the sizing
  thread was on an explicit user instruction, not on the 2×2. The physics
  re-derivation in `SEVENTH_TOPOLOGY_DECISION.md` now supplies a real cause
  (collapse requires a C_off ~16× worse than any shipped tech), which is a
  stronger disposition than the one it replaced.

---

## 5. `r_star` — owned by the specset thread

Review item (6) and the sizing thread's T4 retarget are the same work. The
sizing thread will **not** build it.

One input: the existing D2 numbers (median spread 1.339, 96% of specs above 0.3)
are best-of-K = 32 **uniform random**, which measures *basin width*, not the
envelope, and is confounded by action dimension — Loaded_Line and Switched_Line
have 3 dims, Switched_Filter and VM 4, Reflection_Type 5, All_Pass 6. Probability
of landing within ±10% of the optimum in every dimension at K = 32 is 22.7% at
3 dims and 0.2% at 6, so the ranking partly reflects dimensionality. Relabel
rather than reuse as an envelope baseline.

| quantity | measures | source |
|---|---|---|
| `r_sim` at nominal | bounds placement | D1 |
| best-of-K random | basin width | D2 |
| max over converged optimizer | the envelope | `r_star`, not yet built |

---

## 6. Stale artifacts from the sizing thread

Measured against the v1 600-spec pool, now frozen and no longer live — re-run
against v2 before citing:

- `results/joint/reward_decompose.json` (D1)
- `results/joint/envelope_spread_mna.json` (D2)
- `results/joint/t15_accept.json` (T1.5 acceptance)
- `max_area_mm2` / `area_rank` budgets, appended to the v1 pool

Verified as already absorbed by the v2 refactor, no action needed:
`models/topology_policy.py` defaults to `SPEC_DIM`, and
`inference_topology_select.py` imports it rather than hardcoding 13.

---

## 7. Proposed but unbuilt: Monte Carlo yield term

Switched_Filter's 180° bit sits on a knife edge — the perturbation series from
breaking its HPF/LPF duality is +5% → 5.88°, +20% → 25.48°, +100% → 111.17° of
RMS phase error. Real RF passives are ±5% (0402 caps, chip inductors), so its
headline capability carries ~6° of RMS phase error *as manufactured*, against a
4-bit/180° quantization floor of 3.25°. Nothing in the current reward
distinguishes a knife-edge design from a robust one.

Proposal: 32 draws at ±5% component tolerance, score the **10th percentile**
rather than the mean. Costs 32× MNA (still milliseconds), discriminates at
mmWave where the area term does not, and addresses the paper's own stated
process-variation limitation. Agreed to sit **alongside** the area term, not
replace it. Owner not yet assigned.

---

## Tools and raw results

| path | what |
|---|---|
| `tools/phase_metric_probe.py` | per-topology signed Δφ + RMS error as-shipped vs sign-flipped, over random sizings and fc |
| `tools/allpass_box_probe.py` | All_Pass midpoint √(L/C) ratios, match/loss, box reachability |
| `results/joint/phase_metric_probe.json` | output, K = 64, fc ∈ {2.4, 10, 28} |
| `results/joint/allpass_box_probe.json` | output, K = 3000 |

Both are MNA-only and run in seconds; neither needs a GPU.
