# Work order: joint topology selection + sizing

Reformulate G-DiffPS so topology choice is a trained objective rather than a
byproduct of the sizing critic. Ordered; each task lists acceptance criteria.
Do not start a task until the one before it passes.

Reference: `FRAMEWORK.md` §5 (reward), §6 (training loop), §8.2 (current
selection), §9.5 (open reward issue), §12 (file map).

---

## Root causes being fixed

**R1 — The reward is not comparable across topologies.**
`FRAMEWORK.md` §5.1 normalizes IL by `max_il_db`, RL by `min_rl_db`, gain by
`rms_gain_err_db` — all fields of the spec vector **s**. The phase term, weight
0.40, divides by a hardcoded 90°. The spec vector already carries
`rms_phase_err_deg` (§2) and it is never read. Topologies with different ideal
steps (−22.5°, −90°, −180°) are therefore scored on different scales, and
`argmax_τ V_ψ` compares incommensurable numbers.

**R2 — The expert bonus contaminates the critic.**
`r = r_sim + expert_bonus` (§5.2) puts `specset/phaseshifter_scoring.py` inside
the Q/V targets. Ranking by V_ψ then partly reads back the injected heuristic.

**R3 — V_ψ encodes prior-filter calibration, not topology capability.**
Prior rejections return −5.0 into the same buffer. §9.1 documents the loop:
low pass rate → buffer fills with −5.0 → value net avoids the topology. That is
a property of `sim/physics_priors.py` bounds, not of the circuit.

**R4 — One head, two jobs.**
V_ψ is the IQL expectile baseline (E[r] under the current policy, τ=0.7) and the
ranker (max_a r). Different functionals of different distributions.

**R5 — Wrong metric.**
Top-1 accuracy over 6 near-tied topologies (run_080429: 2.271 vs 2.229) reports
a 0.042-reward mistake as total failure.

**R6 — Selection is under-determined: no area, no power.**
`compute_reward` scores phase, IL, RL and gain flatness only. Area appears
nowhere — not in **s**, not in the reward, not in `sim/physics_priors.py`. Power
is worse: `vdd` and `pmax_mw` are in **s** and condition the actor, but have no
reward term. Four of twelve spec dimensions (`vdd`, `pmax_mw`, `tech`, `app`)
receive no gradient at all.

Consequence: run_080429 shows all six topologies above 1.98 reward. If every
topology meets the S-parameter spec, `argmax_τ` is choosing among near-ties and
the problem is under-determined rather than hard. Area breaks the tie — but only
at low frequency, because distributed topologies are wavelength-scaled while
discrete component and package footprints are not:

| f_c | smallest | largest | spread |
|---|---|---|---|
| 2.4 GHz | Switched_Filter 18.2 mm² | Reflection_Type 400 mm² | 22× |
| 10 GHz | Loaded_Line 11.6 mm² | Reflection_Type 31.9 mm² | 2.7× |
| 28 GHz | Loaded_Line 9.8 mm² | All_Pass 18.9 mm² | 1.9× |
| 38 GHz | Loaded_Line 9.5 mm² | All_Pass 18.9 mm² | 2.0× |

At mmWave the footprint is dominated by switch package count rather than
wavelength, and the ordering inverts — Reflection_Type becomes second smallest.
So area makes the sub-10 GHz end well-posed (where All_Pass sits at 0%) and does
little at 28 GHz. `FRAMEWORK.md` §1 already lists All_Pass as the sub-6 GHz
choice; that is the low-frequency area statement the reward cannot express.

See T1.5a: these numbers assume board-level microstrip, which is what ε_eff = 2.5
implies. The medium must be declared before any of this is implementable.

---

## Target formulation

Per-metric margin, identical form for all four, all spec-relative:

```
g_phase = clip(1 - rms_phase_err_deg_measured / s.rms_phase_err_deg, 0, 1)
g_il    = clip(1 - il_db                      / s.max_il_db,         0, 1)   # 0 if il_db < 0
g_rl    = clip(    abs(rl_db)                 / s.min_rl_db,         0, 1)
g_gain  = clip(1 - abs(gain_err_db)           / s.rms_gain_err_db,   0, 1)

r_sim = Σ_m w_m · g_m  + 1.0 · all_close
```

Sizing path (unchanged): CFM actor, advantage-weighted by `A = Q_φ − V_ψ`.

Ranking path (new): `u(τ|s) = R_ξ(s, z_τ^rank)` estimates the achievable
envelope `r*(τ,s) = max_a r_sim(τ,s,a)`. Listwise (Plackett–Luce) loss over the
six topologies at a shared spec:

```
L_rank = − Σ_s Σ_τ  p*(τ|s) · log softmax_τ( u(τ|s) / T )
         where p*(τ|s) = softmax_τ( r*(τ,s) / T )
```

Total: `L = L_CFM + L_Q + L_V + λ · L_rank`, λ tuned on validation regret.

Encoder: shared trunk `h = f_G(G_τ, s, states)`, two linear projections
`z^size = W_s h` and `z^rank = W_r h`. `L_rank` backprops into `f_G` — this is
the discriminative signal the encoder currently never receives (paper Appendix A
found compliance invariant across encoders spanning L₂ = 0.00 → 4.03).

---

## T0 — Blockers from the previous session (do first, both are cheap)

**T0.1 — Resolve the off-switch contradiction.**
The MLCAD paper §5 states `C_off = 650 fF`. The templates use `R_off = 10 kΩ`.
At 28 GHz, 650 fF is 8.7 Ω — nearly a short — and 10 kΩ is 1143× more open.
Determine which produced Tables 1–5 in the published paper. Report the answer
before anything else; every downstream number depends on it.

Also record, as a benchmark limitation: a frequency-independent 10 kΩ off-state
is ~35× more open than a realistic 20 fF mmWave SPST at 28 GHz (284 Ω), and
unlike hardware it does not degrade with frequency.

**T0.2 — Answer the probe/deploy question.**
Does `tools/contrast_probe.py` build the graph from the *sampled* sizing or from
`nominal_params()`? The deployed encoder runs before the actor picks values, so
it only ever sees the bounds midpoint. If the probe used sampled values, every
R² in that table is inflated and must be re-run on midpoint graphs.

Related: within-topology embedding variance moved 0.0265 → 0.0105 after Level 2.
That is the wrong direction — Level 2 was supposed to make the embedding vary
*more* with the design request. Explain the drop or treat Level 2 as not landed.

**T0.1b — R_on and R_off must stop being action dimensions.**

They are currently policy-sized in all six topologies (`FRAMEWORK.md` §4.3:
R_off log [1e3, 1e6] Ω, R_on linear [0.5, 10] Ω). A designer does not choose
R_on — you buy a switch or you are in a process, and the parasitics come with
it. Sizing them lets the policy purchase a switch that does not exist, reaching
for R_off = 1 MΩ whenever a topology needs isolation.

This is structurally identical to the §9.2 bug (G_I/G_Q > 1.0 buying active
gain) and takes the same three-layer fix: constants in `action_to_params`, gate
in `sim/physics_priors.py`, no reward path that rewards the unphysical value.

**Do not add a C_off model.** All four layers (templates, `mna_scorer`,
`physics_priors`, encoder `R_switch` features) model the off state as a
resistor, and the collapse comes from the magnitude, not the model form. Keep
the resistor and make its value frequency-aware:

```
R_off_eff(tech, fc) = 1 / (2*pi*fc*C_off(tech))
R_on(tech)          = constant per tech
```

A 284 Ω resistor and a 20 fF capacitor present the same impedance at 28 GHz, so
this reproduces the T0.1 measurement with zero structural change.

Action-space consequence — the padded action shrinks from ℝ⁹ to ℝ⁷:

| Topology | params now | after |
|---|---|---|
| Loaded_Line | 5 | 3 |
| Switched_Line | 5 | 3 |
| Reflection_Type | 7 | 5 |
| Switched_Filter | 6 | 4 |
| Vector_Modulator | 6 | 4 |
| All_Pass | 8 | 6 |

**T0.1c — Interpret the collapse correctly before acting on it.**

Series-switch off isolation in a 50 Ω through path:

| Off model | @2.4 GHz | @28 GHz |
|---|---|---|
| R_off = 10 kΩ (templates) | −40.1 dB | −40.1 dB |
| C_off = 20 fF (real SPST) | −30.7 dB | **−11.7 dB** |
| C_off = 650 fF (paper text) | −6.1 dB | −0.7 dB |

At 28 GHz a realistic switch leaks 26% of incident amplitude through the
unselected branch, which is why Switched_Line's median |Δφ| fell 65.7° → 3.4°.
But real switched-line shifters work at 28 GHz — using a **series-shunt SPDT**,
a shunt switch to ground on the off arm that reflects the leakage. One 5 Ω shunt
path takes isolation from −11.7 dB to roughly −27 dB.

So the finding is not "realistic switches make the benchmark unsolvable." It is
**the templates omit the isolation structure branch-selection topologies
require, and only an idealized switch hides the omission.** Note which three
collapse — Switched_Line, Switched_Filter, All_Pass — the same three that had
`cc = 2` in the old adjacency graph and the same three the T6 permutation
theorem covers. Loaded_Line, Reflection_Type and Vector_Modulator tune
parameters rather than select paths and degrade gradually.

**Action:** test the shunt switch in `mna_scorer` first — one device, costs
nothing. If it restores Δφ, add the series-shunt variant as a **seventh
topology** rather than editing `Switched_Line` in place. Published numbers then
stay valid for the topology they describe, and you gain a controlled pair that
is the sharpest available demonstration of why topology selection matters: under
an ideal switch the two variants are equivalent and the choice is arbitrary;
under a realistic switch only one works.

**On invalidation.** MLCAD '26 stands as an *ideal-switch* benchmark. Its flaw
is that it never says so. Scope it rather than retract it: state the switch
model explicitly and present the realistic-switch result as the extension's
contribution. That is a paper, not an erratum.

**T0.3 — Note two more paper/code discrepancies** (do not fix, just log):
- `FRAMEWORK.md` §3.2 and §10.3 say 50 Euler steps. The paper §3.2 says ten, with
  a written justification for why ten is optimal.
- `FRAMEWORK.md` §3.1 still describes SAGEConv; shipped code is GIN.

---

## T1 — Make the reward spec-relative

File: `env/phaseshifter_env.py`, `compute_reward`.

Replace the hardcoded 90° phase denominator with `s.rms_phase_err_deg`. Guard
against zero/absent values with a floor (suggest 1.0°). Keep the weights and the
all-close bonus as they are for now.

The §5.1 rationale for 90° was gradient shaping — an untrained agent at 20–30°
error needs signal. Preserve that with a *soft* floor on the denominator rather
than a constant: `denom = max(s.rms_phase_err_deg, warmup_deg)` where
`warmup_deg` anneals from 45° to 0° over the first 2k steps. Log both the raw and
annealed values so the ablation is reportable.

**Acceptance:** on the 600-spec specset, the per-topology distribution of
`r_sim` at a fixed, known-good design (template defaults) has cross-topology
spread below 0.15. Currently it will be much wider. Produce the before/after
table.

## T1.5 — Add area as a scored objective

Files: new `sim/area_model.py`; `env/phaseshifter_env.py`;
`specset/specset_phaseshifter.json` (new spec field).

**Must land before T2.** See T2's warning.

### T1.5a — Declare the medium (do this first)

The benchmark already committed. `FRAMEWORK.md` §4.2 fixes ε_eff = 2.5, which
back-solves through Hammerstad at Z₀ = 50 Ω to ε_r ≈ 3.2 — a Rogers-class
laminate. Every published number was generated on **board-level microstrip**,
not on-die. Fix the constants to match and do not mix media:

```
eps_r = 3.2,  h = 0.254 mm (10 mil)  ->  W(50 ohm) = 0.611 mm,  eps_eff = 2.55
```

Consequences, all mandatory:

- **No MIM caps, no spiral inductors.** Those are on-die concepts and a 19.8 mm
  line at 2.4 GHz does not exist on a die. Use discrete footprints including
  pads and keepout: 0201 ≈ 0.36 mm², 0402 ≈ 1.0 mm², SPDT package ≈ 4.0 mm².
  Treat the VCVS as a packaged gain block ≈ 6.0 mm².
- **Reinterpret `tech`.** CMOS/SiGe/GaAs are inconsistent with this substrate.
  Redefine `tech` as *switch* technology (PIN diode / GaAs pHEMT / SOI SPDT),
  which is board-compatible and is what actually sets the switch parasitics.
  How that interacts with T0.1 and with the action space is **not** a one-line
  substitution — see T0.1b below before touching anything.
- **Log on-chip as future work.** It would change ε_eff per technology, hence
  λ/4, hence every TL bound in §4.3, hence every published number.
- **State the medium in the paper.** It is currently unstated, yet which
  topology "wins" is entirely medium-dependent. That belongs in the artifact
  argument, not in a footnote.

### T1.5b — Area model

Bounding-box area, applied consistently:

- **Straight transmission line** — `length × W(Z0)`, length from the sized
  parameter, width from Hammerstad.
- **Branchline hybrid (Reflection_Type)** — genuine 2-D structure, bounding box
  `(λ/4)²`. Do not sum arm areas.
- **Discrete passives and switches** — fixed footprint per device from the table
  above.

The constants above are placeholders. Calibrate against a real BOM before
publishing any area number.

Nominal areas this produces (sanity target for the implementation):

| f_c | smallest | largest | spread |
|---|---|---|---|
| 2.4 GHz | Switched_Filter 18.2 mm² | Reflection_Type 400 mm² | 22× |
| 10 GHz | Loaded_Line 11.6 mm² | Reflection_Type 31.9 mm² | 2.7× |
| 28 GHz | Loaded_Line 9.8 mm² | All_Pass 18.9 mm² | 1.9× |
| 38 GHz | Loaded_Line 9.5 mm² | All_Pass 18.9 mm² | 2.0× |

**Read this honestly.** Area discriminates hard at 2.4 GHz and collapses to ~2×
at mmWave, where footprint is dominated by switch package count rather than
wavelength, and the ordering inverts. Area will **not** rescue selection at
28 GHz — it makes the low-frequency end well-posed, which is where All_Pass
currently sits at 0%. Do not oversell it.

Note also that area depends on sized parameters only through TL length, so it
supplies strong topology gradient and almost no sizing gradient. Keep it in
`r_sim` (it does reward shorter lines on distributed topologies) but expect its
value to show up in the T4 ranking target.

### T1.5c — Budget sampling (rank-anchored, not absolute)

An absolute random range is degenerate: a 2.4 GHz spec drawing a 5 mm² budget is
infeasible for all six, `g_area` is 0 everywhere, and the term becomes a constant
offset — the same failure mode as the dead power term.

Instead, compute the six nominal areas at that spec's f_c, sort them, draw a rank
`r ~ Uniform{1..6}`, and set

```
max_area_mm2 = A_(r) · (1 + eps),   eps ~ U[0.02, 0.10]
g_area       = clip(1 - area_mm2 / s.max_area_mm2, 0, 1)
```

Exactly `r` topologies fit, by construction. At least one always fits; not all
always do; `g_area` is never constant across topologies. Store `r` alongside the
spec so results can be stratified by area difficulty — `r = 1` (area decides)
versus `r = 6` (area inert, S-parameters decide) is a reportable axis.

### T1.5d — Power: do not add a continuous term

Five of six topologies are passive; only Vector_Modulator draws DC, through the
VCVS standing in for a gain block. A continuous power term is a Vector_Modulator
indicator in disguise and would hand the selector a shortcut instead of a signal.
Keep `pmax_mw` as a hard gate in `sim/physics_priors.py` — reject VM when
`pmax_mw` falls below the gain block draw — and leave it out of the reward.

### Weights

Five terms, summing to 1.0. Suggest phase 0.32, IL 0.22, RL 0.16, gain 0.12,
area 0.18 — tune on validation regret and report the vector as a hyperparameter.

**Acceptance:** area model reproduces the table above within 10%. Stratified by
`r`, the per-spec spread of `r_sim` across topologies exceeds 0.3 for `r ≤ 3` on
the sub-10 GHz portion of the specset, where today all six sit above 1.98. If
that spread does not appear, T4 has nothing to fit — stop and diagnose before
building the ranking head.

## T2 — Remove the expert bonus from the reward

Files: `train_diffusion.py` (reward assembly), `specset/phaseshifter_scoring.py`.

**Warning — do not run this before T1.5.** Three of the six `score_topology`
rules are area or power constraints in disguise:

- Switched_Line −5 if fc < 5 GHz — area (43 mm² of line and switches at 2.4 GHz)
- Reflection_Type +5 if fc > 15 GHz — area (branchline is 400 mm² at 2.4 GHz)
- Vector_Modulator −5 if pmax < 10 mW — power (VCVS stands in for a gain block)

The expert bonus is currently the **only** place in the system where area and
power knowledge lives. Deleting it before T1.5 strips real information rather
than removing contamination, and will likely degrade selection.

`r` fed to the buffer becomes `r_sim` only. Keep `score_topology` — it moves to
T5 as a baseline and optionally as a logit prior on the ranking head, where its
contribution is ablatable.

**Acceptance:** grep confirms no path adds `expert_bonus` into anything Q_φ or
V_ψ regresses on. A 2k-step smoke run converges (rewards will shift downward by
up to 0.3; that is expected).

## T3 — Separate prior rejection from infeasibility

Files: `train_diffusion.py`, `env/phaseshifter_env.py`.

Tag every buffer entry with `reject_reason ∈ {none, prior, spice_fail,
spice_missing}`. Prior rejections stay in the actor and critic path (they are a
real penalty for the actor) but are **excluded** from the envelope target in T4.
A prior rejection means the actor proposed bad values, not that the topology
cannot serve the spec.

**Acceptance:** buffer entries carry the tag; envelope statistics computed with
and without prior rejections differ measurably, and the difference is reported
per topology (expect the largest gap on Switched_Filter and Vector_Modulator,
per §9.1).

## T4 — Build the envelope target and the ranking head

Files: `models/diffusion_policy.py` (new `RankHead`), `train_diffusion.py`.

Maintain a running per-`(spec_id, τ)` maximum of `r_sim` over non-rejected
entries — this is `r*(τ,s)`. Warm it with best-of-K sampling (K = 8) for each
spec at the start of training so the envelope is not empty early.

`R_ξ`: MLP over `[s (12) ‖ z^rank (64)]`, same shape as ValueNet. Train with the
listwise loss above, temperature T tuned on validation.

Split the encoder into a trunk plus two projections. `z^size` keeps the current
gradient path; `z^rank` receives `L_rank`. Do **not** let `L_rank` flow into the
actor's conditioning directly — the previous session's concern stands: if the
ranking loss shapes `z^size`, the embedding starts encoding "how good is this"
instead of "what is this" and the C3 zero-shot sizing claim degrades.

**Acceptance:** `L_rank` decreases; `R_ξ` predictions correlate with held-out
`r*` at Spearman ρ > 0.5; the actor's in-distribution yield is unchanged from
the pre-T4 baseline within seed noise.

## T5 — Evaluation: regret, and the baselines that matter

New file: `tools/topology_select_eval.py`. Replaces `probe_value_net.py` as the
selection metric (keep the old probe for backward comparison).

Report, over held-out specs, all four:

| Selector | What it is |
|---|---|
| Random | uniform over 6 |
| Heuristic | `score_topology` alone, no learning |
| Spec-only MLP | 6-way head on **s**, no graph at all |
| Graph ranker | `R_ξ(s, z_τ)` |

Metrics, in priority order:
1. **Absolute regret** = `r*(τ_best, s) − r*(τ_selected, s)`, mean and p90.
2. **Normalized regret** = regret / (r*_best − r*_worst).
3. Top-1 and top-2 accuracy (report, but do not lead with them).

**This is the honest-result gate.** If the graph ranker does not beat the
spec-only MLP on regret, that is the finding and it gets reported. With six
fixed topologies the MLP has fewer parameters and no encoder to fit; it may win.
Say so plainly if it does — that register is already how Appendix A and C are
written and it is the paper's strongest quality.

## T6 — Wire in the state-contrast encoder

Use the corrected per-device-row contrast from the previous session (difference
per aligned device row, then pool — not pool then difference).

The claim to test here is narrow and provable: **for a state transition that
acts as a permutation of the device set, any permutation-invariant state pooling
is exactly state-blind**, independent of features or training. Switched_Line's
measured 0.010 contrast under pool-then-difference is that theorem in floating
point.

Investigate All_Pass separately. It is the other branch-swap topology, so the
theorem predicts the symmetric block should fail there too, but the probe showed
both blocks failing (contrast 0.010, mean 0.072). Either the device-row
alignment for the apA/apB sections is not doing what is intended, or Δφ there
depends on cross-section sizing in a way neither block captures. Determine which.

**Acceptance:** the theorem is stated and unit-tested (construct a synthetic
permutation state pair, assert pooled difference is zero to machine precision).
Ablation of contrast on/off is run through T5's regret metric, not just R².

## T7 — Retrain and re-run the matrix

Every checkpoint is invalid: device features 16 → 20, edges 3 → 6, `dev_proj`
reshaped, and now the reward definition itself has changed. The prior joint
checkpoint and the whole `matrix_corrected` set must be re-run, not reloaded.

Run the LOOCV matrix with the new reward so C3 numbers are comparable to the new
selection numbers. Expect Table 5 values to move; the All_Pass 0% is a
frequency-band mismatch (2.4 GHz vs 10–28 GHz training) and should *not* move,
which is a useful sanity check on the reward change.

---

## Do not do

- **Do not drop component values from the encoder for "pure topology" training.**
  That advice is correct for deduplicating generated circuits, where invariance
  to values is the goal. It is wrong here: a λ/4 line at 2.4 GHz and at 38 GHz
  are not the same circuit, and dropping values re-collapses the structural axis
  to a 6-entry lookup.
- **Do not add contrastive net-renaming augmentation yet.** The six netlists are
  hardcoded literals with fixed net names; there is nothing to be invariant to.
  It becomes load-bearing only in the open-topology regime.
- **Do not resurrect `FRAMEWORK.md` §7.4** (the 630× Nelder-Mead table). Several
  rows read "1 call (trivial)", meaning the baseline beat G-DiffPS immediately,
  and the Switched_Filter row matches a 64.65° phase error — not a meaningful
  bar. The paper's Figure 2 budget curve is the defensible version.
- **Do not tighten `physics_priors.py` to enforce coupling constraints.** §9.4
  already established this the hard way (0.4% pass rate, zero learning).
  Reparameterize in `action_to_params` instead.

---

## Order of operations, condensed

```
T0.1  off-switch contradiction        ── blocking, ~1 hour
T0.1b R_on/R_off out of action space  ── blocking; same bug species as 9.2
T0.1c shunt-switch test in mna_scorer ── one device, decides the 7th topology
T0.2  probe/deploy mismatch           ── blocking, ~1 hour
T1    spec-relative phase reward      ── makes scores comparable
T1.5  medium + area objective         ── makes low-freq selection well-posed
T2    remove expert bonus from r      ── ONLY after T1.5
T3    tag reject reasons
T4    envelope target + rank head
T5    regret eval + 4 baselines       ── the honest-result gate
T6    state-contrast + permutation theorem
T7    full retrain, LOOCV matrix
```

Run T5's evaluation three times — after T1, after T1.5, and after T4 — so the
below-chance Appendix C result is attributed rather than merely fixed. T1
addresses comparability (the 90° constant), T1.5 addresses well-posedness (six
topologies tied above 1.98), and T4 is the learned ranker. If most of the gap
closes at T1.5, the paper's finding is that S-parameter-only objectives make
topology selection under-determined — which is a stronger and more general claim
than any architectural result about the encoder.
