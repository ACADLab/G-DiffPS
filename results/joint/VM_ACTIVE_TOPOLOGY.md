# Vector_Modulator is an active topology: decision record

`Vector_Modulator` is modelled with a VCVS I/Q pair, so it can produce gain.
The other five topologies are passive. This records what was measured, what was
decided, and what was explicitly *not* done.

## What the measurement actually showed

The initial report — that free gain biases `argmax_τ r_star` toward VM — was
wrong. Three findings, in the order they matter.

**1. The negative-IL branch does not move `r_star` for any topology.**

`env/reward.py` scores `il_db < 0` as `g_il = 0.0`. This is §9.2's third layer,
applied deliberately: gain is *maximally penalised*, not rewarded. Recomputing
every topology's envelope with gain instead clipped to `g_il = 1.0`:

| topology | r* as-is | r* with gain clipped to 1.0 | negative-IL front points |
|---|---|---|---|
| Vector_Modulator | 1.8455 | 1.8455 | 471 / 2345 |
| Loaded_Line | 1.8314 | 1.8314 | 0 / 1256 |
| Reflection_Type | 1.7615 | 1.7615 | 0 / 1169 |
| Switched_Line | 0.7583 | 0.7583 | 0 / 1401 |
| Switched_Filter | 0.7471 | 0.7471 | 0 / 120 |
| All_Pass | 0.7519 | 0.7519 | 0 / 101 |

Identical to four decimals. VM's envelope-*optimal* design is a positive-IL
design; the 471 gain-producing points are Pareto-optimal on other axes but are
never the reward argmax. So whichever way the branch is resolved, it does not
change the ranking. VM's lead comes from its 16-state constellation's phase
resolution, not from gain.

**2. §9.2's second layer is inoperative.** All 471 negative-IL front points pass
`check_physics_priors`. The guard tests `g_i > 1.0`, but `action_to_params`
already caps `G_I_scale` at 1.0, so it can never fire. The gain arises from the
fixed `_VM_IQ` constellation combined with the 2× divider compensation, not from
the knob the policy controls. Layer 3 (the reward zeroing) is doing all the
work. Impact is nil because layer 3 is effective, but the "three-layer fix"
description in §9.2 overstates the guard.

**3. The real asymmetry is not in the IL term.** Active and passive are
different design commitments. The reward measures every axis on which active
wins and none on which it loses: VM pays nothing for DC power, noise figure,
linearity, or bias supply. Adding those terms is *not* the fix — a VCVS has no
noise or IP3 model, so the numbers would be invented.

## Decisions

### 1. `pmax_mw` is the admissibility gate

`pmax_mw` had been inert in the spec since the beginning. It is now the gate.

`sim/mna_scorer.topology_admits_spec(topology, spec)` is a **(topology, spec)-only**
predicate — it does not depend on the action, so no sizing can escape it:

```
VM_MIN_DRIVE_MW = min over the I/Q table of 5·((G_I·0.7)² + (G_Q·0.7)²) = 2.45 mW
```

using the `[0.7, 1.0]` cap on `G_I_scale`/`G_Q_scale` from §9.2. When
`pmax_mw < 2.45`, no action makes VM feasible, so `tools/compute_envelope.py`
assigns `INADMISSIBLE_REWARD = -5.0` rather than skipping the pair. The ranker
therefore sees VM *lose on power*, instead of never being offered the trade.

Passive topologies always admit — they draw no bias power.

Under the current `pmax_mw` sampling this splits the eval pool 1547
active-allowed / 453 passive-only (77% / 23%), which is a usable spread without
resampling.

### 2. Anchors are passive-only

`min_τ IL*` over a set containing VM can be negative, and `max_il_db = −3·κ` is
meaningless. `tools/compute_envelope.py frontier` therefore restricts the anchor
set:

```
max_il_db = min_{τ ∈ passive} IL*(fc-bin, phase-quality) · κ_il
```

This is a real modelling choice, not an implementation detail: **a designer does
not relax a loss budget because someone might insert an amplifier.** VM's gain
then reads as genuine headroom against a passive-calibrated spec rather than as
a broken anchor. `anchor_topologies` and `excluded_active` are recorded in
`specset/frontier_anchors.json`.

### 3. Active/passive is a reported stratum, not a confound

Selection accuracy and class shares are reported separately for the two regimes.
A system designer chooses "active or passive" at a different level of the
hierarchy than "which passive topology"; presenting six as one flat choice is
the category error, and stratifying dissolves it.

## What stratifying reveals

Class shares from `argmax_τ r_star` on the 2000-spec eval pool
(`results/joint/envelope_class_shares.json`), `realistic` switches:

| topology | active-allowed (n=1547) | passive-only (n=453) |
|---|---|---|
| Vector_Modulator | 91.5% | — (inadmissible) |
| Loaded_Line | 4.6% | 43.5% |
| Switched_Line | 0.3% | 25.4% |
| Reflection_Type | 0.6% | 13.7% |
| All_Pass | 0.1% | 9.1% |
| Switched_Filter | 2.8% | 8.4% |
| **TV from uniform** | **0.749** | **0.355** |

The flat six-way view (TV 0.541) was averaging two different decisions. The
passive-only regime is the most balanced distribution anywhere in this project
and is a genuinely contested choice; the active-allowed regime is a near-foregone
conclusion. Both are now reportable rather than confounded.

VM's dominance where it is admissible is not an artifact to be removed — it is
the correct engineering answer given that the reward does not price bias supply.
The honest presentation is the stratified one, with the caveat stated.

## Not done, deliberately

- **No noise/linearity/DC-power reward terms.** The VCVS has no noise or IP3
  model; any numbers would be invented and an RF reviewer would catch it.
- **No change to the `il_db < 0` branch.** It is §9.2's deliberate
  anti-reward-hacking fix, and the measurement above shows it does not affect
  `r_star` either way. Changing it would move published numbers for no gain.

## Open

§9.2's layer-2 guard should either be corrected to test the effective
constellation drive or its description downgraded to note that layer 3 carries
the fix. Cosmetic — no measured impact.
