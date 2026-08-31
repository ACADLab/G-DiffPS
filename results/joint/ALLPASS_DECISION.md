# All_Pass: two defects, and only one of them is the published one

Two explanations were on the table for All_Pass's 0% zero-shot LOOCV compliance
(paper Table 5, attributed in §5.3):

1. **Band mismatch** (§5.3 as published) — "its 2.4 GHz target lies an order of
   magnitude below the 10–28 GHz training band, so the actor's learned LC
   parameter prior is physically mismatched to All Pass's resonance scale — a
   distributional, not architectural, limitation."
2. **Midpoint degeneracy** — both bridged-T sections were sized from a single
   window, so at `a = 0.5` they are identical, the two switch states are the
   same circuit, and Δφ = 0. No frequency involved.

They were expected to compete. They do not. **Both are real, they sit at
different levels, and only the first explains the 0%.**

## Measurement

`tools/allpass_band_probe.py` (model-free) and a DE search for the maximum
reachable differential phase. Under `bounds='legacy'` — the configuration the
published number came from — All_Pass's `action_to_params` map has no `fc` term
at all, so the *set of reachable circuits is identical at every frequency* and
only the evaluation frequency changes. That makes the comparison clean.

Maximum reachable |Δφ| (differential evolution, popsize 60 × 120 generations):

| fc (GHz) | legacy box | electrical box |
|---|---|---|
| **2.4** | **101.67°** | 180.00° |
| 5.0 | 180.00° | 180.00° |
| 10.0 | 180.00° | 180.00° |
| 14.0 | 180.00° | 179.99° |
| 28.0 | 179.99° | 180.00° |
| 38.0 | 180.00° | 180.00° |

Compliant volume (fraction of the box with Δφ ≥ 60°, RL ≥ 12 dB, IL ≤ 3 dB):

| configuration | in-band | out-of-band | at 2.4 GHz | midpoint Δφ across fc |
|---|---|---|---|---|
| legacy (published) | 0.141 | 0.068 | **0.002** | 0.00° … 0.00° |
| electrical, pre-fix split | 0.148 | 0.152 | 0.158 | 0.00° … 0.00° |
| electrical, re-centred split | 0.176 | 0.195 | 0.227 | 86.68° … 86.71° |

## §5.3 stands, with a sharper mechanism

Scenario S9 asks for a 2.4 GHz All_Pass at **180° coverage**. Under the legacy
box the maximum attainable differential phase at 2.4 GHz is 101.67°. *No design
in the box satisfies the spec.* Compliant volume collapses 70× (0.002 vs 0.141
in-band), and 5 GHz is already degraded (0.045) — a smooth low-frequency
rolloff, not a 2.4-specific artifact.

So the failure is genuinely distributional and genuinely about frequency, and
in-band All_Pass does *not* fail. **No correction to §5.3 is required.** One
refinement is worth making if the sentence is ever revised: the paper locates
the fault in "the actor's learned LC parameter prior", and the measurement
locates it one level lower, in the frequency-independent legacy sizing box,
which contains no compliant 2.4 GHz design for any actor to find. That is a
stronger version of the same claim, and it explains Appendix A's encoder
invariance ("fails at 0% regardless of encoder") as a consequence rather than a
coincidence: no encoder can help when the reachable set is empty.

The `bounds='electrical'` change already fixed this — its code comment
("Frequency-centered ranges — makes 2.4 GHz All Pass reachable") is borne out:
compliant volume is flat at ≈0.148 across the whole 2.4–38 GHz sweep.

## The midpoint degeneracy is a separate, live defect

It is **not** the cause of the 0%: it is identical at every frequency and in
both boxes, so it cannot produce a frequency-specific failure. But it is real
and it mattered elsewhere:

- `nominal_params()` returns the box midpoint, and the topology encoder runs
  before the actor picks sizes — so All_Pass's `z_τ` was computed on a circuit
  with **zero phase shift**, whose two switch states are the same graph. That is
  why T6's state contrast read ≈0.010 for All_Pass in both blocks.
- The midpoint was also badly matched there (RL 6.3 dB), so zero phase was not
  its only problem.

This is a Level-2 defect that would have propagated into T4's ranking head,
where `z_τ` is the topology's representation.

### Fix

`ALLPASS_SECTION_SPLIT = 8.0` in `train_diffusion.py` offsets the two section
windows by √8 either way. Scaling L and C together within a section shifts that
section's resonance while leaving √(L/C) at Z₀, so the split buys differential
phase without spending return loss. Window *width* is unchanged, so symmetric
designs remain reachable — only the midpoint moves.

Chosen by sweep (`tools/allpass_recentre_probe.py`) against All_Pass's ideal
90° step:

| midpoint | before (split 1.0) | after (split 8.0) |
|---|---|---|
| Δφ | 0.00° | 86.70° |
| return loss | 6.30 dB | 19.47 dB |
| insertion loss | 2.20 dB | 0.68 dB |
| encoder state contrast | 0.000 | 0.796 |

All fc-invariant across 2.4–40 GHz. The encoder contrast is now comparable to
Switched_Filter's 0.810, i.e. All_Pass's embedding is a representation of
All_Pass.

Same defect species as the pre-T0.4 Switched_Line arms sharing one window, and
the inverse of Switched_Filter's constant-k midpoint: three cases now of bounds
chosen without checking what `a = 0.5` means.

`G_DIFFPS_ALLPASS_SPLIT=1.0` restores the pre-fix box so the published
configuration stays reproducible.

Locked in by `tests/test_realistic_switch_functional.py`:
`test_allpass_midpoint_is_a_real_design`,
`test_allpass_sections_are_reachably_symmetric`.
