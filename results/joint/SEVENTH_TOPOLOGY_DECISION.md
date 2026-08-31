# Seventh topology (`Switched_Line_SeriesShunt`): decision record

**Decision: abandoned. Reason: there is no collapse left to remedy.**

An earlier turn recorded "seventh topology abandoned" in the hygiene list with no
stated cause, and deleted `results/joint/sl_seriesshunt_matrix.json`, which was
the evidence. The artifact has been restored (flagged as historical — it was
produced under an older `eps_eff`, so its absolute numbers do not match a current
run) and the question has been re-measured from scratch by
`tools/seventh_topology_decision.py` →
[`seventh_topology_decision.json`](seventh_topology_decision.json).

## The four cells that were asked for

Reward at mid-action, fc = 28 GHz, tech 0, `bounds='electrical'`:

| variant | switch model | r_sim | rms φ err | IL | RL |
|---|---|---|---|---|---|
| `Switched_Line` | ideal | **+0.4589** | 6.29° | 0.56 dB | 28.32 dB |
| `Switched_Line_SeriesShunt` | ideal | +0.4557 | 6.30° | 0.60 dB | 29.18 dB |
| `Switched_Line` | realistic | **+0.3547** | 6.03° | 1.93 dB | 21.95 dB |
| `Switched_Line_SeriesShunt` | realistic | +0.2754 | 5.23° | 3.39 dB | 16.41 dB |

No crossover. The shunt loses under *both* switch models, and it loses harder
under realistic — the opposite of the proposed effect.

The reason is visible in the columns: the shunt does slightly improve phase
accuracy under realistic switches (6.03° → 5.23°) but pays 1.46 dB of extra
insertion loss and 5.5 dB of return loss to get it. It buys isolation the
circuit does not need and pays in loss it cannot afford.

## Why the reward 2×2 was the wrong test anyway

T0.1c did not claim the reward dropped. It claimed **Δφ collapsed**, 65.7° → 3.4°.
That is the measurement that decides whether the topology functions:

| topology | Δφ ideal | Δφ realistic | retention |
|---|---|---|---|
| `Switched_Line` | 98.90° | 98.52° | **0.996** |
| `Switched_Filter` | 180.00° | 180.00° | **1.000** |
| `All_Pass` | 0.00° | 0.00° | — (see below) |

**The collapse does not reproduce.** Switched_Line keeps 99.6% of its
differential phase under realistic switches.

## Why the collapse went away

Two candidate mechanisms were tested. Only the second survives.

*Geometry (rejected).* Before T0.4, `L_short_mm` and `L_long_mm` both fell
through to the generic `_mm` rule and shared one window, so at mid-action the
arms were equal and Δφ was ~0 by construction. T0.4 split them (short
0.3–0.8·λ/4, long 0.8–2.5·λ/4). Forcing the arm ratio back down does reproduce a
small Δφ — but retention stays at ~1.0 the whole way:

| arm ratio ρ | Δφ ideal | Δφ realistic | retention |
|---|---|---|---|
| 1.00 | 0.000° | 0.000° | — |
| 1.02 | 0.990° | 1.007° | 1.017 |
| 1.10 | 4.950° | 5.024° | 1.015 |
| 1.50 | 24.738° | 24.883° | 1.006 |
| 3.00 (T0.4 nominal) | 98.950° | 98.576° | 0.996 |

So leakage at 284 Ω does not degrade Δφ at *any* geometry. Equal arms make Δφ
small, but they do not make it switch-sensitive. Geometry alone cannot explain a
65.7° → 3.4° collapse.

*Off-state impedance (accepted).* Δφ collapse requires a far worse off-state
than any tech in the benchmark carries:

| fc | C_off for retention < 0.5 | R_off at that point |
|---|---|---|
| 2.4 GHz | 5000 fF | 13.3 Ω |
| 10 GHz | 650 fF | 24.5 Ω |
| 28 GHz | 400 fF | ~28 Ω |
| 40 GHz | 200 fF | 19.9 Ω |

`TECH_SWITCH` ships C_off = 20, 20, 25 fF. At 28 GHz that is a **16× margin** to
the collapse boundary; at 40 GHz, **8×**. The original 65.7° → 3.4° figure is
consistent with the C_off = 650 fF row of the isolation table (8.7 Ω, −0.7 dB
isolation at 28 GHz) — a cheap board-level part that was never adopted as a tech.

**Conclusion: the collapse was a property of a switch constant that is not in the
benchmark.** The series-shunt variant is a fix for a condition the codebase does
not have. Abandoning it is a physics call, not a bookkeeping one.

## Consequence: `switch_model='realistic'` is a functional benchmark

Five of six topologies work under realistic switches with no modification. What
realistic switches actually cost is **loss and match, not phase** — Switched_Line
goes from 0.56 → 1.93 dB IL and 28.3 → 22.0 dB RL, which is why reward falls
0.459 → 0.355. That is a meaningful and correctly-signed difficulty increase,
which is what the switch model was for.

## The one real defect, and it is not switch-related

`All_Pass` reads Δφ = 0.00° under **ideal** switches too. Its two sections are
identically sized at the box midpoint (`L_apA == L_apB`, `C_brA == C_brB`,
`C_cA == C_cB`), so Δφ vanishes by symmetry. Away from the midpoint the topology
is healthy: box-best 179.8°, median 85.4°, and 71.5% of random draws exceed 45°.

This is a **nominal-point artifact**, and it retroactively explains three
otherwise separate findings:

- All_Pass ranked worst on ~95% of specs at template defaults,
- All_Pass took 0% share in Table 5,
- T6 measured contrast 0.010 for All_Pass in both blocks — at the midpoint the
  two switch states are the same circuit, so contrast is *identically* zero.

It also means any acceptance test evaluated at template defaults is structurally
unfair to All_Pass, which is independent support for retiring the T1 `< 0.15`
gate as unreachable-at-defaults.

Tracked as an open item: All_Pass needs a non-degenerate nominal point (break the
A/B symmetry in the default action, e.g. offset the two sections' mid-action
values) before any per-topology nominal comparison including it is meaningful.

## Regression cover

`tests/test_realistic_switch_functional.py` — 24 combinations (2 branch-select
topologies × 3 techs × 4 carriers) assert Δφ retention > 0.9, plus the off-state
margin, the T0.4 arm partition, and the All_Pass midpoint degeneracy. If a future
`TECH_SWITCH` edit reintroduces the collapse, these fail and name the series-shunt
variant as the remedy to reconsider.
