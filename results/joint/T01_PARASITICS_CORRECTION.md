# T0.1 — downgraded: a wrong sentence, not a wrong result

**Status: no retraction. One sentence in §5 is incorrect and should be
corrected. Every published number stands.**

T0.1 was opened as "the paper's switch parasitics may invalidate every published
number." It resolves as "the paper's parasitics sentence cites a part that was
never adopted."

## The incorrect sentence

§5 (Experimental Evaluation) reads:

> SPICE netlists use realistic switch parasitics (*R*on = 6.5 Ω, *C*off = 650 fF).

Neither number is what the code uses. Both cite a part that was never adopted.

| quantity | paper | code |
|---|---|---|
| `R_on` | 6.5 Ω | 3.0 Ω (PIN), 2.5 Ω (GaAs pHEMT), 2.0 Ω (SOI SPDT) |
| `C_off` | 650 fF | 20 fF (PIN, GaAs), 25 fF (SOI) |
| `R_off` | — | 10 kΩ in the `.sp` templates |

Suggested replacement:

> SPICE netlists use realistic switch parasitics: *R*on = 2.0–3.0 Ω and
> *C*off = 20–25 fF depending on the technology option (PIN diode, GaAs pHEMT,
> SOI SPDT), with *R*off = 10 kΩ in the netlist templates.

The `R_off = 10 kΩ` templates and the 20/25 fF `TECH_SWITCH` values are both
defensible for the parts named, so only the prose changes.

## Why the concern did not survive contact with the measurement

The worry was that 650 fF of off-capacitance would wash out the phase shift in
`Switched_Line`, invalidating the results and motivating a seventh topology
(`Switched_Line_SeriesShunt`) to repair it. Two errors inflated it.

**First, the isolation was quoted per switch.** `Switched_Line` puts *two*
series switches on each arm, so the off-state impedance is 2·Z_off.

**Second, isolation is the wrong figure of merit for phase retention.** Phase in
a two-path sum is set by the dominant path; the leaked path perturbs it by at
most arctan(B), where B is the leakage amplitude. This is a far weaker
requirement than the 30+ dB isolation a switch datasheet would ask for.

Reproduced from `sim/switch_model.py` at 28 GHz, Z₀ = 50 Ω, B = 2Z₀/(2Z₀ + 2·Z_off):

| C_off | Z_off | 2·Z_off (two switches) | isolation | leakage B | max Δφ error |
|---|---|---|---|---|---|
| **20 fF** (adopted) | 284.2 Ω | 568.4 Ω | −16.5 dB | 0.150 | **8.5°** |
| **25 fF** (adopted) | 227.4 Ω | 454.7 Ω | −14.9 dB | 0.180 | **10.2°** |
| 200 fF | 28.4 Ω | 56.8 Ω | −3.9 dB | 0.638 | 32.5° |
| 400 fF | 14.2 Ω | 28.4 Ω | −2.2 dB | 0.779 | 37.9° |
| 650 fF (paper text) | 8.7 Ω | 17.5 Ω | −1.4 dB | 0.851 | 40.4° |

Collapse needs B ≳ 0.6, i.e. the 200–400 fF region — which matches the measured
collapse boundary in `results/joint/SEVENTH_TOPOLOGY_DECISION.md`. At the values
actually adopted the phase error is bounded by 8.5–10.2°, roughly an order of
magnitude inside the collapse regime.

So even under the (incorrect) 650 fF figure the worst case is a 40° phase error,
not a collapse of the mechanism — and under the values the code actually uses
there is comfortable margin.

## Consequences

- **No retraction.** The published numbers were produced with 20/25 fF, which is
  what the corrected sentence describes.
- **The seventh topology abandonment was correct.** `Switched_Line_SeriesShunt`
  existed to repair a collapse that does not occur at the adopted parasitics.
  See `results/joint/SEVENTH_TOPOLOGY_DECISION.md`.
- **`realistic` remains a functional benchmark**, locked in by
  `tests/test_realistic_switch_functional.py`, which asserts branch-select phase
  retention and a margin between each tech's `C_off` and the collapse boundary.
