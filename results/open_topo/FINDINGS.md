# Open topology selection — findings

Selection-only held-out-topology regret experiment on an open set
(6 frozen originals + 34 composed). Does **not** include joint
topology-and-sizing (CFM / T4 / T7).

Artifacts live under `results/open_topo/`; pipeline sketch in `README.md`.

---

## How long

| Span | Duration |
|---|---|
| Session wall clock (plan → LOOCV + README) | **~4 h** (25 Aug 2026, ~18:11–22:13 EDT) |
| Sandboxed archive attempts (stuck / killed) | **~3 h** wasted before restart outside sandbox |
| Successful archive (816 new cells, 6 workers) | **~7 min** (`863492`, 21:43–21:50 EDT) |
| `r_star` assign over 2000 specs × 40 topos | **~3–4 min** (sidecar at 21:53) |
| Light archive-vs-DE noise floor | **~12 s** |
| Full 40-fold LOOCV (ideal, CPU) | **~30 s** per run |

End-to-end compute after the sandbox abort was under ~15 minutes; most of
the evening was the stuck archive jobs.

---

## Chronological checklist

1. **H0 go/no-go** — Confirmed MNA core is device-generic; entry points were
   name-keyed and needed a temporary-register / load-pool shim. Original six
   are single switch-group / ≤12 devices (compose caps: ≤14 devices, one group).
2. **Compose + dedup + emit (H1/H2)** — Enumerate → 1-WL dedup → stratified
   subsample of 34 → emit SPICE under `specset/templates/composed/`.
3. **H2 abort checkpoint — passed** — 34/34 composed netlists round-trip
   `assert_matches_template` and score via MNA at 3 carriers.
4. **Ideal step (H3)** — Pass-1 phase probe → snap to `{22.5, 45, 90, 180}`;
   wrote `ideal_steps.json` + `pool.json` (~18:27 EDT).
5. **Envelope archive (H4)** — Several sandboxed `archive` jobs hung; killed
   and restarted unsandboxed. Final: 960 cells (144 reused + 816 new),
   0 cell errors → `envelope_archive_40.json`.
6. **`r_star` sidecar** — `r_star_40.json` (does not clobber v5 eval pool);
   frozen-area headline + 40-reanchored secondary area variants.
7. **Distribution check** — `distribution_check.json`: 33/40 functional.
8. **Noise floor** — `envelope_validate_40.json` (6 specs × 8 topos subsample):
   **p95 gap = 0.137** (prior six-topo figure was 0.118).
9. **LOOCV** — Smoke (3 folds) then full 40-fold ideal selection;
   final numbers in `loocv_selection.json` (regret fix re-run ~22:10).
10. **README** — Pipeline + headline table written ~22:13.

---

## Findings

### Distribution / difficulty

- **33/40 functional** (best RMS phase err < 45°, IL < 6 dB at mid-bin).
  Non-functional: Gen_T07, T10, T20, T22, T23, T26, T29.
- Ideal `r_star` argmax over 2000 specs is dominated by **Loaded_Line (~56%)**
  and **Vector_Modulator (~39%)**. Composed topologies almost never win
  (**Gen_T30** is the only composed winner: **6/2000**).
- Holding out a never-winning Gen_* fold is easy; **holding out Loaded_Line
  is the hard fold**. Spec difficulty (IL/RL/gain anchors) remains calibrated
  on the original six — composed topologies face more headroom by construction.

### Noise floor

- Archive-vs-DE over the 40-set subsample: **p95 gap = 0.137**, mean gap
  ≈ −0.15, 65% of cells within 0.05. Magnitudes of selector differences must
  be read against this floor.

### LOOCV headline (`ideal`, frozen area budgets, noise floor p95 = 0.137)

All 40 folds (`aggregate_all`):

| Selector | mean of mean regret | mean of p90 |
|---|---|---|
| random | 1.220 | 1.309 |
| maj_regime | 0.062 | 0.037 |
| spec_mlp_39 | 0.062 | 0.058 |
| spec_mlp_40 | 0.061 | 0.058 |
| graph_ranker | 0.045 | 0.037 |

Functional-only (33 folds) is essentially the same (±0.002).

- **Hard fold** (hold out Loaded_Line): all informed selectors ≈ **0.32**
  mean regret.
- **Easy folds** (hold out Gen_* that never win): ≈ **0.03–0.06**.
- Differences among non-random selectors sit **below the noise floor**.
- Structural result: **`spec_mlp_39` is undefined on the held-out class**
  (no output head; reported regret uses forced 39-way fallback).
  **`graph_ranker` is defined** on held-out topologies. Absolute regret
  magnitudes vs. noise remain a separate, currently unresolvable question.

Claim recorded in `loocv_selection.json`:

> On a held-out topology the spec-only MLP is structurally undefined (no
> output head). The graph ranker is defined. Magnitudes sit beside the
> archive-vs-DE noise floor; differences below that floor are unresolvable.

---

## What we do not claim

- Joint topology-and-sizing improvement, CFM sizer gains, T4 joint loss, or T7.
- That the graph ranker is *measurably better* than maj_regime / MLP in
  absolute regret — gaps are inside the 0.137 p95 noise floor.
- That composed topologies are competitive winners under current `r_star`
  (they almost never are).
- Realistic / C_off LOOCV as the headline (headline is **ideal**, frozen area).
- That frozen-area and 40-reanchored-area sidecars are comparable specs.
- Production-ready open-set topology discovery; this is a selection-only
  regret probe on a deliberately capped compose grammar.

---

## Key artifact paths

- `pool.json`, `ideal_steps.json`
- `envelope_archive_40.json`, `r_star_40.json`
- `r_star_40_frozen_area.json`, `r_star_40_reanchored_area.json`
- `distribution_check.json`
- `envelope_validate_40.json`
- `loocv_selection.json`
- `README.md` (this file’s sibling)
