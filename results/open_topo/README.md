# Open topology selection — held-out regret experiment

Selection-only experiment on an open topology set (6 frozen + 34 composed).
Does **not** include joint topology-and-sizing training.

## Pipeline

1. `topology/build_pool.py` — enumerate / 1-WL dedup / subsample / emit SPICE
2. `tools/derive_ideal_step.py` — pass-1 phase probe → snap to {22.5,45,90,180}
3. `tools/compute_envelope.py archive --only-topologies … --merge …` — 816 new cells
4. `tools/compute_envelope.py assign --out results/open_topo/r_star_40.json` — sidecar
5. `tools/open_topo_distribution.py` — functional filter
6. `tools/open_topo_loocv.py` — 40 leave-one-topology-out folds

## H2 abort checkpoint

**Passed.** 34/34 composed netlists emit SPICE that round-trips
`assert_matches_template` and score via MNA at 3 carriers.

## Distribution check

33/40 functional (best RMS phase err < 45°, IL < 6 dB at mid-bin).
`r_star` argmax is dominated by Loaded_Line (~56%) and Vector_Modulator (~39%);
composed topologies almost never win (Gen_T30 wins 6/2000). Holding out a
non-winning topology is easy; holding out Loaded_Line is the hard fold.

Spec difficulty (IL/RL/gain anchors) is calibrated on the original six — new
topologies face more headroom by construction.

## Noise floor

Archive-vs-DE over a 6-spec × 8-topo subsample:
`results/open_topo/envelope_validate_40.json` — **p95 gap = 0.137**
(prior six-topo figure was 0.118).

## LOOCV headline (`ideal`, frozen area budgets)

Noise floor p95 = 0.137 beside every column.

| Selector | mean regret | p90 | On held-out topology |
|---|---|---|---|
| random | 1.220 | 1.309 | defined |
| maj_regime | 0.062 | 0.037 | defined, cannot name held-out |
| spec_mlp_39 | 0.062 | 0.058 | **undefined** (no output head) |
| spec_mlp_40 | 0.061 | 0.058 | defined but held-out logit untrained |
| graph_ranker | 0.045 | 0.037 | defined |

Differences among non-random selectors sit below the noise floor. The
structural result is the undefinedness of the fixed-output-head MLP; the
graph ranker's absolute regret is a separate, currently unresolvable,
magnitude question.

Hard fold (hold out Loaded_Line): all informed selectors ≈ 0.32 mean regret.
Easy folds (hold out Gen_* that never win): ≈ 0.03–0.06.

## Artifacts

- `pool.json` — 34 composed netlists + metadata
- `ideal_steps.json` — raw p90 and snapped steps
- `envelope_archive_40.json` — 960 cells (144 reused + 816 new)
- `r_star_40.json` — sidecar (does not clobber v5 eval pool)
- `r_star_40_frozen_area.json` / `r_star_40_reanchored_area.json`
- `distribution_check.json`
- `envelope_validate_40.json`
- `loocv_selection.json`
- `specset/templates/composed/*.sp`
