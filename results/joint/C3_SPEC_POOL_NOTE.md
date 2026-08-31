# C3 / LOOCV and the spec pool (S4)

## Published numbers (do not retract)

All Table 5 / `results/matrix_corrected/` LOOCV figures were produced against
the 600-spec pool frozen at:

```
specset/specset_v1_frozen.json
```

Pin every published-number reproduction script to that file with
`load_specset(..., expect_version=1)`. Do not regenerate it.

## What those numbers actually measured

`PhaseShifterEnv(restrict_to=[held_out])` filtered the **topology** list only.
`reset()` sampled with replacement from the same 600 specs used in training.
So C3 measured:

> topology transfer **conditional on seen specifications**

not "zero exposure to the held-out design requests."

## What v2 changes

| Pool | File | Size |
|------|------|------|
| train | `specset/specset_train.json` | ~10,000 |
| eval  | `specset/specset_eval.json` | ~2,000 |
| live default | `specset/specset_phaseshifter.json` | = train |

`schema_version=2`, observation dim `SPEC_DIM=19` (one-hot `phase_bits`/`tech`).
Load-time `assert_disjoint_pools` refuses overlapping ids.
`tools/loocv_eval.py --eval-specset` samples from the eval pool.

## Re-reporting

A full LOOCV retrain under v2 (parent plan T7) is required before quoting new
C3 numbers. When available, report the delta vs `matrix_corrected` explicitly.
Until then, keep citing Table 5 with the restatement above.
