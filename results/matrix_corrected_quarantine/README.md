# Quarantined LOOCV cells (bounds-invalid)

`Loaded_Line` and `Reflection_Type` under `--bounds electrical` reported
`prior_pass_rate ≈ 0` because every `*_pf` key used a C0-centered window
(`C0/10…C0*10`). Shunt-loading / tuning caps need a perturbative window
(legacy absolute ranges). See plan `contained_rerun`.

These numbers are **not** valid LOOCV results. Re-run after the electrical
bounds fix lands (`results/matrix_corrected_fixedbounds`).
