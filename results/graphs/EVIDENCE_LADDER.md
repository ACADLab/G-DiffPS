# Parameter-role evidence ladder (D1–D5)

Updated 2026-09-15. Mandate: prove/falsify that explicit parameter-role
representation is the missing ingredient for transferable optimization.
No new GNN architectures. ABCD prior off for these runs.

## Protocol fixes applied mid-ladder

1. **Representation-only heads** for D3/D4 graph variants (no raw action concat).
   Earlier drafts leaked the action vector and made GIN/typed look strong.
2. **Padded flatten of param tokens** instead of mean-pool (mean-pool re-collapses
   sizing identity — the thing R3 is supposed to prevent).
3. **All_Pass** `(centre, ratio, coupling)` semantics in `env/param_semantics.py`
   (D1). Reflection `rms_phase_err` excluded from gates (discontinuous vs −22.5°).

## D1 — semantics correctness

- Tests: `tests/test_param_nodes.py`, `test_reflection_phase_target.py`,
  `test_design_variable_audit.py` → **22 passed**.
- All_Pass: one action slot → centre / ratio / coupling; synthetic perturbation
  moves the intended token only.

## D3 — trained same-topology interpolation

Artifact: `results/graphs/d3_trained_interp.json` (`n=250`, `epochs=300`).

| Variant | Gates passed | Notes |
|---------|-------------:|-------|
| action (ceiling) | 5/6 | Fails Reflection only |
| gin | 0/6 | Topology-static; no sizing |
| circuit-typed | 4/6 | **Fails Vector_Modulator** (IL R² ≈ 0.08) |
| d1a | 5/6 | Recovers VM; fails Reflection |
| r3 | 5/6 | Recovers VM; fails Reflection |

Headline per-topology signals:

- **Vector_Modulator**: typed IL R² `0.08` → d1a `0.96` / r3 `0.98`.
  Clearest evidence that global pooling loses sizing identity that param
  tokens preserve.
- **Switched_Line**: typed phase R² `0.60` → d1a/r3 `≥0.95`.
- **Loaded_Line RL**: typed `0.72` → r3 `0.81` (best).
- **Reflection_Type**: all variants fail (action ceiling also fails) — objective
  pathology, not a representation comparison.

Bottleneck probes (same run): on Switched_Line d1a, pooled-`z` probe IL ≈ 0 while
param-token probe IL ≈ 0.79 — information lost at pooling, retained at tokens.

**D3 gate:** pass for d1a/r3 on all reasonable topologies (exclude Reflection).
Typed fails the hard VM case. Proceed to D4.

## D4 — local sensitivity

### v1 (flawed) — `d4_local_sensitivity_v1.json`
Pooled tokens + knob one-hot + MSE. Inconclusive.

### v2 (corrected) — `d4_local_sensitivity.json`
Token-indexed d1a/r3 + sign BCE. `n=200`, `epochs=250`.

| Variant | Gates | Mean sign | Notes |
|---------|------:|----------:|-------|
| action | 4/6 | 0.780 | Ceiling with full action+knob |
| gin | 0/6 | 0.611 | No sizing |
| circuit-typed | 4/6 | 0.818 | Strong signs; weak Δm R² on RL |
| **d1a** | **6/6** | **0.817** | **Only all-pass; best overall** |
| r3 | 3/6 | 0.776 | Wins VM Δm R²; fails Switched_Line/All_Pass/Reflection |

**D4 verdict:** parameter-role context (**D1a**) clears the sensitivity gate.
R3 graph nodes are not clearly better than D1a — D1a is primary for D5; R3 is a paired control.

## Next

- D5 supervised LOOCV running (`tools/d5_supervised_loocv.py`).
- Then R3/D1a ablations + capacity/prior. **D6 RL only if D5 shows real holdout transfer.**