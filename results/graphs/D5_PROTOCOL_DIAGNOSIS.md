# D5 stopped: preliminary protocol diagnosis

No matching D5 process was active at inspection. Preserved the log and extracted completed results in d5_stopped_partial_summary.json. No RL or replacement matrix launched.

## Confirmed from code

- D4v2 indexes the perturbed parameter token and trains sign BCE plus magnitude MSE. D5 flattens all parameter tokens, appends a topology-local knob index, and trains only standardized magnitude MSE. D5 therefore does not directly test transfer of the successful D4v2 method.
- Flattened token positions and knob indices have different physical meanings across topology families. This introduces a positional confound in the transfer head.
- GIN receives no current sizing in the D5 input. Its failure is not a capacity-matched test of parameter semantics.
- Dataset seeds use Python hash(topology), which varies across processes unless PYTHONHASHSEED is fixed. Model initialization occurs before the training helper seeds PyTorch.
- The report is written only at matrix completion. The partial log preserves rounded IL R2 and IL sign accuracy, plus aggregate gate flags; other per-metric predictions cannot be reconstructed from it.
- R2 alone cannot distinguish large physical error from a small held-out target variance. Need held-out variance, MAE/RMSE and train-mean/majority-sign baselines.

## Interpretation

This configuration has failed the transfer gate. Explicit parameter semantics remain unproven for unseen-topology optimization; the current result neither establishes their benefit nor cleanly falsifies D4v2 transfer. Near-0.5 accuracy is not automatically chance without class balance and majority baselines.

## Next bounded experiment

Before further matrix runs: use the exact D4v2 selected-token/loss protocol, stable seeds set before initialization, train-only normalization, per-fold checkpoints, and a sizing-informed baseline. Test one shared role on one held-out topology with matched input information. Report balanced sign accuracy, majority baseline, per-role errors and target variance. Keep ABCD off and RL gated. Do not change architecture or exclude hard targets based merely on poor fit.

**Implemented:** `tools/d5b_role_transfer.py` → `results/graphs/d5b_role_transfer.json`
(default: impedance role, hold out Vector_Modulator).
