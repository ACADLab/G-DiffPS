# Transfer LOOCV protocol (frozen before Stage 5)

- Branch: `dev/nuerips_workshop`
- Primary metric: `strict_compliance` (per-metric conjunction)
- Secondary: `tolerant_all_close`, `physical_reward` (expert bonus = 0)
- Eval pool: `specset/specset_eval.json` (fail closed if missing)
- Switch model: `ideal` for transfer comparison (matched)
- Bounds: `electrical`
- Seeds: 42, 1337, 2026
- Checkpoint selection: final training checkpoint only (no test-set picking)
- Tags compared: `circuit_device` (candidate) vs `gin_device` (ablation)
- Branch rule: steps ≤ 2000 until C0 green (C0 passed 2026-09-14)

## Pilot (Stage 4)

```bash
export G_DIFFPS_MATRIX_TAGS=circuit_device,gin_device
python tools/run_matrix.py --track corrected --steps 1000 --seeds 42 \
  --folds All_Pass,Vector_Modulator,Reflection_Type --n-eval 40 --workers 2 \
  --out results/transfer_pilot
```

## Full (Stage 5)

```bash
export G_DIFFPS_MATRIX_TAGS=circuit_device,gin_device
python tools/run_transfer_loocv.py --mode full --steps 2000 --n-eval 100 --workers 2
```

## Baselines required alongside trained models

- Midpoint action (a=0.5) under identical decoder/prior
- Uniform random actions under identical decoder/prior
- Untrained (random init) actor under identical eval harness
