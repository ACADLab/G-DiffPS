# Ideal-switch GIN baselines — frozen

Recorded 2026-09-14 for `dev/nuerips_workshop`.

**Do not restart or extend the 50k run.** These 20k/30k runs are the final
ideal-switch GIN regression baselines. Further compute goes to the SKY130
graph → SPICE path and LOOCV transfer experiments.

## Provenance

| Field | Value |
|---|---|
| Branch | `dev/nuerips_workshop` |
| Commit | `90c5ef3e24e3911af55fa398375efe8ae2cba2b5` |
| Message | Load training bounds from run_config in joint inference. |
| Encoder | `gin` |
| Action space | `device` |
| Bounds | `electrical` |
| Switch model | `ideal` |
| Actor | CFM (`--actor cfm --sizing log`) |
| Seed | 42 |
| Batch / buffer | 32 / 5000 |

## Runs

### `train_gin_device_20k`

- Steps: 20000 (complete)
- Success rate (sim returned metrics): 0.5897
- Mean reward: −1.7841
- Best reward: 2.0648 @ step 5065 (Loaded_Line)
- Checkpoints (sha256):
  - `actor.pt` — `241a07cbc24c938e38c30be136e487eb81049ae7c7808a9dac18d0f257f9fcfa`
  - `critic.pt` — `a3ae072af9e5bff1a32523d56a3a46c59c3213537b23e5fa5877292829c77e70`
  - `gnn_encoder.pt` — `34ee71b3396c3f4bedadeba91b734145d263fceedacd4516d719049485f65e15`
  - `value_net.pt` — `aa68236e43977685c463053250d36f0225a75b4e4fa8ad51d5a1d6c68a92ab09`

### `train_gin_device_30k`

- Steps: 30000 (complete)
- Success rate: 0.5865
- Mean reward: −1.7913
- Best reward: 2.0417 @ step 20045 (Loaded_Line)
- Checkpoints (sha256):
  - `actor.pt` — `f0bf736768523516e634bafceeb7076ddf11aa2202ac4d989fa60179115c2411`
  - `critic.pt` — `6df660fa8a0411bef9a1dbb4ccf5eda4032e73ee0b28289674867c9bbd476ba0`
  - `gnn_encoder.pt` — `2951da0e223d4a3d5a76e40450d336e33a965e327fe2d625e628239a4145beca`
  - `value_net.pt` — `12311fefed5d43d40f3401c37cf2d58c073400c70c83a0119e5dd136bdb073e7`

### `train_gin_device_50k` — aborted

- Incomplete (~35298 steps, no checkpoints). Do not resume.

## Evaluation command (reproduction)

```bash
python tools/loocv_eval.py \
  --run results/long_runs/train_gin_device_20k \
  --held-out Loaded_Line \
  --encoder gin --action-space device \
  --bounds electrical --switch-model ideal \
  --eval-specset specset/specset_eval.json \
  --n 200 --seed 42 \
  --out results/long_runs/baseline_eval_ll.json
```

## Notes

- Training "success" means the simulator returned metrics after the physics
  prior; it is **not** specification compliance. See
  `PLATEAU_DIAGNOSIS_2026-09-14.md` for recomputed `all_close` rates (~1.7–2.3%).
- These baselines use the component-adjacency GIN path, not the bipartite
  circuit encoder.
