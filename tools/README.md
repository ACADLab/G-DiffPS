# G-DiffPS tooling

Scripts supporting the circuit-graph encoder rebuild and local experiment
matrix. Activate `.venv` first (`source .venv/bin/activate`).

| Script | Purpose |
|---|---|
| `replay_checkpoint.py` | Re-simulate logged designs under local ngspice-47; gates paper comparability |
| `representation_rank.py` | Table-6 replacement: rank / participation ratio of `z_topo` |
| `plot_graphs.py` | Side-by-side PNGs of current vs bipartite graphs → `results/graphs/` |
| `loocv_eval.py` | Zero-shot held-out topology evaluation |
| `run_matrix.py` | Fan out Track A / Track B LOOCV jobs across CPU cores |
| `make_tables.py` | Reconstruct Table 2 (success-only mean) + LOOCV Wilson CIs |

## Quick start

```bash
# Harness gate (already run; verdict: comparable)
python tools/replay_checkpoint.py --n 60

# Rank diagnostic (no SPICE)
python tools/representation_rank.py

# Graph figures
python tools/plot_graphs.py

# Smoke Track A (2 folds, short)
python tools/run_matrix.py --track A --steps 50 --seeds 42 \
    --folds All_Pass,Switched_Filter --workers 2 --n-eval 20 \
    --out results/matrix_smoke

# Full Track A (paper protocol; multi-hour on Apple Silicon)
python tools/run_matrix.py --track A --steps 5000 --seeds 42,1337,2026 \
    --workers 4 --out results/matrix_trackA
```

## Training flags (defaults preserve paper behavior)

```
--encoder {gin,circuit}          # default gin
--action-space {slot,device}     # default slot
--fc-mode {fixed28,spec}         # default fixed28
--bounds {legacy,electrical}     # default legacy
--run-dir PATH                   # optional explicit checkpoint dir
```
