# G-DiffPS

**Physics-Informed Graph Diffusion Policy for Amortized Multi-Topology RF Phase Shifter Synthesis**

[MLCAD 2026](https://github.com/ACADLab/G-DiffPS) · Paper PDF: [`MLCAD_final.pdf`](MLCAD_final.pdf) · Source: [`paper.tex`](paper.tex)

G-DiffPS is, to our knowledge, the first **graph-conditioned conditional flow matching (CFM)** policy for RF phase-shifter synthesis. Once trained, it emits SPICE-ready component values in ≈2 ms per specification across the 1–40 GHz band. Inference cost does not grow with topology or frequency target.

## Claims (from the paper)

| ID | Claim |
|----|--------|
| **C1** | **Amortization** — search cost is paid once at training; deployment needs no per-spec SPICE search (99% first-pass in-distribution yield) |
| **C2** | **Topology transfer** — one GIN-conditioned actor covers six heterogeneous topologies; LOOCV measures zero-shot transfer |
| **C3** | **ABCD shield** — analytical pre-filter rejects ≈80% of non-resonant candidates before SPICE (≈4.7× less training sim overhead) |

Three mechanisms enable this: (i) physics-informed **log action warping**, (ii) a **GIN** topology encoder conditioning the CFM actor, and (iii) a dual-tier **ABCD-matrix** pre-filter.

## Install

**Requirements:** Python 3.10+, PyTorch with CUDA (recommended), [PyG](https://pytorch-geometric.readthedocs.io/), and a system [`ngspice`](https://ngspice.sourceforge.io/) on `PATH`.

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate gdiffps

# Or pip (after installing a matching PyTorch + PyG build)
pip install -r requirements.txt
```

Optional cluster note (Wulver / similar): load your site CUDA/cuDNN modules before activating the env. Do **not** run GPU training on the login node.

## Quickstart

### Train

```bash
# Syntax / shape check
python train_diffusion.py --dry-run --total-timesteps 5 --actor cfm --sizing log

# Online RL training (CFM + log warping; paper default)
python train_diffusion.py --actor cfm --sizing log --total-timesteps 20000 --gpus 1
```

SLURM example (edit account/QoS/paths for your site):

```bash
sbatch train.sh
```

Checkpoints and `train.log` are written under `runs_diffusion/run_<timestamp>/`.

### Inference (topology select + demo weights)

A demo checkpoint from the paper setting (CFM + GIN) is shipped under `checkpoints/run_20260530_031117/`:

```bash
python inference_topology_select.py \
  --run checkpoints/run_20260530_031117 \
  --spec '{"fc_ghz":28,"bw_pct":20,"phase_coverage_deg":360,"phase_bits":5,
           "rms_phase_err_deg":5,"rms_gain_err_db":1,"max_il_db":5,
           "min_rl_db":10,"vdd":1.8,"pmax_mw":15,"tech":0,"app":2}'
```

Or score a spec from the bundled set by index:

```bash
python inference_topology_select.py --run checkpoints/run_20260530_031117 --spec-idx 0
```

### Baselines / eval (paper comparisons)

```bash
# Classical optimizers (DE / SA / BO / RS) and SAC / LLM sizer live under baselines/
python baselines/diff_evolution.py --help

# CFM scenario eval / topology-selection accuracy / table export
python cfm_scenario_eval.py --help
python topo_selection_accuracy.py --help
python paper_tables.py --help
```

## Training arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--total-timesteps` | 100 | Online RL environment steps |
| `--actor` | `cfm` | `cfm` (flow matching) or `ddpm` |
| `--sizing` | `log` | `log` (physics-informed) or `linear` (ablation) |
| `--seed` | 42 | Random seed |
| `--batch-size` | 16 | Replay sample size per update |
| `--buffer-size` | 1000 | Replay capacity |
| `--gpus` | 1 | GPU count |
| `--dry-run` | off | Short validation run |

## Repository layout

```
train_diffusion.py              Main CFM/DDPM online RL training loop
inference_topology_select.py    ValueNet topology ranking at inference
cfm_scenario_eval.py            Scenario evaluation helpers
topo_selection_accuracy.py      Topology-selection metrics
paper_tables.py                 Export paper tables
train.sh                        Example SLURM job
environment.yml / requirements.txt
FRAMEWORK.md                    Extended technical reference
MLCAD_final.pdf / paper.tex     Paper snapshot + LaTeX source

env/                            Gymnasium env, graphs, exploration/memory
models/                         GIN encoder, CFM/DDPM actors, Critic, ValueNet
sim/                            ABCD physics priors + ngspice runner
specset/                        600 specs, scoring, 6 SPICE templates
netlist/                        Parameter bounds / clamping
baselines/                      DE, SA, BO, RS, SAC, LLM sizer, CBO
scripts/                        Figure regeneration (Fig. 2–5)
figures/                        Paper figures (PDF)
results/paper_tables/           Condensed table CSVs/TeX
checkpoints/run_20260530_031117 Demo CFM+GIN weights
```

## Six phase-shifter topologies

| Topology | Bits | Phase step | Key parameters |
|----------|------|------------|----------------|
| Loaded_Line | 1 | −22.5° | Z0_line, L_quarter_mm, C_load_pf |
| Switched_Line | 1 | −90° | Z0_line, L_short_mm, L_long_mm |
| Reflection_Type | 1 | −22.5° | Z0_main, Z0_branch, L_quarter_mm, C_base/tune_pf |
| Switched_Filter | 1 | −180° | C_hpf/lpf_pf, L_hpf/lpf_nh |
| Vector_Modulator | 4 | −22.5°/state | Z0_line, L_quarter_mm, G_I/Q_scale |
| All_Pass | 1 | −90° | L_apA/B_nh, C_brA/B_pf, C_cA/B_pf |

## Citation

If you use this code, please cite the MLCAD 2026 paper (see `MLCAD_final.pdf` / `paper.tex`):

```bibtex
@inproceedings{fall2026gdiffps,
  title     = {{G-DiffPS}: Physics-Informed Graph Diffusion Policy
               for Amortized Multi-Topology RF Phase Shifter Synthesis},
  author    = {Fall, Shadi and Vungarala, Deepak and Angizi, Shaahin},
  booktitle = {Proceedings of the 2026 ACM/IEEE International Symposium
               on Machine Learning for CAD (MLCAD)},
  year      = {2026},
  note      = {Equal contribution by Fall and Vungarala}
}
```

## License

MIT — see [`LICENSE`](LICENSE).

For implementation detail beyond this README, see [`FRAMEWORK.md`](FRAMEWORK.md).
