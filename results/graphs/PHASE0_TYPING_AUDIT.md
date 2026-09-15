# Phase 0 typing audit + Phase 1 diagnostic results

Filled from `env/netlist_graph.py`, `models/circuit_encoder.py`, and
`models/gnn_encoder.py` on 2026-09-15, then measured with
`tools/graph_diagnostics.py`. Phase 2 (`--encoder circuit-typed`) is the
patch this table said we needed.

## Phase 0 table

| Question | Untyped `circuit` (before) | `circuit-typed` (after) |
|---|---|---|
| Does `build_circuit_graph` put G/D/S/B identity on incidence edges? | **No.** Edges carry `pin_index/3`, `is_control_pin`, KVL `sign`, and three distances. Pin index is an ordinal, not a role. SKY130 MOSFETs keep gate/body in `Dev.control` and never become incidence edges. | **Yes, when `typed=True`.** MOS expands to D/G/S/B; VCVS uses `out_p/out_n/ctrl_p/ctrl_n`; TLine uses `tline_a/tline_b`; symmetric passives share one `passive` relation. Role id is `edge_type`. |
| Does it flag net role (signal / supply / bias / ground)? | **Partial.** `net.x` is `[is_ground, is_port_in, is_port_out, is_internal, log_degree]`. No supply, bias, or RF-path bit. | **Yes.** 7-way one-hot: ground, port_in, port_out, signal, supply, bias, rf_path. Gate nets are `bias`; shortest in→out path is `rf_path`. |
| Does `IncidenceConv` consume any of that if it's there? | **Weakly.** One shared MLP concatenates the 6-dim `edge_attr`. KVL multiplies by `sign`. No per-relation weights. | **Yes.** `TypedIncidenceConv` is R-GCN-style: `W_r x_j` plus a linear on the residual edge features. |

### Additional gaps the table didn't name (still true of untyped `circuit`)

- `Dev.param_role` exists on the dataclass and is unused in `device.x`.
- Component-GIN (`models/gnn_encoder.py`) has no nets, no terminals, no state, no sizing. VM VCVS nodes are typed as `Res`.
- Physics state as a live rollout input remains circular (needs a sim to build the graph that chooses the next sim). Not wired. Hypergraph rewrite (DE-HNN) remains deprioritized: these graphs are small and the bipartite incidence graph is already the star expansion of the netlist hypergraph.

Phase 2 was a **half-day to two-day patch**, not a rebuild: the Levi graph, KCL/KVL rounds, and state-contrast pooling stay. What changed is the relation type on those same incidence edges.

## Phase 1 measurements (untrained encoders, seed 0, 28 GHz)

### G1 — SPICE → graph → SPICE

Exact device / terminal / parameter preservation on all six templates. Pass.

### G2 — permutation invariance

Node-order permutation of nets and devices. Cosine ≥ 0.999999 on every topology for both encoders, with and without RWSE. Pass. (RWSE rather than Laplacian PE specifically so eigenvector sign flips cannot fail this test.)

### G3 — terminal-swap sensitivity (the load-bearing test)

Same incidence set; only pin ordinal / relation type moves. Relative L2 of the state embedding:

| Swap | Expect | `circuit` | `circuit-typed` + RWSE | `circuit-typed` no PE |
|---|---|---|---|---|
| VCVS `out+` ↔ `ctrl+` (VM `E_I`) | move | **0.032** (cos 0.9995) | **0.123** (cos 0.993) | 0.101 |
| Capacitor pin swap (Loaded_Line) | stay | **0.131** (cos 0.992) | **0.052** (cos 0.999) | 0.063 |
| MOS G ↔ D (SKY130 Loaded_Line) | move | *not representable* (gate is not an edge) | **0.102** | 0.100 |
| polar / passive ratio | > 1 | **0.24** | **2.34** | 1.60 |

The untyped encoder is **more sensitive to an electrically-null capacitor pin-swap than to swapping a VCVS output with its control pin**. That is the Phase 0 gap, measured. Typed relations reverse the order and put MOS G/D on the graph at all. Untrained distances are still modest (cosine ~0.99); this is a representation test, not a trained-probe number.

### G7 — perturbation ranking (Vector_Modulator)

| Perturbation | Severity | `circuit` rel_L2 | `circuit-typed`+PE |
|---|---|---|---|
| small param (`G_I_scale` × 1.05) | 1 | 0.001 | 0.0004 |
| wrong param group (gain ← 50) | 2 | 0.068 | 0.031 |
| wrong connection | 3 | 0.150 | 0.106 |
| wrong terminal (`out+`↔`ctrl+` in the netlist) | 4 | 0.032 | 0.054 |
| broken topology (drop `T_quad`) | 5 | 0.159 | 0.156 |
| Spearman (severity vs distance) | | 0.70 | **0.90** |

Not strictly monotonic: wrong-terminal still sits below wrong-connection, because a VCVS pin swap is a local relation change and deleting the quadrature line is a global one. Typed+PE ranks severity better than the untyped encoder; it does not invent a perfect metric.

### Phase 3 ablation

Same typed architecture, RWSE on vs off. G2 still holds either way. G3 polar/passive ratio 2.34 with PE vs 1.60 without. G7 Spearman 0.90 vs 0.70. Cheap, keep PE on for `--encoder circuit-typed`.

## What shipped

- `build_circuit_graph(..., typed=False)` is the original graph; existing `circuit` checkpoints still load.
- `--encoder circuit-typed` → `CircuitTypedEncoder`: typed terminals, net roles, R-GCN incidence conv, RWSE, **and Level-3 RF motif nodes**.
- `--sim mna` / `--skip-prior` on `train_diffusion.py` and `tools/loocv_eval.py` for matched fast ablations.
- `python tools/graph_diagnostics.py` regenerates Phase 1.
- `python tools/supervised_loocv.py` is Phase 5.
- `python tools/run_matrix.py --track typed` is Phase 6.

## Phase 4 — RF motif layer

Hand-defined textbook blocks in `env/rf_motifs.py`, attached as `motif` nodes when `typed=True`:

| Topology | Motifs |
|---|---|
| Loaded_Line | tline, shunt_load × 2 |
| Switched_Line | spdt × 2, tline × 2 |
| Reflection_Type | hybrid, reflective_load × 2 |
| Switched_Filter | spdt × 2, hpf, lpf |
| Vector_Modulator | iq_split, iq_combine |
| All_Pass | spdt × 2, allpass × 2 |

One residual device→motif→device round after KCL/KVL. Action-head device rows are unchanged. No G5/G6 probe-accuracy claim.

## Phase 5 — supervised LOOCV (graph+sizing → MNA reward)

80 MNA-labelled sizings/topology, 60 train / 30 test, 20 epochs, seeds 42/1337/2026. Per-holdout test R² (mean of 3 seeds):

| Holdout | GIN | circuit | circuit-typed |
|---|---|---|---|
| Loaded_Line | −0.45 | −0.46 | **−0.30** |
| Switched_Line | −0.05 | −0.48 | **−0.10** |
| Reflection_Type | −1.25 | **+0.02** | −0.01 |
| Switched_Filter | −1.74 | −2.28 | **−0.32** |
| Vector_Modulator | −0.67 | −0.94 | **−0.55** |
| All_Pass | −0.60 | −1.10 | **−0.53** |
| **macro** | **−0.79** | **−0.87** | **−0.30** |

Train R² is modestly positive (~0.2–0.3); held-out R² is negative for almost every fold. The representation **does not yet transfer supervised**. circuit-typed is the least-bad encoder (and the only one that does not blow up on Switched_Filter), but the Phase-5 gate is not passed. All-Pass is a hard fold for every encoder, not a typed-graph-only bug.

## Phase 6 — matched RL, MNA, 400 steps, seed 42, 40 eval specs

Same policy/critic, device actions, electrical bounds, prior ON. Strict compliance is 0 except Loaded_Line (5% all three encoders). Held-out **mean** physical reward:

| Holdout | GIN | circuit | circuit-typed |
|---|---|---|---|
| Loaded_Line | −0.38 | −0.52 | −0.63 |
| Switched_Line | +0.21 | **+0.25** | +0.14 |
| Reflection_Type | −4.49 | −3.96 | −4.10 |
| Switched_Filter | −0.28 | −0.79 | −0.28 |
| Vector_Modulator | −3.57 | −2.29 | −2.16 |
| All_Pass | −0.86 | **−0.04** | −0.58 |
| **macro** | **−1.56** | **−1.22** | **−1.27** |

400 steps is a matched smoke, not the 2k/10k/20k paper matrix. It does **not** overturn Phase 5: no encoder produces held-out compliance except the easy Loaded_Line fold, and All-Pass remains broken. `--skip-prior` exists for the ABCD on/off cross; not run here. Scale-up command:

`python tools/run_matrix.py --track typed --steps 2000 --seeds 42,1337,2026 --n-eval 200 --out results/matrix_typed_mna_2k`

## Still not a paper matrix

- Phase 6 at 2k/10k/20k × 3 seeds, and prior on/off, is `tools/run_matrix.py --track typed --steps 2000 --seeds 42,1337,2026`.
- Physics state as a policy input and DE-HNN stay deprioritized.

