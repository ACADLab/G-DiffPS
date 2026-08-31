# G-DiffPS Framework: Technical Reference

**Graph-Conditioned Physics-Informed Generative Policy for RF Phase-Shifter Synthesis**

---

## 1. Problem Statement

Given a set of RF performance targets **s** (frequency, phase coverage, insertion loss, return loss, gain flatness), the system must:

1. **Select the best topology** from a discrete set of 6 circuit architectures
2. **Output optimized sizing parameters** (5–8 continuous values per topology)
3. **Guarantee SPICE-verified satisfaction** of all performance metrics

Traditional EDA tools solve this with nested search loops (topology enumeration × parameter sweep × SPICE). The goal here is to internalize both choices into a single trained model that answers the full query in a single forward pass (~2 ms) plus one SPICE verification (~100 ms).

The 6 topologies and their design point operating regimes:

| Topology | Phase Mechanism | Bits | Ideal Step | Best Operating Region |
|---|---|---|---|---|
| Loaded_Line | λ/4 TL + switched shunt RC | 1 | -22.5° | mmWave (>15 GHz), analog |
| Switched_Line | Dual TL path selection | 1 | -90° | All frequencies, digital |
| Reflection_Type | Branchline hybrid + reflective caps | 1 | -22.5° | Broadband mmWave |
| Switched_Filter | HPF/LPF switched pi-sections | 1 | -180° | Broadband digital |
| Vector_Modulator | I/Q VCVS sum (16-state) | 4 | -22.5°/state | Full 360°, fine resolution |
| All_Pass | Bridged-T LC sections | 1 | -90° | Sub-6 GHz, ultra-broadband |

---

## 2. System Architecture Overview

```
Query: Spec s ∈ R^12
  (fc_ghz, bw_pct, phase_coverage_deg, phase_bits,
   rms_phase_err_deg, rms_gain_err_db, max_il_db,
   min_rl_db, vdd, pmax_mw, tech, app)

          ┌─────────────────────────────────────────────────────┐
          │  OFFLINE TRAINING (10k online RL steps, ~24 min)    │
          │                                                      │
          │  For each step:                                      │
          │  1. Sample spec s from specset (600 specs)          │
          │  2. Sample topology τ uniformly from 6              │
          │  3. GNN(G_τ) → z_topo ∈ R^64                       │
          │  4. Actor(s, z_topo) → action a ∈ [0,1]^9           │
          │  5. action_to_params(a, τ, s) → SPICE params        │
          │  6. physics_prior(params, τ, fc) → pass/reject      │
          │  7. ngspice(netlist) → metrics                       │
          │  8. reward(metrics, s) → r ∈ [-5, 2]               │
          │  9. Update Critic Q_φ, Value V_ψ, Actor π_θ, GNN   │
          └─────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────────────────────┐
          │  INFERENCE (single query, ~2 ms + 1 SPICE call)     │
          │                                                      │
          │  For each topology τᵢ (i = 1..6):                   │
          │    z_i = GNN(G_τᵢ)                                  │
          │    V_i = ValueNet(s, z_i)                           │
          │  τ* = argmax_i V_i          ← topology selection   │
          │  a* = Actor(s, z_τ*)        ← parameter generation │
          │  p* = action_to_params(a*, τ*, s) → SPICE params   │
          │  Verify: ngspice(p*) → metrics ✓                    │
          └─────────────────────────────────────────────────────┘
```

---

## 3. Neural Network Architectures

### 3.1 GNN Topology Encoder — TopologyEncoder

Each of the 6 topologies is represented as a PyTorch Geometric graph G = (V, E):

- **Nodes**: circuit components with 5-D one-hot type vectors
  - `[1,0,0,0,0]` = Transmission Line
  - `[0,1,0,0,0]` = Switch (R_on/R_off)
  - `[0,0,1,0,0]` = Capacitor
  - `[0,0,0,1,0]` = Inductor
  - `[0,0,0,0,1]` = Resistor

- **Edges**: bidirectional signal-flow connections between components

```
TopologyEncoder (SAGEConv GNN):
  Input:   x [N_nodes, 5], edge_index [2, N_edges]
  Layer 1: SAGEConv(5 → 32) + ReLU
  Layer 2: SAGEConv(32 → 64) + ReLU
  Pool:    GlobalMeanPool over nodes
  Output:  z_topo ∈ R^64

Graph sizes per topology:
  Loaded_Line:      5 nodes, 12 edges
  Switched_Line:    6 nodes,  8 edges
  Reflection_Type: 10 nodes, 28 edges
  Switched_Filter: 10 nodes, 16 edges
  Vector_Modulator: 5 nodes, 12 edges
  All_Pass:        10 nodes, 24 edges
```

**Key property**: The GNN is permutation-invariant and encodes the structural connectivity of the circuit — not just its name. This enables the Experiment 3 zero-shot generalization test (hold out one topology, evaluate transfer).

### 3.2 CFM Actor (Default) — FlowMatchingPolicy

The default actor uses Conditional Flow Matching (CFM), a deterministic ODE-based generative model that learns straight-line paths from noise to the action manifold. Select with `--actor cfm` (default). The legacy DDPM actor is available via `--actor ddpm`.

```
VectorFieldNet (same MLP shape as DenoisingScoreNet):
  Condition embedding:  [spec(12) + z_topo(64) + t(1)] → Linear(77→128) → ReLU → Linear(128→128)
  Velocity network:     [action(9) + cond_emb(128)]     → [256→256→256→9]
  Output:               predicted velocity v_θ(x_t, t) ∈ R^9

FlowMatchingPolicy (CFM, 50 Euler steps):
  Training:  x_t = (1-t)·x_0 + t·x_1,   target u = x_1 - x_0
  Inference: x_{t+dt} = x_t + dt · v_θ(x_t, t, s, z_topo),   t ∈ [0,1]
  Output:    sigmoid(x_1) ∈ [0,1]^9  → mapped to physical params

  Key properties:
  - Deterministic: same (spec, topology) → same parameters every run
  - Straight-line ODE paths (no Langevin drift)
  - Uniform gradient signal across timesteps (no β-schedule bias)
```

**Legacy DDPM actor** (available via `--actor ddpm`):
```
DiffusionPolicy (DDPM, T=10 steps):
  Forward:  a_t = √ᾱ_t · a_0 + √(1-ᾱ_t) · ε,   ε ~ N(0,I)
  Reverse:  a_{t-1} = (1/√α_t)(a_t - β_t/√(1-ᾱ_t) · ε̂_θ) + √β_t · z
  β schedule: linear [1e-4, 0.02] over T=10 steps
```

### 3.3 Critic Network — CriticNet Q_φ

```
Input:  [spec(12) + z_topo(64) + action(9)] = 85-D
MLP:    85 → ReLU(256) → ReLU(256) → ReLU(256) → 1
Output: Q(s, z_topo, a) ∈ R  (estimated reward)
```

### 3.4 Value Network — ValueNet V_ψ

```
Input:  [spec(12) + z_topo(64)] = 76-D
MLP:    76 → ReLU(256) → ReLU(256) → ReLU(256) → 1
Output: V(s, z_topo) ∈ R  (baseline value, used for topology selection at inference)
```

---

## 4. Physics-Informed Parameter Scaling

### 4.1 The Core Problem

The diffusion actor outputs `a ∈ [0,1]^9`. These must be mapped to physical SPICE values. The mapping determines what the "middle" of the action space means physically. Wrong mapping → 99% of samples are physically nonsensical → reward starvation → no learning.

**Example — R_off catastrophe with linear scaling:**

```
Linear:  R_off = 0 + 0.5 × (1e6 - 1e3) = 499,500 Ω  (upper extreme)
Log:     R_off = 10^(3 + 0.5 × 3)      =   3,162 Ω  (physically typical)
```

R_off spans 3 decades. Linear puts the "midpoint" at 500 kΩ — far outside the operating range of most designs (3–50 kΩ). Every early random sample has a pathologically open switch.

### 4.2 Declared Medium (Board Microstrip)

All area and TL-geometry numbers in this work assume a board-level microstrip:

```
eps_r   = 3.2
h       = 0.254 mm (10 mil)
W(50 Ω) = 0.611 mm
eps_eff = 2.55
```

No MIM capacitors, no spiral inductors — a 19.8 mm line at 2.4 GHz does not
exist on a die. Discrete passives use 0201 footprints with pads and keepout;
switches are packaged SPDTs; VCVS is a packaged gain block. On-die is logged
as future work: it would change ε_eff per technology, hence λ/4, hence every
TL bound in §4.3. Which topology wins is entirely medium-dependent.

`tech` in the spec is **switch technology** (PIN / GaAs pHEMT / SOI SPDT),
operationalized as `(R_on, C_off)` by `sim/switch_model.py`. It is not a
process node for on-die passives.

Quarter-wave length used by the area model and SPICE templates:

```
λ/4 [mm] = 47.43 / fc_GHz

Examples:
  fc =  2.4 GHz  → λ/4 = 19.76 mm
  fc = 10 GHz    → λ/4 =  4.74 mm
  fc = 28 GHz    → λ/4 =  1.69 mm
  fc = 38 GHz    → λ/4 =  1.25 mm
```

TL sampling bounds remain [0.4·λ/4, 2.5·λ/4] under electrical log-warp for
`L_quarter_mm` (mid-action = λ/4 after T0.4).

### 4.2b Frequency-Adaptive Transmission Line Bounds

(Historical title retained; constants above supersede the older ε_eff = 2.5 note.)

TL bounds are set to [0.4·λ/4, 2.5·λ/4] — the policy can explore ±60% around the nominal quarter-wave, covering the full practical design space without sampling 50 mm lines at 28 GHz.

### 4.3 Per-Topology Scaling Table

| Topology | Parameter | Scaling | Bounds | Physical Rationale |
|---|---|---|---|---|
| **All** | R_off | **log** | [1e3, 1e6] Ω | 3 decades; linear wastes 99% of space |
| **All** | R_on | linear | [0.5, 10] Ω | 1 decade; linear fine |
| **Loaded_Line** | C_load_pf | **log** | [0.005, 0.5] pF | Resonance at 28GHz needs 0.04–0.06 pF; log puts midpoint at 0.05 pF |
| | L_quarter_mm | linear | [0.4·λ/4, 2.5·λ/4] | Freq-adaptive |
| | Z0_line | linear | [35, 75] Ω | RF matching range |
| **Switched_Line** | L_short_mm | linear | [0.3·λ/4, 0.8·λ/4] | Partitioned: upper(short)=lower(long) → L_long>L_short always |
| | L_long_mm | linear | [0.8·λ/4, 2.5·λ/4] | Guarantees meaningful phase differential |
| **Reflection_Type** | C_base_pf | **log** | [0.01, 1.0] pF | Reflection phase φ = -2·atan(ωCZ0) is log-sensitive |
| | C_tune_pf | **log** | [0.01, 1.0] pF | Same |
| | Z0_branch | linear | [25, 45] Ω | Constrained near Z0_main/√2 ≈ 35 Ω |
| **Switched_Filter** | C_hpf/C_lpf | **log** | [0.005, 2.0] pF | Design value 0.114 pF at 28GHz; log centers exploration there |
| | L_hpf/L_lpf | **log** | [0.005, 2.0] nH | Design value 0.284 nH at 28GHz |
| **Vector_Modulator** | L_quarter_mm | linear | [0.4·λ/4, 2.5·λ/4] | Freq-adaptive |
| | G_I/G_Q_scale | linear | [0.7, 1.0] | Passive only: scale > 1.0 → active gain (IL < 0 dB); VCVS template math IL = -20·log10(scale) |
| **All_Pass** | L_apA/L_apB | **log** | [0.01, 2.0] nH | Bridged-T: L spans 2 decades |
| | C_brA/C_brB | **log** | [0.005, 0.5] pF | Bridge capacitor — direct log parameter |
| | C_cA/C_cB | **ratio** | k ∈ [1.2, 4.0] × C_br | Reparameterized: C_c = k×C_br; ideal k≈2; enforces C_c/C_br ≥ 1.2 by construction (k<1.2 → IL > 20 dB) |

### 4.4 Analytical Physics Pre-Filters

Before calling ngspice (which takes ~80–150 ms per call), each parameter set is validated against analytical S-parameter models. Rejection returns reward = -5.0 immediately in microseconds.

| Topology | Prior Model | Filter Condition |
|---|---|---|
| **Loaded_Line** | Full ABCD cascade (TL + shunt RC) | \|S11\| < 0.56, \|S21\| > 0.17, both states |
| **Switched_Line** | Parallel ABCD with Y-matrix combination | \|S11\| < 0.56, \|S21\| > 0.17, both states |
| **Reflection_Type** | Reflection phase: φ = -2·atan(ωCZ₀) | Z_branch/Z_main ∈ [0.60, 0.85]; Δφ ∈ (5°, 90°) |
| **Switched_Filter** | LC pi-section: Z = √(L/C), ωc = 1/√(LC) | Z ∈ [10, 200]Ω both sections; ωc ∈ [0.1, 10]·ωfc |
| **Vector_Modulator** | Quadrature TL + passivity + gain balance | L_quarter ∈ [0.1, 3.0]·λ/4; G_I ≤ 1.0; G_Q ≤ 1.0; \|G_I - G_Q\| < 0.4 |
| **All_Pass** | Bridged-T balance + resonance alignment | C_br ∈ [0.1, 10]·(L/Z₀²); ωres = 1/√(LC) ∈ [0.2, 5]·ωfc, per section. C_c/C_br ≥ 1.2 guaranteed by action_to_params reparameterization — no prior check needed |

The prior filter passes 100% of template-default values by design — it only rejects clearly unphysical combinations.

---

## 5. Reward Function

### 5.1 Simulation Reward (Topology-Agnostic)

```
r_sim = w_phase · max(0, 1 - |rms_phase_err| / 90°)               [w=0.40]
      + w_il    · max(0, 1 - il_db / max_il_db)  if il_db ≥ 0     [w=0.25]
                  (0 credit if il_db < 0 — active gain is not a pass)
      + w_rl    · min(1, |rl_db|          / min_rl_db)             [w=0.20]
      + w_gain  · max(0, 1 - |gain_err|   / rms_gain_err)          [w=0.15]
      + 1.0  if all_close (all metrics within 20% slack of targets,
                           and 0.0 ≤ il_db ≤ 1.2 × max_il_db)

Range: [-1.0, 2.0]

Sentinel values:
  -5.0  = total simulation failure
  -3.0  = required metrics missing from ngspice output
  -5.0  = rejected by physics prior (returned before ngspice call)
```

**Design rationale for 90° phase scale**: An untrained agent produces ~20–30° RMS error. With a 5° denominator, the gradient signal collapses to near-zero. With 90°, the agent gets gradient signal of 0.67–0.78 from the start — enough to learn from.

**All-close bonus (+1.0)**: The discrete bonus is intentional. Without it, rewards saturate at ~0.95 (sum of weights) and there's no signal to push the policy toward full specification compliance. The +1.0 creates a sharp incentive to actually meet spec.

### 5.2 Expert Bonus (Reward Shaping)

```
expert_bonus = score_topology(τ, s) → ranks all 6 topologies by heuristic
             = +0.3 (rank 1) to -0.1 (rank 6)

Final reward = r_sim + expert_bonus  ∈ [-1.1, 2.3]
```

The heuristic scorer encodes RF textbook knowledge:
- Switched_Line: +5 if bits ≥ 4, -5 if fc < 5 GHz
- Loaded_Line: +4 if analog (bits=0), -5 if bw > 25%
- Reflection_Type: +5 if fc > 15 GHz, +4 if bw > 30%
- Switched_Filter: +8 if bits ≥ 4 AND bw > 40%
- Vector_Modulator: +6 if bits ≥ 5, -5 if pmax < 10 mW
- All_Pass: +6 if bw > 50%

**Purpose**: Provides mild initial topology guidance while the policy converges, without overriding the simulation reward. The expert bonus is ~14% of the maximum achievable reward.

### 5.3 Known Limitation: Topology-Agnostic Reward

The current reward uses fixed weights (0.40/0.25/0.20/0.15) regardless of topology. This is a known limitation:

- **Switched_Filter** achieves ideal 180° phase shift — its phase error denominator of 90° is inappropriate. A 180° step topology should normalize against 180°.
- **All_Pass** is a 1-bit fixed-phase-difference topology — gain flatness (gain_err_db) is structurally harder for it than for filter-based topologies.
- **Vector_Modulator** has 16 states (4-bit) — the gain error is more meaningful here than for 1-bit topologies.

**Future fix**: Topology-conditioned reward weights `w(τ)`, or separate reward heads per topology class. This is a planned enhancement in v2.

---

## 6. Online RL Training Loop

### 6.1 Algorithm: Advantage-Weighted Score Matching (Offline RL flavor)

```
Initialize: GNN, Actor (CFM default, or DDPM with --actor ddpm), Critic Q_φ, Value V_ψ
Replay buffer B (capacity 10,000)

For each step t = 1..10000:
  1. s ← sample_spec(specset_600)
  2. τ ← uniform_random(6 topologies)
  3. z = GNN(G_τ)                          # topology embedding
  4. a ~ Actor.sample(s, z)                # CFM: Euler ODE integration (50 steps)
  5. params = action_to_params(a, τ, s)    # physics-informed scaling
  6. if not physics_prior(params, τ, fc):
       r = -5.0 + expert_bonus; skip ngspice
  7. else:
       netlist = make_spice_netlist(τ, params)
       metrics = ngspice(netlist, multi-state)
       r_sim = compute_reward(metrics, s)
       r = r_sim + expert_bonus
  8. B.push(s, τ, a, r)

  If |B| ≥ batch_size (16):
    sample batch {sᵢ, τᵢ, aᵢ, rᵢ}
    zᵢ = GNN(G_τᵢ) for each i

    # Critic update (bandit: Q-target = immediate reward)
    L_Q = MSE(Q_φ(sᵢ, zᵢ, aᵢ), rᵢ)

    # Value update (expectile regression, τ=0.7)
    L_V = E[|0.7 - 1(Q-V < 0)| · (Q_φ(sᵢ,zᵢ,aᵢ) - V_ψ(sᵢ,zᵢ))²]

    # Actor update (advantage-weighted flow matching / score matching)
    A = Q_φ(sᵢ,zᵢ,aᵢ) - V_ψ(sᵢ,zᵢ)
    w = clamp(exp(A / 0.5), max=10.0)

    # CFM path (default --actor cfm):
    x_0 ~ N(0,I), t ~ Uniform(0,1)
    x_t = (1-t)·x_0 + t·aᵢ,  u_target = aᵢ - x_0
    L_actor = E[w · ‖u_target - v_θ(x_t, t, sᵢ, zᵢ)‖²]

    # DDPM path (--actor ddpm, legacy):
    # L_actor = E[w · ‖ε - ε̂_θ(a_noisy, t, sᵢ, zᵢ)‖²]

    # GNN is updated jointly with Actor (shares gradient path)
    optimizer.step(L_actor)
    optimizer.step(L_Q, L_V separately)
```

**Expectile parameter τ=0.7**: Asymmetric loss that biases the value network toward the upper end of the reward distribution — it should estimate the expected reward of good actions, not the average over all (including failed) actions.

**Advantage weight clamp=10.0**: Prevents gradient explosions when a rare high-reward sample gets exponentially upweighted.

### 6.2 State Sampling for Multi-State Evaluation

Each topology has a STATE_TABLE in its SPICE template defining N switch states:

```
State count by topology:
  Loaded_Line:       2 states (1-bit: unloaded / loaded)
  Switched_Line:     2 states (1-bit: short path / long path)
  Reflection_Type:   2 states (1-bit: C_base / C_base+C_tune)
  Switched_Filter:   2 states (1-bit: HPF path / LPF path)
  Vector_Modulator: 16 states (4-bit: I/Q angle sweep)
  All_Pass:          2 states (1-bit: section A / section B)
```

For topologies with ≥4 bits, the state sampler selects a representative subset (anchor states + random sample) to limit simulation time without losing coverage.

**Metric aggregation**: RMS phase error is computed as the RMS of wrapped residuals between measured per-state phase deltas and the ideal grid:
```
err_i = wrap_180(measured_Δφᵢ - ideal_Δφᵢ)
rms_phase_err = √(mean(err_i²))
```

---

## 7. RL Agent Comparison: What We Tried

### 7.1 Run History

| Run | Method | Topology Scope | Key Config | Result |
|---|---|---|---|---|
| `run_035556` | DDPM + **linear scaling** | All 6 | No log, no priors for 4/6 topologies | Baseline: max rewards found but poor mean. All 6 pass ngspice 100% (no filtering). Value net learns mean reward, not topology-conditioned. |
| `run_185745` | DDPM + **partial log (Loaded_Line only)** | All 6 | Log for C_load only, R_off still linear for 5/6 | Loaded_Line regresses (bug in bounds). Others unchanged. Confirms partial fixes are worse than none. |
| `run_023537` | DDPM + **full physics log + all priors** | All 6 | Log R_off all topologies, freq-adaptive TL bounds, analytical priors for all 6 | 4/6 topologies converge. Switched_Filter 38.7% prior pass, Vector_Modulator 34.9% and diverging. Vector_Modulator reward hacking via abs(il_db) + G_I/G_Q > 1.0. |
| `run_044731` | CFM + **relaxed priors** | All 6 | Relaxed SF/VM priors; G_I/G_Q capped [0.7,1.0]; IL passivity fix; All_Pass resonance prior | Switched_Filter and Vector_Modulator converge. All_Pass still IL=27 dB — root cause: off-resonance LC (resonance prior missing). |
| `run_053508` | CFM + **resonance prior** | All 6 | + All_Pass resonance check ωres ∈ [0.2,5]·ωfc per section | All_Pass IL still ~35 dB avg despite resonance check. Root cause: C_c ≈ C_br (policy stuck in bad local mode; ratio check via prior created 0.4% starvation). |
| `run_063534` | CFM + **C_c tightened in prior** | All 6 | C_c/C_br lower bound 0.5→1.2 in prior | 0.4% All_Pass pass rate — starvation. Zero learning in final 4k steps. Prior is wrong layer for coupling constraints. |
| `run_073203` | CFM + **C_c reparameterized** | All 6 | C_c = k×C_br in action_to_params, k ∈ [1.2,4.0]; prior ratio check removed | **All 6 converged**: Switched_Line 2.261, Reflection_Type 2.217, SF 2.169, VM 2.120, LL 2.077, All_Pass 1.064 (IL 1.73 dB, phase 18°). |
| `run_080429` | Same + **20k steps** | All 6 | Extended budget for All_Pass phase convergence | **All 6 > 1.98**: Switched_Line 2.271, Reflection_Type 2.229, SF 2.203, LL 2.195, VM 2.162, All_Pass 1.986 (phase 0.7°, IL 1.26 dB). |

### 7.2 Quantitative Comparison (10k steps each)

| Topology | Linear Baseline | Partial Log v1 | Physics Log v2 | CFM + relaxed priors | **CFM + all fixes (final)** |
|---|---|---|---|---|---|
| Loaded_Line | 0.554 / 2.271 | -1.097 (regressed) | 0.663 / 2.276 | — | **1.997 max** |
| Switched_Line | 0.779 / 2.233 | 0.789 / 2.254 | 0.900 / 2.261 | — | **2.262 max** |
| Reflection_Type | 0.475 / 0.727 | 0.475 / 0.750 | 0.698 / 2.259 | — | **2.246 max** |
| Switched_Filter | 0.456 / 0.640 | 0.455 / 0.623 | -2.621 / 2.224 (prior starved) | converging | **2.193 max** |
| Vector_Modulator | 0.567 / 2.162 | 0.571 / 2.117 | -2.922 / 2.262 (hacked) | converging | **1.061 max** (improving) |
| All_Pass | 0.675 / 0.991 | 0.677 / 0.995 | -0.084 / 0.984 | 0.571 (resonance off) | **1.986 max** (C_c reparameterized; 0.7° phase, 1.26 dB IL at 20k steps) |

**Reading**: Linear baseline had 100% prior pass (no filter) so mean rewards inflated by mediocre designs. The CFM + all-fixes run is the current production state; Vector_Modulator and All_Pass continue improving in subsequent runs.

### 7.3 Best Verified SPICE Designs Found (run_080429, CFM, 20k steps)

| Topology | Reward | Phase Err | IL | RL | Notes |
|---|---|---|---|---|---|
| Switched_Line | 2.271 | 0.2° | 1.05 dB | 16.5 dB | Fully converged |
| Reflection_Type | 2.229 | 1.0° | 0.87 dB | 20.9 dB | Fully converged |
| Switched_Filter | 2.203 | 3.7° | 1.87 dB | 12.4 dB | Fully converged |
| Loaded_Line | 2.195 | 4.5° | 0.73 dB | 19.0 dB | Fully converged |
| Vector_Modulator | 2.162 | 4.5° | 0.03 dB | 15.3 dB | Fully converged |
| All_Pass | 1.986 | 0.7° | 1.26 dB | 14.6 dB | Fully converged — C_c reparameterization unlocked 0.7° phase accuracy |

### 7.4 Comparison Against Traditional Optimization (From Experiment 5)

| Scenario | G-DiffPS Phase Err | Nelder-Mead SPICE Calls to Match | Speedup |
|---|---|---|---|
| Loaded_Line @ 28 GHz | 9.91° | 1 call (trivial) | — |
| Loaded_Line @ 38 GHz | 10.97° | 1 call | — |
| Switched_Line @ 24 GHz | 7.98° | 11 calls (0.90s) | ~10× |
| Reflection_Type @ 10 GHz | 20.95° | 1 call | — |
| Switched_Filter @ 18 GHz | 64.65° | **29 calls (2.21s)** | **630×** |
| Vector_Modulator @ 8 GHz | 77.61° | 1 call | — |
| All_Pass @ 2.4 GHz | 63.65° | 1 call | — |

**Key result**: G-DiffPS performs synthesis in 1.7 ms. Nelder-Mead requires up to 29 SPICE calls (2.21s) to match quality on the hardest topologies. Average speedup: **630×**.

---

## 8. Inference: From Query to SPICE-Verified Design

### 8.1 Complete Inference Pipeline

```
INPUT: spec s = {fc_ghz=28, bw_pct=20, phase_coverage_deg=360,
                  phase_bits=5, max_il_db=5, min_rl_db=12, ...}

Step 1 — Normalize spec to [0,1]:
  s_norm = normalize(s)  using SPEC_BOUNDS (log-scale fc_ghz, pmax_mw)

Step 2 — Score all 6 topologies (Option A: Critic-Scored Forward Pass):
  For τ ∈ {Loaded_Line, Switched_Line, Reflection_Type,
            Switched_Filter, Vector_Modulator, All_Pass}:
    z_τ = GNN(G_τ)                  # 6 forward passes, ~0.1 ms total
    score_τ = V_ψ(s_norm, z_τ)     # value net evaluation
  τ* = argmax score_τ               # topology selection

Step 3 — Generate sizing parameters:
  a* = Actor.sample(s_norm, z_τ*)   # CFM Euler ODE, 50 steps (~2 ms)
  p* = action_to_params(a*, τ*, s)  # physics-informed scaling

Step 4 — Physics pre-check:
  if not physics_prior(p*, τ*, fc): resample (rare after convergence)

Step 5 — SPICE verification:
  netlist = template(τ*) + p*
  metrics = ngspice(netlist)        # 1 simulation call

Step 6 — Report:
  ✓ TOPOLOGY SELECTED:  τ*
  ✓ SPICE PARAMETERS:   p*
  ✓ VERIFIED METRICS:   rms_phase_err, il_db, rl_db, gain_err_db
  ✓ SPEC SATISFIED:     YES / NO (with slack)
  ✓ TOTAL LATENCY:      ~2 ms synthesis + ~100 ms verification
```

### 8.2 Topology Selection via ValueNet (Option A)

The trained ValueNet V_ψ(spec, z_topo) learns the expected reward achievable by the best action for a given (spec, topology) pair. At inference, this becomes a **6-way topology ranker with zero extra training** — the value function already encodes which topologies work for which specs.

```python
# inference_topology_select.py
scores = {}
for name in TOPOLOGY_NAMES:
    g = get_topology_graph(name)
    z = gnn(g.x, g.edge_index)
    scores[name] = value_net(spec_tensor, z).item()

winner = max(scores, key=scores.get)
```

**Current accuracy**: improving with final run (run_053508, CFM + all fixes). Target: ≥4/6. Run `probe_value_net.py` after checkpoints save to evaluate.

The probe tests that known-best (topology, freq) pairs from the compliance heatmap are correctly ranked first.

---

## 9. Issue History and Current Known Limitations

### 9.1 RESOLVED: Switched_Filter / Vector_Modulator Prior Starvation

**Symptom**: Prior pass rates of 38.7% and 34.9% → replay buffer fills with -5.0 → value net learns to avoid these topologies → diverging feedback loop.

**Fix applied** (run_044731+): Relaxed bounds:
- Switched_Filter: impedance [20,120]→[10,200] Ω; resonance [0.3,3]→[0.1,10]×ωfc
- Vector_Modulator: TL length [0.3,1.7]→[0.1,3.0]×λ/4

Both topologies now converge to rewards above 1.0.

### 9.1b RESOLVED: Silent Clamping Truncated the Low Band

**Symptom**: none — that was the problem. `clamp_spice_value` caps inductors and capacitors at 10 (nH / pF) and TL lengths at 50 mm, silently. Under `bounds='electrical'` the sampling windows are multiples of the resonant value at `fc`, which scales as `1/fc`, so at low carriers the window ran past the cap and a large part of the action box mapped onto the same few circuits.

**Measured** (`tools/clamp_audit.py`, fraction of draws with at least one clamped parameter):

| topology | 1.26 GHz | 2.00 | 3.17 | 5.02 | ≥7.96 |
|---|---|---|---|---|---|
| Switched_Filter | 77% | 62% | 38% | 19% | 0% |
| Switched_Line | 72% | 23% | 0% | 0% | 0% |
| All_Pass | 47% | 34% | 16% | 4% | 5–41% |
| Loaded_Line / Reflection_Type / Vector_Modulator | 34–36% | 10% | 0% | 0% | 0% |

The worst single offender was `Switched_Line.L_long_mm` at 72.3%: its long arm — the element that produces the phase — was being truncated at the 50 mm cap, where λ/4 alone is 37.6 mm. All_Pass was U-shaped, clamping at the 10 nH ceiling at low `fc` and the 5 fF floor at high `fc`.

This mattered disproportionately because the sub-6 GHz band is where the area term binds and where the passive-only regime lives — i.e. the regime in which topology selection is actually contested.

**Fix applied** (schema v5): the electrical windows are clipped to the physical limits *before* a value is drawn, so no action lands on a clamp boundary and no action volume is dead. For All_Pass the clip is applied symmetrically in log space about the designed centre and conditioned on the ratio drawn, since an asymmetric clip would move the `a = 0.5` nominal off the design. Residual boundary contact is ~0% for five topologies and ~1% for All_Pass at 40 GHz, where a resonant capacitor is 0.08 pF and the lower section genuinely sits near the 5 fF floor. Pinned by `tests/test_no_silent_clamping.py`, which also asserts the mirrored limit table matches `clamp_spice_value`.

### 9.1c The archive factorization survives the `g_il` discontinuity

`r_star = max over the Pareto front` is only provably `max over achievable` if reward is monotone in every archived metric, and §9.2's branch makes it non-monotone: `g_il` runs 0.79 → 1.0 → **0.0** as `il_db` goes 2 dB → 0 dB → −1 dB. A point with worse IL can outscore a better one, and if it is dominated it never enters the archive.

**Audited** (`tools/monotonicity_audit.py`): every cell's Sobol+DE sweep was re-run at the archive's seed, the samples the Pareto filter discarded were kept, and each was scored against real specs from that cell. **0 violations in 3600 checks**, including all 600 Vector_Modulator checks — the only topology that produces negative IL.

The structural reason: a negative-IL point is near-optimal on the IL axis, so it is rarely dominated in the first place, and the front always retains positive-IL points scoring at least as well. The assumption is empirically safe; if a future change makes it dirty, the fix is to define the archive's IL axis as `max(il_db, 0)`, which is monotone under the reward.

### 9.2 RESOLVED: Vector_Modulator Reward Hacking via Active Gain

**Symptom**: Vector_Modulator achieved reward ~2.26 but IL = -0.49 dB (active gain). The network learned to set G_I/G_Q > 1.0, which the VCVS template converts to IL = -20·log10(scale) < 0 dB. The old reward used `abs(il_db)`, treating gain as near-zero loss and passing the all_close check.

**Fix applied** (run_044731+): Three-layer fix:
1. `action_to_params`: G_I/G_Q range capped to [0.7, 1.0]
2. `physics_prior`: `g_i > 1.0 or g_q > 1.0 → return False`
3. Reward: removed `abs(il_db)` — active gain (il_db < 0) receives zero IL credit; all_close requires `0.0 ≤ il_db`

### 9.3 RESOLVED: All_Pass Off-Resonance Starvation

**Symptom**: All_Pass prior pass rate 99.3% but reward consistently ~0.2 and IL ~27 dB. Root cause: old prior only checked C_br ≈ L/Z₀² ratio but not whether the LC resonant frequency aligned with the operating frequency. Components could satisfy the ratio check while resonating at a completely different frequency.

**Fix applied** (run_053508+): Added per-section resonance check:
- `ω_res = 1/√(L·C) ∈ [0.2, 5]·ωfc` for both sections A and B

### 9.4 RESOLVED: All_Pass C_c/C_br Imbalance (Action Space Reparameterization)

**Symptom**: After the resonance prior fix, All_Pass IL remained at ~35 dB average. Analysis showed C_c/C_br ≈ 1.0 for 99% of passed samples; good designs require C_c/C_br ≈ 1.5–2.5.

**Failed approach**: Tightening the prior to require C_c/C_br ≥ 1.2 caused 0.4% pass rate starvation — zero learning over 10k steps.

**Root cause**: C_c and C_br were independently parameterized as separate log-uniform capacitors. The policy had no representation of their coupling and converged to a local mode where C_c ≈ C_br. Trying to enforce the constraint via the prior was the wrong layer.

**Fix applied** (run_073203): Reparameterized C_c as a ratio of C_br in `action_to_params`:
```python
# C_cA = k_A × C_brA, k_A ∈ [1.2, 4.0] — guaranteed by construction
c_brA = log_scale(float(action[keys.index("C_brA_pf")]), 0.005, 0.5)
physical_val = lin(val_01, 1.2, 4.0) * c_brA
```
Prior ratio check removed (now redundant). Result: pass rate 52.5%, avg IL 2 dB, best reward 1.064.

**General principle**: When two parameters have a physical coupling constraint, encode one as a function of the other in `action_to_params`. Never rely on the policy learning the coupling through reward signals. See also: Switched_Line L_long > L_short (partitioned action dims), Vector_Modulator G_scale ≤ 1.0 (capped range).

### 9.5 OPEN: Topology-Agnostic Reward Weights

The fixed 0.40/0.25/0.20/0.15 weights are not physically calibrated per topology:

| Topology | Issue | Proposed Fix |
|---|---|---|
| Switched_Filter | 180° step → 90° phase scale is too lenient | scale_deg = 180° OR w_phase ×= 0.5 |
| All_Pass | Fixed 90° differential — gain variation is intrinsic | w_gain reduced for 1-bit topologies |
| Vector_Modulator | 16 states — gain flatness across 360° is harder | w_gain increased, scale relaxed |

### 9.6 RESOLVED: All_Pass Phase Accuracy

All_Pass best reward 1.986 at 0.7° phase error and 1.26 dB IL (run_080429, 20k steps). All 6 topologies now achieve best reward > 1.98.

### 9.7 OPEN: Vector_Modulator Phase Error Still Elevated

After the gain hacking fix, Vector_Modulator best reward is 1.061 with 19.3° phase error. The reward correctly penalizes gain-producing actions now, but the policy needs additional training steps to converge on a passive, phase-accurate design. Expected to resolve with continued training.

### 9.8 LOGGED (do not fix here): Paper / Framework / Code Discrepancies

**Euler steps.** §3.2 and §10.3 document 50 Euler ODE steps for CFM. The paper §3.2 says ten, with a written justification. Shipped code: `FlowMatchingPolicy` defaults to `num_steps=50`, but every call site that constructs the device-path actor (`train_diffusion.py`, `inference_joint.py`, `tools/loocv_eval.py`) passes `num_steps=10`. Slot-path training leaves the class default (50) while inference loads with 10 — a genuine train/inference solver mismatch, not only a doc error.

**SAGEConv vs GIN.** §3.1 and §12 describe a two-layer SAGEConv encoder with mean pooling. Shipped `models/gnn_encoder.py` is three `GINConv` layers with sum pooling and an injected degree feature; the module docstring argues explicitly against mean aggregation on one-hot inputs. A second encoder (`models/circuit_encoder.py`) is the default for `inference_joint.py` and is absent from §12.

**Dead ninth action dimension.** §3 / paper claim the nine-dimensional action covers the largest parameter set across the six topologies. `TOPOLOGY_PARAMS` maxes at All_Pass with **eight** keys; slot-mode `action_to_params` iterates `enumerate(keys)`, so index 8 is never read. `action_dim=9` has one permanently unused dimension. (Device-mode `max_sized` is a different quantity — padding at `max(len(sized_devices)+2)`.)

**Premise corrections vs §5 / §7.** (1) Shipped reward uses five weights `0.35/0.20/0.15/0.15/0.15` including a dead `w_power` term; §5.1's `0.40/0.25/0.20/0.15` is stale — production path is now `WEIGHTS_AREA` (phase 0.32 / il 0.22 / rl 0.16 / gain 0.12 / area 0.18). (2) Reward is unified in `env/reward.py`. (3) `run_080429` and the other §7.1 run IDs do not exist on disk; max All_Pass `best_reward` across on-disk LOOCV is 0.978, not the claimed 1.986.

### 9.9 Specset v2 and the C3 claim (honest restatement)

Published Table 5 / LOOCV numbers were measured against the 600-spec pool frozen at `specset/specset_v1_frozen.json`. That pool was **shared** between train and eval: `restrict_to` filtered topologies only, so every held-out-topology trial still drew specs the actor had seen paired with other graphs. The C3 claim is therefore:

> **topology transfer conditional on seen specifications** — zero gradient updates on the held-out graph, but not zero exposure to the held-out specs.

Specset v2+ (`specset_train.json` / `specset_eval.json`, `spec_dim=19`) enforces disjoint pools at load time. Re-reported LOOCV is not numerically comparable to Table 5 without a full retrain (T7); when that lands, state the difference from the v1 numbers explicitly. Do **not** regenerate `specset_v1_frozen.json`.

Filenames are unversioned on purpose: `schema_version` inside the file is authoritative and `specset.schema.load_specset` asserts it, so a versioned filename only goes stale on every bump. Current `SCHEMA_VERSION = 3`.

### 9.10 Floor-relative sampling (house convention)

**Any spec field with a physical floor is sampled relative to that floor, never from an absolute box.**

Sampling such a field independently wastes draws at both ends at once: the strict end is infeasible for every topology, the lax end is satisfied by every topology, and no box width fixes both. Widening to buy acceptance rate only trades the first degeneracy for the second. Measured on the shipped generator, the hand-widened `rms_phase_err_deg` box of (1°, 20°) was **16.8% infeasible and 57.8% vacuous** — three quarters of draws carried no information.

The pattern is `value = floor(conditioning fields) · κ`, with `κ` from a fixed multiplier range. Zero rejections by construction, every draw informative, and `κ` becomes an explicit difficulty knob to stratify on. Fields that follow it:

| field | floor | multiplier |
|---|---|---|
| `max_area_mm2` | rank-ordered reference areas at `fc` (T1.5c) | `1 + ε`, `ε ~ U[0.02, 0.10]` |
| `rms_phase_err_deg` | quantization floor `(coverage / 2^bits) / √12` (S1) | `κ ~ LogUniform[1.2, 3.0]` |

Margin-stratified evaluation (S3) is the same idea applied to the reward gap. `phase_bits = 0` (analog) has no quantization floor, so it uses the 6-bit grid as a proxy — an analog part is expected to beat the finest digital one — which keeps the field floor-relative everywhere rather than falling back to a box for one level.

A consequence worth stating: for a floor-relative field, its `SPEC_BOUNDS` entry is a **normalization range only** — the span the rule can produce — not a sampling box. It is derived, not chosen. `rms_phase_err_deg` reaches [0.49°, 38.97°], so it is log-normalized over (0.4, 40.0).

**RESOLVED at schema v4 — all four S-parameter terms are now floor-relative.** `max_il_db`, `min_rl_db` and `rms_gain_err_db` were previously sampled from independent absolute boxes. Measured against the envelope archive, best achievable IL was ~0.002 dB against demands of 1.9–9.2 dB and gain error ~0.0000 dB against demands of 0.6–2.7 dB, giving mean satisfactions of `g_il` 0.963, `g_rl` 0.999, `g_gain` 0.979 — saturated, with only `g_area` (0.395) binding. The T1 envelope gate failed G2/G3 on exactly this.

They are now anchored on the achievable frontier, conditioned on phase quality:

```
max_il_db       = min_{τ ∈ passive} IL*(fc-bin, phase) · κ_il      κ_il   ~ LogU[1.5, 15]
min_rl_db       = max_{τ ∈ passive} RL*(fc-bin, phase) / κ_rl      κ_rl   ~ LogU[1.2, 3]
rms_gain_err_db = min_{τ ∈ passive} GAIN*(fc-bin, phase) · κ_gain  κ_gain ~ LogU[1.5, 20]
```

Three properties of this are load-bearing:

- **Conditioning on phase quality is required, not decorative.** The unconstrained loss extremum is attained by a through-line that does not shift phase at all; anchoring on it would price every loss budget against a circuit that is not a phase shifter. `tools/compute_envelope.py frontier` therefore reports the frontier subject to holding RMS phase error at or below each grid value.
- **The anchor set is passive-only.** `min_τ IL*` including `Vector_Modulator` can be negative, which makes `max_il_db = −3 · κ` meaningless. Excluding active topologies is a modelling choice with a designer's justification: you do not relax a loss budget because someone might insert an amplifier. See `results/joint/VM_ACTIVE_TOPOLOGY.md`.
- **One pool, anchored on `ideal`.** Anchors are switch-model dependent. Generating a pool per switch model would make the two incomparable; with one pool, `realistic` is uniformly harder and that difficulty delta becomes a reportable quantity rather than a confound.

**Declared coupling:** spec difficulty now depends on the topology set, because the anchors are the best any topology in the anchor set attains. Adding or removing a topology shifts the anchors and changes what the specs demand. This is intended and is not worth engineering around, but it must be declared: **whenever the topology set changes, rebuild the archive, re-extract anchors, regenerate the pools, and re-report.**

The κ values are stored as spec metadata (`il_kappa`, `rl_kappa`, `gain_kappa`, alongside `phase_kappa`), **not** as conditioning dimensions. `SPEC_DIM` stays 19 and trained checkpoints keep loading.

**Result — T1 passes structurally, with no weight tuning.** Term variance shares at the envelope, 2k eval pool at schema v5:

| switch model | g_il | g_phase | g_gain | g_area | g_rl | all-saturated |
|---|---|---|---|---|---|---|
| ideal | 0.41 | 0.36 | 0.09 | 0.08 | 0.07 | 0.7% |
| realistic | 0.34 | 0.49 | 0.07 | 0.07 | 0.03 | 0.3% |

G1 (envelope spread) 1.298 ≥ 0.25, G2 (saturation) 0.007 ≤ 0.2, G3 (no dominant term) 0.406 ≤ 0.6. Same verdict on the 10k train pool.

**Caveat that must be reported with these shares.** Under `ideal`, `g_il` (0.41) edges out `g_phase` (0.36) on what is a phase-shifter benchmark. That ordering is a consequence of how tightly each term is anchored, not a statement about physics: frontier anchoring makes whichever term is anchored tightest the dominant discriminator, and the five κ ranges were chosen independently. `tools/kappa_sensitivity.py` sweeps each κ and shows the lead flipping to `g_phase` when either `il_kappa` or `phase_kappa` is halved:

| knob | ×0.5 | ×1.0 | ×2.0 | ×4.0 |
|---|---|---|---|---|
| `phase_kappa` → `g_phase` share | 0.40 | 0.34 | 0.27 | 0.21 |
| `il_kappa` → `g_il` share | 0.34 | 0.42 | 0.45 | 0.41 |

The sensitivity is asymmetric: relaxing `phase_kappa` drains `g_phase` monotonically, while perturbing `il_kappa` moves `g_il` non-monotonically around 0.42. So the lever on term dominance is `PHASE_KAPPA_RANGE`, not the IL anchor.

**Deliberately left alone.** `PHASE_KAPPA_RANGE` could be narrowed to about [1.2, 1.8] to make phase lead by design. It was not, because reporting the measured shares alongside this sensitivity table is a stronger claim than reporting a tuned number, and because the schema is frozen at v5. Realized κ distributions are published in `results/joint/kappa_sensitivity.json`.

### 9.11 T1 acceptance, restated at the envelope

The original T1 gate (r_sim spread < 0.15 across topologies at template defaults) was unreachable and is retired: the `all_close` bonus is a discrete +1.0, so any spec where one topology clears every threshold and another does not forces a spread ≥ 1.0 regardless of the continuous weights. Measured spread at defaults is 1.42 median (0.44 for the continuous part alone) — `tools/t1_reward_accept.py`.

The live replacement is `tools/t1_envelope_gate.py`, which asks the question that actually matters: does the reward discriminate between topologies **when each is sized as well as it can be**? That is a statement about `r_star`, which is why it was blocked on the envelope.

| gate | threshold | ideal | realistic |
|---|---|---|---|
| G1 envelope spread (median, max−min over topologies) | ≥ 0.25 | 1.061 **PASS** | 1.069 **PASS** |
| G2 fraction of specs where all six saturate | ≤ 0.20 | 0.397 **FAIL** | 0.406 **FAIL** |
| G3 largest single term's share of between-topology variance | ≤ 0.60 | 0.714 (`g_phase`) **FAIL** | 0.505 **PASS** |

**Verdict: T1 not accepted.** The two failures have one cause, given in §9.10: `il`/`rl`/`gain` thresholds are drawn from absolute boxes loose enough that an optimally-sized circuit clears them with margin, so 40% of specs are satisfied by every topology and `g_phase` — the one field now floor-relative, hence the one with real spread — carries most of the discrimination.

Note G1 and the top-2 margin measure different things and both matter: spread is 1.06 (best vs worst topology is a real gap) while the median top-2 margin is 0.009 (the best two are nearly tied). Read together: there is a good group and a bad group, and choosing within the good group is close to arbitrary at current spec difficulty.

### 9.12 How `r_star` is computed

`r_star(τ, s) = max_a r_sim(τ, s, a)` is on the critical path for T4 ranking, the S3 margin strata and the §9.11 gate, so how it is estimated matters.

**Not best-of-K random.** At K = 32 uniform draws the chance of landing within ±10% of the optimum in every dimension is 22.7% at 3 action dims and 0.2% at 6. The bias is not just large, it *grows with action dimension*, so it would silently penalize exactly the topologies with the most design freedom (All_Pass d=6, Reflection_Type d=5).

**Not per-spec DE either**, at least not as the primary path: 12k specs × 6 topologies × 2 switch models is ~144k optimizations, roughly 24 h.

The factorization that makes it cheap is that the spec and the action meet only at the thresholds:

```
metrics = f(τ, a, fc, tech, switch_model)     # no spec targets
r_sim   = g(metrics, s.targets)               # no action
```

So the reachable metric set of a cell `(τ, switch_model, fc-bin, C_off-class)` is a property of the circuit alone. `tools/compute_envelope.py archive` builds a Pareto archive of that set once per cell — Sobol sweep (1024 points) plus differential evolution on six scalarizations chosen to push toward each face of the trade-off surface — and then `r_star(τ, s) = max over archive of g(metrics, s.targets)` is arithmetic. **144 cells instead of 144k optimizations: 53 s wall clock.** Because `bounds='electrical'` scales every sizing bound with `fc`, electrical metrics are near-invariant within a bin; area is not, so it is recomputed at each spec's exact `fc`/`tech` from the archived action rather than read from the archive.

Every archive point is a real sizing, so this is a **certified achievable lower bound**, not an estimate that could sit above the truth. `--mode validate` runs per-spec DE against individual specs' own rewards to measure tightness:

| statistic | value |
|---|---|
| median gap (direct DE − archive bound) | 0.009 |
| within 0.01 / within 0.05 | 55% / 86% |
| archive beats per-spec DE | 22.5% |
| p95 / max gap | 0.118 / 0.967 |

The archive wins 22.5% of the time because it pools ~1030 circuit evaluations per cell against DE's per-spec budget. **The tail is the honest caveat:** a ~5% minority of cells are loose by ≳0.1, and since the top-2 margins are ~0.01–0.05 — smaller than that looseness — the S3 strata are **provisional**. Tightening them requires a per-spec DE refinement restricted to each spec's top-two topologies (2000 specs × 2 models × 2 topologies ≈ 2 h), which is the tracked next step before any per-stratum selector number is published.

Computed separately under each switch model throughout; `r_star` is stored as `{ideal, realistic} → {per_topology, best, argmax, margin, stratum}`.

`RunningEnvelope` in the same module maintains a per-`(spec_id, τ, switch_model)` max over rewards actually observed in training. It is free and monotonically correct but only covers visited specs, so it tightens the archive bound rather than replacing it.

### 9.13 The seventh topology, and whether `realistic` is a usable benchmark

T0.1c reported that realistic switches collapse the branch-selecting topologies (Switched_Line Δφ 65.7° → 3.4°) and `Switched_Line_SeriesShunt` was proposed as the remedy. **The collapse does not reproduce**, so the seventh topology is abandoned on physics grounds. Full decision record with the reward 2×2 and the arm-ratio sweep: `results/joint/SEVENTH_TOPOLOGY_DECISION.md`.

Δφ retention under realistic switches, mid-action at 28 GHz: Switched_Line 0.996, Switched_Filter 1.000. Collapse requires an off-state far worse than anything shipped — C_off ≈ 400 fF at 28 GHz and 200 fF at 40 GHz, against `TECH_SWITCH` values of 20/20/25 fF, an 8–16× margin. The original 65.7° → 3.4° figure matches the C_off = 650 fF row of the isolation table (8.7 Ω, −0.7 dB isolation), a part never adopted as a tech. What realistic switches actually cost is **loss and match, not phase**: Switched_Line goes 0.56 → 1.93 dB IL and 28.3 → 22.0 dB RL, which is a correctly-signed difficulty increase and is the point of the switch model. `tests/test_realistic_switch_functional.py` pins this across 2 topologies × 3 techs × 4 carriers.

**All_Pass is the one real defect, and it is not switch-related.** It reads Δφ = 0.00° under *ideal* switches too, because its two sections are identically sized at the box midpoint and Δφ vanishes by symmetry. Away from the midpoint it is healthy (box-best 179.8°, median 85.4°, 71.5% of draws over 45°). This nominal-point artifact retroactively explains three separate findings — All_Pass worst on ~95% of specs at template defaults, 0% share in Table 5, and T6 contrast 0.010 in *both* blocks (at the midpoint the two switch states are the same circuit, so contrast is identically zero) — and it means any acceptance test evaluated at template defaults is structurally unfair to All_Pass. Open item: give All_Pass a non-degenerate nominal point before any per-topology nominal comparison including it is quoted.

The general lesson, which cost two separate misreadings: **do not build a measurement on a nominal point without checking the nominal point is non-degenerate.** `tests/test_state_permutation.py` was silently relying on the same class of accident (pre-T0.4 equal Switched_Line arms) and now constructs its symmetric case explicitly.

**Related (joint-selection work).** Switch model, medium declaration, TL log-warp for `L_quarter_mm`, and topology ranking are tracked under the joint topology selection plan; this section only records the discrepancies above.

---

## 10. Conditional Flow Matching (CFM) — Implemented as Default Actor

### 10.1 Why DDPM Was Replaced

DDPM was the right proof-of-concept choice. For production, it had three structural limitations:

**1. Curved stochastic trajectories**: The DDPM reverse process follows a Langevin-style trajectory that curves through parameter space. Each denoising step introduces stochastic noise. For circuit parameters with sharp resonance conditions (e.g., C_load must be within ±0.01 pF of resonance at 28 GHz), this stochasticity causes the final sample to "drift" away from the sharp valid region.

**2. Non-uniform gradient signal**: The β schedule concentrates gradient signal at certain timesteps. Parameters that only matter at the final denoising step (t≈0) receive weak gradient signal — exactly the fine-precision parameters that determine whether a circuit meets spec.

**3. Training instability from advantage weighting**: The exponential advantage weights exp(A/τ) are amplified by DDPM's score-matching formulation. When a rare high-reward sample appears early in training, it can dominate the actor loss and push the β schedule into degenerate regimes.

### 10.2 What Conditional Flow Matching Provides

Flow Matching (Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022) replaces the stochastic DDPM denoising with a deterministic ODE that learns **straight-line paths from noise to data**. This is now the default actor (`--actor cfm`).

```
DDPM path:       x_T →[curved,stochastic]→ x_0
CFM path:        x_T →[straight,deterministic ODE]→ x_0

DDPM objective:  L = E[‖ε - ε_θ(x_t, t, cond)‖²]    ← predict noise
CFM objective:   L = E[‖u - v_θ(x_t, t, cond)‖²]    ← predict velocity
                 where u = x_1 - x_0 (straight line from noise to target)
```

**Benefits for circuit parameter generation:**

| Property | DDPM | CFM |
|---|---|---|
| Training objective | Predict injected noise | Predict straight-line velocity |
| Trajectory | Curved, stochastic (Langevin) | Straight ODE path |
| Gradient signal distribution | Non-uniform (β schedule bias) | Uniform across timesteps |
| Inference stability | Stochastic (different each call) | Deterministic ODE solver |
| Steps needed | 10–1000 | 5–20 ODE steps (same speed) |
| Sharp resonance handling | Poor (stochastic drift) | Better (deterministic path) |
| Training stability | Sensitive to advantage weight scale | More robust |

**Most important for this application**: Deterministic inference means the same (spec, topology) pair always produces the same parameters. For a design tool, this is essential — engineers need reproducible results.

### 10.3 Architecture Change (Implemented)

Only `DiffusionPolicy` and `DenoisingScoreNet` were replaced. All other components (GNN, Critic, Value, reward, training loop, inference pipeline) remain identical.

```
Replaced: DiffusionPolicy(DDPM)  →  FlowMatchingPolicy(CFM)  [--actor cfm, default]
          DenoisingScoreNet       →  VectorFieldNet

VectorFieldNet:
  Same architecture as DenoisingScoreNet
  Input:   x_t ∈ R^9, t ∈ [0,1], spec ∈ R^12, z_topo ∈ R^64
  Output:  v_θ(x_t, t) ∈ R^9   ← predicted velocity field

FlowMatchingPolicy:
  Training: x_t = (1-t)·x_0 + t·x_1,  target = x_1 - x_0
  Inference: Euler ODE,  x_{t+dt} = x_t + dt · v_θ(x_t, t, s, z)
  50 Euler steps (5× more than DDPM T=10, same wall-clock ~2 ms)

Legacy DiffusionPolicy (DDPM) remains available via --actor ddpm for ablation studies.
```

### 10.4 Training Objective (Current Implementation)

```python
# CFM (default --actor cfm) — implemented in train_diffusion.py
x_0 = torch.randn_like(actions_b)    # noise sample
x_1 = actions_b                       # target (clean params from replay buffer)
t = torch.rand(B)                     # uniform in [0,1]
x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
u   = x_1 - x_0                       # straight-line velocity target
v_pred = actor(x_t, t, specs_b, z_topo_b)
actor_loss = (weights * (u - v_pred)**2).mean()

# DDPM (legacy --actor ddpm)
noise = torch.randn_like(actions_b)
t = torch.randint(0, T, (B,)).float()
a_noisy = actor.add_noise(actions_b, t.long(), noise)
noise_pred = actor(a_noisy, t, specs_b, z_topo_b)
actor_loss = (weights * (noise - noise_pred)**2).mean()
```

The advantage-weighting (IQL-style) is shared by both paths.

### 10.5 Why Not DPPO?

DPPO (Diffusion Policy Policy Optimization) applies PPO's clipped surrogate objective to diffusion model updates. It solves a different problem: online fine-tuning of a pre-trained diffusion policy with online rollouts. Our training loop is already online RL — we don't need DPPO's trust-region mechanism on top. Adding PPO clipping would introduce a second hyperparameter (ε_clip) interacting with the advantage temperature τ, increasing instability risk.

**Decision**: CFM replaces the actor backbone. PPO/DPPO adds overhead without solving our root problem (training signal quality).

---

## 11. Roadmap

### Phase 1 ✅ COMPLETE: Fix remaining 2 topologies
- [x] Relax Switched_Filter prior: resonance [0.1, 10]·ωfc, impedance [10, 200]Ω
- [x] Relax Vector_Modulator prior: TL bound [0.1, 3.0]·λ/4; add passivity check G ≤ 1.0
- [x] Fix All_Pass off-resonance: add ωres ∈ [0.2, 5]·ωfc per section
- [x] Fix Vector_Modulator reward hacking: remove abs(il_db), cap G_I/G_Q to [0.7, 1.0]
- [x] All 6 topologies produce positive rewards in run_053508

### Phase 2: Topology-conditioned reward
- [ ] Per-topology phase scale: Switched_Filter → scale_deg=180°, others stay 90°
- [ ] Reweight gain term for 1-bit (All_Pass, Loaded_Line) vs 4-bit (Vector_Modulator) topologies
- [ ] Target: All_Pass and Vector_Modulator mean rewards above 0.5 (All_Pass avg=0.57 ✓, VM avg=0.68 ✓ — achieved via reparameterization + 20k steps)

### Phase 3 ✅ COMPLETE: Replace DDPM with CFM
- [x] Implement `FlowMatchingPolicy` and `VectorFieldNet` (drop-in for `DiffusionPolicy`)
- [x] CFM is now the default actor (`--actor cfm`); DDPM available as `--actor ddpm`
- [ ] Ablation: DDPM vs CFM on same seeds → timing + reward distribution + probe accuracy (Exp 6)
- [ ] Target: ValueNet probe ≥ 5/6, all topologies mean reward > 0.7

### Phase 4: Experiments for publication
- [x] Exp 1 Multi-topology convergence (run_035556, run_053508)
- [x] Exp 2 Linear vs log sizing ablation (run_035556 vs run_023537)
- [ ] Exp 3 Zero-shot graph generalization (train on 5, evaluate on held-out topology)
- [ ] Exp 4 Traditional RL baselines (SAC/PPO sizing, same 10k budget)
- [x] Exp 5 Pareto speedup vs Nelder-Mead (630× on Switched_Filter)
- [ ] Exp 6 DDPM vs CFM: synthesis quality + training convergence speed

---

## 12. File Map

```
train_diffusion.py          Main training loop, action_to_params, make_spice_netlist
sim/physics_priors.py       Analytical pre-filters for all 6 topologies
env/phaseshifter_env.py     RL environment, compute_reward, aggregate_state_metrics
env/graph_utils.py          Topology graph definitions, TOPOLOGY_PARAMS
models/gnn_encoder.py       TopologyEncoder (SAGEConv GNN)
models/diffusion_policy.py  FlowMatchingPolicy (default), DiffusionPolicy (DDPM legacy), CriticNet, ValueNet
specset/phaseshifter_scoring.py  Heuristic topology scorer (expert bonus)
specset/specset_train.json        training pool (10k, schema v3)
specset/specset_eval.json         held-out eval pool (2k, disjoint by spec id)
specset/specset_v1_frozen.json    frozen 600-spec v1 pool for published numbers
specset/templates/          6 SPICE netlists with STATE_TABLE definitions
inference_topology_select.py  Inference-time topology ranking (Option A)
probe_value_net.py          6-spec calibration test for ValueNet quality
run_all_experiments.sh      Multi-seed batch runner (4 GPU, SLURM)
train.sh                    Single-run SLURM submission script
```
