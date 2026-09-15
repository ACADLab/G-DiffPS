# Transfer-first implementation and experiment plan

Updated 2026-09-15 for `dev/nuerips_workshop`.

## Active mandate (narrow)

Prove or falsify that **explicit parameter-role representation** is the missing
ingredient for transferable optimization. Do **not** invent new GNN architectures.
Freeze `circuit-typed + RWSE` as the topology backbone.

Causal chain we must evidence:

> Old graph can represent topology → typed graph fixes terminal semantics →
> global pooling loses sizing identity → explicit parameter-role tokens preserve
> sizing → R3 predicts local physical sensitivities better → R3 transfers those
> sensitivities across unseen topologies → that produces better RL compliance.

If the chain breaks at D4 or D5, **stop before RL** and diagnose there.

### Evidence ladder (gates)

| Stage | Question | Gate / artifact |
|-------|----------|-----------------|
| **D1** | Semantics correct? All_Pass `(centre,ratio)`; Reflection phase audit | One action index ↔ one physical variable; role/bounds/norm correct; synthetic perturbation moves intended token only. Tests green. |
| **D3** | Same-topo interpolation after end-to-end training | GIN / typed / D1a / R3; raw metrics; per-metric R². R3/D1a remain strong after training. `tools/d3_trained_interp.py` → `results/graphs/d3_trained_interp.json` |
| **D3b** | Where is information lost? | Probes: raw device.x → param token → pre-pool → pooled `z` → head. |
| **D4** | Local sensitivity (most important next science) | Predict `Δm_i` / sign for ±Δ on θ_i. R3 should know which knob moves which metric. `tools/d4_local_sensitivity.py` |
| **D4b** | Held-out parameter-role transfer (if practical) | Same role (e.g. Z0, length) across topologies; test held-out topology. |
| **D5** | Corrected supervised LOOCV | Only after D3/D4. Raw metrics, 3 seeds, 6 holdouts. Macro + per-topo + metric-wise R², sensitivity sign, calibration. |
| **R3 ablations** | Which semantics matter? | Drop role / value / bounds / log-flag / coupling / RWSE one at a time. |
| **Capacity** | Fairness | Match hidden dim / param count; capacity-matched GIN control if needed. |
| **Prior** | Independence | Core runs ABCD off; then GIN±prior × R3±prior. |
| **D6 RL** | Expensive LOOCV | Only if D5 passes. GIN vs best R3. Compliance, TTFP, physical reward, curves. |

Primary deliverables now: **D1, D3, D4, D5, R3 ablations**. Everything else secondary.

## Main objective

Learn reusable circuit structure and sizing behavior that transfers to a topology
excluded from training. At evaluation, construct that topology's graph and run
the frozen model without retraining, fine-tuning, updating normalization, or
optimizing the proposed design. Choose the graph representation on demonstrated
transfer, not its name: bipartite incidence is a candidate, not an assumed winner.

Higher all-topology training reward, topology separation in embedding space, and
simulator success alone do not satisfy this objective.

## Experimental contract

- Six outer leave-one-topology-out folds: train on five; evaluate the sixth.
  Each fold needs its own model trained from scratch on its permitted data. The
  no-retraining claim concerns deployment on the excluded topology, not sharing
  a model trained on all six across folds.
- Exclude held-out topology trajectories, replay, simulator labels, envelope
  targets, and checkpoints from gradient training and model selection. Audit any
  pretraining. Use validation data from the training side for checkpoint and
  hyperparameter selection; never choose the best checkpoint on the outer test.
- Preserve a labeled legacy track for topology transfer conditional on seen
  specs. Primary track: held-out topology AND disjoint held-out specifications.
  Freeze and hash the train/validation/test manifests and report their overlap.
- Predeclare shared physics rules, graph features, bounds, and nominal designs.
  Known hand-designed decoder priors are permitted but must be disclosed and
  tested as inductive biases. They are not evidence that the network learned
  the transferred behavior.
- Freeze all learned modules at evaluation. Record checkpoint hashes before and
  after evaluation and verify no state changes or optimizer steps occurred.
- Primary metric uses one generated design per specification. Separately report
  best-of-K with a fixed equal sampling/simulator budget across methods. No
  gradient-based or optimizer-based refinement in the zero-shot result.
- Report optimizer baselines as separate reference methods, including their
  simulator cost. If an approximate achievable envelope is used, label reward
  gaps accordingly; its lower-bound nature does not guarantee nonnegative gaps.
- Distinguish (a) sizing a supplied unseen graph and (b) selecting an unseen graph
  from a candidate set and sizing it. Neither result substitutes for the other.

## Stage 1 — Repair measurement and preserve provenance

Primary files: `tools/loocv_eval.py`, `tools/run_matrix.py`,
`train_diffusion.py`, `env/reward.py`.

Verified harness defects to address:

1. `COMPLIANCE_THRESHOLD = 0.5` with a heuristic bonus is not specification
   compliance. Compute explicit per-metric compliance, strict conjunction,
   existing tolerant all_close, and physical reward separately.
2. LOOCV calls `check_physics_priors` without pmax_mw. Pass identical power,
   switch, bounds, and physical configuration through training and evaluation.
3. `load_models` silently leaves models random if checkpoint files are absent.
   Fail on missing/incompatible checkpoints and configuration mismatches.
4. The matrix runner does not pass the now-required evaluation switch-model
   argument. Propagate it consistently to both training and evaluation.
5. Missing evaluation data falls back to train data. Primary evaluation must
   fail closed; legacy seen-spec evaluation must be explicitly requested.
6. Use deterministic specification manifests. Seeding torch/numpy alone does
   not seed the environment's independent specification sampler.
7. Stop after failed training and reject stale outputs from previous executions.
   Record full command/configuration, code/spec hashes, fold, seed, checkpoint
   provenance, simulator identity, sample budget, and per-attempt results.

Outcome: a small repeatable evaluation produces correct denominators and cannot
silently evaluate an untrained model, the wrong spec pool, or a different switch
model. Meaningful regression tests cover these failure modes and frozen-state
evaluation. Do not interpret the old reward-threshold column as strict compliance.

## Stage 2 — Align the physical search space and objective

Primary files: `train_diffusion.py`, `sim/physics_priors.py`,
`sim/mna_scorer.py`, `sim/area_model.py`.

- Repair the demonstrated All-Pass decoder/prior inconsistency. Its current
  nominal passes direct RF evaluation but fails the section resonance filter.
  Validate good nominal and perturbed designs plus bad circuits across carrier
  frequencies and switch models; do not tune a threshold solely to raise yield.
- Log detailed prior reasons, simulator errors, and topology-level power
  ineligibility separately. Keep both unconditional and eligible-only outcomes.
  In selection, an ineligible candidate must not be selected; in sizing, do not
  interpret an impossible power budget as avoidable sizing error.
- Establish the baseline physical reward without a hidden topology-ranking bonus.
  Keep the old heuristic as a labeled baseline/ablation, not part of compliance.
- Keep area/power fidelity explicit: current area is a board footprint model;
  current VM power is a proxy gate. Do not invent measured DC power. A calibrated
  technology-specific area/power experiment is a separate objective version with
  frozen weights and budgets shared by all encoders.
- Run midpoint and random/Sobol references with exactly the same decoder and
  prior. Check whether apparent improvements are inherited from these priors.

Outcome: valid circuits are not systematically discarded, and every compared
model sees the same reachable design space and objective.

## Stage 3 — Make graph and action semantics transferable

Primary files: `env/netlist_graph.py`, `env/graph_utils.py`,
`models/circuit_encoder.py`, `models/diffusion_policy.py`.

- Preserve the current component-only GIN as a baseline. Evaluate circuit
  incidence with terminal roles, device type, ports/ground, switch-state contrast,
  electrical values, and technology as the first richer representation.
- Represent action tokens as device PLUS parameter role, units/normalization,
  bounds, and coupling role. Length and impedance on the same line must be
  distinguishable; All-Pass centre and ratio must be distinguishable.
- Compare the existing independent scalar device flow with a shared action model
  that can condition on other action coordinates, such as message passing or
  attention over parameter tokens. Do not create a head for each topology.
- Check graph/device permutation equivariance, role distinction, state contrast,
  and variable action counts. Avoid topology-ID embeddings that cannot support
  a new topology. Higher embedding rank is a diagnostic, not the acceptance test.
- Use identical action architecture for paired graph comparisons. When changing
  action architecture, hold the graph fixed. Attribute coupled decoder changes
  through shared baselines rather than crediting them to graph learning.

Outcome: a new graph can be encoded and its parameter roles addressed by shared
weights; representation and action-head effects can be separated experimentally.

## Stage 4 — Bounded pilot before the full matrix

- Start with All-Pass, VM, and Reflection-Type held-out folds, one seed, and
  1k–2k training steps per configuration. Never include the held-out topology
  in that fold's training to rescue its result.
- Compare GIN and the richer graph under the same corrected decoder and shared
  action head. Add controlled role/coupling ablations as needed. Treat the pilot
  evaluation specs as development data, not the final test report.
- Include trained, randomly initialized, and midpoint/random baselines so that
  learned transfer can be separated from graph construction and decoder priors.
- Record strict compliance, tolerant all_close, physical reward distributions,
  individual RF errors, area, available power metrics, rejection reasons, sample
  cost, and known-topology validation performance.
- Estimate wall time and memory from these runs. Proceed only if harness checks
  pass and the pilot provides evidence of learned transfer beyond priors. If
  it fails, diagnose the bottleneck instead of extending runs blindly.

Outcome: a defensible architecture/configuration shortlist and a measured budget.

## Stage 5 — Six-fold transfer result

- Freeze the protocol before final evaluation. Use all six held-out topologies
  and seeds 42, 1337, 2026. A two-graph, two-action-design factorial is 72 trained
  fold/seed configurations; two matched primary configurations require 36.
  Select the matrix using the pilot budget and retain the controls needed for
  the claimed attribution.
- Match training/evaluation budgets. Evaluate fixed checkpoints at predeclared
  training milestones; choose checkpoints with training-side validation only.
- Report each held-out topology, each seed, macro-average, worst fold, uncertainty,
  and paired method differences on identical specifications. Do not hide All-Pass
  or VM behind easier folds. Avoid treating repeated samples as independent seeds.
- Evaluate unseen-topology selection separately with a shared graph scoring head
  and no held-out envelope supervision. Report selection regret, selection of
  infeasible candidates, and end-to-end compliance after sizing.
- Compare with the historical LOOCV only through an explicitly reproduced legacy
  protocol. Keep changed bounds, reward, switch model, and spec split visible.

Acceptance: a richer graph must improve zero-shot held-out performance over the
matched GIN and untrained/physics-only baselines across folds/seeds, including
credible hard-topology gains. If gains are limited to certain folds, state that
scope; do not claim universal transfer. Repeated final-test tuning requires a
new untouched test set for the final claim.

## Stage 6 — Broader topology generalization, if claimed

Six-template LOOCV establishes transfer within the registered circuit families.
It does not by itself establish support for arbitrary unseen netlists, because
graph construction and action decoding still contain template-specific knowledge.
For a broader claim, use generic netlist-to-graph/action construction and held-out
generated circuit families. Deduplicate structural equivalents and keep close
variants in the same split. Audit the existing open-topology experiment separately:
its frozen-embedding selection task is not a substitute for learned sizing transfer.

## Execution boundaries

Work in this development branch and preserve existing experiments. Save fixes
and experiments in distinct, traceable increments. After Stage 1/2 verification,
run the bounded pilot before expensive training. Report blockers and negative
results as well as improvements. Do not launch another 20k/30k/50k all-topology
run as evidence for LOOCV. Periodic resumable checkpoints are required before
longer jobs.
