# Long-run diagnosis — 2026-09-14

Inspected checkout: `dev/nuerips_workshop`, HEAD
`90c5ef3e24e3911af55fa398375efe8ae2cba2b5`. Existing uncommitted changes were
preserved. This is a diagnosis, not an implementation change or a new training run.

## Conclusion

The current component-only GIN should remain a baseline. The intended method is
the circuit representation, physically meaningful action coordinates, validated
prior, and simulation-based objective together. The long runs establish a plateau
for this configuration; they do not isolate GIN as its sole cause or establish
that the bipartite encoder alone resolves it.

The most immediate demonstrated defect is that the All-Pass prior rejects the
repaired nominal design, although both MNA and ngspice score that design well.

## Recomputed results

Source: `train_gin_device_20k/train.log` and `train_gin_device_30k/train.log`.
Every logged frequency matches one specification in the current 10,000-spec
training pool. Recomputing the raw reward plus expert bonus gives exactly the
logged reward_raw for all 11,795 and 17,595 successful evaluations respectively
(maximum absolute difference 0.0). This validates the recovered target mapping.

- 20k: prior/simulation success 58.975%; reward all_close 335/20,000 = 1.675%.
- 30k: prior/simulation success 58.650%; reward all_close 698/30,000 = 2.327%.
- These all_close rates use the existing tolerant reward thresholds, not strict
  specification compliance or held-out checkpoint evaluation. They measure
  sampled training attempts; training samples can differ from a fixed-checkpoint
  evaluation distribution.
- All-Pass all_close: 0/3,366 at 20k and 0/5,069 at 30k.
- VM all_close: 1/3,361 at 20k and 0/5,038 at 30k.
- 20k all_close for the other topologies: Loaded_Line 310/3,250;
  Reflection_Type 5/3,381; Switched_Filter 1/3,205; Switched_Line 18/3,437.
- 30k all_close for the other topologies: Loaded_Line 664/4,826;
  Reflection_Type 4/5,024; Switched_Filter 5/4,830; Switched_Line 25/5,213.

In these logs, every prior-passing sample produced simulator metrics. Thus the
headline success rates primarily measure the filter. `parallel_eval_worker`
defines success as `agg is not None`. The training source calls the ngspice
environment path; the presence of a separate MNA scorer does not make these
logs an MNA training experiment.

## All-Pass: a demonstrated false rejection

The current electrical decoder uses centre offset 0.25 and nominal section ratio
5. The nominal section-A resonance is approximately sqrt(5)/0.25 = 8.94 times
the carrier. `check_physics_priors` permits only 0.2 to 5 times the carrier for
each section. The decoder's deliberately displaced section is therefore rejected.

At 20k, 3,279 of 3,294 All-Pass rejections (99.545%) include failure of section
A's resonance check; 2,590 fail only that check. This does not mean every rejected
sample is good. The nominal-point experiment below proves at least one systematic
class of good designs is removed by the filter.

Direct diagnostic: decode six actions of 0.5 with electrical bounds and ideal
switches. Check the prior, then evaluate independently using `mna_evaluate` and
`make_spice_netlist` / `parallel_eval_worker` without applying the pre-filter.

- MNA at 2.4, 10, and 28 GHz: prior fails at all three; IL approximately 0.560 dB,
  aggregate RL approximately 22.8 dB, RMS phase error approximately 1.33 degrees.
- ngspice at 2.4 GHz: prior false, simulation succeeds, IL 0.560514 dB,
  aggregate RL 22.763880 dB, RMS phase error 1.339331 degrees, reward 1.839108.
- ngspice at 28 GHz: prior false, simulation succeeds, IL 0.560405 dB,
  aggregate RL 22.786055 dB, RMS phase error 1.324765 degrees, reward 1.840049.

Diagnostic targets: bandwidth 20%, phase RMS 5 degrees, IL max 3 dB, RL min
12 dB, gain RMS 1 dB, power budget 15 mW, area budget 100 mm2. Expert bonus was
zero. These are nominal-design checks at the listed carriers under the existing
metric extraction, not a claim of compliance across the training spec distribution
or a broadband/PDK validation.

## VM: distinguish ineligible specifications from sizing failures

The VM minimum drive proxy is 2.45 mW. Training chooses a random topology before
checking whether the specification can afford it.

- 20k: 1,697/3,361 VM attempts are topology-ineligible. Of 1,664 eligible
  attempts, 1,484 pass the prior: 89.183%.
- 30k: 2,537/5,038 are topology-ineligible. Of 2,501 eligible attempts,
  2,264 pass: 90.524%.
- Rechecking every logged VM parameter set without the power budget rejects
  zero samples. All observed VM prior failures are attributable to the power
  gate under the current implementation.

This explains much of the 44–45% headline. It does not establish good VM sizing:
the all_close counts above remain essentially zero.

## Representation and action limitations

`env/graph_utils.py` constructs the old component-only graph with five type
features. VM's controlled sources are represented as Res nodes. Nets, terminal
roles, switch-state electrical values, and sizing are absent from that graph.

Verified on the saved 20k GIN checkpoint: the maximum absolute embedding
difference is exactly 0.0 for both All-Pass L_apA/L_apB and VM E_I/E_Q. Increasing
training time cannot remove an equality imposed by the graph's symmetries.
For All-Pass this is particularly problematic because the decoder interprets
the A and B coordinates differently: centre versus ratio.

`CircuitEncoder` instead uses device/net incidence, terminal features, electrical
values, and per-device switch-state contrasts. It also uses GIN-style sum
aggregation: bipartite graph construction and GIN aggregation are different
choices, not mutually exclusive concepts. Training still defaults to the older
GIN path; the long-run metadata explicitly selects it. The available 300-step
GIN-versus-circuit comparison also changes ideal to realistic switches, so it
does not isolate encoder quality.

Both device paths still have action-head limitations. `sized_devices` returns
(device, parameter) pairs, but the actor is given only the device embedding.
For example, Loaded_Line line length and impedance use the same T_main row.
`NodeVectorFieldNet` processes each scalar action independently, with shared
weights and independent initial noise, so it cannot learn arbitrary conditional
correlations across action coordinates. Decoder-imposed couplings still exist.
A bipartite encoder alone does not fix missing parameter roles or this
factorization.

## Objective fidelity

`WEIGHTS_AREA` weights phase/IL/RL/gain/area as 0.32/0.22/0.16/0.12/0.18;
continuous power weight is 0.0. VM power uses a VCVS drive proxy as a prior gate,
not measured DC supply consumption. Missing power is also treated as zero in
the reward's all_close calculation.

Area is computed from the selected topology and decoded sizing, but the current
model is a board-level geometry/footprint estimate. Lumped passives and switches
have fixed footprints; All-Pass area is consequently 18.88 mm2 independent of
its L/C values in this model. It is not extracted chip layout area. Physical
area and power objectives require a consistent technology and calibrated models.

The expert topology bonus remains in the replay reward. Therefore reported
total reward, including values above 2, is not solely the physical objective.
The workorder already proposes removing this bonus after physical constraints
are represented and recording explicit rejection reasons; those changes are
not implemented in this training path.

## Recommended order of work

1. Reconcile All-Pass prior with the current decoder and simulated transfer
   function. Add a regression that the repaired nominal passes, while invalid
   circuits remain rejected. Audit false rejections against independent simulator
   results; do not merely remove or arbitrarily widen the guard.
2. Log spec IDs, full run configuration, code/spec hashes, rejection reasons,
   prior pass, simulation validity, strict compliance, tolerant all_close,
   individual RF/area/power terms, and physical reward separately from bonuses.
   Separate topology-ineligible VM requests from avoidable sizing failures.
3. Use the bipartite circuit encoder as the main candidate. Add explicit
   parameter roles, units/normalized bounds, and coupled-coordinate roles to
   device action tokens. Compare a joint action model with the current
   independent scalar model. Keep the old GIN as an ablation.
4. Make area/power claims match the model fidelity. Calibrate physical metrics
   before assigning a continuous power weight; expose weights and hard budgets
   explicitly. Separate the topology heuristic from the physical objective.
5. Run matched GIN/circuit x slot/device controls on identical fixed specs,
   seeds, decoder, simulator, switch model, reward, and evaluation sample budget.
   Include midpoint, random/Sobol, and optimization baselines. Evaluate frozen
   checkpoints on held-out specs with per-topology compliance and reward gap
   to the achievable envelope. Use multiple seeds before inferring superiority.
6. Resume long runs only once corrected hard-topology compliance and held-out
   checkpoint performance improve. Add periodic resumable checkpoints first.

Older best-reward anecdotes, LOOCV tables, and these all-topology training means
are different protocols. FRAMEWORK section 9.8 explicitly flags missing historical
run evidence and changed reward definitions. Some older matrix cells are also
quarantined. They cannot establish a controlled regression without reproduction.
