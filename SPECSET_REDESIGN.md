# Specset redesign

Companion to `JOINT_TOPOLOGY_SIZING_WORKORDER.md`. Addresses the specset audit
(items #1–#8). Ordered; each task lists acceptance criteria.

The audit filed eight items under "distribution." They are three different
problems and one operational hazard:

| Problem | Audit items | Fixed by |
|---|---|---|
| Heuristic labels used as ground truth | #2, #3, part of #6 | S0 |
| Specs that are physically contradictory or incoherent | #5 | S1, S2 |
| Sampling design and pool hygiene | #6, #7, size | S3, S4 |
| Schema/encoding hazards | #1, #8 | S5, S6 |

**Freeze before you regenerate.** Copy the current 600-spec file to
`specset/specset_v1_frozen.json` and pin every published-number reproduction
script to it. Everything below produces `v2`. Same principle as the switch
model: scope it, do not retract it.

---

## S0 — Stop using heuristic labels as ground truth

**This is the highest-leverage change in the document.** Items #2 and #3 and part
of #6 are all consequences of treating `score_topology`'s argmax as the label.

Under the T4 design the ranking target is the empirical envelope
`r*(τ, s) = max_a r_sim(τ, s, a)`, obtained by best-of-K sampling on the MNA
scorer. The heuristic is a *baseline*, not a label source.

Actions:

- Remove `entry["topology"]` as ground truth. Retire or rename the field so
  nothing reads it by accident (`heuristic_topology_deprecated`).
- Update every consumer: `topo_selection_accuracy.py` (repo root, not
  `tools/`), the LLM RAG path, `inference_topology_select.py`.
- Keep `specset/phaseshifter_scoring.py` intact — it becomes the "Heuristic"
  row in the T5 baseline table.

**Why this settles #3 (heuristic-label dominance).** Treating `score_topology`'s
argmax as ground truth invents class imbalances that physics does not support —
most visibly Switched_Filter's 35.3% share driven by the conjunctive `+8` bonus
(`bits ≥ 4 AND bw > 40%`). Under empirical envelope labels
`r*(τ,s) = max_a r_sim(τ,s,a)` a topology wins only where it genuinely scores
highest. Do not fix imbalance by editing heuristic constants; fix it by deleting
the label.

**Note on the abandoned SeriesShunt seventh topology.** An earlier probe added
`Switched_Line_SeriesShunt` temporarily; `results/joint/sl_seriesshunt_matrix.json`
recorded `crossover: false` (SeriesShunt 0.517 vs plain SL 0.566 under
`realistic`), so the seventh topology was abandoned and the registries stay at
six. The `+1` SeriesShunt heuristic rule cited in earlier drafts never existed
in `score_topology` — the only `+1` is the `tech == 2` rule for Switched_Line /
Reflection_Type. S0's argument rests on the `+8` normalization experiment
below, not on a crossover that did not occur.

**Labels are switch-model dependent.** Which topology wins depends on the switch
model (`ideal` vs `realistic`). So the envelope must be computed and stored
*per* `switch_model`, e.g. `r_star.ideal` and `r_star.realistic`, and every
consumer must specify which it reads. A single unlabelled `r_star` field will
silently mix the two regimes.

**Before blaming the box for #6:** `Switched_Filter: +8 if bits ≥ 4 AND bw > 40%`
is the largest single bonus in `score_topology` (+8 against +5/+6 elsewhere).
With `bits` and `bw_pct` drawn independently that condition fires often.
Re-score all specs with every bonus normalized to equal magnitude and report
whether the 35.3% share flattens. If it does, the imbalance was one hand-tuned
constant, not the spec box, and S3 may be unnecessary.

**Acceptance:** grep confirms no training or evaluation path reads a stored
topology label. The `+8` normalization experiment is run and its result
reported. Envelope fields exist for both switch models.

---

## S1 — Reject infeasible specs at sampling time

A spec demanding `rms_phase_err_deg` below the quantization floor implied by its
own `phase_bits` and `phase_coverage_deg` is not difficult, it is contradictory.
No sizing of any topology satisfies it. The audit measures 145/600 (24%).

```
floor(bits, cov) = (cov / 2**bits) / sqrt(12)

 bits   cov     step     floor
    3   180   22.50°    6.50°
    3   360   45.00°   12.99°
    4   180   11.25°    3.25°
    4   360   22.50°    6.50°
    5   180    5.62°    1.62°
    5   360   11.25°    3.25°
    6   180    2.81°    0.81°
    6   360    5.62°    1.62°
```

### RESOLVED — but not by rejection

Rejection was the wrong instrument and widening the box to compensate was worse.
Measured on the shipped generator, rejection ran at 42%; widening the box to
(1°, 20°) brought that to 28.4% but the widened box was **16.8% infeasible and
57.8% vacuous** — 74.6% of draws either unsatisfiable or satisfiable by anything.
A spec demanding 20° RMS phase error is worse than a 3-bit shifter's own
quantization error; it is not a loose spec, it is a non-spec.

The box is the problem, not its width. Sample the field relative to its own floor:

```
rms_phase_err_deg = floor(bits, cov) · κ,    κ ~ LogUniform[1.2, 3.0]

 bits   cov     floor    κ=1.2    κ=3.0
    4   180     3.25°    3.90°    9.74°
    5   180     1.62°    1.95°    4.87°
    6   180     0.81°    0.97°    2.44°
    3   360    12.99°   15.59°   38.97°
```

Zero rejections by construction, every spec informative, and `κ` is stored per
entry as `phase_kappa` so it can be stratified on directly. For
`phase_bits = 0` (analog) there is no quantization floor, so the 6-bit grid is
used as a proxy — an analog part is expected to beat the finest digital one —
which avoids special-casing one level back onto a box.

This is now the house convention for the whole benchmark: **any spec field with a
physical floor is sampled relative to that floor, never absolutely.** See
FRAMEWORK.md §9.10. `max_area_mm2` already followed it (rank-anchored, T1.5c);
`max_il_db`, `min_rl_db` and `rms_gain_err_db` still do not, and FRAMEWORK.md
§9.11 shows that is now the binding defect in the reward.

**Acceptance (met):** zero floor violations, asserted per draw as a postcondition
in `_sample_one` rather than checked after the fact. `SPEC_BOUNDS` for the field
is a normalization range only — the reachable span [0.49°, 38.97°], log-scaled —
not a sampling box.

---

## S2 — Couple the fields that are physically coupled

Independent per-field draws are the root of #5. Sample in dependency order
rather than drawing all fields independently.

**`app` → `fc_ghz`.** Draw `app` first, then `fc` from that band's real range.
Currently all four band classes span 1–39 GHz, producing 71 sub-6 specs above
6 GHz and 118 FR2/mmWave specs below 20 GHz. This also rescues `app` as a
conditioning input — it presently correlates with nothing, so it is a dead
dimension by construction.

**`phase_bits` ↔ `phase_coverage_deg`.** Reject combinations demanding ≥5 bits
over <180° of coverage (78 specs today). Either draw coverage first and cap
bits, or reject-and-resample.

**`tech` → plausible `fc_ghz`.** Now that `tech` sets switch parasitics
(work order T0.1b), a PIN diode at 38 GHz is a substantively different claim
from a pHEMT at 38 GHz. Constrain to plausible pairings and document the table.

**`max_area_mm2`.** Generated by the rank-anchored procedure in work order
T1.5c, which needs nominal areas at that spec's `fc`. **Sequencing: this field
cannot be generated until T0.4 lands**, because switching `L_quarter_mm` to log
scaling moves the nominal transmission-line length from 1.45·λ/4 to 1.00·λ/4 and
therefore moves every distributed-topology area by up to 2.1×.

**Acceptance:** zero `app`/`fc` mismatches, zero bits/coverage violations.
Report the joint marginals so the couplings are auditable.

---

## S3 — Sampling design: natural for training, stratified for evaluation

Reframe the goal. "Class imbalance" presumes balanced classes are correct, but
for a *realistic* spec distribution imbalance is the truth — All_Pass genuinely
is a narrow-application topology. The defect is not that the split is 35%/3%,
it is that **nobody designed that split; it fell out of `fc ~ U[1,40]` and
hand-chosen bounds.**

So split by purpose:

**Training pool — natural sampling.** Coupled and feasibility-filtered, but not
rebalanced. The policy should see the design space as it is.

**Evaluation pool — stratified by decision margin.** Define

```
m(s) = r*_(1)(s) - r*_(2)(s)        # best minus second-best topology
```

Bin `m` into `easy / medium / boundary` and sample evenly across bins. Report
selector performance per stratum.

Margin is the right stratification axis because it measures the thing that
matters. A selector at 95% on easy specs and 55% at boundaries is a different
object from one at 80% everywhere, and a uniform box reveals which regime you
are in only by accident. This is the same principle as T1.5c's rank-anchored
area budget: make benchmark difficulty an explicit, reportable knob.

Apply per-class stratification *within* the evaluation pool as a secondary
guarantee, so every topology has measurable support (see S4).

Note `m(s)` is switch-model dependent, per S0. Stratify under whichever model
the experiment reports, and say which.

**Acceptance:** evaluation pool has roughly equal mass in each margin stratum
and ≥1500 specs per topology in the per-class view. Training pool marginals are
reported, not rebalanced.

### REQUIRED — an earlier decision to skip this was wrong

S3 was briefly marked skippable on the strength of Switched_Filter's share
falling 35.3% → 20.6% → 9.1% after S0/S2. That reading was wrong: uniform over
six classes is 16.7%, so 9.1% is *below* chance. The mass did not spread, it
moved. All six shares, from `tools/class_balance_report.py`:

| topology | original scorer | normalized ±1 |
|---|---|---|
| Switched_Line | 21.8% | **49.7%** |
| Reflection_Type | **42.0%** | 29.7% |
| Switched_Filter | 21.1% | 9.6% |
| Loaded_Line | **4.3%** | 9.3% |
| Vector_Modulator | 6.2% | **0.3%** |
| All_Pass | **4.6%** | **1.4%** |

Total variation from uniform gets **worse** under normalization, 0.349 → 0.461.
Three classes violate the [5%, 30%] gate under each scorer. Eval-pool shares
match within 1.5 points.

The correction to the S0 finding: the `+8` bonus was indeed the driver of
Switched_Filter's *specific* dominance, and that part of the experiment stands.
But removing it relocated the peak rather than flattening it. The heuristic
scorer is degenerate under both magnitude schemes, which strengthens the case for
retiring it as a label source and removes any basis for skipping S3.

Margin strata are now computed and stored (`r_star[model].stratum`, terciles of
`m(s)`) and `topo_selection_accuracy.py` reports top-1/top-2 per stratum
alongside a **majority-class baseline** — the number that matters when one class
holds ~50%, since aggregate accuracy must be read against a constant predictor
rather than against 1/6. Strata are provisional pending the `r_star` refinement
described in FRAMEWORK.md §9.12: the top-two margins (~0.01–0.05) are currently
smaller than the archive bound's tail looseness.

---

## S4 — Size, and disjoint pools

**#7 is a correctness issue.** `tools/loocv_eval.py` builds
`PhaseShifterEnv(restrict_to=[held_out])` and calls `reset()`, which samples with
replacement from the same pool training used. `restrict_to` filters the topology
list, not the spec pool. 200 draws from 600 give 168 unique specs.

Be precise about what this compromises. The paper's C3 claim is about *topology*
transfer — "zero gradient updates on the held-out graph" — and that remains
true. What is weaker is that the actor saw those specs paired with other
topologies, so it can partly recall a good parameter region rather than
generalize to it. The honest restatement: **C3 measured topology transfer
conditional on seen specifications.** Correct the wording; do not defend it.

Fix: disjoint train / eval pools with no spec appearing in both, enforced by an
assertion at load time, not by convention.

**Size.** 600 is too small to measure anything per-class:

| Class support | 95% CI half-width at p = 0.9 |
|---|---|
| All_Pass, n = 19 | ±13.5 pp |
| Loaded_Line, n = 24 | ±12.0 pp |
| Switched_Filter, n = 212 | ±4.0 pp |
| proposed, n = 1500 | ±1.5 pp |

Table 5's Loaded_Line 76.7% and Reflection_Type 61.2% sit inside each other's
error bars at n = 24. Specs cost nothing to generate and the envelope labels are
microseconds of MNA, so:

```
train:  ~10,000 specs, natural distribution
eval:   ~2,000 specs, margin-stratified, disjoint
```

**Acceptance:** load-time assertion that the pools are disjoint. LOOCV numbers
re-reported with the spec pool held out as well as the topology, and the
difference from the previous numbers stated explicitly.

---

## S5 — Fix the categorical encodings

Two entries in `_normalize` encode nominal variables as ordinals, creating false
distributional structure:

- **`phase_bits = 0`** means analog-continuous but normalizes to `0.000`, the
  *coarsest* end of the scale, adjacent to 3 bits. It should be the finest, or
  better, a separate indicator.
- **`tech`** becomes `0.0 / 0.5 / 1.0`, asserting that GaAs pHEMT is halfway
  between PIN and SOI. This is meaningless in general and actively harmful now
  that `tech` determines switch parasitics — the model would interpolate between
  switch technologies.

One-hot both. **s** grows from 13 to roughly 17 dimensions. Update the CFM
actor's input width, `SPEC_KEYS`, and the ranking head.

**Acceptance:** no nominal field is represented as a scalar ordinal. A synthetic
test confirms `phase_bits = 0` is not nearest-neighbour to `phase_bits = 3` in
normalized space.

---

## S6 — Schema safety (the live hazard, #1)

`generate_specset.py` and `specset_phaseshifter.json` changed within the same
minute and were briefly inconsistent: `SPEC_BOUNDS` had 13 fields while the JSON
had 12, and `PhaseShifterEnv.reset()` died with `KeyError: 'max_area_mm2'`.
`_normalize()` indexes the spec dict directly while iterating `SPEC_KEYS`, so any
mismatch is a hard crash. A run started in that window would have died on first
reset. Nothing prevents a recurrence.

Actions:

- Add a `schema_version` integer to the JSON and a matching constant in code.
  Assert equality at load; fail with a clear message, not a `KeyError`.
- Write the JSON atomically (temp file plus rename) so a partial file is never
  loadable.
- Have `_normalize()` use `.get()` with an explicit missing-field error naming
  the field and both schema versions.

**Housekeeping from #8**, same pass:

- The module docstring claims "**Grid**-samples ~600 specifications." It is
  i.i.d. sampling with `default_rng(42)`. Fix the docstring.
- `vdd`, `app` and `bw_pct` are near open-loop — one or two references each
  across `env/`, `sim/` and `train_diffusion.py`. S2 rescues `app`. Decide
  explicitly for `vdd` and `bw_pct`: either score them or remove them from **s**.
  Conditioning on an unscored dimension spends actor capacity on noise.

**Acceptance:** a deliberately mismatched schema produces a named, actionable
error rather than a `KeyError`. Every field in `SPEC_KEYS` has at least one
scoring or gating consumer, or is documented as deliberately unscored.

---

## Order of operations

```
S6   schema guard + atomic write     ── do first, prevents recurrence
S0   heuristic labels out            ── dissolves #2, #3, part of #6
     └─ run the +8 normalization check here
S1   feasibility filter              ── removes the contradictory 24%
S2   field coupling                  ── needs T0.4 for max_area_mm2
S5   one-hot the categoricals        ── changes actor input width
S4   regenerate at 10k / 2k, disjoint
S3   margin stratification on eval   ── may be unnecessary; decide after S0–S2
```

S3 is deliberately last. Once the contradictory quarter is gone, the fields are
coupled, and the `+8` question is answered, re-measure the class distribution
before deciding whether stratification is still needed. It may not be.

---

## Do not do

- **Do not rebalance the training pool to equal class frequencies.** That
  replaces one undesigned distribution with a different undesigned one and makes
  the policy's spec prior less realistic, not more. Stratify the *evaluation*
  pool; report the training marginals.
- **Do not fix #3 by editing the heuristic's `+1`.** S0 removes the reason #3
  matters, and editing the heuristic would suppress the crossover result.
- **Do not regenerate `specset_v1_frozen.json`.** Published numbers must remain
  reproducible against the specset that produced them.
- **Do not stratify by argmax topology alone.** Sampling only where each
  topology clearly wins concentrates specs in region interiors, and the selector
  never sees the boundaries — which is where selection is hard and where it
  matters.
