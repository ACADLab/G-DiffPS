# Circuit graph rebuild: component-adjacency → bipartite device/net incidence

What the topology encoder consumes, before and after. Companion to
[`comparison.html`](comparison.html), which renders the same six pairs as images.

## The netlist is a hypergraph

A netlist is not a graph. It is a hypergraph \(H = (N, D)\): the vertices \(N\)
are nets, and each device \(d \in D\) is a hyperedge covering the set of nets its
pins touch. A two-terminal capacitor is a 2-edge; a T-junction is a single net
shared by three devices; a VCVS is a 4-edge with distinguished pin roles.

Any graph the encoder consumes is a *reduction* of that hypergraph, and the two
reductions in play here differ in whether they are invertible.

**Component-adjacency** is the intersection graph (line graph) of \(H\): one node
per device, an edge whenever two devices share a net. This is what
`get_topology_graph()` in [`env/graph_utils.py`](../../env/graph_utils.py) builds
— a hand-written `edge_index` literal per topology, with 5-dim one-hot node
features over `{TLine, Switch, Cap, Ind, Res}`.

**Bipartite incidence** is the star expansion (Levi graph) of \(H\): vertices are
\(N \cup D\), with an edge per pin. This is what `build_circuit_graph()` in
[`env/netlist_graph.py`](../../env/netlist_graph.py) builds, and it is the same
object SPICE assembles — the incidence matrix \(A\), with KCL as \(A\mathbf{i} = 0\)
and KVL as \(\mathbf{v} = A^{\top}\mathbf{e}\). Ground is an explicit net at index 0
and the RF ports are labelled. The netlists are transcribed from
`specset/templates/*.sp` and checked against them by `assert_matches_template()`,
so the graph cannot silently drift from what actually gets simulated.

### Star expansion is lossless; intersection is not

The Levi graph determines \(H\): the neighbourhood of each device node *is* its
net set, so \(H\) is recovered exactly. The map is injective, and any function of
\(H\) is in principle computable from it.

The intersection graph is not injective, and the counterexample is a circuit
primitive rather than a pathology. Three devices meeting at one net — a
T-junction — and three devices joined pairwise by three separate nets both
induce the triangle \(K_3\). They are different circuits: the first has one node
with a single KCL equation, the second has three. Adjacency cannot tell them
apart, because it records "these two devices touch" and discards *where*.
Hyperedge arity and multiplicity are destroyed by the same collapse, and pin
order — hence the KVL sign convention — was never representable at all.

## The rank bound, and why it holds for any encoder

The stronger and simpler defect is not that adjacency is lossy in general, but
that as constructed here it is constant in everything except the topology label.

> **Proposition.** `get_topology_graph(τ)` depends only on the topology index
> \(τ \in T\): it consults no spec, no frequency, no switch state, and no sizing.
> Therefore for *any* encoder \(f\), the embedding \(z_τ = f(G_τ)\) factors through
> \(T\). Its image has at most \(|T| = 6\) distinct values, so the embedding matrix
> over any spec set has rank \(\le 6\) and within-topology variance exactly \(0\).

This bound is a property of the *input*, not of the architecture. No choice of
layer type, width, depth, or training budget can lift it. Conditioning the policy
on \(z_τ\) therefore carries exactly the information of a 6-way one-hot, and the
GNN contributes nothing beyond that label.

Measured over 500 random specs
([`results/rank_diagnostic/rank_report.json`](../rank_diagnostic/rank_report.json)):

| Encoder | Rank | Participation ratio | Within-topology variance | Unique vectors / 500 |
|---|---|---|---|---|
| `dying_relu` | 5 | 1.94 | 5.9e-15 | 6 |
| `sage_ln` | 6 | 2.46 | 3.9e-13 | 6 |
| `gin` (paper) | 6 | 1.59 | 3.6e-12 | 6 |
| **`circuit`** (new) | **23** | 1.83 | **0.0105** | **99** |

Within-topology variance of 4e-12 is numerically zero: every spec mapping to the
same topology received a bit-identical embedding, exactly as the proposition
requires. The three legacy rows are not three failures to be explained
separately; they are three instances of one bound.

The 99 distinct `circuit` vectors are \(6\) topologies \(\times\) roughly 17
frequency buckets, not 500 distinct specs. Graph construction is cached at
`FC_BUCKETS_PER_DECADE = 10`, so specs sharing a bucket share an embedding.
That is a deliberate cost trade rather than a property of the representation,
and it is the ceiling the count is hitting.

## Structure, per topology

| Topology | old n | old e | old cc | new nets | new devices | new pins | new cc |
|---|---|---|---|---|---|---|---|
| Loaded_Line | 5 | 6 | 1 | 5 | 7 | 14 | 1 |
| Switched_Line | 6 | 4 | **2** | 7 | 8 | 16 | 1 |
| Reflection_Type | 10 | 14 | 1 | 7 | 12 | 24 | 1 |
| Switched_Filter | 10 | 8 | **2** | 7 | 12 | 24 | 1 |
| Vector_Modulator | 5 | 6 | 1 | 6 | 7 | 18 | 1 |
| All_Pass | 10 | 12 | **2** | 9 | 14 | 28 | 1 |

`cc` = connected components. Device counts include the two synthetic `Port`
terminations described below.

Half the topologies were disconnected under adjacency. Switched_Line,
Switched_Filter and All_Pass each had `cc = 2`. All three are two-branch
circuits, and with no shared ground or port net there was nothing joining the
branches — message passing could not carry information between the short and
long path of a switched-line phase shifter, which is the entire mechanism of the
device. The incidence graph joins them through the real `in`, `out` and `0` nets.

## Features

| | Before | After |
|---|---|---|
| Net nodes | none | 5-dim: `is_ground, is_port_in, is_port_out, is_internal, log1p(degree)` |
| Device nodes | 5-dim one-hot type | 20-dim: 7-way type one-hot + `is_sized, is_shunt, n_terminals, switch_is_on, log10(fc_ghz), electrical_size, z0_norm`, and (min, max) over pins of `dist_to_port_in, dist_to_port_out, dist_to_gnd` |
| Edges | untyped adjacency | 6-dim: `pin_index/3, is_control_pin, sign` (KVL sign convention) + that pin's three distances |
| Device vocabulary | 5 types | 7 types, adding `R_fixed`, `VCVS` and `Port` |
| Varies with `fc_ghz` | no | **yes** |
| Varies with switch state | no | **yes** |
| Varies with device sizing | no | **yes** |

### Electrical semantics

`electrical_size` is the log of a normalized reactance at the spec frequency —
\(\beta\ell\) for a line, \(\omega L / Z_0\) for an inductor,
\(1/(\omega C Z_0)\) for a capacitor, \(R/Z_0\) for a resistor — computed from
the device's actual value rather than from its type. A controlled source stores
its signed gain instead, since a VCVS has no reactance and the Vector
Modulator's states differ precisely by the sign and ratio of its I/Q drives.

Sizing reaches the encoder through `nominal_params()`, the midpoint of the
bounds the actor will sample within. The encoder necessarily runs *before* the
actor picks values — its output is what the actor is conditioned on — so the
midpoint is a stand-in, not the realised design. It varies with the spec
through the fc-centred bounds and stays in sync with `action_to_params` by
construction. Callers who do have values (the diagnostics below, or a two-pass
scheme) can pass `params=` directly.

Two consequences of that choice are worth stating, because both look like bugs
and are not:

- At the nominal, `Switched_Line`'s short and long lines take the *same* length,
  and `All_Pass`'s two branches take the same values. The bounds are symmetric;
  only the sampled action breaks the tie. So at nominal sizing those circuits
  really are trivial phase shifters with \(\Delta\phi = 0\), and their state
  contrast is correctly zero.
- Switches are resistors here, not capacitors. Both the SPICE templates and the
  MNA scorer model the off state as `R_off = 10 kΩ`, so the graph does too:
  `electrical_size` reads \(\log_{10}(R/Z_0)\), which separates the on and off
  states by more than three decades without desynchronising the graph from the
  simulator. See the switch-model discrepancy below before quoting either
  number.

`Port` is now instantiated: `P_in` and `P_out` are appended to each graph,
carrying \(Z_0\) and standing for the `Rsrc` and `Rload` terminations that the
templates contain and `solve_sparams` stamps. They are appended last, so the
device row indices returned by `device_names()` are unchanged and policy-side
lookups by name stay valid; `graph_device_names()` returns the full list.
Distances are computed on the net graph induced by the *real* devices only —
including the terminations would short every port to ground and collapse
`dist_to_gnd`.

Distances are now per-pin. Each pin's three distances ride on its incidence
edge, and the device-level feature is the (min, max) over its pins, which does
not depend on the order the pin tuple happened to be written in.

## States: the contrast is the device

A phase shifter is not one circuit. It is an indexed family \(\{G_s\}\), one
member per switch state, and the specified quantity — the phase step — is a
*difference* between two members. Any encoder that pools over states
symmetrically destroys exactly the thing being specified.

The encoder therefore encodes each state separately and summarises the family as

\[
z_\tau = \Big[\ \textstyle\sum_d \mathrm{mean}_s\, h_s[d]\ ;\ \sum_d \mathrm{mean}_{s<s'} \big| h_s[d] - h_{s'}[d] \big|\ \Big]
\]

with the same construction giving the per-device rows the actor consumes.

The order of operations matters more than it looks. Differencing *after* the
device pool — \(|\sum_d h_s[d] - \sum_d h_{s'}[d]|\) — is strictly weaker,
because a state change that permutes the device set leaves any
permutation-invariant readout unchanged. That is not a corner case: it is what a
switched-line phase shifter does when it swaps its short and long branch. With
the pooled difference, `Switched_Line`'s contrast measures 0.010 at
initialisation, effectively nothing. Differencing per device first, it measures
0.910.

Two state-selection details fall out of the same reasoning. The Vector Modulator
has no switches at all — it selects phase through its I/Q drive gains — so
without `_vcvs_gain()` feeding `VM_IQ[state]` into the graph, every VM state
encoded identically. And the previous default of contrasting states
\(\{0, n/2\}\) was the one degenerate choice for the VM: at \(n/2\) the drive is
exactly negated, so \(\Delta\phi = 180°\) for *every* sizing. The default is now
\(\{0, 1, n/2\}\), which includes the adjacent pair the spec's `ideal_step_deg`
actually names.

### The blindness is a theorem, not a measurement

State the transition as a group action. If a phase state change \(s \to s'\)
acts on the device set as a permutation \(\pi\), so that
\(h_{s'}[d] = h_s[\pi(d)]\) for every device \(d\), then for **any**
permutation-invariant readout \(P\),

\[
P\big(\{h_s[d]\}_d\big) \;=\; P\big(\{h_{s'}[d]\}_d\big)
\]

exactly — independent of node features, weights, depth, or training. A symmetric
state pool is not merely weak on such a transition; it is blind to it. Sum
pooling is permutation invariant, so \(\big|\sum_d h_s[d] - \sum_d h_{s'}[d]\big|\)
is identically zero, while \(\sum_d \big|h_s[d] - h_{s'}[d]\big|\) is not.

Branch-switched networks are exactly the circuits whose states are related this
way. `tests/test_state_permutation.py` checks the precondition programmatically
and finds it holds for precisely three of the six topologies —
`Switched_Line`, `Switched_Filter`, `All_Pass` — and for no others. Measured on
an untrained encoder at 28 GHz:

| Topology | permutes devices | pooled-then-differenced | differenced-then-pooled |
|---|---|---|---|
| Loaded_Line | no | 8.89 | 10.37 |
| **Switched_Line** | **yes** | **0.000003** | **9.66** |
| Reflection_Type | no | 9.64 | 10.58 |
| Switched_Filter | yes | 0.85 | 11.24 |
| Vector_Modulator | no | 0.54 | 0.61 |
| **All_Pass** | **yes** | **0.000008** | **10.19** |

Switched_Line and All_Pass are exact automorphisms at symmetric sizing, and
their pooled difference is numerically zero — 3e-6 and 8e-6 against a
per-device contrast near 10. Switched_Filter permutes the feature multiset
without being a full automorphism (its two branches differ in connectivity, not
just values), so message passing leaves a small residue rather than zero.
Sizing the branches apart breaks the automorphism, and the pooled difference
becomes small but nonzero — 0.12 for Switched_Line against a per-device 10.5,
still a factor of ninety.

This is the result worth carrying: it is a structural statement about a *class*
of circuits, it is provable rather than empirical, and it names the failure of
every representation that pools states symmetrically. The Vector Modulator row
is the counterpoint — its states differ only in two VCVS gains, so both
quantities are small and the theorem has nothing to say about it.

### What the phase-step probe does and does not show

`tools/contrast_probe.py` regresses the two blocks of \(z_\tau\) against the
phase step measured by the MNA scorer, over 800 random sizings per topology at
2–40 GHz, targets \(\cos\Delta\phi, \sin\Delta\phi\), 5-fold cross-validated
ridge \(R^2\), untrained encoder.

The mode that matters is `--z-sizing nominal`, because that is what the
deployed encoder sees: it runs *before* the actor picks values, so it is given
the bounds midpoint, never the design being evaluated. Under that setting the
result is unambiguous
([`contrast_probe_deploy.json`](contrast_probe_deploy.json)):

| Topology | contrast | mean | both | sd(Δφ)° |
|---|---|---|---|---|
| Loaded_Line | 0.083 | 0.083 | 0.082 | 30.9 |
| Switched_Line | −0.012 | −0.007 | −0.013 | 80.0 |
| Reflection_Type | 0.008 | 0.010 | 0.008 | 26.9 |
| Switched_Filter | −0.005 | −0.002 | −0.006 | 109.4 |
| Vector_Modulator | −0.016 | −0.016 | −0.016 | 14.9 |
| All_Pass | −0.013 | −0.014 | −0.014 | 101.1 |

**The deployed embedding carries essentially no information about the phase step
of the design it conditions.** Every within-topology \(R^2\) is at or below
noise. The pooled figure of 0.262 is entirely a between-topology effect — it
reflects that different families occupy different phase-step ranges, which a
6-way one-hot would also tell you.

Running the same probe with `--z-sizing sampled`
([`contrast_probe_oracle.json`](contrast_probe_oracle.json)) gives much higher
numbers — Loaded_Line 0.882, Switched_Line contrast 0.176 against mean −0.002 —
but that configuration hands the encoder the component values it does not have
at run time. It measures the capacity of the representation, not the content of
the deployed embedding, and it must not be quoted as the latter.

### Level 2 has not landed

The honest reading of the above, and of the diagnostic, is that wiring
`electrical_size` to `nominal_params()` did **not** make the embedding vary with
the design request:

- Within-topology variance moved 0.0265 → **0.0105**, the wrong direction.
- The bounds midpoint is a constant per (topology, parameter, fc bucket), so
  `electrical_size` is still constant across specs at fixed frequency. One
  constant was replaced by a different constant.
- `Port` devices and per-pin (min, max) distances added further constant
  dimensions, so the single genuinely varying quantity — frequency — is now a
  smaller share of the vector norm. Rank 20 → 23 is consistent with three new
  constant-but-independent directions and no new spec-dependent ones.
- Unique embeddings held at 99, which is \(6 \times\) the number of frequency
  buckets. The deployed \(z_\tau\) is a function of (topology, `fc_bucket`)
  alone, exactly as before.

The `--dphi-sizing nominal` run makes the same point from the other side: with
both the embedding and the phase step taken at nominal sizing, Loaded_Line
reaches \(R^2 = 0.987\) — the encoder recovers \(\Delta\phi\) perfectly, because
both sides are functions of one scalar. Switched_Line and All_Pass are skipped
outright there, since symmetric midpoint sizing makes their branches identical
and \(\Delta\phi\) is constant at zero.

The chicken-and-egg is real: the embedding conditions the actor that chooses the
sizes. The resolution is a two-pass encoder — pass one on nominal to condition
the actor, pass two on the emitted values to condition the critic and the
topology ranker, which run after sizing anyway and are where value dependence is
actually wanted. That is not implemented here.

## The switch model: paper and code disagree

The MLCAD paper, Section 5, states: *"SPICE netlists use realistic switch
parasitics (\(R_\text{on} = 6.5\,\Omega\), \(C_\text{off} = 650\,\text{fF}\))."*
All six templates in `specset/templates/` instead declare
`.PARAM R_on=3 R_off=10k`, and no capacitance appears in any switch. The two
descriptions disagree on every term: on-resistance, off-model, and whether the
off state is frequency dependent at all.

| Off-state model | 2.4 GHz | 28 GHz | 38 GHz |
|---|---|---|---|
| \(C_\text{off} = 650\) fF (paper text) | 102 Ω | 8.7 Ω | 6.4 Ω |
| \(C_\text{off} = 20\) fF (typical mmWave SPST) | 3.3 kΩ | 284 Ω | 209 Ω |
| \(R_\text{off} = 10\) kΩ (templates) | 10 kΩ | 10 kΩ | 10 kΩ |

Which one produced the published tables is settled by an internal consistency
check rather than by inspection. At 28 GHz the paper's stated parasitics give an
off-switch of 8.7 Ω against an on-switch of 6.5 Ω — a ratio of 1.35, meaning the
switch barely switches. Driving the MNA scorer with those values over 200 random
Switched_Line sizings gives a median \(|\Delta\phi|\) of **3.4°**, with 61.5% of
designs under 5°. The template values give a median of **65.7°**, with 3% under
5°. A circuit that cannot produce a phase step cannot report 100% yield and 100%
zero-shot LOOCV compliance, which is what Tables 2 and 5 report for
Switched_Line.

So the simulations ran on `R_on = 3 Ω, R_off = 10 kΩ`, and the paper's
parasitics sentence is incorrect as written. It needs a correction; the measured
results themselves are unaffected, since they never used the quoted numbers.

Separately, and on its own merits: a frequency-independent 10 kΩ off-state is
about 35× more open than a realistic 20 fF mmWave switch at 28 GHz, and it does
not degrade with frequency the way hardware does. For a shunt-switched loaded
line that inflates achievable phase coverage and isolation. This is a benchmark
artifact of the same class the representation critique is about — a modelling
choice that flatters the measured quantity — and it is quantifiable, so it
belongs in the limitations either way.

## Caveats

- This is a change of **representation**. Whether it improves held-out
  performance is what the corrected LOOCV matrix (`results/matrix_corrected`,
  2×2 encoder × action-space) is measuring; do not claim the improvement from
  the rank diagnostic alone. Higher embedding rank is a necessary condition for
  the encoder to matter, not evidence that it helps.
- The rank diagnostic and the contrast probe both use **untrained** encoders, so
  they measure the capacity of the representation, not of a learned model.
- Rank 23 is measured on a 64-dim embedding, so it is well short of saturating.
  The remaining 41 directions are unused capacity, not evidence of richness.
- The sizing the encoder sees is the bounds midpoint, not the design the actor
  goes on to produce. Feature values are therefore *representative* of the spec,
  not descriptive of the candidate being scored — and the deploy-mode probe
  shows this costs essentially all within-topology information about Δφ.
- Quote the probe only with its `z_sizing` stated. The `sampled` numbers
  describe a representation given values the deployed encoder never has.
- Checkpoints written before this change cannot be loaded: device features went
  from 16 to 20 dims, edges from 3 to 6, and `dev_proj` from `hidden → out` to
  `2·hidden → out`.
- `FRAMEWORK.md` §3.1 still describes the legacy encoder as
  `SAGEConv → SAGEConv → GlobalMeanPool`. The shipped legacy code is three
  `GINConv` layers with `global_add_pool`; the doc is stale.
