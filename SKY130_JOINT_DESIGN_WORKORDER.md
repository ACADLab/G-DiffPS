# Work order: joint topology + sizing with SKY130

Research objective: a circuit graph object that supports joint decisions about
connectivity, device implementation, geometry, bias, and switching behavior,
evaluated with SKY130 simulation for phase response, loss, matching, power, and
robustness.

This document supersedes the ideal-component joint path for *new* PDK-backed
experiments. Legacy ideal-switch templates remain for paper reproduction only.

Branch: `dev/nuerips_workshop`

---

## Milestone status

| ID | Milestone | Status |
|----|-----------|--------|
| A | Working SKY130 device and circuit simulations | **In progress — harness green** |
| B | Characterized device vocabulary and initial frequency range | Not started |
| C | PDK-aware graph object and realizable sizing decoder | Not started |
| D | Joint topology–sizing on a small circuit grammar | Not started |
| E | Robustness across operating conditions | Not started |
| F | Layout and extracted-parasitic feedback | Not started |

### A — pinned reference (done locally)

- PDK: SKY130A via volare hash `c6d73a35f524070e85faff4a6a9eef49553ebc2b`
- Pin file: `pdk/sky130_pin.yaml`
- Simulator: ngspice-47 (Homebrew), verified with harness
- Harness: `sim/sky130/`, runner `tools/run_sky130_harness.py`
- Device decks: `nfet_01v8`, `pfet_01v8`, `res_xhigh_po_0p35`, `cap_mim_m3_1`
- OTA integration: OP+AC and TRAN decks inspired by
  [opensrc_analog](https://github.com/eescottie/opensrc_analog) miller OTA
  examples (vendored under `third_party/opensrc_analog/` as xschem schematics;
  SPICE decks are project-local)
- Results: `results/sky130/harness_results.json`, `results/sky130/environment.json`

```bash
export PDK_ROOT="$(pwd)/pdk"
export PATH="/opt/homebrew/bin:$PATH"   # or wherever ngspice lives
python3 tools/run_sky130_harness.py
```

PDK install (once per machine):

```bash
export PDK_ROOT="$(pwd)/pdk"
pip install volare
volare fetch --pdk sky130 c6d73a35f524070e85faff4a6a9eef49553ebc2b
volare enable --pdk sky130 c6d73a35f524070e85faff4a6a9eef49553ebc2b
```

---

## Ordered tasks (from research plan)

1. **Reproducible PDK simulation reference** — pin PDK, libraries, simulator;
   batch OP/AC/TRAN; individual MOS/R/C testbenches. *(harness established)*
2. **Physically useful design space** — characterize switches and passives over
   geometry, bias, frequency, corners; choose first frequency range and specs.
3. **PDK-aware circuit graph object** — device–net bipartite + terminal roles +
   PDK identity + geometry/bias/control + evaluation context (context ≠ topology).
4. **Realizable sizing outputs** — transistor geometry/control instead of
   independent R_on/R_off; map R/C to PDK devices; tied parameters; **resolve CFM
   train/sample coordinate mismatch before new data collection**.
5. **Small phase-shifter grammar** — MOS switches/TGs, R/C networks, switchable
   caps, bias/buffers; RC/all-pass and active phase shifters first.
6. **Candidate-dependent electrical representations** — structural → sized →
   evaluated views; auxiliary predictors must not leak eval-only features.
7. **Couple topology decisions to sizing refinement** — alternating loop with
   equal budgets; transfer compatible parameters across edits.
8. **Evaluation hierarchy** — structural checks → OP → small-signal → switch
   states → transient/periodic as needed; validate MNA/ABCD against PDK SPICE.
9. **Physical robustness** — corners, supply, temp; mismatch only where verified.
10. **Controlled graph ablations** — which graph information improves joint design.
11. **Layout loop on a small subset** — DRC/LVS, extraction, ranking stability.

First concrete joint experiment: a few PDK-realizable phase-shifting structures
under the same specs and simulation budget, refining both structure and geometry.

---

## Immediate next work (after A)

1. Sweep nfet/pfet as switches (Ron/Roff, feedthrough, Cparasitic vs W, Vgs, freq).
2. Record model-validity notes and choose an initial RF or IF band.
3. Sketch the PDK-aware circuit object API that emits the same netlists the
   harness already simulates.
4. Keep CFM `clamp` (not sigmoid) and `cfm_steps=10` for any new actor training
   on this branch (`models/diffusion_policy.py`).

---

## Explicit non-claims

- Do not claim bipartite graphs as already implemented for the *legacy*
  `env/graph_utils.py` component graphs.
- Do not treat ValueNet topology selection as a deployment path.
- Do not equate simulation success with specification compliance.
- Do not treat ideal R_on/R_off templates as SKY130-realizable devices.
