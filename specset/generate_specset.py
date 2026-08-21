"""Generate SpecSet-PhaseShifter benchmark.

Grid-samples ~600 phase-shifter design specifications across a 12-D space,
labels each with the expert-recommended topology from the heuristic scorer,
and writes the dataset to JSON for use by the RL environment and LLM RAG.
"""

import json
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from phaseshifter_scoring import score_topology, TOPOLOGY_LABELS

N_SAMPLES = 600

# 12-D phase-shifter specification space
SPEC_BOUNDS = {
    "fc_ghz":             (1.0, 40.0),    # log-sampled
    "bw_pct":             (5.0, 60.0),
    "phase_coverage_deg": (90.0, 360.0),
    "phase_bits":         [0, 3, 4, 5, 6],   # 0 = analog continuous
    "rms_phase_err_deg":  (1.0, 10.0),
    "rms_gain_err_db":    (0.3, 3.0),
    "max_il_db":          (1.0, 10.0),
    "min_rl_db":          (8.0, 20.0),
    "vdd":                (0.8, 3.3),
    "pmax_mw":            (1.0, 50.0),    # log-sampled
    "tech":               [0, 1, 2],       # 0=CMOS, 1=SiGe, 2=GaAs
    "app":                [0, 1, 2, 3],    # 0=sub-6, 1=FR1, 2=FR2/mmW, 3=Ku/Ka
}


def _sample_log(lo, hi, rng):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def generate_samples():
    rng = np.random.default_rng(42)
    dataset = []

    for i in range(N_SAMPLES):
        spec = {
            "fc_ghz":             _sample_log(*SPEC_BOUNDS["fc_ghz"], rng),
            "bw_pct":             float(rng.uniform(*SPEC_BOUNDS["bw_pct"])),
            "phase_coverage_deg": float(rng.uniform(*SPEC_BOUNDS["phase_coverage_deg"])),
            "phase_bits":         int(rng.choice(SPEC_BOUNDS["phase_bits"])),
            "rms_phase_err_deg":  float(rng.uniform(*SPEC_BOUNDS["rms_phase_err_deg"])),
            "rms_gain_err_db":    float(rng.uniform(*SPEC_BOUNDS["rms_gain_err_db"])),
            "max_il_db":          float(rng.uniform(*SPEC_BOUNDS["max_il_db"])),
            "min_rl_db":          float(rng.uniform(*SPEC_BOUNDS["min_rl_db"])),
            "vdd":                float(rng.uniform(*SPEC_BOUNDS["vdd"])),
            "pmax_mw":            _sample_log(*SPEC_BOUNDS["pmax_mw"], rng),
            "tech":               int(rng.choice(SPEC_BOUNDS["tech"])),
            "app":                int(rng.choice(SPEC_BOUNDS["app"])),
        }

        scores = {t: score_topology(t, spec) for t in TOPOLOGY_LABELS}
        best_topology = max(scores, key=scores.get)

        entry = {
            "id": f"spec_{i:04d}",
            "spec": spec,
            "topology": best_topology,
            "netlist_path": f"templates/{best_topology.lower()}.sp",
            "sizing_hints": {
                "varactor_c_max_pf": 1.0,
                "tline_zo_ohm": 50.0,
                "switch_w_um": 50.0,
            },
        }
        dataset.append(entry)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "specset_phaseshifter.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset generated: {out_path} ({len(dataset)} entries)")

    # Print topology distribution as a sanity check
    from collections import Counter
    counts = Counter(e["topology"] for e in dataset)
    print("\nTopology distribution:")
    for t in TOPOLOGY_LABELS:
        print(f"  {t}: {counts.get(t, 0)}")


if __name__ == "__main__":
    generate_samples()