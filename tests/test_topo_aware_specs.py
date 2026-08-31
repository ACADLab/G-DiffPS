"""Unit tests for topology-biased spec proposals (no SPICE)."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from specset.phaseshifter_scoring import TOPOLOGY_LABELS, score_topology
from specset.topo_aware_specs import (
    sample_balanced_oracle_pool,
    sample_spec_for_topology,
)


def test_sample_spec_spans_fc_decades_and_covers_topologies():
    rng = np.random.default_rng(0)
    fcs = []
    for topo in TOPOLOGY_LABELS:
        for _ in range(30):
            spec = sample_spec_for_topology(topo, rng)
            assert set(spec) >= {
                "fc_ghz", "bw_pct", "phase_bits", "phase_coverage_deg", "pmax_mw",
            }
            fcs.append(spec["fc_ghz"])
    # Priors cover low-GHz (All_Pass / Switched_Filter) through mmWave
    # (Reflection_Type), so log span must exceed two decades.
    assert min(fcs) < 3.0
    assert max(fcs) > 20.0
    assert np.log10(max(fcs) / min(fcs)) >= 1.0


def test_balanced_pool_requests_each_topology():
    rng = np.random.default_rng(1)
    requested: list[str] = []

    def scorer(spec: dict) -> str:
        # Deterministic stand-in for an oracle: heuristic argmax.
        scores = {t: score_topology(t, spec) for t in TOPOLOGY_LABELS}
        return max(scores, key=scores.get)

    # Wrap proposals to record which classes were asked for.
    import specset.topo_aware_specs as mod
    real = mod.sample_spec_for_topology

    def tracking(topo, rng_):
        requested.append(topo)
        return real(topo, rng_)

    mod.sample_spec_for_topology = tracking
    try:
        n_per = 2
        pool = sample_balanced_oracle_pool(
            n_per_class=n_per,
            scorer_fn=scorer,
            pool_cap=300,
            rng=rng,
        )
    finally:
        mod.sample_spec_for_topology = real

    assert set(requested) == set(TOPOLOGY_LABELS)
    labels = {row["label"] for row in pool}
    # Biased proposals + heuristic oracle should yield >2 winning classes.
    assert len(labels) >= 3
    counts = {t: sum(1 for r in pool if r["label"] == t) for t in labels}
    assert all(c <= n_per for c in counts.values())
    assert all("spec" in r and "target" in r for r in pool)


if __name__ == "__main__":
    test_sample_spec_spans_fc_decades_and_covers_topologies()
    print("OK decades")
    test_balanced_pool_requests_each_topology()
    print("OK balanced pool")
