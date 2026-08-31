"""Topology-biased specification proposals for stratified oracle evaluation.

Uniform draws from SPEC_BOUNDS make only a couple of topologies win SPICE/MNA
oracles.  These helpers bias fc / bw / phase_bits (and related fields) toward
regions where ``score_topology`` gives each class a competitive prior, so
rejection sampling can fill more than two oracle buckets.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from specset.generate_specset import SPEC_BOUNDS
from specset.phaseshifter_scoring import TOPOLOGY_LABELS

# Per-topology proposal boxes aligned with score_topology bonuses/penalties.
# Ranges stay inside SPEC_BOUNDS; only the conditional prior changes.
_TOPO_PRIORS: dict[str, dict] = {
    # Schiffman / all-pass: broadband, analog, modest coverage; low-GHz friendly.
    "All_Pass": {
        "fc_ghz": (1.0, 8.0),
        "bw_pct": (48.0, 60.0),
        "phase_bits": [0],
        "phase_coverage_deg": (90.0, 180.0),
        "pmax_mw": (5.0, 40.0),
    },
    # Loaded-line varactor: mid-band, narrowband, analog, limited coverage.
    "Loaded_Line": {
        "fc_ghz": (4.0, 12.0),
        "bw_pct": (5.0, 22.0),
        "phase_bits": [0],
        "phase_coverage_deg": (90.0, 170.0),
        "pmax_mw": (3.0, 30.0),
    },
    # Switched-line: digital mid/high band (avoid bulky <5 GHz).
    "Switched_Line": {
        "fc_ghz": (6.0, 22.0),
        "bw_pct": (28.0, 55.0),
        "phase_bits": [4, 5, 6],
        "phase_coverage_deg": (180.0, 360.0),
        "pmax_mw": (5.0, 40.0),
    },
    # Reflection-type coupler: mmWave + wideband, needs DC budget.
    "Reflection_Type": {
        "fc_ghz": (16.0, 40.0),
        "bw_pct": (32.0, 60.0),
        "phase_bits": [0, 3, 4],
        "phase_coverage_deg": (180.0, 360.0),
        "pmax_mw": (8.0, 50.0),
    },
    # Switched HP/LP filter: low-GHz digital + very wide BW.
    "Switched_Filter": {
        "fc_ghz": (1.0, 8.0),
        "bw_pct": (42.0, 60.0),
        "phase_bits": [4, 5, 6],
        "phase_coverage_deg": (180.0, 360.0),
        "pmax_mw": (5.0, 40.0),
    },
    # Vector modulator: high bits, full coverage, power-hungry.
    "Vector_Modulator": {
        "fc_ghz": (10.0, 35.0),
        "bw_pct": (28.0, 55.0),
        "phase_bits": [5, 6],
        "phase_coverage_deg": (320.0, 360.0),
        "pmax_mw": (12.0, 50.0),
    },
}


def _sample_log(lo: float, hi: float, rng: np.random.Generator) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _clip_range(key: str, lo: float, hi: float) -> tuple[float, float]:
    """Intersect a proposal range with global SPEC_BOUNDS."""
    glo, ghi = SPEC_BOUNDS[key]
    return (max(float(lo), float(glo)), min(float(hi), float(ghi)))


def sample_spec_for_topology(
    topo: str,
    rng: np.random.Generator,
) -> dict:
    """Draw one spec biased toward regions where ``topo`` is competitive.

    Unbiased fields (error floors, Vdd, tech, app) still use SPEC_BOUNDS.
    """
    if topo not in _TOPO_PRIORS:
        raise ValueError(f"Unknown topology: {topo}")
    prior = _TOPO_PRIORS[topo]

    fc_lo, fc_hi = _clip_range("fc_ghz", *prior["fc_ghz"])
    bw_lo, bw_hi = _clip_range("bw_pct", *prior["bw_pct"])
    cov_lo, cov_hi = _clip_range("phase_coverage_deg", *prior["phase_coverage_deg"])
    p_lo, p_hi = _clip_range("pmax_mw", *prior["pmax_mw"])
    bits_choices = [b for b in prior["phase_bits"] if b in SPEC_BOUNDS["phase_bits"]]
    if not bits_choices:
        bits_choices = list(SPEC_BOUNDS["phase_bits"])

    return {
        "fc_ghz": _sample_log(fc_lo, fc_hi, rng),
        "bw_pct": float(rng.uniform(bw_lo, bw_hi)),
        "phase_coverage_deg": float(rng.uniform(cov_lo, cov_hi)),
        "phase_bits": int(rng.choice(bits_choices)),
        "rms_phase_err_deg": float(rng.uniform(*SPEC_BOUNDS["rms_phase_err_deg"])),
        "rms_gain_err_db": float(rng.uniform(*SPEC_BOUNDS["rms_gain_err_db"])),
        "max_il_db": float(rng.uniform(*SPEC_BOUNDS["max_il_db"])),
        "min_rl_db": float(rng.uniform(*SPEC_BOUNDS["min_rl_db"])),
        "vdd": float(rng.uniform(*SPEC_BOUNDS["vdd"])),
        "pmax_mw": _sample_log(p_lo, p_hi, rng),
        "tech": int(rng.choice(SPEC_BOUNDS["tech"])),
        "app": int(rng.choice(SPEC_BOUNDS["app"])),
        "max_area_mm2": float(_sample_log(*SPEC_BOUNDS["max_area_mm2"], rng)),
    }


def sample_balanced_oracle_pool(
    n_per_class: int,
    scorer_fn: Callable[[dict], str],
    pool_cap: int,
    rng: Optional[np.random.Generator] = None,
    topologies: Optional[list[str]] = None,
) -> list[dict]:
    """Propose topo-biased specs until each class has ``n_per_class`` oracle wins.

    ``scorer_fn(spec)`` returns the oracle-winning topology label (e.g. SPICE
    argmax).  Proposals cycle over under-filled classes via
    ``sample_spec_for_topology``; accepted rows are bucketed by the *oracle*
    label, not the proposal target.

    Returns a list of ``{"spec", "label", "target"}`` dicts (at most
    ``n_per_class`` per label).  Stops early if ``pool_cap`` draws are used.
    """
    if n_per_class < 1:
        raise ValueError("n_per_class must be >= 1")
    if pool_cap < 1:
        raise ValueError("pool_cap must be >= 1")

    topos = list(topologies) if topologies is not None else list(TOPOLOGY_LABELS)
    if not topos:
        return []

    rng = rng if rng is not None else np.random.default_rng()
    buckets: dict[str, list[dict]] = {t: [] for t in topos}
    drawn = 0
    rr = 0

    def _need() -> list[str]:
        return [t for t in topos if len(buckets[t]) < n_per_class]

    while drawn < pool_cap:
        need = _need()
        if not need:
            break
        target = need[rr % len(need)]
        rr += 1
        spec = sample_spec_for_topology(target, rng)
        label = scorer_fn(spec)
        drawn += 1
        if label not in buckets:
            # Oracle returned an unexpected class; ignore for stratification.
            continue
        if len(buckets[label]) < n_per_class:
            buckets[label].append({
                "spec": spec,
                "label": label,
                "target": target,
            })

    out: list[dict] = []
    for t in topos:
        out.extend(buckets[t][:n_per_class])
    return out
