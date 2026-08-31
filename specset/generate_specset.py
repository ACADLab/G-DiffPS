"""Generate SpecSet-PhaseShifter benchmark (v3).

I.i.d.-samples phase-shifter design specifications with default_rng(seed),
applying physically coupled field draws (S2) and floor-relative draws for the
two fields that have physical floors: rank-anchored area budgets (T1.5c) and
quantization-floor-anchored phase error (S1). Neither can produce an infeasible
spec, so there is no rejection loop for either — see the house convention note
in specset/schema.py.

Does NOT use score_topology argmax as ground truth (S0) — heuristic scores are
stored only under heuristic_topology_deprecated for the T5 baseline row.

Writes a schema-versioned wrapper (atomic temp+rename) so a partial file is
never loadable (S6).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specset.phaseshifter_scoring import score_topology, TOPOLOGY_LABELS  # noqa: E402
from sim.area_model import reference_areas_mm2  # noqa: E402
from sim.mna_scorer import VM_MIN_DRIVE_MW, topology_admits_spec
from specset.schema import (  # noqa: E402
    APP_FC_BANDS,
    FEASIBILITY_MARGIN,
    SCHEMA_VERSION,
    SPEC_BOUNDS,
    SPEC_DIM,
    SPEC_KEYS,
    PHASE_KAPPA_RANGE,
    TECH_FC_BANDS,
    assert_disjoint_pools,
    atomic_write_json,
    effective_phase_floor_deg,
    is_feasible_phase_err,
    ACTIVE_ALLOWED_FRACTION,
    sample_anchored_targets,
    sample_pmax_mw,
    sample_rms_phase_err_deg,
    wrap_specset,
)

# Re-export for back-compat imports (`from specset.generate_specset import SPEC_BOUNDS`)
__all__ = [
    "SPEC_BOUNDS",
    "SPEC_KEYS",
    "SPEC_DIM",
    "SCHEMA_VERSION",
    "N_SAMPLES_TRAIN",
    "N_SAMPLES_EVAL",
    "generate_samples",
    "rank_anchored_area_budget",
]

N_SAMPLES_TRAIN = 10_000
N_SAMPLES_EVAL = 2_000
N_TOPO = len(TOPOLOGY_LABELS)


def _sample_log(lo: float, hi: float, rng) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def rank_anchored_area_budget(fc_ghz: float, tech: int, rng) -> tuple[float, int]:
    """Draw r ~ U{1..N}, set max_area_mm2 = A_(r) * (1+eps)."""
    areas = reference_areas_mm2(fc_ghz, tech=tech)
    ordered = sorted(areas.values())
    n = len(ordered)
    r = int(rng.integers(1, n + 1))
    eps = float(rng.uniform(0.02, 0.10))
    max_area = float(ordered[r - 1] * (1.0 + eps))
    return max_area, r


def _sample_one(rng, *, max_tries: int = 200) -> tuple[dict, int, float, str, dict] | None:
    """Draw one feasible, coupled spec.

    Returns (spec, area_rank, kappas, heur_topo, scores).
    """
    for _ in range(max_tries):
        app = int(rng.choice(SPEC_BOUNDS["app"]))
        fc_lo, fc_hi = APP_FC_BANDS[app]
        fc = _sample_log(fc_lo, fc_hi, rng)

        tech = int(rng.choice(SPEC_BOUNDS["tech"]))
        t_lo, t_hi = TECH_FC_BANDS[tech]
        if not (t_lo <= fc <= t_hi):
            continue

        coverage = float(rng.uniform(*SPEC_BOUNDS["phase_coverage_deg"]))
        # Cap bits given coverage: reject bits>=5 with coverage<180 (S2).
        bits_choices = list(SPEC_BOUNDS["phase_bits"])
        if coverage < 180.0:
            bits_choices = [b for b in bits_choices if b < 5]
        phase_bits = int(rng.choice(bits_choices))

        # Regime first, then the budget within it: the passive-only regime is
        # where selection is contested, so its support is designed rather than
        # inherited from the pmax_mw box.
        active_allowed = bool(rng.random() < ACTIVE_ALLOWED_FRACTION)

        # Floor-relative, so feasible by construction (S1 + house convention).
        rms_phase, phase_kappa = sample_rms_phase_err_deg(phase_bits, coverage, rng)
        spec = {
            "fc_ghz": fc,
            "bw_pct": float(rng.uniform(*SPEC_BOUNDS["bw_pct"])),
            "phase_coverage_deg": coverage,
            "phase_bits": phase_bits,
            "rms_phase_err_deg": rms_phase,
            "vdd": float(rng.uniform(*SPEC_BOUNDS["vdd"])),
            "pmax_mw": sample_pmax_mw(rng, active_allowed, VM_MIN_DRIVE_MW),
            "tech": tech,
            "app": app,
        }
        # The remaining three S-parameter terms, anchored on the achievable
        # frontier at this carrier and phase quality (house convention, v4).
        anchored, term_kappas = sample_anchored_targets(fc, rms_phase, rng)
        spec.update(anchored)
        kappas_all = {"phase_kappa": phase_kappa, **term_kappas}

        # Postcondition, not a filter: floor-relative sampling cannot violate it.
        if not is_feasible_phase_err(spec, FEASIBILITY_MARGIN):
            raise AssertionError(
                f"floor-relative draw produced an infeasible spec "
                f"(bits={phase_bits}, cov={coverage:.1f}, "
                f"rms={rms_phase:.3f}, kappa={phase_kappa:.3f}) — "
                f"sample_rms_phase_err_deg has regressed"
            )

        max_area, area_rank = rank_anchored_area_budget(fc, tech, rng)
        spec["max_area_mm2"] = max_area

        scores = {t: score_topology(t, spec) for t in TOPOLOGY_LABELS}
        heur = max(scores, key=scores.get)
        return spec, area_rank, kappas_all, heur, scores
    return None


def generate_pool(
    n: int,
    rng,
    *,
    id_prefix: str,
    id_offset: int = 0,
) -> tuple[list[dict], dict]:
    """Generate n feasible specs. Returns (entries, stats)."""
    dataset: list[dict] = []
    attempts = 0
    kappas: list[float] = []
    while len(dataset) < n:
        attempts += 1
        result = _sample_one(rng)
        if result is None:
            raise RuntimeError(
                f"failed to sample a coupled spec after many tries "
                f"(have {len(dataset)}/{n}). Check TECH_FC_BANDS coverage."
            )
        spec, area_rank, spec_kappas, heur, scores = result
        kappas.append(spec_kappas["phase_kappa"])
        i = id_offset + len(dataset)
        entry = {
            "id": f"{id_prefix}_{i:05d}",
            "spec": spec,
            "area_rank": area_rank,
            # Difficulty knobs from floor-relative sampling; stratify on these.
            # Metadata only -- deliberately NOT observation dimensions, so
            # SPEC_DIM stays 19 and trained checkpoints keep loading.
            **spec_kappas,
            # Whether this spec's power budget can cover an active stage at all.
            # A (topology, spec)-only fact, so it is a property of the spec.
            "active_allowed": bool(
                topology_admits_spec("Vector_Modulator", spec)),
            "admissible": {
                t: topology_admits_spec(t, spec) for t in TOPOLOGY_LABELS},
            # S0: heuristic is baseline metadata, not a training label.
            "heuristic_topology_deprecated": heur,
            "heuristic_scores": {k: float(v) for k, v in scores.items()},
            # Envelope placeholders (filled by tools/compute_envelope.py / T4).
            "r_star": {"ideal": None, "realistic": None},
            "sizing_hints": {
                "varactor_c_max_pf": 1.0,
                "tline_zo_ohm": 50.0,
                "switch_w_um": 50.0,
            },
        }
        dataset.append(entry)

    # Floor rejections are now structurally zero. What is still worth reporting
    # is the counterfactual: how many draws the old independent-box sampler
    # would have thrown away, and how much of its accepted mass was vacuous
    # (rms already so far above the floor that any circuit satisfies it).
    probe_n = 5000
    probe_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    old_box = (1.0, 20.0)  # the hand-widened box this rule replaces
    rejects_tech = 0
    old_reject_floor = 0
    old_vacuous = 0
    n_scored = 0
    for _ in range(probe_n):
        app = int(probe_rng.choice(SPEC_BOUNDS["app"]))
        fc = _sample_log(*APP_FC_BANDS[app], probe_rng)
        tech = int(probe_rng.choice(SPEC_BOUNDS["tech"]))
        t_lo, t_hi = TECH_FC_BANDS[tech]
        if not (t_lo <= fc <= t_hi):
            rejects_tech += 1
            continue
        coverage = float(probe_rng.uniform(*SPEC_BOUNDS["phase_coverage_deg"]))
        bits_choices = list(SPEC_BOUNDS["phase_bits"])
        if coverage < 180.0:
            bits_choices = [b for b in bits_choices if b < 5]
        phase_bits = int(probe_rng.choice(bits_choices))
        floor = effective_phase_floor_deg(phase_bits, coverage)
        n_scored += 1
        old_draw = float(probe_rng.uniform(*old_box))
        if old_draw < FEASIBILITY_MARGIN * floor:
            old_reject_floor += 1
        elif old_draw > PHASE_KAPPA_RANGE[1] * floor:
            # Above the difficulty range entirely: satisfiable by anything.
            old_vacuous += 1

    kap = np.asarray(kappas)
    stats = {
        "n": n,
        "attempts_to_fill": attempts,
        "probe_n": probe_n,
        "probe_reject_tech_fc": rejects_tech / probe_n,
        # Floor-relative sampling: zero by construction, asserted per draw.
        "probe_reject_floor": 0.0,
        "probe_reject_rate": rejects_tech / probe_n,
        "counterfactual_independent_box": {
            "box": list(old_box),
            "reject_infeasible_frac": old_reject_floor / max(n_scored, 1),
            "vacuous_frac": old_vacuous / max(n_scored, 1),
            "note": "share of independent-box draws that were below the "
                    "quantization floor (infeasible) or above kappa_max*floor "
                    "(satisfiable by anything)",
        },
        "phase_kappa": {
            "range": list(PHASE_KAPPA_RANGE),
            "mean": float(kap.mean()),
            "p10": float(np.percentile(kap, 10)),
            "p50": float(np.percentile(kap, 50)),
            "p90": float(np.percentile(kap, 90)),
        },
        "rms_phase_err_deg_realized": {
            "min": float(min(e["spec"]["rms_phase_err_deg"] for e in dataset)),
            "p50": float(np.percentile(
                [e["spec"]["rms_phase_err_deg"] for e in dataset], 50)),
            "max": float(max(e["spec"]["rms_phase_err_deg"] for e in dataset)),
        },
        "feasibility_margin": FEASIBILITY_MARGIN,
        "schema_version": SCHEMA_VERSION,
        "spec_dim": SPEC_DIM,
    }
    return dataset, stats


def generate_samples(
    *,
    n_train: int = N_SAMPLES_TRAIN,
    n_eval: int = N_SAMPLES_EVAL,
    seed: int = 42,
    out_dir: str | None = None,
) -> dict:
    """Generate disjoint train/eval pools and write schema-versioned JSON."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(seed)

    train, train_stats = generate_pool(n_train, rng, id_prefix="train")
    eval_pool, eval_stats = generate_pool(
        n_eval, rng, id_prefix="eval", id_offset=0,
    )
    assert_disjoint_pools(train, eval_pool)

    # Unversioned names: schema_version inside the file is authoritative, and the
    # load-time guard catches mismatches. A versioned filename goes stale on bump.
    train_path = os.path.join(out_dir, "specset_train.json")
    eval_path = os.path.join(out_dir, "specset_eval.json")
    atomic_write_json(train_path, wrap_specset(train, pool="train"))
    atomic_write_json(eval_path, wrap_specset(eval_pool, pool="eval"))

    # Heuristic class marginals (for S0/+8 experiment comparison, not labels).
    from collections import Counter
    train_heur = Counter(e["heuristic_topology_deprecated"] for e in train)
    eval_heur = Counter(e["heuristic_topology_deprecated"] for e in eval_pool)

    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "train": {"path": train_path, "n": len(train), **train_stats,
                  "heuristic_marginals": dict(train_heur)},
        "eval": {"path": eval_path, "n": len(eval_pool), **eval_stats,
                 "heuristic_marginals": dict(eval_heur)},
        "app_fc_bands": {str(k): v for k, v in APP_FC_BANDS.items()},
        "tech_fc_bands": {str(k): v for k, v in TECH_FC_BANDS.items()},
        "field_roles": {
            "bw_pct": "KEEP — SPICE sweep half-bw + heuristic baseline",
            "vdd": "DELIBERATELY UNSCORED — retained for future power conditioning",
            "app": "KEEP — coupled to fc_ghz via APP_FC_BANDS",
            "tech": "KEEP — one-hot; sets switch parasitics under realistic",
        },
    }
    report_path = os.path.join(out_dir, "..", "results", "joint", "specset_generate.json")
    report_path = os.path.normpath(report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    atomic_write_json(report_path, report)

    print(f"train: {train_path} ({len(train)})")
    print(f"eval:  {eval_path} ({len(eval_pool)})")
    cf = train_stats["counterfactual_independent_box"]
    rk = train_stats["rms_phase_err_deg_realized"]
    print(f"probe reject rate: {train_stats['probe_reject_rate']:.1%} "
          f"(floor=0.0% by construction, "
          f"tech_fc={train_stats['probe_reject_tech_fc']:.1%})")
    print(f"floor-relative phase: rms in [{rk['min']:.2f}, {rk['max']:.2f}]° "
          f"(p50 {rk['p50']:.2f}°), kappa p10/p50/p90 = "
          f"{train_stats['phase_kappa']['p10']:.2f}/"
          f"{train_stats['phase_kappa']['p50']:.2f}/"
          f"{train_stats['phase_kappa']['p90']:.2f}")
    print(f"counterfactual independent box {cf['box']}: "
          f"{cf['reject_infeasible_frac']:.1%} infeasible + "
          f"{cf['vacuous_frac']:.1%} vacuous")
    print(f"report: {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=N_SAMPLES_TRAIN)
    ap.add_argument("--n-eval", type=int, default=N_SAMPLES_EVAL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    generate_samples(
        n_train=args.n_train, n_eval=args.n_eval, seed=args.seed, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
