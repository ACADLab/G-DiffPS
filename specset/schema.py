"""Specset schema: versioning, observation encoding, atomic I/O.

SCHEMA_VERSION bumps whenever SPEC_BOUNDS keys, the observation layout, or the
encoding of any field changes. Loaders must assert equality; a mismatch raises
SchemaVersionError, not KeyError.

House convention — floor-relative sampling
------------------------------------------
Any spec field with a physical floor is sampled *relative to that floor*, never
from an absolute box. Sampling such a field independently wastes draws at both
ends simultaneously: the strict end is infeasible (no circuit can meet it) and
the lax end is under-determined (every circuit meets it), and no single box width
fixes both. Widening the box to raise the acceptance rate only trades the first
degeneracy for the second.

The pattern is: value = floor(conditioning fields) * kappa, with kappa drawn from
a fixed multiplier range. This gives zero rejections by construction, makes every
spec informative, and turns kappa into an explicit difficulty knob that can be
stratified on.

Three fields now follow it:
  max_area_mm2       kappa over the rank-ordered reference areas (T1.5c)
  rms_phase_err_deg  kappa over the quantization floor (S1, below)
and margin-stratified evaluation is the same idea applied to the reward gap.

The consequence for SPEC_BOUNDS is that a floor-relative field's entry is a
*normalization range only* — the min and max the rule can produce — not a
sampling box. It is derived, not chosen.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np

# v2: one-hot phase_bits/tech, coupled sampling, feasibility filter, heuristic
#     labels retired, disjoint train/eval pools.
# v3: rms_phase_err_deg is drawn floor-relative (kappa) instead of from a box, so
#     its normalization range and scale both changed. Loading a v2 file with v3
#     code would silently mis-encode that field, which is what the guard is for.
# v4: max_il_db, min_rl_db and rms_gain_err_db join rms_phase_err_deg on the
#     floor-relative convention, anchored on the achievable frontier in
#     specset/frontier_anchors.json. Their scales change, so v3 files would be
#     mis-encoded. The observation layout is *unchanged* -- the new kappas are
#     spec metadata, not conditioning dimensions, so SPEC_DIM stays 19 and
#     existing checkpoints keep loading.
# v5: electrical sampling windows are clipped to the netlist's physical limits
#     before drawing, so no action lands on a clamp boundary. This widens the
#     honestly-achievable set at low fc (72-77% of Switched_Line and
#     Switched_Filter draws used to clamp at 1.26 GHz), which moves the frontier
#     anchors and therefore every anchored target. Also: pmax_mw is now drawn
#     regime-first at ACTIVE_ALLOWED_FRACTION rather than from a bare box.
#     Observation layout still unchanged -- SPEC_DIM stays 19.
SCHEMA_VERSION = 5

# Frozen published-number snapshot (600 specs, schema v1 layout, no wrapper).
V1_FROZEN_PATH = os.path.join(os.path.dirname(__file__), "specset_v1_frozen.json")

# Canonical live pools. Reference these rather than repeating string literals:
# the previous arrangement kept a third file, specset_phaseshifter.json, that was
# a byte-identical copy of the train pool, so the two could drift silently and
# 20 MB of duplicate was headed for git history.
TRAIN_SPECSET_PATH = os.path.join(os.path.dirname(__file__), "specset_train.json")
EVAL_SPECSET_PATH = os.path.join(os.path.dirname(__file__), "specset_eval.json")

# ---------------------------------------------------------------------------
# Spec field bounds (JSON stores scalars; observation expands categoricals)
# ---------------------------------------------------------------------------

SPEC_BOUNDS = {
    "fc_ghz":             (1.0, 40.0),    # log-sampled within app band
    "bw_pct":             (5.0, 60.0),
    "phase_coverage_deg": (90.0, 360.0),
    "phase_bits":         [0, 3, 4, 5, 6],   # 0 = analog continuous
    # DERIVED normalization range, not a sampling box. Sampled floor-relative as
    # floor(bits, coverage) * kappa; see PHASE_KAPPA_RANGE and the module note.
    # Reachable span is [0.406*1.2, 12.99*3.0] = [0.49, 38.97]; log-normalized
    # because it covers ~2 decades.
    "rms_phase_err_deg":  (0.4, 40.0),
    "rms_gain_err_db":    (0.3, 3.0),
    "max_il_db":          (1.0, 10.0),
    "min_rl_db":          (8.0, 20.0),
    "vdd":                (0.8, 3.3),
    "pmax_mw":            (1.0, 50.0),    # log-sampled
    "tech":               [0, 1, 2],       # 0=PIN, 1=GaAs pHEMT, 2=SOI SPDT
    "app":                [0, 1, 2, 3],    # 0=sub-6, 1=FR1, 2=FR2/mmW, 3=Ku/Ka
    "max_area_mm2":       (5.0, 500.0),   # log-normalized; set by rank draw
}

# Keys as stored in each entry["spec"] dict (scalar categoricals).
SPEC_KEYS = list(SPEC_BOUNDS.keys())

# One-hot levels for nominal fields (S5). app stays ordinal (frequency-ordered).
PHASE_BITS_LEVELS = list(SPEC_BOUNDS["phase_bits"])  # [0, 3, 4, 5, 6]
TECH_LEVELS = list(SPEC_BOUNDS["tech"])              # [0, 1, 2]

# Observation layout: continuous/log fields + one-hot phase_bits + one-hot tech
# + ordinal app.  (13 scalars) - 2 ordinal cats + 5 + 3 = 19.
_SCALAR_KEYS = [k for k in SPEC_KEYS if k not in ("phase_bits", "tech")]
SPEC_DIM = (
    len([k for k in _SCALAR_KEYS if k != "app"])  # continuous/log
    + len(PHASE_BITS_LEVELS)                       # one-hot phase_bits
    + len(TECH_LEVELS)                             # one-hot tech
    + 1                                            # ordinal app
)
assert SPEC_DIM == 19, SPEC_DIM

# Field-role notes (S6 housekeeping):
#   bw_pct  — scored indirectly via heuristic baseline; also sets SPICE sweep
#             half-bandwidth in train_diffusion.rewrite_control_block. KEEP.
#   vdd     — deliberately unscored in r_sim (board supply is not an RF term);
#             retained so future power models can condition without a schema bump.
#   app     — rescued by S2 coupling to fc_ghz band.
#   pmax_mw — hard gate in physics_priors (not a continuous reward term).
#   tech    — sets switch parasitics under switch_model=realistic.

# app → fc_ghz band ranges (GHz), log-sampled within (S2).
APP_FC_BANDS = {
    0: (1.0, 6.0),     # sub-6
    1: (6.0, 20.0),    # FR1 / mid
    2: (20.0, 40.0),   # FR2 / mmWave
    3: (12.0, 40.0),   # Ku/Ka
}

# tech → plausible fc_ghz ranges (GHz). Outside → reject-and-resample (S2).
TECH_FC_BANDS = {
    0: (1.0, 18.0),    # PIN diode — microwave / low mmWave
    1: (1.0, 40.0),    # GaAs pHEMT — full band
    2: (1.0, 40.0),    # SOI SPDT — full band
}

FEASIBILITY_MARGIN = 1.2

# Difficulty multiplier over the quantization floor. kappa=1.2 sits just above
# the floor (hard but reachable); kappa=3.0 is comfortably slack. Log-uniform so
# difficulty is uniform in ratio, not in absolute degrees.
PHASE_KAPPA_RANGE = (1.2, 3.0)

# An analog shifter (phase_bits=0) has no quantization floor, but it is not
# floorless: it is expected to beat the finest digital grid in the pool. Using
# the 6-bit floor as its proxy keeps the field floor-relative everywhere and
# keeps analog specs scaling with coverage instead of falling back to a box.
ANALOG_PROXY_BITS = 6


# Difficulty multipliers for the three terms anchored on the achievable frontier
# rather than on an absolute box. Same role as PHASE_KAPPA_RANGE: kappa near 1
# sits on the frontier (hard), larger is slack. Loss and gain-error targets are
# the anchor *times* kappa; return loss is the anchor *divided by* kappa, since
# more return loss is better.
IL_KAPPA_RANGE = (1.5, 15.0)
RL_KAPPA_RANGE = (1.2, 3.0)
GAIN_KAPPA_RANGE = (1.5, 20.0)

# Absolute guards so an anchored target never lands somewhere unphysical or
# vacuous if a frontier value is pathological.
IL_TARGET_CLAMP_DB = (0.10, 6.0)
RL_TARGET_CLAMP_DB = (6.0, 30.0)
GAIN_TARGET_CLAMP_DB = (0.03, 3.0)

# Fraction of specs whose power budget admits an active stage. Sampled as a
# regime *first*, then pmax_mw within it, rather than left to fall out of an
# absolute box: the passive-only regime is where topology selection is actually
# contested, and an undesigned 77/23 split starves it. Same house convention as
# phase_kappa and the frontier anchors -- design the difficulty, don't inherit
# it from bounds nobody set for this purpose.
ACTIVE_ALLOWED_FRACTION = 0.5

FRONTIER_ANCHORS_PATH = os.path.join(
    os.path.dirname(__file__), "frontier_anchors.json")


def sample_pmax_mw(rng, active_allowed: bool, min_active_draw_mw: float) -> float:
    """Log-uniform pmax_mw drawn *within* the chosen regime.

    Splitting at the active topology's minimum draw makes regime membership
    exact rather than approximate, and keeps pmax_mw inside its declared bounds.
    """
    lo, hi = SPEC_BOUNDS["pmax_mw"]
    a, b = ((min_active_draw_mw, hi) if active_allowed
            else (lo, min_active_draw_mw))
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))

_ANCHOR_CACHE: dict | None = None


class SchemaVersionError(ValueError):
    """Raised when on-disk schema_version disagrees with SCHEMA_VERSION."""


class MissingAnchorsError(RuntimeError):
    """Raised when the achievable-frontier anchors have not been built."""


def load_frontier_anchors(path: str = FRONTIER_ANCHORS_PATH) -> dict:
    """Achievable IL/RL/gain frontier, keyed by fc bin and phase-quality grid.

    Built by `tools/compute_envelope.py frontier` from the envelope archive.
    Note the coupling this creates: the anchors are the best any topology in the
    *anchor set* attains, so adding or removing a topology shifts spec
    difficulty. That is intended, and it means the pools must be regenerated
    and re-reported whenever the topology set changes.
    """
    global _ANCHOR_CACHE
    if _ANCHOR_CACHE is None:
        if not os.path.exists(path):
            raise MissingAnchorsError(
                f"{path} not found. Build it with:\n"
                f"  python tools/compute_envelope.py archive\n"
                f"  python tools/compute_envelope.py frontier"
            )
        with open(path) as fh:
            _ANCHOR_CACHE = json.load(fh)
    return _ANCHOR_CACHE


def anchor_for(fc_ghz: float, rms_phase_err_deg: float,
               anchors: dict | None = None) -> dict:
    """Achievable (il_db, rl_db, gain_err_db) at this carrier and phase quality.

    Conditioned on phase quality because the terms trade against each other: a
    design held to tight phase error cannot also be the lowest-loss design in
    the box. Anchoring on the unconstrained extremum would price every loss
    target against a through-line that does not shift phase at all.
    """
    anchors = anchors or load_frontier_anchors()
    edges = anchors["fc_edges_ghz"]
    bi = int(np.clip(np.searchsorted(edges, float(fc_ghz), side="right") - 1,
                     0, len(anchors["anchors"]) - 1))
    by_phase = anchors["anchors"][str(bi)]["by_phase"]
    grid = sorted(float(k) for k in by_phase)
    # Nearest grid point at or above the spec's phase target; the frontier is
    # monotone in phase quality, so rounding up is the conservative direction.
    pick = next((g for g in grid if g >= float(rms_phase_err_deg)), grid[-1])
    return by_phase[f"{pick:g}"]


def sample_anchored_targets(
    fc_ghz: float, rms_phase_err_deg: float, rng, anchors: dict | None = None,
) -> tuple[dict, dict]:
    """Floor-relative IL/RL/gain targets. Returns (targets, kappas).

    Anchored on `switch_model='ideal'`. Generating a pool per switch model would
    make the two incomparable; with one pool, `realistic` is uniformly harder
    and that difficulty delta is a reportable quantity rather than a confound.
    """
    a = anchor_for(fc_ghz, rms_phase_err_deg, anchors)

    def _logu(rng, rng_range):
        lo, hi = rng_range
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    k_il = _logu(rng, IL_KAPPA_RANGE)
    k_rl = _logu(rng, RL_KAPPA_RANGE)
    k_gain = _logu(rng, GAIN_KAPPA_RANGE)

    il = float(np.clip(a["il_db"] * k_il, *IL_TARGET_CLAMP_DB))
    rl = float(np.clip(a["rl_db"] / k_rl, *RL_TARGET_CLAMP_DB))
    gain = float(np.clip(a["gain_err_db"] * k_gain, *GAIN_TARGET_CLAMP_DB))

    return (
        {"max_il_db": il, "min_rl_db": rl, "rms_gain_err_db": gain},
        {"il_kappa": k_il, "rl_kappa": k_rl, "gain_kappa": k_gain},
    )


def quantization_floor_deg(phase_bits: int, phase_coverage_deg: float) -> float | None:
    """RMS phase error floor from uniform quantization. None if analog (bits=0)."""
    bits = int(phase_bits)
    if bits == 0:
        return None
    step = float(phase_coverage_deg) / (2 ** bits)
    return step / np.sqrt(12.0)


def effective_phase_floor_deg(phase_bits: int, phase_coverage_deg: float) -> float:
    """Floor used for sampling: quantization floor, or the analog proxy."""
    bits = int(phase_bits)
    if bits == 0:
        bits = ANALOG_PROXY_BITS
    step = float(phase_coverage_deg) / (2 ** bits)
    return step / float(np.sqrt(12.0))


def sample_rms_phase_err_deg(
    phase_bits: int,
    phase_coverage_deg: float,
    rng,
    kappa_range: tuple[float, float] = PHASE_KAPPA_RANGE,
) -> tuple[float, float]:
    """Floor-relative draw. Returns (rms_phase_err_deg, kappa).

    kappa ~ LogUniform(kappa_range). Feasible by construction: the result is
    never below the quantization floor, so no rejection loop is needed.
    """
    floor = effective_phase_floor_deg(phase_bits, phase_coverage_deg)
    lo, hi = kappa_range
    kappa = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return floor * kappa, kappa


def is_feasible_phase_err(spec: dict, margin: float = FEASIBILITY_MARGIN) -> bool:
    """Invariant check. Under floor-relative sampling this always holds; kept as
    a postcondition so a regression in the sampler is caught at generation."""
    floor = quantization_floor_deg(spec["phase_bits"], spec["phase_coverage_deg"])
    if floor is None:
        return True
    return float(spec["rms_phase_err_deg"]) >= margin * floor


def atomic_write_json(path: str, obj: Any, indent: int = 2) -> None:
    """Write JSON via temp file + os.replace so a partial file is never loadable."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".specset_", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_specset(path: str, *, expect_version: int | None = SCHEMA_VERSION) -> list[dict]:
    """Load a specset list, asserting schema_version when the file is wrapped.

    Accepts:
      - bare list (v1 frozen / legacy) — only when expect_version is None or 1
      - {"schema_version": N, "pool": ..., "specs": [...]}  (v2+)
    """
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if expect_version is not None and expect_version != 1:
            raise SchemaVersionError(
                f"{path}: bare list (implicit schema_version=1) but code expects "
                f"schema_version={expect_version}. Pin published scripts to "
                f"specset_v1_frozen.json or regenerate to v{expect_version}."
            )
        return payload

    if not isinstance(payload, dict) or "specs" not in payload:
        raise SchemaVersionError(
            f"{path}: expected a list or a dict with 'specs'; got {type(payload).__name__}"
        )

    file_ver = payload.get("schema_version")
    if expect_version is not None and file_ver != expect_version:
        raise SchemaVersionError(
            f"{path}: schema_version={file_ver!r} but code expects "
            f"schema_version={expect_version}. Refusing to load (would KeyError "
            f"or silently mis-encode). Regenerate or pin to the matching freeze."
        )
    return list(payload["specs"])


def wrap_specset(specs: list[dict], *, pool: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "pool": pool,
        "n_specs": len(specs),
        "spec_dim": SPEC_DIM,
        "specs": specs,
    }


def assert_disjoint_pools(train_specs: list[dict], eval_specs: list[dict]) -> None:
    train_ids = {e["id"] for e in train_specs}
    eval_ids = {e["id"] for e in eval_specs}
    overlap = train_ids & eval_ids
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(
            f"train/eval spec pools overlap on {len(overlap)} ids "
            f"(e.g. {sample}). Disjointness is required by S4."
        )


def normalize_spec(spec: dict, *, schema_version: int = SCHEMA_VERSION) -> np.ndarray:
    """Map a scalar spec dict to a SPEC_DIM-vector in [0, 1].

    One-hots phase_bits and tech; log-scales fc_ghz / pmax_mw / max_area_mm2;
    leaves app ordinal. Missing fields raise a named error citing both versions.
    """
    if schema_version not in (1, SCHEMA_VERSION) and schema_version != SCHEMA_VERSION:
        # Allow callers to pass file version; encoding always follows current code.
        pass

    def _get(key: str):
        val = spec.get(key)
        if val is None:
            if key == "max_area_mm2":
                return 50.0  # back-compat for pre-T1.5
            raise KeyError(
                f"spec missing required key {key!r} "
                f"(file schema may be {schema_version}, code SCHEMA_VERSION="
                f"{SCHEMA_VERSION})."
            )
        return val

    vec: list[float] = []
    for k in SPEC_KEYS:
        if k == "phase_bits":
            level = int(_get(k))
            for lv in PHASE_BITS_LEVELS:
                vec.append(1.0 if level == lv else 0.0)
            continue
        if k == "tech":
            level = int(_get(k))
            for lv in TECH_LEVELS:
                vec.append(1.0 if level == lv else 0.0)
            continue

        bnd = SPEC_BOUNDS[k]
        val = _get(k)
        if isinstance(bnd, tuple):
            mn, mx = bnd
            if k in ("fc_ghz", "pmax_mw", "max_area_mm2", "rms_phase_err_deg"):
                mn, mx = np.log10(mn), np.log10(mx)
                val = np.log10(max(float(val), 1e-12))
            v = (float(val) - mn) / (mx - mn)
            vec.append(float(np.clip(v, 0.0, 1.0)))
        elif isinstance(bnd, list):
            # ordinal (app)
            max_val = max(bnd) if max(bnd) > 0 else 1
            vec.append(float(val) / float(max_val))
        else:
            raise TypeError(f"bad bound for {k}: {bnd!r}")

    out = np.asarray(vec, dtype=np.float32)
    if out.shape != (SPEC_DIM,):
        raise RuntimeError(f"normalize_spec produced shape {out.shape}, expected ({SPEC_DIM},)")
    return out
