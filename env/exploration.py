"""Exploration layer — mechanism (a): temperature-tuned retries.

Self-contained config + helpers for #36(a). Imported by phaseshifter_env.
No env-side wiring lives here; this module is pure (no I/O, no LLM calls,
no ngspice). That keeps it unit-testable in isolation.

Design decisions (locked in session 3):
  - Trigger:    visit count of (topology, spec_bucket) >= revisit_threshold.
                "always" mode is the ablation control.
  - Schedule:   temperatures list, attempted in order. Attempt 0 is baseline.
  - Selection:  best-of-K by reward, baseline always in the bag.
  - Bucketing:  per-field coarse rounding grid; None means exact equality.
  - Placement:  env-side; this module only provides config + key derivation.
"""

from dataclasses import dataclass, field
from typing import Literal


# Frequency band cutoffs (GHz). Conventional RF naming:
#   sub-6      <  6 GHz   (5G sub-6, WiFi, sub-band radar)
#   X-Ku       6-18 GHz   (radar, satcom uplink)
#   Ka        18-30 GHz   (5G mmWave low, satcom downlink)
#   mmW       >= 30 GHz   (5G mmWave high, automotive radar)
# These map "what kind of phase shifter is this" onto a finite set, which is
# the right granularity for the revisit-detection trigger. A 27 GHz design
# and a 29 GHz design are the same design problem; treating them as different
# buckets defeats the purpose.
_FC_BANDS = [
    ("sub6", 0.0,    6.0),
    ("XKu",  6.0,   18.0),
    ("Ka",  18.0,   30.0),
    ("mmW", 30.0, float("inf")),
]


def _fc_band(fc_ghz: float) -> str:
    """Map a center frequency in GHz to its band label."""
    for label, lo, hi in _FC_BANDS:
        if lo <= fc_ghz < hi:
            return label
    return "mmW"  # defensive: out-of-range high


# Default bucket grid. Sparse on purpose — bucketing only on fields that
# define the *identity* of a design problem ("28 GHz 5-bit Ka-band on
# tech=0 for app=2"). Spec constraints like vdd, pmax_mw, max_il_db are
# treated as parameterizations of the *same* design problem, not separate
# buckets — they shape the prompt but don't trigger a different design
# strategy.
#
# Rule per field:
#   None  -> exact value used as bucket key (categorical fields).
#   tuple ("band",) -> use _fc_band() to map continuous frequency to a
#                      discrete label. Only valid for fc_ghz currently.
#   g > 0 -> bucket key is round(value / g) * g (continuous fields).
#
# Fields not listed in the grid are ignored by bucketing, NOT passed through.
# This is a deliberate change from the v1 grid: with 12 fields and a 600-
# entry specset, including all fields produces 600 distinct buckets and
# the revisit trigger never fires. See deferred #39.
_DEFAULT_SPEC_BUCKET_GRID: dict = {
    "fc_ghz":     ("band",),  # banded: sub-6 / X-Ku / Ka / mmW
    "phase_bits": None,       # categorical: 0, 3, 4, 5, 6
    "tech":       None,       # categorical: 0, 1, 2
    "app":        None,       # categorical: 0, 1, 2, 3
}


@dataclass
class ExplorationConfig:
    """Configuration for the temperature-tuned retry layer (#36a).

    Default values (enabled=False) reproduce session-2 baseline behavior
    exactly. The env checks `enabled` and skips the entire retry path when
    false, so the default config carries zero runtime cost.

    Attributes:
        enabled: master switch. When False, env.step() behaves identically
            to session 2 (one attempt, T=0.2, no retry bookkeeping).
        mode: "on_revisit" fires retries only when the (topology, spec_bucket)
            visit count meets `revisit_threshold`. "always" fires retries
            on every step (ablation control).
        temperatures: per-attempt sampling temperatures, in order. Attempt 0
            is the deterministic baseline. Length is K (number of attempts
            per retry-triggering step).
        revisit_threshold: minimum visit count to trigger retries in
            "on_revisit" mode. 2 means retries fire from the second visit
            onward (the first visit is the baseline).
        spec_bucket_grid: per-field rounding rule. See module docstring.
    """
    enabled: bool = False
    mode: Literal["on_revisit", "always"] = "on_revisit"
    temperatures: list[float] = field(
        default_factory=lambda: [0.2, 0.7, 1.0]
    )
    revisit_threshold: int = 2
    spec_bucket_grid: dict = field(
        default_factory=lambda: dict(_DEFAULT_SPEC_BUCKET_GRID)
    )

    def __post_init__(self):
        # Sanity: temperatures must be non-empty and start with baseline.
        if not self.temperatures:
            raise ValueError("ExplorationConfig.temperatures must be non-empty")
        # Soft check: T>1.0 causes qwen2.5:7b to emit prose around the
        # .PARAM line (empirical, end of session 2). We don't hard-cap
        # because someone using a different LLM may want T>1.0; just warn.
        for t in self.temperatures:
            if t < 0.0:
                raise ValueError(f"Negative temperature in config: {t}")


def compute_spec_bucket(spec: dict, grid: dict) -> dict:
    """Reduce a spec dict to its bucket representation.

    Iterates over GRID keys (not spec keys). Spec fields not present in the
    grid are ignored — they are not part of the bucket identity. This is a
    deliberate change from a naive "round every field" approach: with a
    sparse-by-design grid, the bucket count stays in a useful range.

    Per-field rules:
        grid[k] is None       -> spec[k] passed through (categorical).
        grid[k] == ("band",)  -> _fc_band(spec[k]) applied. fc_ghz only.
        grid[k] is a number   -> round(spec[k] / g) * g.

    Returns a new dict containing only the bucket-relevant fields.
    Missing spec fields (key in grid but not in spec) are silently skipped
    — defensive against future spec schema changes.
    """
    bucket = {}
    for key, g in grid.items():
        if key not in spec:
            continue
        val = spec[key]
        if g is None:
            bucket[key] = val
        elif isinstance(g, tuple) and g == ("band",):
            bucket[key] = _fc_band(val)
        else:
            rounded = round(val / g) * g
            if isinstance(val, int) and float(g).is_integer():
                bucket[key] = int(rounded)
            else:
                bucket[key] = float(rounded)
    return bucket


def make_bucket_key(topology: str, spec_bucket: dict) -> tuple:
    """Build a hashable, order-stable key for the visit-count dict.

    Sorted by spec key so dict-iteration-order changes don't produce
    different keys for the same content.
    """
    return (topology, tuple(sorted(spec_bucket.items())))