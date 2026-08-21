"""State table parsing and per-step state sampling for phase-shifter templates.

Each template embeds a STATE_TABLE comment block declaring the bit-count, the
ideal per-state phase increment, and a per-state list of parameter overrides
referencing other .PARAM symbols (Option B: ngspice resolves cross-references
natively at simulation time).

Sampling policy (two-tier, locked end of session 2):
  - bits <= 3: every step runs exhaustive (all 2**bits states)
  - bits >= 4: every step runs anchors + random subset; checkpoints run
    exhaustive (called from env at baseline / every N steps / final).

Per-step sampled budgets for bits >= 4:
    4-bit: 4 anchors + 4 random   = 8 of 16 states (50%)
    5-bit: 4 anchors + 6 random   = 10 of 32 states (~31%)
    6-bit: 8 anchors + 8 random   = 16 of 64 states (25%)

Anchors are the indices most diagnostic of phase-error monotonicity:
    4-bit: 0, 7, 8, 15           (extremes + MSB transition)
    5-bit: 0, 15, 16, 31         (extremes + MSB transition)
    6-bit: 0, 8, 16, 24, 32, 40, 48, 63 (eight states stepping the upper-3 bits)

Random states are sampled deterministically with seed = base_seed + step_count,
giving (a) reproducibility across runs with the same seed and (b) rotation
across steps so different states are explored over time.
"""

from __future__ import annotations
import re
import numpy as np


# ---------- Anchor table for sampled mode (bits >= 4) ----------

_ANCHORS_BY_BITS = {
    4: [0, 7, 8, 15],
    5: [0, 15, 16, 31],
    6: [0, 8, 16, 24, 32, 40, 48, 63],
}

# Per-step total budget (anchors + random) for sampled mode
_BUDGET_BY_BITS = {
    4: 8,
    5: 10,
    6: 16,
}


# ---------- Parser ----------

# Matches the opening and closing fences. Tolerant of leading whitespace
# and number of `=` characters in the fence (we look for the keyword).
_OPEN_FENCE  = re.compile(r"^\s*\*\s*=+\s*STATE_TABLE\s*=+\s*$", re.IGNORECASE)
_CLOSE_FENCE = re.compile(r"^\s*\*\s*=+\s*END_STATE_TABLE\s*=+\s*$", re.IGNORECASE)

# Match `* bits=N` or `* ideal_step_deg=X` (case insensitive, flexible spaces)
_BITS_LINE  = re.compile(r"^\s*\*\s*bits\s*=\s*(\d+)\s*$", re.IGNORECASE)
_STEP_LINE  = re.compile(r"^\s*\*\s*ideal_step_deg\s*=\s*([+\-0-9\.eE]+)\s*$",
                         re.IGNORECASE)

# Match `* state_K: key1=val1 key2=val2 ...  $ optional comment`
# - Index K is captured.
# - Body is everything after the colon up to an optional `$` comment.
_STATE_LINE = re.compile(
    r"^\s*\*\s*state_(\d+)\s*:\s*(.+?)\s*(?:\$.*)?$",
    re.IGNORECASE,
)

# Within a state body, match `key=value` tokens. Value can be either a
# brace-wrapped reference like {R_off} or a bare literal like 10k.
_KV_TOKEN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\{[^}]+\}|[^\s]+)"
)


def parse_state_table(template_text: str) -> dict:
    """Extract the STATE_TABLE block from a template's text.

    Returns a dict:
        {
            "bits":            int,    # 0 if no table found
            "ideal_step_deg":  float,  # 0.0 if no table found
            "states":          list of dicts,
                               # each dict maps param_name -> value_string,
                               # value_string is the raw token from the table
                               # (typically {RefName}, sometimes a literal).
        }

    A template with no STATE_TABLE block returns the empty default. This
    allows analog-continuous or future state-less templates to coexist
    with the multi-state runner without special-casing.

    The parser does NOT validate that the referenced parameters actually
    exist in the .PARAM lines of the template — that's a runtime concern
    handled by ngspice when it instantiates the netlist.
    """
    result = {"bits": 0, "ideal_step_deg": 0.0, "states": []}

    lines = template_text.splitlines()
    in_block = False
    states_by_index: dict[int, dict] = {}

    for raw in lines:
        if not in_block:
            if _OPEN_FENCE.match(raw):
                in_block = True
            continue

        if _CLOSE_FENCE.match(raw):
            break

        m = _BITS_LINE.match(raw)
        if m:
            result["bits"] = int(m.group(1))
            continue

        m = _STEP_LINE.match(raw)
        if m:
            try:
                result["ideal_step_deg"] = float(m.group(1))
            except ValueError:
                pass
            continue

        m = _STATE_LINE.match(raw)
        if m:
            idx = int(m.group(1))
            body = m.group(2)
            overrides = {}
            for km in _KV_TOKEN.finditer(body):
                overrides[km.group(1)] = km.group(2)
            states_by_index[idx] = overrides
            continue

        # Any other comment line inside the block is ignored.

    # Assemble states list ordered by index. Missing indices are skipped
    # silently; that's the template author's bug to discover via validation.
    if states_by_index:
        max_idx = max(states_by_index.keys())
        ordered = []
        for i in range(max_idx + 1):
            if i in states_by_index:
                ordered.append(states_by_index[i])
            else:
                ordered.append(None)  # placeholder for gap
        result["states"] = ordered

    return result


# ---------- Sampling policy ----------

def select_states(
    bits: int,
    total_states: int,
    step_count: int,
    mode: str = "auto",
    seed: int = 42,
) -> list[int]:
    """Choose which state indices to simulate this step.

    Args:
        bits: phase-shifter bit count from the spec (1, 2, 3 = exhaustive;
            4, 5, 6 = sampled by default).
        total_states: number of states actually declared in the template
            (typically 2**bits, but the parser is tolerant of mismatches).
        step_count: monotonically increasing step counter from the env;
            used both to vary which random states are drawn and to drive
            checkpoint detection (via is_checkpoint_step).
        mode: one of "auto" | "exhaustive" | "sampled".
            "auto" picks "exhaustive" for bits <= 3 and "sampled" otherwise.
            Callers (env) explicitly pass "exhaustive" at checkpoints.
        seed: base seed for the rng. Different seeds give different
            rotation patterns; same seed reproduces.

    Returns:
        Sorted list of state indices in [0, total_states).
    """
    if total_states <= 0:
        return []

    if mode == "auto":
        mode = "exhaustive" if bits <= 3 else "sampled"

    if mode == "exhaustive":
        return list(range(total_states))

    if mode == "sampled":
        anchors = _ANCHORS_BY_BITS.get(bits, [])
        # Clip anchors to the actually-declared state count, in case a
        # template under-specifies its table.
        anchors = [a for a in anchors if a < total_states]

        budget = _BUDGET_BY_BITS.get(bits, total_states)
        budget = min(budget, total_states)

        n_random = max(0, budget - len(anchors))
        candidates = [i for i in range(total_states) if i not in set(anchors)]

        if n_random == 0 or not candidates:
            return sorted(set(anchors))

        rng = np.random.default_rng(seed + step_count)
        n_random = min(n_random, len(candidates))
        picked = rng.choice(candidates, size=n_random, replace=False).tolist()

        return sorted(set(anchors) | set(picked))

    raise ValueError(f"Unknown mode: {mode}")


def is_checkpoint_step(step_count: int, ckpt_interval: int = 20) -> bool:
    """Whether this step should run exhaustive evaluation.

    Step 0 (baseline) is always a checkpoint. After that, every
    ckpt_interval steps. Callers (env) may also flip to exhaustive
    independently for "promising candidate" or "final" evaluations.
    """
    if step_count <= 0:
        return True
    return (step_count % ckpt_interval) == 0


# ---------- Self-tests ----------

def _run_self_tests() -> int:
    """Inline tests. Returns 0 on pass, nonzero on fail."""
    failures = 0

    def check(cond, label, detail=""):
        nonlocal failures
        if cond:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}  {detail}")
            failures += 1

    # --- Parser tests against synthesized templates ---
    print("\n[parser: switched_line]")
    sw_text = """* Switched-Line ...
* === STATE_TABLE ===
* bits=1
* ideal_step_deg=90.0
* state_0: R_short_in={R_on} R_short_out={R_on} R_long_in={R_off} R_long_out={R_off}
* state_1: R_short_in={R_off} R_short_out={R_off} R_long_in={R_on} R_long_out={R_on}
* === END_STATE_TABLE ===
.PARAM R_on=3 R_off=10k
.end
"""
    p = parse_state_table(sw_text)
    check(p["bits"] == 1, "bits parsed", f"got {p['bits']}")
    check(abs(p["ideal_step_deg"] - 90.0) < 1e-9, "ideal_step_deg parsed",
          f"got {p['ideal_step_deg']}")
    check(len(p["states"]) == 2, "2 states", f"got {len(p['states'])}")
    check(p["states"][0]["R_short_in"] == "{R_on}", "state_0 R_short_in",
          f"got {p['states'][0]}")
    check(p["states"][1]["R_long_out"] == "{R_on}", "state_1 R_long_out",
          f"got {p['states'][1]}")

    print("\n[parser: loaded_line]")
    ll_text = """* Loaded-Line ...
* === STATE_TABLE ===
* bits=1
* ideal_step_deg=22.5
* state_0: R_path_in={R_off} R_path_out={R_off}
* state_1: R_path_in={R_on} R_path_out={R_on}
* === END_STATE_TABLE ===
.PARAM R_on=3 R_off=10k
.end
"""
    p = parse_state_table(ll_text)
    check(p["bits"] == 1, "bits parsed", f"got {p['bits']}")
    check(abs(p["ideal_step_deg"] - 22.5) < 1e-9, "ideal_step_deg parsed",
          f"got {p['ideal_step_deg']}")
    check(len(p["states"]) == 2, "2 states")
    check(p["states"][0]["R_path_in"] == "{R_off}",
          "state_0 unloaded reference",
          f"got {p['states'][0]}")

    print("\n[parser: no STATE_TABLE block]")
    p = parse_state_table("* just some template\n.PARAM x=1\n.end\n")
    check(p["bits"] == 0, "empty template -> bits=0")
    check(p["ideal_step_deg"] == 0.0, "empty template -> step=0")
    check(p["states"] == [], "empty template -> no states")

    print("\n[parser: tolerates trailing $ comment on state line]")
    txt = """* === STATE_TABLE ===
* bits=1
* ideal_step_deg=180.0
* state_0: K=1  $ this is a comment
* state_1: K=2  $ another
* === END_STATE_TABLE ===
"""
    p = parse_state_table(txt)
    check(p["states"] == [{"K": "1"}, {"K": "2"}],
          "$ comment stripped", f"got {p['states']}")

    # --- Sampler tests ---
    print("\n[sampler: bits=1 auto -> exhaustive]")
    s = select_states(bits=1, total_states=2, step_count=0)
    check(s == [0, 1], "all 2 states", f"got {s}")

    print("\n[sampler: bits=3 auto -> exhaustive]")
    s = select_states(bits=3, total_states=8, step_count=5)
    check(s == [0, 1, 2, 3, 4, 5, 6, 7], "all 8 states")

    print("\n[sampler: bits=4 auto -> sampled, 8 of 16]")
    s = select_states(bits=4, total_states=16, step_count=0)
    check(len(s) == 8, "budget=8", f"got {len(s)}")
    check(set([0, 7, 8, 15]).issubset(set(s)), "anchors present",
          f"got {s}")

    print("\n[sampler: bits=5 auto -> sampled, 10 of 32]")
    s = select_states(bits=5, total_states=32, step_count=0)
    check(len(s) == 10, "budget=10", f"got {len(s)}")
    check(set([0, 15, 16, 31]).issubset(set(s)), "anchors present")

    print("\n[sampler: bits=6 auto -> sampled, 16 of 64]")
    s = select_states(bits=6, total_states=64, step_count=0)
    check(len(s) == 16, "budget=16", f"got {len(s)}")
    expected_anchors = {0, 8, 16, 24, 32, 40, 48, 63}
    check(expected_anchors.issubset(set(s)), "anchors present",
          f"missing: {expected_anchors - set(s)}")

    print("\n[sampler: determinism]")
    s1 = select_states(bits=6, total_states=64, step_count=10, seed=42)
    s2 = select_states(bits=6, total_states=64, step_count=10, seed=42)
    check(s1 == s2, "same seed+step -> same states")

    print("\n[sampler: rotation across steps]")
    s_a = select_states(bits=6, total_states=64, step_count=0, seed=42)
    s_b = select_states(bits=6, total_states=64, step_count=1, seed=42)
    check(s_a != s_b, "different steps -> different states",
          f"a={s_a}\n    b={s_b}")

    print("\n[sampler: explicit exhaustive overrides auto]")
    s = select_states(bits=6, total_states=64, step_count=5, mode="exhaustive")
    check(s == list(range(64)), "64 states forced")

    print("\n[sampler: handles total_states=0]")
    s = select_states(bits=0, total_states=0, step_count=0)
    check(s == [], "empty -> empty")

    # --- Checkpoint tests ---
    print("\n[checkpoint detection]")
    check(is_checkpoint_step(0) is True, "step 0 is checkpoint")
    check(is_checkpoint_step(1) is False, "step 1 not checkpoint")
    check(is_checkpoint_step(19) is False, "step 19 not checkpoint")
    check(is_checkpoint_step(20) is True, "step 20 is checkpoint")
    check(is_checkpoint_step(40) is True, "step 40 is checkpoint")
    check(is_checkpoint_step(7, ckpt_interval=5) is False,
          "step 7 with interval=5 not checkpoint")
    check(is_checkpoint_step(10, ckpt_interval=5) is True,
          "step 10 with interval=5 is checkpoint")

    return failures


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("state_sampler.py self-tests")
    print("=" * 60)
    n_fail = _run_self_tests()
    print()
    if n_fail == 0:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"{n_fail} TEST(S) FAILED")
        sys.exit(1)
