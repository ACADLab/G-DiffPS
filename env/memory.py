"""Exploration layer — mechanism (b): memory of past sizing attempts.

Self-contained module for #36(b). Imported by phaseshifter_env (write path)
and llm_netlist_gen (read path). Like exploration.py, this module is pure
in the sense that it does not invoke ngspice or the LLM; it only reads/writes
its own JSONL store and computes similarity scores.

Design decisions (locked in session 4, B-design):
  - Unit of memory:    (spec, params, full_metrics_dict, reward, topology).
                       Q1(ii): metrics broken out so the LLM can see *why*
                       a past attempt scored, not just *that* it scored.
  - Similarity:        Hybrid. Filter to same topology + fc_band + phase_bits,
                       then cosine on spec_to_vec within the filtered set.
                       Q2(iv). Reuses session-3 _fc_band semantics.
  - Write policy:      Only the best-of-K selected attempt per env.step().
                       Q3(iii). Lowest write rate that still feeds retrieval.
                       Negative-example tagging is a clean extension.
  - Retrieval format:  Caller (llm_netlist_gen) renders top-K as a markdown
                       table. This module returns the structured top-K list;
                       formatting is the caller's concern. Q4(iii).
  - Persistence:       Append-only JSONL. One line per write. Q5(ii). Load
                       on __init__; rewrite never happens in v0.
  - Empty-on-no-match: strict. If the filter excludes every entry, return
                       []. The LLM prompt then gets no memory block, which
                       is the honest signal "new design problem, no prior."

Two engineering choices flagged for the deferred list:
  - #46 (new): spec_to_vec imported from netlist.llm_netlist_gen rather than
    duplicated. Soft env->netlist dependency; matches existing env imports.
  - #47 (new): full-load on __init__ is O(n) but trivial below ~10k entries.
    Revisit (mmap / sqlite / faiss) when production training run is sized.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Reuse the band labels from exploration so the hybrid filter agrees with
# the bucket grid. Single source of truth for "which band is this fc in?"
from env.exploration import _fc_band

# spec_to_vec is the canonical spec->numeric mapping; cosine similarity is
# computed on this vector. Imported (not duplicated) so spec schema changes
# propagate to exactly one place. See deferred #46.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from netlist.llm_netlist_gen import spec_to_vec  # noqa: E402


# A vanishingly small denominator guard for cosine. We never compare a
# zero-vector spec against anything in practice, but defensive against
# degenerate test inputs.
_EPS = 1e-12


@dataclass
class MemoryConfig:
    """Configuration for the memory layer (#36b).

    Default values (enabled=False) make the memory layer a no-op: env.step()
    skips the write path and llm_netlist_gen.generate() skips the read path.
    Like ExplorationConfig, this default reproduces session-3 baseline
    behavior exactly when enabled=False.

    Attributes:
        enabled: master switch. When False, all Memory operations short-
            circuit. read_top_k returns []; write is a no-op.
        path: filesystem path to the JSONL store. Appended one line per
            write. Loaded fully into memory on Memory.__init__. Relative
            paths are resolved against the cwd of the process.
        top_k: number of retrieved entries to return from read_top_k.
            v0 default is 3; matches the few-shot triplet pattern.
        min_reward_for_inclusion: writes below this threshold are skipped.
            -inf default = always write. Reserved for Q3(iv) extension
            (negative-example tagging) but not used in v0.
        bucket_keys: which fields define the hybrid filter. Hardcoded in
            v0 to ("topology", "fc_band", "phase_bits") and read directly
            in _hybrid_filter; this field is here for future config-driven
            redesign (see deferred #34) and is currently informational only.
    """
    enabled: bool = False
    path: str = "memory.jsonl"
    top_k: int = 3
    min_reward_for_inclusion: float = float("-inf")
    bucket_keys: tuple = ("topology", "fc_band", "phase_bits")

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError(f"MemoryConfig.top_k must be >= 1, got {self.top_k}")


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two numeric vectors of equal length.

    Returns 0.0 if either vector has zero norm (defensive; spec_to_vec
    will not produce a zero vector for any realistic spec, but unit tests
    occasionally do).
    """
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class Memory:
    """Append-only JSONL store of past sizing attempts with hybrid retrieval.

    Lifecycle:
      m = Memory(MemoryConfig(enabled=True, path="memory.jsonl"))
      m.write({"spec": ..., "params": ..., "metrics": ..., "reward": ...,
               "topology": "loaded_line"})
      hits = m.read_top_k(spec, topology, k=3)   # list of entries

    When config.enabled is False, write() is a no-op and read_top_k()
    returns []. Constructing a Memory with enabled=False does not touch
    the filesystem.
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config if config is not None else MemoryConfig()
        # In-memory cache. Each entry is a dict with at minimum:
        #   topology: str
        #   spec: dict (full spec, all fields preserved)
        #   params: dict (parameter name -> value)
        #   metrics: dict (rms_phase_err_deg, il_db, rl_db, gain_err_db, ...)
        #   reward: float
        # Optional keys (for future extensions) are passed through unchanged.
        self._entries: list[dict[str, Any]] = []

        if self.config.enabled and os.path.exists(self.config.path):
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Read the entire JSONL file into memory.

        Tolerant of malformed lines (skips with a stderr print, does not
        raise). This is deliberate: if a previous run crashed mid-write,
        the last line may be truncated; that should not prevent the
        current run from using everything before it.
        """
        try:
            with open(self.config.path, "r") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self._entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(
                            f"[memory] skipping malformed JSONL at "
                            f"{self.config.path}:{lineno}: {e}",
                            file=sys.stderr,
                        )
        except OSError as e:
            print(
                f"[memory] could not read {self.config.path}: {e}; "
                f"starting with empty store",
                file=sys.stderr,
            )

    def write(self, entry: dict[str, Any]) -> bool:
        """Append a single entry to the JSONL store and the in-memory cache.

        Returns True if the entry was written, False if it was skipped
        (memory disabled, or reward below min_reward_for_inclusion).

        Required keys in entry: topology (str), spec (dict), params (dict),
        metrics (dict | None), reward (float). Extra keys pass through.
        """
        if not self.config.enabled:
            return False

        required = ("topology", "spec", "params", "reward")
        missing = [k for k in required if k not in entry]
        if missing:
            raise ValueError(
                f"Memory.write: entry missing required keys: {missing}"
            )

        if entry["reward"] < self.config.min_reward_for_inclusion:
            return False

        # Append to in-memory cache first; then disk. If disk write fails,
        # the in-memory entry is still valid for the rest of this run.
        self._entries.append(entry)

        try:
            # Ensure directory exists. Cheap; useful when path has a dir
            # prefix the caller didn't create.
            parent = os.path.dirname(self.config.path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(self.config.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            print(
                f"[memory] disk write to {self.config.path} failed: {e}; "
                f"entry kept in memory only",
                file=sys.stderr,
            )

        return True

    def _hybrid_filter(self, spec: dict, topology: str) -> list[dict]:
        """Return entries with same topology + fc_band + phase_bits.

        Empty list if no entries match; the caller is responsible for the
        empty-on-no-match prompt behavior.
        """
        if "fc_ghz" not in spec or "phase_bits" not in spec:
            # Defensive: hybrid filter requires phase-shifter spec keys.
            # Return empty if the spec is missing them.
            return []

        target_band = _fc_band(spec["fc_ghz"])
        target_bits = spec["phase_bits"]

        matches = []
        for e in self._entries:
            if e.get("topology") != topology:
                continue
            e_spec = e.get("spec", {})
            if "fc_ghz" not in e_spec or "phase_bits" not in e_spec:
                continue
            if _fc_band(e_spec["fc_ghz"]) != target_band:
                continue
            if e_spec["phase_bits"] != target_bits:
                continue
            matches.append(e)
        return matches

    def read_top_k(
        self,
        spec: dict,
        topology: str,
        k: int | None = None,
    ) -> list[dict]:
        """Return up to k entries most similar to (spec, topology).

        Procedure:
          1. Filter to same topology + fc_band + phase_bits (strict).
          2. Score remaining entries by cosine similarity on spec_to_vec.
          3. Return top-k by similarity, descending. Ties broken by recency
             (later entries first), which is also descending by index.

        If memory is disabled or no entries pass the filter, returns [].

        Each returned entry is the original dict augmented with a
        '_similarity' field for caller inspection / debugging. The original
        in-memory entries are not mutated.
        """
        if not self.config.enabled or not self._entries:
            return []

        if k is None:
            k = self.config.top_k

        filtered = self._hybrid_filter(spec, topology)
        if not filtered:
            return []

        target_vec = spec_to_vec(spec)

        scored = []
        for idx, e in enumerate(filtered):
            sim = _cosine(target_vec, spec_to_vec(e["spec"]))
            # Annotate a shallow copy so callers get the score but the
            # store entries stay clean. Preserve insertion-order for tie
            # break (later entries win ties, hence -idx in sort key).
            scored.append((sim, -idx, e))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        top = []
        for sim, _, e in scored[:k]:
            out = dict(e)
            out["_similarity"] = sim
            top.append(out)
        return top

    def __len__(self) -> int:
        return len(self._entries)

    def all_entries(self) -> list[dict]:
        """Read-only view of the cache. Used by tests; not for env code."""
        return list(self._entries)
