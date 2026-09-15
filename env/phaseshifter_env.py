"""Phase-shifter RL environment.

Multi-state evaluation (added Step 4 of multi-state arc):
  - On step(), env parses the LLM-generated netlist's STATE_TABLE block,
    invokes the state sampler to choose a subset of states to simulate,
    runs ngspice once per selected state via param_overrides, and computes
    aggregated metrics across states:
      * rms_phase_err_deg: RMS of (measured_delta_phase - ideal_delta_phase),
        with wrap-around correction.
      * gain_err_db: standard deviation of il_db across states.
      * il_db, rl_db: mean across states (for monitoring; not headline).
  - The reward function uses rms_phase_err_deg with smooth normalization
    against scale_deg=90 (gradient-friendly when error is far from target),
    plus a discrete all_close bonus when the spec target is actually met.

Exploration layer (#36a, session 3):
  - Optional `exploration: ExplorationConfig` kwarg on __init__.
  - When enabled, step() runs K attempts at the configured temperatures
    and picks the best by reward (best-of-K selection).
  - Trigger: visit count of (topology, spec_bucket) >= revisit_threshold,
    or always-on for ablation. See ps_syn/env/exploration.py.
  - Default config (enabled=False) reproduces session-2 baseline exactly.

Decisions captured (end of session 2):
  - No env step counter yet; always sampled mode in env.step().
  - Exhaustive sweeps are final-validation-only, run externally.
  - Failure semantics: drop None states; full failure if <50% succeed.
  - State_0 in each STATE_TABLE is the phase reference; all phases are
    measured as deltas from state_0.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys

# Ensure imports work from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import sim.ngspice_runner as ngspice_runner
import netlist.llm_netlist_gen as llm_netlist_gen
from specset.schema import (
    TRAIN_SPECSET_PATH,
    SPEC_BOUNDS,
    SPEC_KEYS,
    SPEC_DIM,
    SCHEMA_VERSION,
    SchemaVersionError,
    assert_disjoint_pools,
    load_specset,
    normalize_spec,
)
from specset.phaseshifter_scoring import TOPOLOGY_LABELS, score_topology
from specset import state_sampler
from env.exploration import (
    ExplorationConfig, compute_spec_bucket, make_bucket_key
)
from env.memory import Memory, MemoryConfig
from env.reward import (
    PHASE_FLOOR_DEG, compute_sim_reward, phase_warmup_deg, WEIGHTS_AREA,
)

# Per-state metric keys extracted from each ngspice run
PS_METRIC_KEYS = ["phase_deg", "il_db", "rl_db", "gain_err_db"]

# Minimum fraction of states that must simulate successfully for the episode
# to count. Below this, the env returns the metrics-missing failure reward.
_MIN_SUCCESS_FRACTION = 0.5


def _wrap_180(deg: float) -> float:
    """Wrap a phase difference (degrees) into the half-open interval [-180, 180).

    Equivalent magnitudes at the boundary (180 vs -180) represent the same
    phase difference; the sign is convention-dependent but the squared
    error is unambiguous.
    """
    return ((deg + 180.0) % 360.0) - 180.0


def aggregate_state_metrics(
    state_indices: list[int],
    per_state_metrics: list[dict | None],
    ideal_step_deg: float,
) -> dict | None:
    """Reduce per-state metrics to a single aggregated dict.

    Args:
        state_indices: which state indices were simulated, same length and
            order as per_state_metrics.
        per_state_metrics: ngspice_runner.run output (list of dicts or None
            per state).
        ideal_step_deg: the per-state ideal phase increment from STATE_TABLE.

    Returns:
        Aggregated metric dict with:
            "rms_phase_err_deg": RMS phase error vs the ideal grid (degrees).
            "il_db": mean insertion loss across successful states.
            "rl_db": mean return loss across successful states.
            "gain_err_db": std-dev of il_db across successful states.
            "n_states_run": total states simulated.
            "n_states_succeeded": of which returned non-None metrics.
            "per_state": the raw list, for debugging/info.
        Returns None if fewer than _MIN_SUCCESS_FRACTION of states succeeded,
        or if the reference state (lowest index that succeeded) is missing
        a phase reading.
    """
    n_run = len(per_state_metrics)
    succ_pairs = [
        (idx, m)
        for idx, m in zip(state_indices, per_state_metrics)
        if m is not None and m.get("phase_deg") is not None
    ]

    n_succ = len(succ_pairs)
    if n_run == 0 or (n_succ / n_run) < _MIN_SUCCESS_FRACTION:
        return None

    # Reference state for delta computation: lowest successful state index.
    # State_0 is the canonical reference, but if it failed we fall back to
    # the next-lowest successful index. This keeps the math defined even
    # under partial failures.
    succ_pairs.sort(key=lambda p: p[0])
    ref_idx, ref_metrics = succ_pairs[0]
    ref_phase = ref_metrics["phase_deg"]

    # Compute wrapped delta from reference for each successful state.
    # The ideal delta for state i (relative to ref_idx) is (i - ref_idx) * ideal_step_deg.
    squared_errors = []
    il_values = []
    rl_values = []
    for idx, m in succ_pairs:
        measured_delta = _wrap_180(m["phase_deg"] - ref_phase)
        ideal_delta    = (idx - ref_idx) * ideal_step_deg
        # Wrap the ideal too so e.g. 6-bit cumulative ideal=360 maps to 0
        ideal_delta_wrapped = _wrap_180(ideal_delta)
        # Phase error is the wrapped residual
        err = _wrap_180(measured_delta - ideal_delta_wrapped)
        squared_errors.append(err * err)

        # IL is reported as positive dB by the templates (sign already
        # flipped in the testbench). Accumulate raw for mean and std.
        if m.get("il_db") is not None:
            il_values.append(m["il_db"])
        if m.get("rl_db") is not None:
            rl_values.append(m["rl_db"])

    rms_phase_err = float(np.sqrt(np.mean(squared_errors)))
    il_mean       = float(np.mean(il_values)) if il_values else 99.0
    rl_mean       = float(np.mean(rl_values)) if rl_values else 0.0
    # Standard deviation of insertion loss across states is the amplitude-
    # error definition standard in phase-shifter literature (RMS amplitude
    # variation).
    gain_err      = float(np.std(il_values)) if len(il_values) >= 2 else 0.0

    return {
        "rms_phase_err_deg": rms_phase_err,
        "il_db":             il_mean,
        "rl_db":             rl_mean,
        "gain_err_db":       gain_err,
        "n_states_run":      n_run,
        "n_states_succeeded": n_succ,
        "per_state":         per_state_metrics,
    }


class PhaseShifterEnv(gym.Env):
    def __init__(self, specset_path=None, restrict_to=None,
                 exploration: ExplorationConfig | None = None,
                 memory: MemoryConfig | None = None,
                 eval_specset_path=None,
                 pool: str = "train",
                 spec_ids=None,
                 expert_bonus_scale: float = 0.0):
        """
        Args:
            specset_path: path to the train specset JSON. None uses default.
            restrict_to: optional list of topology names to allow. If given,
                the action space is reduced to len(restrict_to) and the
                action index maps to the restricted list. Used for smoke
                tests when not all templates are implemented yet.
            exploration: optional ExplorationConfig for #36(a) temperature
                retries. None or default-disabled config reproduces
                session-2 baseline behavior (single attempt, T=0.2, no
                retry bookkeeping). When enabled, env.step() runs K
                attempts at the configured temperatures and selects the
                highest-reward result. See ps_syn/env/exploration.py.
            memory: optional MemoryConfig for #36(b) past-attempt memory.
                None or default-disabled config reproduces session-3
                behavior exactly: no read injected into prompts, no write
                at end of step. When enabled, env.step() writes the
                selected best-of-K attempt to JSONL, and llm_netlist_gen
                receives top-K similar past attempts as prompt priors.
                See ps_syn/env/memory.py.
            eval_specset_path: optional held-out eval pool. When set, loads
                and asserts disjointness vs the train pool at init.
            pool: which pool reset() samples from ("train" or "eval").
            spec_ids: optional allowlist of entry ids; filters the active
                pool after load.
            expert_bonus_scale: scale for heuristic topology-ranking bonus.
                Default 0.0 — physical reward only. Set 1.0 for the labeled
                ablation that restores the old expert bonus.
        """
        super().__init__()
        if pool not in ("train", "eval"):
            raise ValueError(f"pool must be 'train' or 'eval', got {pool!r}")
        self.pool = pool
        self.expert_bonus_scale = float(expert_bonus_scale)
        self.exploration = exploration if exploration is not None else ExplorationConfig()
        # Visit-count bookkeeping for the "on_revisit" trigger mode.
        # Keyed by (topology_name, sorted-tuple of bucketed spec). Survives
        # env.reset() — exploration is supposed to fire on revisits within
        # a training run, not within a single episode.
        self._visit_counts: dict[tuple, int] = {}

        # Memory layer (#36b). One instance per env lifetime. Disabled by
        # default; JSONL store is the durable state and is loaded on
        # construction when enabled and the path exists.
        self._memory_config = memory if memory is not None else MemoryConfig()
        self._memory = Memory(self._memory_config)

        # Build the active topology list (full set or restricted subset)
        if restrict_to is not None:
            self._active_topologies = list(restrict_to)
            for t in self._active_topologies:
                if t not in TOPOLOGY_LABELS:
                    raise ValueError(f"Unknown topology in restrict_to: {t}")
        else:
            self._active_topologies = list(TOPOLOGY_LABELS)

        self.action_space = spaces.Discrete(len(self._active_topologies))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(SPEC_DIM,), dtype=np.float32
        )

        if specset_path is None:
            specset_path = TRAIN_SPECSET_PATH

        self._specset_path = specset_path
        self._specset_load_error = None
        try:
            self.dataset = load_specset(specset_path)
        except SchemaVersionError as e:
            raise SchemaVersionError(
                f"Train specset schema mismatch ({specset_path}): {e}"
            ) from e
        except Exception as e:
            # Keep construction working for eval scripts that only need
            # normalize/expert_bonus. reset() must not invent a dummy spec.
            self.dataset = []
            self._specset_load_error = (
                f"Could not load specset {specset_path}: {e}. "
                "Generate it with `python specset/generate_specset.py` "
                "or pass specset_path to a tracked file."
            )
            print(f"[Warning] {self._specset_load_error}")

        self.eval_dataset: list = []
        if eval_specset_path is not None:
            try:
                self.eval_dataset = load_specset(eval_specset_path)
            except SchemaVersionError as e:
                raise SchemaVersionError(
                    f"Eval specset schema mismatch ({eval_specset_path}): {e}"
                ) from e
            if self.dataset:
                assert_disjoint_pools(self.dataset, self.eval_dataset)

        if spec_ids is not None:
            allow = set(spec_ids)
            if self.pool == "train":
                self.dataset = [e for e in self.dataset if e.get("id") in allow]
            else:
                self.eval_dataset = [
                    e for e in self.eval_dataset if e.get("id") in allow
                ]

        self.current_spec = None
        self.current_spec_id = None
        self.current_entry = None

    def _normalize(self, spec):
        """Normalize each spec field to [0, 1] (schema v2 observation layout)."""
        return normalize_spec(spec)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        active = self.dataset if self.pool == "train" else self.eval_dataset
        if not active:
            which = "dataset" if self.pool == "train" else "eval_dataset"
            extra = f" {self._specset_load_error}" if self._specset_load_error else ""
            raise RuntimeError(
                f"PhaseShifterEnv.reset() has an empty {which} "
                f"(pool={self.pool!r}). Refusing to serve a dummy spec.{extra}"
            )
        idx = self.np_random.integers(0, len(active))
        entry = active[idx]
        self.current_entry = entry
        self.current_spec_id = entry.get("id")
        self.current_spec = entry["spec"]

        return self._normalize(self.current_spec), {}

    def _evaluate_netlist(self, netlist_path, warmup_deg: float = 0.0):
        """Run multi-state simulation on a netlist and compute its reward.

        Extracted from step() so the retry loop can call it K times. Pure
        — does not touch self state, does not log, does not clean up.
        Caller owns the temp netlist's lifecycle.

        Returns:
            (agg_metrics, sim_reward, state_indices, bits, ideal_step_deg)
              - agg_metrics: aggregated multi-state metric dict, or None
                on failure (matches compute_reward's None contract).
              - sim_reward: float in [-5.0, 2.0] from compute_reward.
              - state_indices: which state indices were simulated (info).
              - bits, ideal_step_deg: parsed from STATE_TABLE (info).
        """
        try:
            with open(netlist_path) as f:
                netlist_text = f.read()
        except Exception as e:
            print(f"[Warning] Could not read netlist {netlist_path}: {e}")
            netlist_text = ""

        table = state_sampler.parse_state_table(netlist_text)
        bits = table["bits"]
        ideal_step_deg = table["ideal_step_deg"]
        all_states = table["states"]
        total_states = len(all_states)

        # Single-state fallback for templates without STATE_TABLE.
        if total_states == 0:
            agg = ngspice_runner.run(netlist_path, metric_keys=PS_METRIC_KEYS)
            state_indices: list[int] = []
        else:
            state_indices = state_sampler.select_states(
                bits=bits, total_states=total_states,
                step_count=0, mode="auto",
            )
            overrides_list = [all_states[i] for i in state_indices]
            per_state = ngspice_runner.run(
                netlist_path, metric_keys=PS_METRIC_KEYS,
                param_overrides=overrides_list,
            )
            agg = aggregate_state_metrics(state_indices, per_state, ideal_step_deg)

        # Board-level area for WEIGHTS_AREA (T1.5).
        if agg is not None:
            try:
                from sim.area_model import estimate_area_mm2
                # Prefer sized params if caller stashed them; else nominal.
                params = getattr(self, "_last_params", None) or {}
                agg["area_mm2"] = estimate_area_mm2(
                    getattr(self, "_last_topology", "Loaded_Line"),
                    params if params else None,
                    fc_ghz=float(self.current_spec.get("fc_ghz", 28.0)),
                    tech=int(self.current_spec.get("tech", 0)),
                )
            except Exception:
                agg["area_mm2"] = None

        sim_reward = self.compute_reward(agg, self.current_spec, warmup_deg=warmup_deg)
        return agg, sim_reward, state_indices, bits, ideal_step_deg

    def step(self, action):
        topology_name = self._active_topologies[action]

        # Visit-count bookkeeping. Increment BEFORE the trigger check so a
        # spec's second visit gets visit_count=2 and fires the trigger.
        spec_bucket = compute_spec_bucket(
            self.current_spec, self.exploration.spec_bucket_grid
        )
        bucket_key = make_bucket_key(topology_name, spec_bucket)
        self._visit_counts[bucket_key] = self._visit_counts.get(bucket_key, 0) + 1
        visit_count = self._visit_counts[bucket_key]

        # Decide which temperatures to attempt this step.
        # Attempt 0 (baseline T=0.2) is always present. Higher temperatures
        # fire only when the exploration layer is enabled AND the trigger
        # mode requires it.
        if not self.exploration.enabled:
            temps_to_try = [self.exploration.temperatures[0]]
        elif self.exploration.mode == "always":
            temps_to_try = list(self.exploration.temperatures)
        elif self.exploration.mode == "on_revisit":
            if visit_count >= self.exploration.revisit_threshold:
                temps_to_try = list(self.exploration.temperatures)
            else:
                temps_to_try = [self.exploration.temperatures[0]]
        else:
            raise ValueError(f"Unknown exploration mode: {self.exploration.mode}")

        # Expert bonus is topology-and-spec dependent only; identical
        # across attempts on this step. Compute once.
        expert_bonus = self.compute_expert_bonus(topology_name, self.current_spec)

        # Memory read (#36b): fetch top-K past attempts for this (spec,
        # topology) once per step. All K best-of-K attempts on this step
        # share the same memory hits — memory is about *what to try*, not
        # *which temperature*. read_top_k returns [] when memory is
        # disabled OR when no past attempts match the hybrid filter.
        memory_hits = self._memory.read_top_k(self.current_spec, topology_name)

        # Run each attempt. Each generates its own netlist, simulates,
        # and computes its own reward. We keep all netlists for cleanup
        # at the end of the step.
        retry_log = []
        netlists_to_clean = []

        for attempt_idx, temperature in enumerate(temps_to_try):
            netlist_path, params_dict = llm_netlist_gen.generate(
                self.current_spec, topology_name,
                temperature=temperature,
                memory_hits=memory_hits,
            )
            netlists_to_clean.append(netlist_path)

            # Area term (WEIGHTS_AREA) reads these; must match the netlist under test.
            self._last_topology = topology_name
            self._last_params = params_dict or {}

            agg, sim_reward, state_indices, bits, ideal_step_deg = (
                self._evaluate_netlist(netlist_path)
            )
            total_reward = sim_reward + expert_bonus

            # Compact metrics summary for the retry log (full dict kept
            # in `info["metrics"]` for the selected attempt only).
            if agg is not None:
                metrics_summary = {
                    k: agg.get(k) for k in
                    ("rms_phase_err_deg", "il_db", "rl_db", "gain_err_db")
                }
            else:
                metrics_summary = None

            retry_log.append({
                "attempt": attempt_idx,
                "temperature": temperature,
                "ngspice_succeeded": agg is not None,
                "sim_reward": sim_reward,
                "total_reward": total_reward,
                "metrics_summary": metrics_summary,
                # Keep these for the selected attempt's info reconstruction
                "_netlist_path": netlist_path,
                "_agg": agg,
                "_state_indices": state_indices,
                "_bits": bits,
                "_ideal_step_deg": ideal_step_deg,
                # Captured for memory write (#36b). Always present even
                # when ngspice failed; the LLM's parameter choice is
                # itself a signal worth remembering as a negative example.
                "_params": params_dict,
            })

        # Best-of-K selection: highest total_reward wins. Ties broken by
        # lower attempt index (baseline preferred when tied). This is the
        # monotonic-non-regression guarantee from the design note.
        best_idx = max(
            range(len(retry_log)),
            key=lambda i: (retry_log[i]["total_reward"], -i),
        )
        best = retry_log[best_idx]

        reward = best["total_reward"]
        agg = best["_agg"]
        state_indices = best["_state_indices"]
        bits = best["_bits"]
        ideal_step_deg = best["_ideal_step_deg"]
        sim_reward = best["sim_reward"]
        netlist_path = best["_netlist_path"]

        done = True
        truncated = False

        # Memory write (#36b): persist the selected best-of-K attempt.
        # ngspice failures get reward=-1.0 (the failure floor) and
        # metrics=None — they are kept as negative examples. Memory.write
        # is a no-op when memory is disabled.
        if best["ngspice_succeeded"]:
            mem_metrics = best["metrics_summary"]
            mem_reward = reward
        else:
            mem_metrics = None
            mem_reward = -1.0
        memory_wrote = self._memory.write({
            "topology": topology_name,
            "spec": dict(self.current_spec),
            "params": dict(best["_params"]) if best["_params"] else {},
            "metrics": mem_metrics,
            "reward": mem_reward,
            "temperature": best["temperature"],
            "ngspice_succeeded": best["ngspice_succeeded"],
        })

        # Strip private fields from retry_log before exposing in info.
        public_retry_log = [
            {k: v for k, v in entry.items() if not k.startswith("_")}
            for entry in retry_log
        ]

        info = {
            "metrics": agg,
            "netlist": netlist_path,
            "topology": topology_name,
            "sim_reward": sim_reward,
            "expert_bonus": expert_bonus,
            "bits": bits,
            "ideal_step_deg": ideal_step_deg,
            "state_indices": state_indices,
            # Selected attempt's parameters as emitted by the LLM (post
            # clamp, pre netlist write). Always present; useful for
            # downstream callbacks (memory introspection, learning-curve
            # logging, ablation analysis) that need to know what the LLM
            # actually wrote without re-parsing the netlist file (which
            # the env cleans up before returning).
            "params": dict(best["_params"]) if best["_params"] else {},
            # Exploration layer (#36a). Always present so downstream
            # callbacks have a uniform schema regardless of enabled state.
            "exploration": {
                "enabled": self.exploration.enabled,
                "mode": self.exploration.mode if self.exploration.enabled else None,
                "spec_bucket": spec_bucket,
                "visit_count": visit_count,
                "n_attempts": len(temps_to_try),
                "selected_attempt": best_idx,
                "retry_log": public_retry_log,
            },
            # Memory layer (#36b). Always present so callbacks have a
            # uniform schema regardless of enabled state.
            "memory": {
                "enabled": self._memory_config.enabled,
                "n_hits_injected": len(memory_hits),
                "wrote": memory_wrote,
                "store_size": len(self._memory),
            },
        }

        # Cleanup: every attempt's temp netlist and ngspice .lis dump.
        for path in netlists_to_clean:
            try:
                if os.path.exists(path):
                    os.remove(path)
                lis_path = f"{path}.lis"
                if os.path.exists(lis_path):
                    os.remove(lis_path)
            except Exception:
                pass

        return self._normalize(self.current_spec), reward, done, truncated, info

    def compute_expert_bonus(self, topology_name, spec):
        """Heuristic topology-ranking bonus (ablation only; default disabled).

        Physical training reward must not include this term. Callers pass
        ``expert_bonus_scale`` (0.0 default) to gate it.
        """
        scale = float(getattr(self, "expert_bonus_scale", 0.0))
        if scale == 0.0:
            return 0.0
        scores = {t: score_topology(t, spec) for t in TOPOLOGY_LABELS}
        sorted_topos = sorted(scores.items(), key=lambda x: -x[1])
        rank = [t for t, s in sorted_topos].index(topology_name)
        # Best topology: +0.3, worst: -0.1  (then scaled)
        bonus = 0.3 - (rank / (len(TOPOLOGY_LABELS) - 1)) * 0.4
        return float(scale) * bonus

    def compute_reward(self, metrics, targets, warmup_deg: float = 0.0):
        """Delegate to env.reward.compute_sim_reward (single implementation)."""
        return compute_sim_reward(
            metrics, targets, weights=WEIGHTS_AREA, warmup_deg=warmup_deg,
        )
