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
import json
import os
import sys

# Ensure imports work from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import sim.ngspice_runner as ngspice_runner
import netlist.llm_netlist_gen as llm_netlist_gen
from specset.generate_specset import SPEC_BOUNDS
from specset.phaseshifter_scoring import TOPOLOGY_LABELS, score_topology
from specset import state_sampler
from env.exploration import (
    ExplorationConfig, compute_spec_bucket, make_bucket_key
)
from env.memory import Memory, MemoryConfig

SPEC_KEYS = list(SPEC_BOUNDS.keys())

# Per-state metric keys extracted from each ngspice run
PS_METRIC_KEYS = ["phase_deg", "il_db", "rl_db", "gain_err_db"]

# Minimum fraction of states that must simulate successfully for the episode
# to count. Below this, the env returns the metrics-missing failure reward.
_MIN_SUCCESS_FRACTION = 0.5

# Phase-error reward shape: smooth normalization against scale_deg, with a
# discrete bonus when spec target is met.
# scale_deg = 90 was chosen so an untrained agent at ~20-30 deg RMS error
# still gets meaningful gradient signal (~0.6-0.8 phase_term) rather than
# being crushed to zero by a 5-deg denominator.
_PHASE_SCALE_DEG = 90.0


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
                 memory: MemoryConfig | None = None):
        """
        Args:
            specset_path: path to the specset JSON. None uses default.
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
        """
        super().__init__()
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
            low=0.0, high=1.0, shape=(len(SPEC_KEYS),), dtype=np.float32
        )

        if specset_path is None:
            specset_path = os.path.join(
                os.path.dirname(__file__),
                "../specset/specset_phaseshifter.json"
            )

        try:
            with open(specset_path, "r") as f:
                self.dataset = json.load(f)
        except Exception as e:
            print(f"[Warning] Could not load specset, will use dummy spec on reset: {e}")
            self.dataset = []

        self.current_spec = None

    def _normalize(self, spec):
        """Normalize each spec field to [0, 1] using SPEC_BOUNDS."""
        vec = []
        for k in SPEC_KEYS:
            bnd = SPEC_BOUNDS[k]
            val = spec[k]
            if isinstance(bnd, tuple):
                mn, mx = bnd
                # Log-normalize fc_ghz and pmax_mw to match the sampler
                if k in ("fc_ghz", "pmax_mw"):
                    mn, mx = np.log10(mn), np.log10(mx)
                    val = np.log10(max(val, 1e-12))
                v = (val - mn) / (mx - mn)
                vec.append(max(0.0, min(1.0, v)))
            elif isinstance(bnd, list):
                # Categorical: normalize by max value of category set
                max_val = max(bnd) if max(bnd) > 0 else 1
                vec.append(float(val) / float(max_val))
        return np.array(vec, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.dataset:
            idx = self.np_random.integers(0, len(self.dataset))
            self.current_spec = self.dataset[idx]["spec"]
        else:
            self.current_spec = {
                "fc_ghz": 28.0, "bw_pct": 30.0, "phase_coverage_deg": 360.0,
                "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0,
                "max_il_db": 5.0, "min_rl_db": 10.0, "vdd": 1.8, "pmax_mw": 15.0,
                "tech": 0, "app": 2,
            }

        return self._normalize(self.current_spec), {}

    def _evaluate_netlist(self, netlist_path):
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

        sim_reward = self.compute_reward(agg, self.current_spec)
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
        """Reward shaping: bonus if chosen topology ranks high under heuristic scorer."""
        scores = {t: score_topology(t, spec) for t in TOPOLOGY_LABELS}
        sorted_topos = sorted(scores.items(), key=lambda x: -x[1])
        rank = [t for t, s in sorted_topos].index(topology_name)
        # Best topology: +0.3, worst: -0.1
        bonus = 0.3 - (rank / (len(TOPOLOGY_LABELS) - 1)) * 0.4
        return bonus

    def compute_reward(self, metrics, targets):
        """Multi-state phase-shifter reward.

        Smooth normalized penalty on RMS phase error + binary all_close
        bonus when spec target is met. Designed for gradient-friendly
        early-training signal even when the agent is far from the spec.

        Returns a float in [-1.0, 2.0]. -5.0 / -3.0 sentinels for total
        sim failure / metrics missing.
        """
        if metrics is None:
            return -5.0

        # Required keys for the aggregated multi-state metric dict.
        # Note: per_state and counts are info-only; not used here.
        required = ["rms_phase_err_deg", "il_db", "rl_db", "gain_err_db"]
        if any(metrics.get(k) is None for k in required):
            return -3.0

        # Per-metric weights — phase error dominates since it's the headline
        w_phase = 0.40
        w_il    = 0.25
        w_rl    = 0.20
        w_gain  = 0.15

        r = 0.0

        # Phase error: smooth normalized penalty against a wide scale_deg.
        # Gives meaningful gradient signal even when error is tens of degrees
        # away from the spec target (which would be e.g. 5 deg).
        # Spec target is used only for the all_close bonus below.
        m_phase = abs(metrics["rms_phase_err_deg"])
        r += w_phase * max(0.0, 1.0 - m_phase / _PHASE_SCALE_DEG)

        # Insertion loss: simulated IL vs max acceptable.
        # Do NOT use abs() — negative IL means active gain (unphysical for
        # passive circuits) and must not be rewarded as low loss.
        t_il = max(targets["max_il_db"], 0.1)
        m_il = metrics["il_db"]
        if m_il < 0.0:
            # Active gain: no IL credit (passive network constraint violated)
            pass
        else:
            r += w_il * max(0.0, 1.0 - m_il / t_il)

        # Return loss: simulated |S11| in dB. Saturates at 1.0 when met.
        t_rl = max(targets["min_rl_db"], 1.0)
        m_rl = abs(metrics["rl_db"])
        r += w_rl * max(0.0, min(1.0, m_rl / t_rl))

        # Gain (amplitude) error across states: std-dev of IL, in dB
        t_gain = max(targets["rms_gain_err_db"], 0.1)
        m_gain = abs(metrics["gain_err_db"])
        r += w_gain * max(0.0, 1.0 - m_gain / t_gain)

        # all_close bonus: rewards meeting the actual spec target (not
        # the wide scale_deg). Phase error against the spec target with
        # 20% slack on each metric.
        t_phase_target = max(targets["rms_phase_err_deg"], 0.1)
        all_close = (
            m_phase <= 1.2 * t_phase_target and
            0.0 <= m_il <= 1.2 * t_il       and
            m_rl    >= 0.8 * t_rl           and
            m_gain  <= 1.2 * t_gain
        )
        if all_close:
            r += 1.0

        return float(np.clip(r, -1.0, 2.0))