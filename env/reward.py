"""Single source of truth for the simulation reward.

Both `PhaseShifterEnv.compute_reward` and `sim.mna_scorer.score_from_metrics`
delegate here so the two paths cannot drift.

Phase denominator is spec-relative with an annealed warmup floor:
  denom = max(s.rms_phase_err_deg, warmup_deg, PHASE_FLOOR_DEG)
warmup_deg anneals 45 → 0 over the first WARMUP_STEPS of training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

PHASE_FLOOR_DEG = 1.0
WARMUP_START_DEG = 45.0
WARMUP_STEPS = 2000


@dataclass(frozen=True)
class RewardWeights:
    phase: float = 0.35
    il: float = 0.20
    rl: float = 0.15
    gain: float = 0.15
    power: float = 0.15  # removed in T1.5d; kept for T1 continuity
    area: float = 0.0

    def __post_init__(self):
        s = self.phase + self.il + self.rl + self.gain + self.power + self.area
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"RewardWeights must sum to 1.0, got {s}")


# T1 default (includes dead power term; T1.5d replaces this).
WEIGHTS_T1 = RewardWeights()

# T1.5d / production: five RF+area terms, no continuous power.
WEIGHTS_AREA = RewardWeights(
    phase=0.32, il=0.22, rl=0.16, gain=0.12, power=0.0, area=0.18,
)


def phase_warmup_deg(step: Optional[int]) -> float:
    """Anneal 45° → 0° over WARMUP_STEPS; 0 if step is None (eval / MNA)."""
    if step is None:
        return 0.0
    if step >= WARMUP_STEPS:
        return 0.0
    return WARMUP_START_DEG * (1.0 - float(step) / float(WARMUP_STEPS))


def phase_denom(targets: dict, warmup_deg: float = 0.0) -> float:
    t = float(targets.get("rms_phase_err_deg", PHASE_FLOOR_DEG) or PHASE_FLOOR_DEG)
    return max(t, float(warmup_deg), PHASE_FLOOR_DEG)


def compute_sim_reward(
    metrics: Optional[dict],
    targets: dict,
    weights: RewardWeights = WEIGHTS_AREA,
    warmup_deg: float = 0.0,
    return_parts: bool = False,
):
    """Spec-relative soft margins + all_close bonus.

    Returns float in [-1, 2], or (reward, parts_dict) if return_parts.
    Sentinels: -5.0 if metrics is None; -3.0 if required keys missing.
    Default weights are WEIGHTS_AREA (T1.5d production). Pass WEIGHTS_T1
    explicitly for the pre-area ablation.
    """
    if metrics is None:
        return (-5.0, {"sentinel": -5.0}) if return_parts else -5.0

    required = ["rms_phase_err_deg", "il_db", "rl_db", "gain_err_db"]
    if any(metrics.get(k) is None for k in required):
        return (-3.0, {"sentinel": -3.0}) if return_parts else -3.0

    parts = {}
    r = 0.0

    m_phase = abs(float(metrics["rms_phase_err_deg"]))
    denom = phase_denom(targets, warmup_deg)
    g_phase = max(0.0, 1.0 - m_phase / denom)
    parts["g_phase"] = g_phase
    parts["phase_denom"] = denom
    r += weights.phase * g_phase

    t_il = max(float(targets.get("max_il_db", 5.0)), 0.1)
    m_il = float(metrics["il_db"])
    if m_il < 0.0:
        g_il = 0.0
    else:
        g_il = max(0.0, 1.0 - m_il / t_il)
    parts["g_il"] = g_il
    r += weights.il * g_il

    t_rl = max(float(targets.get("min_rl_db", 10.0)), 1.0)
    m_rl = abs(float(metrics["rl_db"]))
    g_rl = max(0.0, min(1.0, m_rl / t_rl))
    parts["g_rl"] = g_rl
    r += weights.rl * g_rl

    t_gain = max(float(targets.get("rms_gain_err_db", 1.0)), 0.1)
    m_gain = abs(float(metrics["gain_err_db"]))
    g_gain = max(0.0, 1.0 - m_gain / t_gain)
    parts["g_gain"] = g_gain
    r += weights.gain * g_gain

    if weights.power > 0.0:
        t_pwr = max(float(targets.get("pmax_mw", 15.0)), 0.1)
        m_pwr = metrics.get("pwr_mw")
        m_pwr = 0.0 if m_pwr is None else abs(float(m_pwr))
        g_power = max(0.0, 1.0 - m_pwr / t_pwr)
        parts["g_power"] = g_power
        r += weights.power * g_power
    else:
        t_pwr = max(float(targets.get("pmax_mw", 15.0)), 0.1)
        m_pwr = metrics.get("pwr_mw")
        m_pwr = 0.0 if m_pwr is None else abs(float(m_pwr))
        parts["g_power"] = None

    if weights.area > 0.0:
        t_area = max(float(targets.get("max_area_mm2", 1e9)), 1e-6)
        m_area = metrics.get("area_mm2")
        if m_area is None:
            g_area = 0.0
        else:
            g_area = max(0.0, min(1.0, 1.0 - float(m_area) / t_area))
        parts["g_area"] = g_area
        r += weights.area * g_area

    t_phase_target = max(float(targets.get("rms_phase_err_deg", 5.0)), 0.1)
    all_close = (
        m_phase <= 1.2 * t_phase_target
        and 0.0 <= m_il <= 1.2 * t_il
        and m_rl >= 0.8 * t_rl
        and m_gain <= 1.2 * t_gain
        and m_pwr <= 1.2 * t_pwr
    )
    if weights.area > 0.0:
        t_area = max(float(targets.get("max_area_mm2", 1e9)), 1e-6)
        m_area = float(metrics.get("area_mm2") or 1e18)
        all_close = all_close and (m_area <= 1.2 * t_area)
    parts["all_close"] = all_close
    if all_close:
        r += 1.0

    reward = float(np.clip(r, -1.0, 2.0))
    parts["reward"] = reward
    # Raw (no warmup) for ablation logging.
    if warmup_deg > 0.0:
        raw = compute_sim_reward(
            metrics, targets, weights=weights, warmup_deg=0.0, return_parts=False,
        )
        parts["reward_raw"] = raw
    else:
        parts["reward_raw"] = reward

    if return_parts:
        return reward, parts
    return reward


def strict_compliance(metrics: Optional[dict], targets: dict) -> bool:
    """Every hard RF (+ area if present) threshold met — no tolerance margin."""
    if metrics is None:
        return False
    required = ["rms_phase_err_deg", "il_db", "rl_db", "gain_err_db"]
    if any(metrics.get(k) is None for k in required):
        return False
    if abs(float(metrics["rms_phase_err_deg"])) > float(targets.get("rms_phase_err_deg", 5.0)):
        return False
    if float(metrics["il_db"]) > float(targets.get("max_il_db", 5.0)):
        return False
    if abs(float(metrics["rl_db"])) < float(targets.get("min_rl_db", 10.0)):
        return False
    if abs(float(metrics["gain_err_db"])) > float(targets.get("rms_gain_err_db", 1.0)):
        return False
    if "max_area_mm2" in targets and metrics.get("area_mm2") is not None:
        if float(metrics["area_mm2"]) > float(targets["max_area_mm2"]):
            return False
    return True


def tolerant_all_close(metrics: Optional[dict], targets: dict,
                       weights: RewardWeights = WEIGHTS_AREA) -> bool:
    """Existing reward all_close condition (1.2× / 0.8× margins)."""
    if metrics is None:
        return False
    _, parts = compute_sim_reward(
        metrics, targets, weights=weights, warmup_deg=0.0, return_parts=True,
    )
    return bool(parts.get("all_close", False))


def classify_attempt(
    *,
    topology: str,
    spec: dict,
    prior_pass: bool,
    metrics: Optional[dict],
    physical_reward: float,
    expert_bonus: float = 0.0,
) -> dict:
    """Separate eligibility / prior / sim validity / compliance outcomes."""
    from sim.mna_scorer import topology_admits_spec

    eligible = bool(topology_admits_spec(topology, spec))
    sim_success = metrics is not None
    strict = strict_compliance(metrics, spec) if sim_success else False
    tolerant = tolerant_all_close(metrics, spec) if sim_success else False
    return {
        "eligible": eligible,
        "prior_pass": bool(prior_pass),
        "sim_success": sim_success,
        "strict_compliance": strict,
        "tolerant_all_close": tolerant,
        "physical_reward": float(physical_reward),
        "expert_bonus": float(expert_bonus),
        "total_reward": float(physical_reward) + float(expert_bonus),
    }
