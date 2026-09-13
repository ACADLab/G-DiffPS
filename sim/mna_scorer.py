"""Differentiable nodal MNA S-parameter scorer for TOPOLOGY_NETLIST circuits.

Builds a complex admittance system from bipartite incidence + device values
at omega = 2*pi*fc, solves with 50-ohm source/load (matching the SPICE
templates), and returns S11/S21 plus a soft score aligned with
PhaseShifterEnv.compute_reward (without expert_bonus).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from env.netlist_graph import (
    TOPOLOGY_NETLIST, PORT_NETS, N_STATES, VM_IQ, VP, Z0_REF, _normalize_name,
)
from env.reward import compute_sim_reward, WEIGHTS_AREA
from sim.area_model import estimate_area_mm2
from sim.physics_priors import compute_tline_abcd, abcd_to_y

# Fixed resistors present in SPICE templates but not policy-sized.
_FIXED_R = {
    "Vector_Modulator": {"R_q_term": 50.0, "R_drv_out": 50.0},
}

# Canonical VM state table — shared with the encoder so the graph and the
# solver cannot disagree about what a state means.
_VM_IQ = VM_IQ

# Return loss above this is not physically meaningful — it indicates a port
# decoupled by construction (e.g. VCVS control pins draw no current) rather
# than a well-matched design. SPICE reaches 300 dB via its |S11|^2 floor of
# 1e-30; MNA is unfloored and reaches 400+. Clamping equalizes the two and
# keeps the reported number interpretable. Does not change the reward, since
# mu_RL already saturates at min_rl_db <= 20.
RL_CEILING_DB = 60.0

# |S21| above this counts as active gain: the network delivers more power than
# the source makes available, which no passive phase shifter can do.
_ACTIVE_S21_TOL = 1.0 + 1e-6


def parse_spice_number(v, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace(" ", "")
    if not s:
        return default
    mult = 1.0
    if s.endswith("meg"):
        mult, s = 1e6, s[:-3]
    elif s.endswith("mil"):
        mult, s = 25.4e-6, s[:-3]
    else:
        suf = {"t": 1e12, "g": 1e9, "k": 1e3, "m": 1e-3,
               "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
        if s[-1] in suf:
            mult, s = suf[s[-1]], s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return default


def _wrap_180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _param_float(params: dict, key: str, default: float) -> float:
    return parse_spice_number(params.get(key), default)


def _switch_R(dev, state: int, r_on: float, r_off: float) -> float:
    if not dev.switch_param:
        return r_on
    return r_on if state in dev.on_in_states else r_off


def _net_index(nets: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(nets)}


def _collect_nets(topology: str) -> list[str]:
    """Ordered net list: ground '0' first, then remaining unique nets."""
    name = _normalize_name(topology)
    seen = {"0"}
    nets = ["0"]
    for dev in TOPOLOGY_NETLIST[name].values():
        for n in dev.nets:
            if n not in seen and n not in ("GND", "gnd", "ground"):
                if n in ("0",):
                    continue
                seen.add(n)
                nets.append(n)
            elif n in ("GND", "gnd", "ground") and "0" not in seen:
                pass
    # Normalize GND aliases already handled; ensure ports present
    for p in PORT_NETS[name]:
        if p not in seen:
            seen.add(p)
            nets.append(p)
    return nets


def _stamp_admittance(Y: torch.Tensor, i: int, j: int, y: torch.Tensor):
    """Stamp 2-terminal admittance between nodes i,j (ground index 0 skipped via i/j>=0)."""
    if i >= 0:
        Y[i, i] = Y[i, i] + y
    if j >= 0:
        Y[j, j] = Y[j, j] + y
    if i >= 0 and j >= 0:
        Y[i, j] = Y[i, j] - y
        Y[j, i] = Y[j, i] - y


def solve_sparams(
    topology: str,
    params: dict,
    fc_ghz: float,
    state: int = 0,
    G_I: float | None = None,
    G_Q: float | None = None,
) -> tuple[complex, complex]:
    """Return (S11, S21) at fc for one switch / VM state.

    Uses the same port convention as the SPICE templates:
      s11 = (v_in - 0.5) / 0.5,  s21 = v_out / 0.5
    with Vsrc=1 behind Rsrc=50 and Rload=50.
    """
    name = _normalize_name(topology)
    netlist = TOPOLOGY_NETLIST[name]
    nets = _collect_nets(name)
    nidx = _net_index(nets)
    # Working nodes exclude ground (index 0). Map net -> reduced index or -1 for gnd.
    reduced = {}
    k = 0
    for n in nets:
        if n == "0":
            reduced[n] = -1
        else:
            reduced[n] = k
            k += 1
    n_nodes = k

    # Count VCVS for MNA extras
    vcvs = [(dn, d) for dn, d in netlist.items() if d.dtype == "VCVS"]
    n_extra = len(vcvs)
    N = n_nodes + n_extra
    dtype = torch.complex128
    Y = torch.zeros((N, N), dtype=dtype)
    rhs = torch.zeros(N, dtype=dtype)

    omega = 2.0 * math.pi * fc_ghz * 1e9
    r_on = _param_float(params, "R_on", 3.0)
    r_off = _param_float(params, "R_off", 1e4)
    fixed = _FIXED_R.get(name, {})

    def ri(net: str) -> int:
        if net in ("0", "GND", "gnd", "ground"):
            return -1
        return reduced[net]

    # --- Device stamps ---
    for dname, dev in netlist.items():
        pins = list(dev.nets)
        if dev.dtype in ("R_switch", "R_fixed"):
            if len(pins) < 2:
                continue
            if dev.dtype == "R_switch":
                R = _switch_R(dev, state, r_on, r_off)
            else:
                R = fixed.get(dname, 50.0)
            y = torch.tensor(1.0 / max(R, 1e-12), dtype=dtype)
            _stamp_admittance(Y, ri(pins[0]), ri(pins[1]), y)

        elif dev.dtype == "C":
            C = _param_float(params, dev.sizes, 0.1) * 1e-12
            y = torch.tensor(1j * omega * C, dtype=dtype)
            _stamp_admittance(Y, ri(pins[0]), ri(pins[1]), y)

        elif dev.dtype == "L":
            L = _param_float(params, dev.sizes, 0.3) * 1e-9
            y = torch.tensor(1.0 / (1j * omega * L + 1e-30), dtype=dtype)
            _stamp_admittance(Y, ri(pins[0]), ri(pins[1]), y)

        elif dev.dtype == "TLine":
            Z0 = _param_float(
                params,
                (dev.aux_sizes[0] if dev.aux_sizes else "Z0_line"),
                50.0,
            )
            L_mm = _param_float(params, dev.sizes, 1.69)
            A, B, C, D = compute_tline_abcd(Z0, L_mm, fc_ghz * 1e9)
            Y11, Y12, Y21, Y22 = abcd_to_y(A, B, C, D)
            i, j = ri(pins[0]), ri(pins[1])
            # 2-port Y stamp (grounded reference already in ABCD)
            for (a, b, yab) in (
                (i, i, Y11), (i, j, Y12), (j, i, Y21), (j, j, Y22),
            ):
                if a >= 0 and b >= 0:
                    Y[a, b] = Y[a, b] + torch.as_tensor(yab, dtype=dtype)
                elif a >= 0 and b < 0:
                    Y[a, a] = Y[a, a] + torch.as_tensor(yab, dtype=dtype)
                elif a < 0 and b >= 0:
                    Y[b, b] = Y[b, b] + torch.as_tensor(yab, dtype=dtype)

        elif dev.dtype == "VCVS":
            pass  # stamped below with branch unknowns

    # VCVS stamps
    g_i_scale = _param_float(params, "G_I_scale", 1.0)
    g_q_scale = _param_float(params, "G_Q_scale", 1.0)
    if G_I is None:
        G_I = _VM_IQ[state % 16][0] if name == "Vector_Modulator" else 1.0
    if G_Q is None:
        G_Q = _VM_IQ[state % 16][1] if name == "Vector_Modulator" else 0.0

    for bi, (dname, dev) in enumerate(vcvs):
        out_p, out_m, ctrl_p, ctrl_m = dev.nets[:4]
        if dname == "E_I":
            gain = 2.0 * G_I * g_i_scale
        elif dname == "E_Q":
            gain = 2.0 * G_Q * g_q_scale
        else:
            gain = _param_float(params, dev.sizes, 1.0)
        row = n_nodes + bi
        op, om = ri(out_p), ri(out_m)
        cp, cm = ri(ctrl_p), ri(ctrl_m)
        # KCL: +i at out+, -i at out-
        if op >= 0:
            Y[op, row] = Y[op, row] + 1.0
        if om >= 0:
            Y[om, row] = Y[om, row] - 1.0
        # Branch: Vop - Vom - gain*(Vcp - Vcm) = 0
        if op >= 0:
            Y[row, op] = Y[row, op] + 1.0
        if om >= 0:
            Y[row, om] = Y[row, om] - 1.0
        if cp >= 0:
            Y[row, cp] = Y[row, cp] - gain
        if cm >= 0:
            Y[row, cm] = Y[row, cm] + gain

    # Port excitation: Rsrc from virtual src, but we inject Norton equivalent.
    # Template: Vsrc=1 behind Rsrc=50 into port_in; Rload=50 at port_out.
    # Norton: I = Vsrc/Rsrc into port_in, and Y += 1/Rsrc at port_in;
    #         Y += 1/Rload at port_out.
    port_in, port_out = PORT_NETS[name]
    ii, io = ri(port_in), ri(port_out)
    y50 = torch.tensor(1.0 / Z0_REF, dtype=dtype)
    if ii >= 0:
        Y[ii, ii] = Y[ii, ii] + y50
        rhs[ii] = rhs[ii] + torch.tensor(1.0 / Z0_REF, dtype=dtype)  # Vsrc=1
    if io >= 0:
        Y[io, io] = Y[io, io] + y50

    # Solve
    try:
        v = torch.linalg.solve(Y, rhs)
    except Exception:
        return complex(1.0, 0.0), complex(0.0, 0.0)

    def vnode(net: str) -> complex:
        r = ri(net)
        if r < 0:
            return 0j
        return complex(v[r].real.item(), v[r].imag.item())

    vin = vnode(port_in)
    vout = vnode(port_out)
    s11 = (vin - 0.5) / 0.5
    s21 = vout / 0.5
    return s11, s21


def sparams_to_metrics(s11: complex, s21: complex) -> dict:
    mag_s21 = abs(s21)
    mag_s11 = abs(s11)
    phase = math.degrees(math.atan2(s21.imag, s21.real))
    il_db = -20.0 * math.log10(max(mag_s21, 1e-30))
    rl_db = -20.0 * math.log10(max(mag_s11, 1e-30))
    return {
        "phase_deg": phase,
        "il_db": il_db,
        "rl_db": min(rl_db, RL_CEILING_DB),
        "rl_db_raw": rl_db,
        "gain_err_db": 0.0,
        "is_active": bool(mag_s21 > _ACTIVE_S21_TOL),
        "rl_saturated": bool(rl_db > RL_CEILING_DB),
        "s11": s11,
        "s21": s21,
    }


def _states_for_topo(topology: str, align_spice: bool = True) -> list[int]:
    """States to score for a topology.

    For 1-bit topologies (N=2) this is always [0, 1]. For Vector_Modulator
    (N=16), SPICE's env uses select_states(..., mode='auto') which returns 8
    indices — not the full 16. When align_spice=True (default), MNA uses that
    same subset so gain_err_db / rms_phase_err_deg are comparable to the SPICE
    oracle. Set align_spice=False to force a full enumeration.
    """
    name = _normalize_name(topology)
    n = N_STATES[name]
    if not align_spice or n <= 2:
        return list(range(n))
    # Mirror PhaseShifterEnv / state_sampler for 4-bit VM (bits=4, total=16).
    from specset.state_sampler import select_states
    bits = int(round(math.log2(n))) if n > 0 else 1
    return select_states(bits=bits, total_states=n, step_count=0, mode="auto")


def aggregate_mna_metrics(
    per_state: list[dict],
    ideal_step_deg: float,
) -> dict:
    """Mirror env.aggregate_state_metrics for MNA per-state dicts."""
    if not per_state:
        return None
    phases = [m["phase_deg"] for m in per_state]
    # State 0 is phase reference
    ref = phases[0]
    deltas = [_wrap_180(p - ref) for p in phases]
    # Ideal grid: i * ideal_step for state index i (use sample indices)
    # Caller passes states in order; for endpoints-only VM, approximate with
    # consecutive ideal steps between sampled indices via stored state ids.
    ils = [m["il_db"] for m in per_state]
    rls = [m["rl_db"] for m in per_state]
    state_ids = [m.get("state", i) for i, m in enumerate(per_state)]
    errs = []
    for sid, d in zip(state_ids, deltas):
        ideal = sid * ideal_step_deg
        errs.append(_wrap_180(d - ideal))
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else 99.0
    return {
        "rms_phase_err_deg": rms,
        "il_db": float(np.mean(ils)),
        "rl_db": float(np.mean(rls)),
        "gain_err_db": float(np.std(ils)) if len(ils) > 1 else 0.0,
        "n_states_run": len(per_state),
        "n_states_succeeded": len(per_state),
        # Idealness diagnostics: a model that is lossless and perfectly matched
        # in every state is an artifact of its circuit description, not a good
        # design, and cannot be fairly ranked against physical passive networks.
        "any_active": bool(any(m.get("is_active") for m in per_state)),
        "all_rl_saturated": bool(
            per_state and all(m.get("rl_saturated") for m in per_state)
        ),
        "min_il_db": float(np.min(ils)) if ils else 0.0,
        "per_state": per_state,
    }



def estimate_pwr_mw(
    topology: str,
    params: dict,
    state: int = 0,
    G_I: float | None = None,
    G_Q: float | None = None,
) -> float:
    """QUARANTINED (T1.5d). Continuous power left the reward.

    Raises RuntimeError — callers must not use this for scoring. Vector_Modulator
    power is enforced as a hard prior gate via ``vm_drive_pwr_mw``.
    """
    raise RuntimeError(
        "estimate_pwr_mw is quarantined (T1.5d); use vm_drive_pwr_mw for the "
        "pmax_mw prior gate only"
    )


def vm_drive_pwr_mw(
    params: dict,
    state: int = 0,
    G_I: float | None = None,
    G_Q: float | None = None,
) -> float:
    """VCVS drive proxy in mW: 5 * ((G_I*scale)^2 + (G_Q*scale)^2)."""
    g_i_s = _param_float(params, "G_I_scale", 1.0)
    g_q_s = _param_float(params, "G_Q_scale", 1.0)
    if G_I is None:
        G_I = _VM_IQ[state % 16][0]
    if G_Q is None:
        G_Q = _VM_IQ[state % 16][1]
    g_eff_sq = (G_I * g_i_s) ** 2 + (G_Q * g_q_s) ** 2
    return float(5.0 * g_eff_sq)


# Topologies that need a bias supply. Kept as data so the distinction is
# inspectable rather than implied by a name check.
ACTIVE_TOPOLOGIES = frozenset({"Vector_Modulator"})

# Smallest drive the Vector_Modulator can be sized to draw, over the whole
# action box and the worst I/Q state: G_I_scale and G_Q_scale are capped to
# [0.7, 1.0] by `action_to_params` (§9.2), so the floor is attained at 0.7.
# Being action-*independent* is the point -- it makes "this spec cannot afford
# an active stage" a property of (topology, spec) alone.
VM_MIN_DRIVE_MW = min(
    5.0 * ((gi * 0.7) ** 2 + (gq * 0.7) ** 2) for gi, gq in _VM_IQ
)


def topology_admits_spec(topology_name: str, spec: dict) -> bool:
    """(topology, spec)-only feasibility, independent of sizing.

    An active topology needs a bias supply. When a specification's power budget
    cannot cover the smallest draw the topology can be sized to, no action makes
    it feasible -- so this is a reject that belongs in the envelope rather than a
    penalty the sizing policy could ever learn its way out of.

    Passive topologies always admit: they draw no bias power.
    """
    if topology_name not in ACTIVE_TOPOLOGIES:
        return True
    pmax = spec.get("pmax_mw")
    if pmax is None:
        return True
    return float(pmax) >= VM_MIN_DRIVE_MW


def score_from_metrics(metrics: Optional[dict], targets: dict,
                       warmup_deg: float = 0.0) -> float:
    """Thin wrapper over env.reward.compute_sim_reward."""
    return compute_sim_reward(
        metrics, targets, weights=WEIGHTS_AREA, warmup_deg=warmup_deg,
    )


_IDEAL_STEP = {
    "Loaded_Line": -22.5,
    "Switched_Line": -90.0,
    "Reflection_Type": -22.5,
    "Switched_Filter": -180.0,
    "Vector_Modulator": -22.5,
    "All_Pass": -90.0,
}


def mna_evaluate(
    topology: str,
    params: dict,
    spec: dict,
    states: list[int] | None = None,
) -> tuple[float, dict | None]:
    """Multi-state MNA evaluate → (score, aggregated metrics)."""
    name = _normalize_name(topology)
    fc = float(spec.get("fc_ghz", 28.0))
    if states is None:
        states = _states_for_topo(name)
    per = []
    for s in states:
        try:
            s11, s21 = solve_sparams(name, params, fc, state=s)
            m = sparams_to_metrics(s11, s21)
            m["state"] = s
            per.append(m)
        except Exception:
            continue
    if len(per) < max(1, len(states) // 2):
        return -5.0, None
    agg = aggregate_mna_metrics(per, _IDEAL_STEP.get(name, -22.5))
    try:
        agg["area_mm2"] = estimate_area_mm2(
            name, params, fc_ghz=fc, tech=int(spec.get("tech", 0)),
        )
    except Exception:
        agg["area_mm2"] = None
    return score_from_metrics(agg, spec), agg


def mna_score(topology: str, params: dict, spec: dict) -> float:
    s, _ = mna_evaluate(topology, params, spec)
    return s
