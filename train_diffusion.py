import os
import sys
import argparse
import datetime
import json
import random
import tempfile
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# Ensure imports work from project root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import (
    get_topology_graph, TOPOLOGY_PARAMS, SLOT_ACTION_DIM, gin_device_rows,
)
from env.action_tokens import encode_action_tokens
from env.netlist_graph import sized_devices, _normalize_name
from env.param_semantics import (
    PARAM_CONTEXT_DIM,
    PARAM_ROLES,
    N_PARAM_ROLES,
    param_role_name,
)
from models.gnn_encoder import TopologyEncoder
from models.circuit_encoder import is_circuit_encoder, make_encoder, uses_param_nodes
from models.diffusion_policy import (
    DiffusionPolicy, FlowMatchingPolicy, CriticNet, ValueNet,
    NodeFlowMatchingPolicy, NodeCriticNet, CoupledNodeFlowMatchingPolicy,
)
from sim.physics_priors import check_physics_priors
from sim.switch_model import switch_params, format_switch_params
from env.reward import phase_warmup_deg
from specset.schema import SPEC_DIM
import netlist.llm_netlist_gen as llm_netlist_gen
import re
import math

# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, spec, topo_name, action, reward, fc_ghz=28.0, spec_dict=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        stored_spec = dict(spec_dict) if spec_dict else None
        self.buffer[self.position] = (
            spec, topo_name, action, reward, float(fc_ghz), stored_spec,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        specs, topos, actions, rewards, fcs, spec_dicts = zip(*batch)
        return (
            torch.tensor(np.array(specs), dtype=torch.float),
            topos,
            torch.tensor(np.array(actions), dtype=torch.float),
            torch.tensor(np.array(rewards), dtype=torch.float),
            list(fcs),
            list(spec_dicts),
        )

    def __len__(self):
        return len(self.buffer)


def _replay_spec(spec_dict, fc_ghz):
    """Full spec for circuit-encoder replay; never drop tech/switch fields."""
    if spec_dict:
        out = dict(spec_dict)
        out.setdefault("fc_ghz", float(fc_ghz))
        return out
    return {"fc_ghz": float(fc_ghz)}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# All_Pass builds differential phase from two bridged-T sections. Sizing them
# from two independent windows gives the policy no coordinate for the thing that
# actually produces phase -- the asymmetry between the sections -- and makes them
# identical at a=0.5, so the two switch states are the same circuit: Δφ = 0 and
# RL = 6.3 dB at the nominal point, and the topology encoder sees a graph with no
# phase shift in it.
#
# This is the §9.4 failure repeating: C_c and C_br were independently
# parameterized, the policy had no representation of their coupling, and it
# collapsed to C_c ≈ C_br. The fix there was to reparameterize as a ratio, and
# the same fix applies here. Each section pair is drawn as a (centre, ratio)
# pair rather than two independent values:
#
#     L_apA = L_centre / sqrt(rho_L)      L_apB = L_centre * sqrt(rho_L)
#
# so the asymmetry is a coordinate the policy controls directly, and the pair
# stays centred on the resonant value. Scaling L and C together within a section
# shifts that section's resonance while leaving sqrt(L/C) at Z0, so the ratio
# buys differential phase without spending return loss.
#
# Both sections sit *below* the resonant value, by ALLPASS_CENTRE_OFFSET. That
# is forced rather than chosen: `clamp_spice_value` caps inductors at 10 nH, and
# L0 at the low end of the sampled band (1 GHz) is already 7.96 nH, so any
# section pushed above the resonant centre clamps there. A centred pair reads
# Δφ = 86.7° but silently clamps below 2.25 GHz.
#
# (ratio 5.0, offset 0.25) chosen by 2-D sweep (tools/allpass_recentre_probe.py):
# Δφ = 91.9° against the ideal −90° step, RL ≥ 22.8 dB, IL ≤ 0.56 dB, and
# fc-invariant across the whole 1–40 GHz sampled band with no clamping.
#
# Overridable so the pre-fix parameterization (ratio=1.0) stays reproducible:
# the published All_Pass LOOCV number was produced under it, and testing what
# caused that number means being able to rebuild it.
# Absolute physical limits enforced downstream by `clamp_spice_value`. Mirrored
# here so the sampling windows can be clipped to them *before* a value is drawn
# rather than silently truncated after. tests/test_no_silent_clamping.py pins
# the two together.
PHYSICAL_LIMITS_BY_SUFFIX = {
    "_nh": (0.005, 10.0),
    "_pf": (0.005, 10.0),
    "_mm": (0.1, 50.0),
}


def _suffix_of(key: str) -> str | None:
    k = key.lower()
    for suf in PHYSICAL_LIMITS_BY_SUFFIX:
        if k.endswith(suf):
            return suf
    return None


ALLPASS_SECTION_RATIO = float(os.environ.get("G_DIFFPS_ALLPASS_SPLIT", "5.0"))
ALLPASS_CENTRE_OFFSET = 0.25

# Log-uniform span of the section ratio, geometric mean = ALLPASS_SECTION_RATIO.
ALLPASS_RATIO_SPAN = 3.0

# Upper end of the C_c = k * C_br coupling ratio (§9.4).
ALLPASS_CC_MAX = 4.0


def _allpass_ratio_range() -> tuple[float, float]:
    r = ALLPASS_SECTION_RATIO
    if r <= 1.0:  # pre-fix reproduction: sections stay equal
        return 1.0, 1.0
    return r / ALLPASS_RATIO_SPAN, r * ALLPASS_RATIO_SPAN


def _allpass_centre_offset() -> float:
    return 1.0 if ALLPASS_SECTION_RATIO <= 1.0 else ALLPASS_CENTRE_OFFSET


def _allpass_section(action, keys, kind: str, centre: float, is_section_b: bool,
                     log_scale) -> float:
    """One All_Pass section value from a (centre, ratio) coordinate pair.

    `kind` is "L" or "C". The A-slot action carries the pair's centre and the
    B-slot action carries the ratio between the sections, so both original
    coordinates are still used and the asymmetry is directly representable.
    """
    a_key, b_key = ("L_apA_nh", "L_apB_nh") if kind == "L" else \
                   ("C_brA_pf", "C_brB_pf")
    v_centre = float(action[keys.index(a_key)])
    v_ratio = float(action[keys.index(b_key)])
    c = centre * _allpass_centre_offset()

    lo_r, hi_r = _allpass_ratio_range()
    rho = (10 ** (math.log10(lo_r) + v_ratio * (math.log10(hi_r) - math.log10(lo_r)))
           if hi_r > lo_r else lo_r)
    s = math.sqrt(rho)

    # Clip the centre window against the *derived* section values rather than
    # the centre itself, conditioned on the ratio actually drawn. Section A sits
    # at mid/s and section B at mid*s, and the coupling cap reaches
    # ALLPASS_CC_MAX * (mid*s), so those are the values that must stay inside
    # the physical limits -- clipping only the centre lets the derived values
    # land on a clamp boundary anyway.
    # Clip *symmetrically in log space about c*, not to the raw feasible
    # interval: an asymmetric clip moves the a=0.5 point off the designed
    # nominal, which is the whole thing this parameterization exists to fix.
    # The window narrows at the band edges, which is physics -- at 40 GHz a
    # resonant cap is 0.08 pF and the smallest modelable one is 5 fF -- rather
    # than a truncation that silently maps many actions onto one circuit.
    half = 1.0
    lim = PHYSICAL_LIMITS_BY_SUFFIX.get(_suffix_of(a_key))
    if lim is not None:
        clo, chi = lim
        top = s * (ALLPASS_CC_MAX if kind == "C" else 1.0)
        half = max(0.0, min(half,
                            math.log10(max(c / (clo * s), 1.0)),
                            math.log10(max((chi / top) / c, 1.0))))

    c_l = math.log10(max(c, 1e-12))
    mid = 10 ** (c_l + (2.0 * v_centre - 1.0) * half)
    return mid * s if is_section_b else mid / s


def action_to_params(action, topology_name, spec_dict, sizing="log", bounds="electrical",
                     switch_model="ideal"):
    """
    Scale continuous [0, 1] action vector to physical SPICE values.

    spec_dict must contain 'fc_ghz' for frequency-adaptive transmission-line bounds.
    sizing='log'    → physics-informed log scaling (default, used for training).
    sizing='linear' → flat linear scaling across same bounds (ablation baseline).
    bounds='electrical' → ranges centered on resonant (C0, L0, lam4) at fc (train default).
    bounds='legacy'     → original per-topology hard-coded ranges (paper numbers).
    switch_model='ideal'|'realistic' → R_on/R_off from tech constants (not actions).
    """
    keys = TOPOLOGY_PARAMS[topology_name]
    params_dict = {}

    fc_ghz = spec_dict.get("fc_ghz", 28.0)
    # Quarter-wavelength in mm at fc using eps_eff=2.5 (matches SPICE templates):
    # vp = c/sqrt(eps_eff) = 3e8/sqrt(2.5) = 1.897e8 m/s → λ/4 = vp/(4fc)
    lam4 = 47.43 / fc_ghz
    omega = 2.0 * math.pi * fc_ghz * 1e9
    Z0 = 50.0
    C0_pf = (1.0 / (omega * Z0)) * 1e12
    L0_nh = (Z0 / omega) * 1e9

    # Under bounds='electrical' the windows are multiples of the resonant value
    # at fc, but `clamp_spice_value` enforces absolute limits. Since the
    # resonant value scales as 1/fc, the two disagree badly at low carriers: at
    # 1.26 GHz, 72% of Switched_Line draws and 77% of Switched_Filter draws used
    # to land on a clamp boundary, so most of the action box mapped onto the same
    # few circuits and the achievable set was silently truncated exactly where
    # the passive-only regime lives (tools/clamp_audit.py).
    #
    # Clipping the *window* to the reachable range instead keeps the map onto
    # physically admissible parts injective: no draw clamps, no action volume is
    # dead, and what the archive explores is what the netlist can express.
    _cur_key = [""]

    def _clip(lo, hi):
        rng_ = PHYSICAL_LIMITS_BY_SUFFIX.get(_suffix_of(_cur_key[0]))
        if rng_ is None:
            return lo, hi
        clo, chi = rng_
        lo, hi = max(lo, clo), min(hi, chi)
        if hi <= lo:  # window collapsed: pin to the reachable end
            lo = hi = min(max(clo, lo), chi)
        return lo, hi

    def lin(v, lo, hi):
        lo, hi = _clip(lo, hi)
        return lo + v * (hi - lo)

    def log_scale(v, lo, hi):
        if sizing == "linear":
            return lin(v, lo, hi)
        lo, hi = _clip(lo, hi)
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.000001)
        lo_l = np.log10(lo)
        hi_l = np.log10(hi)
        return 10 ** (lo_l + v * (hi_l - lo_l))

    for i, key in enumerate(keys):
        val_01 = float(action[i]) if i < len(action) else 0.5
        _cur_key[0] = key

        # ── SKY130 PDK geometry windows (Milestone C action coordinates) ──
        if bounds == "sky130":
            # Switches / transistors: map onto pin-file W/L/nf ranges.
            # Passives that remain ideal keep electrical-style windows until
            # the topology is fully PDK-mapped; geometry keys use PDK limits.
            try:
                from sim.sky130 import load_pin
                pin = load_pin()
                nmos_g = pin["devices"]["nmos"]["geometry"]
                res_g = pin["devices"]["resistor"]["geometry"]
                cap_g = pin["devices"]["capacitor"]["geometry"]
            except Exception:
                nmos_g = {"L_um": [0.15, 10.0], "W_um": [0.42, 100.0], "nf": [1, 32]}
                res_g = {"L_um": [0.35, 100.0], "mult": [1, 64]}
                cap_g = {"W_um": [2.0, 30.0], "L_um": [2.0, 30.0], "mf": [1, 64]}

            if key in ("W_um",) or key.endswith("_W_um"):
                physical_val = log_scale(val_01, nmos_g["W_um"][0], nmos_g["W_um"][1])
            elif key in ("L_um",) or key.endswith("_L_um"):
                physical_val = log_scale(val_01, nmos_g["L_um"][0], nmos_g["L_um"][1])
            elif key == "nf":
                physical_val = round(lin(val_01, nmos_g["nf"][0], nmos_g["nf"][1]))
            elif key == "mult":
                physical_val = round(lin(val_01, res_g["mult"][0], res_g["mult"][1]))
            elif key == "mf":
                physical_val = round(lin(val_01, cap_g["mf"][0], cap_g["mf"][1]))
            elif key.endswith("_pf"):
                # Keep electrical resonant window for still-ideal caps.
                physical_val = log_scale(val_01, max(C0_pf / 10.0, 0.005), C0_pf * 10.0)
            elif key.endswith("_nh"):
                physical_val = log_scale(val_01, max(L0_nh / 10.0, 0.005), L0_nh * 10.0)
            elif key.endswith("_mm"):
                physical_val = log_scale(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key.startswith("Z0"):
                physical_val = lin(val_01, 25.0, 75.0)
            elif key in ("G_I_scale", "G_Q_scale"):
                physical_val = lin(val_01, 0.7, 1.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Electrical / per-topology branches (R_on/R_off are NOT actions) ──
        elif bounds == "electrical":
            # Frequency-centered ranges — makes 2.4 GHz All Pass reachable.
            # All-Pass coupling caps stay ratio-encoded against their bridge caps.
            # Perturbative / shunt-loading / tuning caps use a sub-resonant window
            # (not C0/10…C0*10): at 28 GHz C0≈0.114 pF, so mid of the resonant
            # window is ~0.29 pF and fails Loaded_Line / Reflection_Type priors.
            if key in ("L_apA_nh", "L_apB_nh"):
                physical_val = _allpass_section(
                    action, keys, "L", L0_nh, key.endswith("B_nh"), log_scale)
            elif key in ("C_brA_pf", "C_brB_pf"):
                physical_val = _allpass_section(
                    action, keys, "C", C0_pf, key.endswith("B_pf"), log_scale)
            elif key in ("C_cA_pf", "C_cB_pf"):
                # Coupling cap stays ratio-encoded against its own bridge cap
                # (§9.4), which is now itself a (centre, ratio) draw.
                is_b = key == "C_cB_pf"
                c_br = _allpass_section(
                    action, keys, "C", C0_pf, is_b, log_scale)
                physical_val = lin(val_01, 1.2, ALLPASS_CC_MAX) * c_br
            elif key == "C_load_pf":
                # Shunt loading: legacy absolute window (template default 0.04 pF)
                physical_val = log_scale(val_01, 0.005, 0.5)
            elif key in ("C_base_pf", "C_tune_pf"):
                # Reflection-type switched/tuning caps (defaults 0.10 / 0.20 pF)
                physical_val = log_scale(val_01, 0.01, 1.0)
            elif key.endswith("_pf"):
                # Series / filter / bridge resonant caps: C0/10 … C0*10
                physical_val = log_scale(val_01, C0_pf / 10.0, C0_pf * 10.0)
            elif key.endswith("_nh"):
                physical_val = log_scale(val_01, L0_nh / 10.0, L0_nh * 10.0)
            elif key == "L_short_mm":
                # Switched_Line partition: short arm stays below the long arm.
                physical_val = lin(val_01, 0.3 * lam4, 0.8 * lam4)
            elif key == "L_long_mm":
                physical_val = lin(val_01, 0.8 * lam4, 2.5 * lam4)
            elif key.endswith("_mm"):
                # L_quarter_mm etc.: log-symmetric about λ/4 so a=0.5 → 90°.
                physical_val = log_scale(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key == "Z0_main":
                # Hybrid main arm. A shared 25–75 Ω window with Z0_branch made
                # the electrical midpoint Z0_branch/Z0_main = 1, which the prior
                # rejects ([0.60, 0.85], ideal 1/√2).
                physical_val = lin(val_01, 40.0, 60.0)
            elif key == "Z0_branch":
                physical_val = lin(val_01, 25.0, 45.0)
            elif key.startswith("Z0"):
                physical_val = lin(val_01, 25.0, 75.0)
            elif key in ("G_I_scale", "G_Q_scale"):
                physical_val = lin(val_01, 0.7, 1.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Loaded_Line ──────────────────────────────────────────────────────
        elif topology_name == "Loaded_Line":
            if key == "Z0_line":
                physical_val = lin(val_01, 35.0, 75.0)
            elif key == "L_quarter_mm":
                physical_val = log_scale(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key == "C_load_pf":
                physical_val = log_scale(val_01, 0.005, 0.5)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Switched_Line ────────────────────────────────────────────────────
        elif topology_name == "Switched_Line":
            if key == "Z0_line":
                physical_val = lin(val_01, 35.0, 75.0)
            elif key == "L_short_mm":
                physical_val = lin(val_01, 0.3 * lam4, 0.8 * lam4)
            elif key == "L_long_mm":
                physical_val = lin(val_01, 0.8 * lam4, 2.5 * lam4)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Reflection_Type ──────────────────────────────────────────────────
        elif topology_name == "Reflection_Type":
            if key == "Z0_main":
                physical_val = lin(val_01, 40.0, 60.0)
            elif key == "Z0_branch":
                physical_val = lin(val_01, 25.0, 45.0)
            elif key == "L_quarter_mm":
                physical_val = log_scale(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key == "C_base_pf":
                physical_val = log_scale(val_01, 0.01, 1.0)
            elif key == "C_tune_pf":
                physical_val = log_scale(val_01, 0.01, 1.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Switched_Filter ──────────────────────────────────────────────────
        elif topology_name == "Switched_Filter":
            if key in ("C_hpf_pf", "C_lpf_pf"):
                physical_val = log_scale(val_01, 0.005, 2.0)
            elif key in ("L_hpf_nh", "L_lpf_nh"):
                physical_val = log_scale(val_01, 0.005, 2.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Vector_Modulator ─────────────────────────────────────────────────
        elif topology_name == "Vector_Modulator":
            if key == "Z0_line":
                physical_val = lin(val_01, 35.0, 75.0)
            elif key == "L_quarter_mm":
                physical_val = log_scale(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key in ("G_I_scale", "G_Q_scale"):
                physical_val = lin(val_01, 0.7, 1.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── All_Pass ─────────────────────────────────────────────────────────
        elif topology_name == "All_Pass":
            if key in ("L_apA_nh", "L_apB_nh"):
                physical_val = log_scale(val_01, 0.01, 2.0)
            elif key in ("C_brA_pf", "C_brB_pf"):
                physical_val = log_scale(val_01, 0.005, 0.5)
            elif key == "C_cA_pf":
                c_brA = log_scale(float(action[keys.index("C_brA_pf")]), 0.005, 0.5)
                k_A = lin(val_01, 1.2, 4.0)
                physical_val = k_A * c_brA
            elif key == "C_cB_pf":
                c_brB = log_scale(float(action[keys.index("C_brB_pf")]), 0.005, 0.5)
                k_B = lin(val_01, 1.2, 4.0)
                physical_val = k_B * c_brB
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Fallback ─────────────────────────────────────────────────────────
        else:
            info = llm_netlist_gen._PARAM_INFO.get(key)
            if info is not None:
                lo, hi = info[0], info[1]
            else:
                lo, hi = 0.1, 10.0
            physical_val = lin(val_01, lo, hi)

        clamped_str = llm_netlist_gen.clamp_spice_value(key, f"{physical_val:.4e}")
        params_dict[key] = clamped_str

    # Switch parasitics from tech constants (not sized by the actor).
    tech = int(spec_dict.get("tech", 0))
    sw = format_switch_params(switch_params(tech, fc_ghz, model=switch_model))
    params_dict.update(sw)
    return params_dict


def device_action_to_params(action_by_param, topology_name, spec_dict,
                            sizing="log", bounds="electrical", switch_model="ideal"):
    """Map {param_key: a_01} (or ordered array over sized_devices) to params_dict.

    Re-indexes the EXISTING action_to_params bounds by device/param rather
    than by a global slot vector. R_on/R_off are tech constants, not actions.
    """
    keys = TOPOLOGY_PARAMS[topology_name]
    if isinstance(action_by_param, dict):
        action = np.zeros(len(keys), dtype=np.float64)
        for i, k in enumerate(keys):
            action[i] = float(action_by_param.get(k, 0.5))
    else:
        sized = sized_devices(topology_name)
        action = np.full(len(keys), 0.5, dtype=np.float64)
        arr = np.asarray(action_by_param, dtype=np.float64).reshape(-1)
        for i, (_, param) in enumerate(sized):
            if i < len(arr) and param in keys:
                action[keys.index(param)] = arr[i]
    return action_to_params(action, topology_name, spec_dict,
                            sizing=sizing, bounds=bounds,
                            switch_model=switch_model)


def rewrite_control_block(skeleton: str, fc_ghz: float, bw_pct: float = 30.0,
                          fc_mode: str = "fixed28") -> str:
    """Rewrite AC sweep and meas AT= lines to track the specification.

    fc_mode='fixed28' leaves the template unchanged (paper-faithful).
    fc_mode='spec' substitutes sweep and measurement frequency from the spec.
    """
    if fc_mode == "fixed28":
        return skeleton

    fc_hz = fc_ghz * 1e9
    half_bw = max(bw_pct, 5.0) / 200.0  # fractional half-bandwidth
    lo_hz = max(fc_hz * (1.0 - half_bw), 1e8)
    hi_hz = fc_hz * (1.0 + half_bw)
    # Prefer G-suffix for readability when >= 1 GHz
    def fmt(hz):
        if hz >= 1e9:
            return f"{hz/1e9:.6g}G"
        if hz >= 1e6:
            return f"{hz/1e6:.6g}Meg"
        return f"{hz:.6g}"

    lo_s, hi_s, fc_s = fmt(lo_hz), fmt(hi_hz), fmt(fc_hz)

    out = re.sub(
        r"ac\s+lin\s+\d+\s+\S+\s+\S+",
        f"ac lin 201 {lo_s} {hi_s}",
        skeleton,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"(meas\s+ac\s+\S+\s+FIND\s+\S+\s+AT=)\S+",
        lambda m: m.group(1) + fc_s,
        out,
        flags=re.IGNORECASE,
    )
    # Also rewrite .PARAM fc=... FRAMEWORK_CONTROLLED if present
    out = re.sub(
        r"(\.PARAM\s+[^\n]*\bfc=)\S+",
        lambda m: m.group(1) + f"{fc_hz:.6e}",
        out,
        flags=re.IGNORECASE,
    )
    return out


def make_spice_netlist(topology_name, params_dict, spec_dict=None, fc_mode="spec"):
    """
    Load topology template, inject param overrides, and write temporary netlist.
    """
    skeleton_path = os.path.join(REPO_ROOT, f"specset/templates/{topology_name.lower()}.sp")
    with open(skeleton_path, "r") as f:
        skeleton_content = f.read()

    if spec_dict is not None:
        skeleton_content = rewrite_control_block(
            skeleton_content,
            fc_ghz=float(spec_dict.get("fc_ghz", 28.0)),
            bw_pct=float(spec_dict.get("bw_pct", 30.0)),
            fc_mode=fc_mode,
        )

    param_str = ".PARAM " + " ".join([f"{k}={v}" for k, v in params_dict.items()]) + "\n"

    lines = skeleton_content.split('\n')
    new_lines = []
    for l in lines:
        stripped = l.strip().upper()
        if stripped.startswith(".PARAM"):
            if 'DERIVED' in stripped or 'FRAMEWORK_CONTROLLED' in stripped:
                new_lines.append(l)
        else:
            new_lines.append(l)

    if new_lines:
        final_netlist = new_lines[0] + '\n' + param_str + '\n'.join(new_lines[1:])
    else:
        final_netlist = f"* {topology_name}\n" + param_str

    fd, path = tempfile.mkstemp(suffix=".sp", prefix=f"diff_{topology_name.lower()}_")
    with os.fdopen(fd, 'w') as f:
        f.write(final_netlist)
    return path


def parallel_eval_worker(args):
    """
    Multiprocessing worker to evaluate a single netlist in parallel.

    args tuple: (netlist_path, spec_dict, topology_name, expert_bonus,
                 env_restrict_to[, warmup_deg[, params_dict]])
    """
    warmup_deg = 0.0
    params_dict = None
    if len(args) == 5:
        netlist_path, spec_dict, topology_name, expert_bonus, env_restrict_to = args
    elif len(args) == 6:
        (netlist_path, spec_dict, topology_name, expert_bonus,
         env_restrict_to, warmup_deg) = args
    else:
        (netlist_path, spec_dict, topology_name, expert_bonus,
         env_restrict_to, warmup_deg, params_dict) = args

    # Instantiate thread-local/process-local env to evaluate the netlist safely
    env = PhaseShifterEnv(restrict_to=env_restrict_to)
    env.current_spec = spec_dict
    env._last_topology = topology_name
    env._last_params = params_dict or {}

    try:
        agg, sim_reward, state_indices, bits, ideal_step_deg = env._evaluate_netlist(
            netlist_path, warmup_deg=float(warmup_deg),
        )
        total_reward = sim_reward + expert_bonus
        success = (agg is not None)
    except Exception as e:
        print(f"[Worker Error] {e}")
        agg, total_reward, success = None, -5.0 + expert_bonus, False

    # Cleanup temp netlist files
    try:
        if os.path.exists(netlist_path):
            os.remove(netlist_path)
        lis_path = f"{netlist_path}.lis"
        if os.path.exists(lis_path):
            os.remove(lis_path)
    except Exception:
        pass

    return total_reward, agg, success


# =============================================================================
# DDP AND TRAINING SETUP
# =============================================================================

def train(rank, world_size, args):
    # Set seed for reproducibility
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize Distributed Data Parallel
    is_ddp = world_size > 1
    if is_ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.ddp_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    print(f"[Rank {rank}] Running on {device} (Seed: {seed})")

    # Initialize environment and dataset specs
    env = PhaseShifterEnv(
        restrict_to=args.restrict_to,
        expert_bonus_scale=float(getattr(args, "expert_bonus_scale", 0.0)),
    )
    if not env.dataset:
        extra = getattr(env, "_specset_load_error", None) or ""
        raise RuntimeError(
            "Training specset is empty; refusing to run against a dummy spec. "
            + extra
        )
    
    # Neural Networks initialization
    use_circuit = is_circuit_encoder(getattr(args, "encoder", "gin"))
    use_param = uses_param_nodes(getattr(args, "encoder", "gin"))
    use_device = getattr(args, "action_space", "slot") == "device"
    use_coupled = bool(getattr(args, "coupled_actions", False)) or use_param
    fc_mode = getattr(args, "fc_mode", "spec")
    bounds = getattr(args, "bounds", "electrical")
    switch_model = getattr(args, "switch_model", "ideal")

    if use_circuit:
        gnn_encoder = make_encoder(
            getattr(args, "encoder", "circuit"), hidden=64, out_dim=64,
        ).to(device)
    else:
        gnn_encoder = TopologyEncoder().to(device)

    if use_device:
        if use_coupled:
            actor = CoupledNodeFlowMatchingPolicy(
                spec_dim=SPEC_DIM, graph_dim=64, num_steps=10,
                role_dim=PARAM_CONTEXT_DIM,
            ).to(device)
        else:
            actor = NodeFlowMatchingPolicy(spec_dim=SPEC_DIM, graph_dim=64, num_steps=10).to(device)
        critic = NodeCriticNet(spec_dim=SPEC_DIM, graph_dim=64).to(device)
    elif args.actor == "cfm":
        actor = FlowMatchingPolicy(
            action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64, num_steps=10
        ).to(device)
        critic = CriticNet(action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64).to(device)
    else:
        actor = DiffusionPolicy(action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64).to(device)
        critic = CriticNet(action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64).to(device)
    value_net = ValueNet(spec_dim=SPEC_DIM, graph_dim=64).to(device)

    # Wrap models in DDP if running distributed
    if is_ddp:
        gnn_encoder = DDP(gnn_encoder, device_ids=[rank], find_unused_parameters=True)
        actor = DDP(actor, device_ids=[rank], find_unused_parameters=True)
        critic = DDP(critic, device_ids=[rank])
        value_net = DDP(value_net, device_ids=[rank])

    # Optimizers
    gnn_opt = optim.Adam(gnn_encoder.parameters(), lr=args.lr)
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)
    value_opt = optim.Adam(value_net.parameters(), lr=args.lr)

    # Shared Replay Buffer (only rank 0 does logging & checkpoint saving)
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    # Output run setup
    if getattr(args, "run_dir", None):
        run_dir = args.run_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            REPO_ROOT, f"runs_diffusion/run_{stamp}_sw{switch_model}"
        )
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        log_fh = open(os.path.join(run_dir, "train.log"), "w", buffering=1)
        print(f"[Rank 0] Saving results to: {run_dir}")
        meta = {
            "event": "training_start",
            "time": datetime.datetime.utcnow().isoformat() + "Z",
            "switch_model": switch_model,
            "bounds": bounds,
            "encoder": getattr(args, "encoder", "gin"),
            "action_space": getattr(args, "action_space", "slot"),
            "coupled_actions": bool(use_coupled),
            "expert_bonus_scale": float(getattr(args, "expert_bonus_scale", 0.0)),
            "sim": getattr(args, "sim", "spice"),
            "skip_prior": bool(getattr(args, "skip_prior", False)),
            "slot_action_dim": SLOT_ACTION_DIM,
        }
        log_fh.write(json.dumps(meta) + "\n")
        with open(os.path.join(run_dir, "run_config.json"), "w") as cf:
            json.dump(meta, cf, indent=2)
    # Dry-run override
    total_timesteps = args.total_timesteps
    if args.dry_run:
        total_timesteps = 5
        print(f"[Rank {rank}] Running dry-run validation (5 steps)")

    # Prep graph PyG data for the 6 topologies on target device (GIN path only)
    topo_graphs = {}
    if not use_circuit:
        for name in TOPOLOGY_PARAMS.keys():
            g = get_topology_graph(name)
            g.x = g.x.to(device)
            g.edge_index = g.edge_index.to(device)
            topo_graphs[name] = g

    # Pre-fill replay buffer with warm start random runs if desired
    obs, info = env.reset()

    # Max sized-device count across topologies (for padding device actions).
    # R_on/R_off are tech constants — no longer padded as action dims.
    max_sized = max(len(sized_devices(t)) for t in TOPOLOGY_PARAMS)    
    # Main online training loop
    for step in range(total_timesteps):
        # 1. Reset Spec and select random/heuristic topologies to explore
        obs_vec, info = env.reset()
        spec_dict = env.current_spec
        
        # Sample random active topology to evaluate continuous action on
        topology_name = random.choice(env._active_topologies)
        
        # 2. Extract topology / parameter embeddings
        enc = gnn_encoder.module if is_ddp else gnn_encoder
        spec_tensor = torch.tensor(obs_vec, dtype=torch.float, device=device).unsqueeze(0)
        encoder_name = getattr(args, "encoder", "gin")
        if use_device:
            z_topo, h_act, ctx = encode_action_tokens(
                enc, topology_name, spec_dict,
                encoder_name=encoder_name,
                bounds=bounds, switch_model=switch_model,
                gin_graph=topo_graphs.get(topology_name),
                device=device,
            )
            n_act = h_act.size(0)
            mask = torch.ones(n_act, dtype=torch.bool, device=device)
            with torch.no_grad():
                if use_coupled:
                    action_tensor = actor.sample(
                        spec_tensor, h_act, mask=mask, role=ctx,
                    )
                else:
                    action_tensor = actor.sample(spec_tensor, h_act, mask=mask)
                action = action_tensor.detach().cpu().numpy()
                action_pad = np.zeros(max_sized, dtype=np.float32)
                action_pad[: len(action)] = action
                action = action_pad
        else:
            if use_circuit:
                z_topo = enc(
                    topology_name, spec_dict, return_device=False,
                    bounds=bounds, switch_model=switch_model,
                )
            else:
                graph_data = topo_graphs[topology_name]
                z_topo = gnn_encoder(graph_data.x, graph_data.edge_index)
            with torch.no_grad():
                action_tensor = actor.sample(spec_tensor, z_topo)
                action = action_tensor.squeeze(0).cpu().numpy()
        
        # 4. Map actions to physical parameters
        if use_device:
            params_dict = device_action_to_params(
                action, topology_name, spec_dict,
                sizing=args.sizing, bounds=bounds, switch_model=switch_model,
            )
        else:
            params_dict = action_to_params(
                action, topology_name, spec_dict,
                sizing=args.sizing, bounds=bounds, switch_model=switch_model,
            )        
        passed_prior = True if getattr(args, "skip_prior", False) else check_physics_priors(
            topology_name, params_dict, spec_dict["fc_ghz"],
            pmax_mw=float(spec_dict.get("pmax_mw", 1e9)),
        )
        expert_bonus = env.compute_expert_bonus(topology_name, spec_dict)
        
        agg_metrics = None
        if not passed_prior:
            total_reward = -5.0 + expert_bonus
            success = False
            if rank == 0:
                print(f"[Step {step:04d}] [{topology_name}] Rejected by physics prior. RL penalty assigned.")
        else:
            if getattr(args, "sim", "spice") == "mna":
                from sim.mna_scorer import mna_evaluate
                from env.reward import compute_sim_reward, WEIGHTS_AREA
                _score, agg_metrics = mna_evaluate(
                    topology_name, params_dict, spec_dict,
                )
                if agg_metrics is None:
                    total_reward = -5.0 + expert_bonus
                    success = False
                else:
                    total_reward = float(compute_sim_reward(
                        agg_metrics, spec_dict, weights=WEIGHTS_AREA,
                        warmup_deg=phase_warmup_deg(step),
                    )) + expert_bonus
                    success = True
            else:
                netlist_path = make_spice_netlist(
                    topology_name, params_dict, spec_dict=spec_dict, fc_mode=fc_mode,
                )
                total_reward, agg_metrics, success = parallel_eval_worker(
                    (netlist_path, spec_dict, topology_name, expert_bonus,
                     args.restrict_to, phase_warmup_deg(step), params_dict)
                )
            
        # 6. Push transition to Replay Buffer (keep fc for circuit-encoder re-encode)
        replay_buffer.push(
            obs_vec, topology_name, action, total_reward,
            fc_ghz=float(spec_dict.get("fc_ghz", 28.0)),
            spec_dict=spec_dict,
        )
        
        # Log to file on rank 0 (annealed reward + raw for ablation)
        if rank == 0:
            wdeg = phase_warmup_deg(step)
            reward_raw = None
            if agg_metrics is not None and passed_prior:
                from env.reward import WEIGHTS_AREA, compute_sim_reward
                reward_raw = float(compute_sim_reward(
                    agg_metrics, spec_dict, weights=WEIGHTS_AREA, warmup_deg=0.0,
                )) + float(expert_bonus)
            metrics_log = None
            if isinstance(agg_metrics, dict):
                metrics_log = {
                    k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in agg_metrics.items()
                    if k != "per_state" and not isinstance(v, complex)
                }
            log_entry = {
                "step": step,
                "topology": topology_name,
                "passed_prior": passed_prior,
                "reward": float(total_reward),
                "reward_raw": reward_raw,
                "success": success,
                "metrics": metrics_log,
                "params": params_dict,
                "fc_ghz": spec_dict.get("fc_ghz"),
                "encoder": getattr(args, "encoder", "gin"),
                "action_space": getattr(args, "action_space", "slot"),
                "fc_mode": fc_mode,
                "bounds": bounds,
                "switch_model": switch_model,
                "warmup_deg": wdeg,
            }
            log_fh.write(json.dumps(log_entry) + "\n")
            print(f"[Step {step:04d}] [{topology_name}] Reward: {total_reward:+.3f} (Succeeded: {success})")
            
        # 7. Update networks via Q-value reweighted score matching
        if len(replay_buffer) >= args.batch_size:
            specs_b, topos_b, actions_b, rewards_b, fcs_b, spec_dicts_b = replay_buffer.sample(args.batch_size)
            
            specs_b = specs_b.to(device)
            actions_b = actions_b.to(device)
            rewards_b = rewards_b.to(device)

            # Encode batch
            z_topo_list = []
            h_dev_list = []
            mask_list = []
            role_list = []
            encoder_name = getattr(args, "encoder", "gin")
            for bi, topo_name in enumerate(topos_b):
                replay_spec = _replay_spec(spec_dicts_b[bi], fcs_b[bi])
                if use_device:
                    z, h_act, ctx = encode_action_tokens(
                        enc, topo_name, replay_spec,
                        encoder_name=encoder_name,
                        bounds=bounds, switch_model=switch_model,
                        gin_graph=topo_graphs.get(topo_name),
                        device=device,
                    )
                    pad = torch.zeros(max_sized, h_act.size(-1), device=device)
                    pad[: h_act.size(0)] = h_act
                    m = torch.zeros(max_sized, dtype=torch.bool, device=device)
                    m[: h_act.size(0)] = True
                    pad_r = torch.zeros(max_sized, ctx.size(-1), device=device)
                    pad_r[: ctx.size(0)] = ctx
                    z_topo_list.append(z)
                    h_dev_list.append(pad)
                    mask_list.append(m)
                    role_list.append(pad_r)
                elif use_circuit:
                    z = enc(
                        topo_name, replay_spec, return_device=False,
                        bounds=bounds, switch_model=switch_model,
                    )
                    z_topo_list.append(z)
                else:
                    g = topo_graphs[topo_name]
                    z = gnn_encoder(g.x, g.edge_index)
                    z_topo_list.append(z)
            z_topo_b = torch.cat(z_topo_list, dim=0)

            if use_device:
                h_dev_b = torch.stack(h_dev_list, dim=0)  # [B, max_sized, D]
                mask_b = torch.stack(mask_list, dim=0)

                q_pred = critic(specs_b, h_dev_b.detach(), actions_b, mask_b)
                critic_loss = F.mse_loss(q_pred, rewards_b)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                with torch.no_grad():
                    q_pred = critic(specs_b, h_dev_b.detach(), actions_b, mask_b)
                v_pred = value_net(specs_b, z_topo_b.detach())
                diff = q_pred - v_pred
                weight = torch.where(diff > 0, 0.7, 0.3)
                value_loss = (weight * (diff ** 2)).mean()
                value_opt.zero_grad()
                value_loss.backward()
                value_opt.step()

                with torch.no_grad():
                    q_pred = critic(specs_b, h_dev_b.detach(), actions_b, mask_b)
                    v_pred = value_net(specs_b, z_topo_b.detach())
                advantage = q_pred - v_pred
                weights = torch.clamp(torch.exp(advantage / args.tau), max=10.0)

                x_0 = torch.randn_like(actions_b)
                t_cfm = torch.rand(args.batch_size, device=device)
                x_t = (1 - t_cfm.unsqueeze(-1)) * x_0 + t_cfm.unsqueeze(-1) * actions_b
                u_target = actions_b - x_0
                if use_coupled:
                    roles_b = torch.stack(role_list, dim=0)
                    u_pred = actor(x_t, t_cfm, specs_b, h_dev_b, role=roles_b)
                else:
                    u_pred = actor(x_t, t_cfm, specs_b, h_dev_b)
                # Mask unused slots
                m = mask_b.float()
                actor_loss = (weights.unsqueeze(-1) * m * (u_target - u_pred) ** 2).sum() / m.sum().clamp(min=1.0)

                actor_opt.zero_grad()
                gnn_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                gnn_opt.step()
            else:
                # --- Original slot-indexed path ---
                q_pred = critic(specs_b, z_topo_b.detach(), actions_b)
                critic_loss = F.mse_loss(q_pred, rewards_b)
                
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                
                z_topo_list = []
                for bi, topo_name in enumerate(topos_b):
                    if use_circuit:
                        z = enc(
                            topo_name,
                            _replay_spec(spec_dicts_b[bi], fcs_b[bi]),
                            return_device=False,
                            bounds=bounds,
                            switch_model=switch_model,
                        )
                    else:
                        g = topo_graphs[topo_name]
                        z = gnn_encoder(g.x, g.edge_index)
                    z_topo_list.append(z)
                z_topo_b = torch.cat(z_topo_list, dim=0)
                
                with torch.no_grad():
                    q_pred = critic(specs_b, z_topo_b.detach(), actions_b)
                    
                v_pred = value_net(specs_b, z_topo_b.detach())
                diff = q_pred - v_pred
                weight = torch.where(diff > 0, 0.7, 0.3)
                value_loss = (weight * (diff ** 2)).mean()
                
                value_opt.zero_grad()
                value_loss.backward()
                value_opt.step()
                
                with torch.no_grad():
                    q_pred = critic(specs_b, z_topo_b.detach(), actions_b)
                    v_pred = value_net(specs_b, z_topo_b.detach())
                    
                advantage = q_pred - v_pred
                weights = torch.clamp(torch.exp(advantage / args.tau), max=10.0)
                
                module_actor = actor.module if is_ddp else actor
                if isinstance(module_actor, FlowMatchingPolicy):
                    x_0 = torch.randn_like(actions_b)
                    t_cfm = torch.rand(args.batch_size, device=device)
                    x_t = (1 - t_cfm.unsqueeze(-1)) * x_0 + t_cfm.unsqueeze(-1) * actions_b
                    u_target = actions_b - x_0
                    u_pred = actor(x_t, t_cfm, specs_b, z_topo_b)
                    actor_loss = (weights.unsqueeze(-1) * (u_target - u_pred) ** 2).mean()
                else:
                    noise = torch.randn_like(actions_b)
                    t = torch.randint(0, module_actor.num_timesteps, (args.batch_size,), device=device).float()
                    a_noisy = module_actor.add_noise(actions_b, t.long(), noise)
                    noise_pred = actor(a_noisy, t, specs_b, z_topo_b)
                    actor_loss = (weights.unsqueeze(-1) * (noise - noise_pred) ** 2).mean()
                
                actor_opt.zero_grad()
                gnn_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                gnn_opt.step()
            
    # Clean up distributed processes
    if rank == 0:
        log_fh.close()
        # Save checkpoints
        torch.save(gnn_encoder.state_dict(), os.path.join(run_dir, "gnn_encoder.pt"))
        torch.save(actor.state_dict(), os.path.join(run_dir, "actor.pt"))
        torch.save(critic.state_dict(), os.path.join(run_dir, "critic.pt"))
        torch.save(value_net.state_dict(), os.path.join(run_dir, "value_net.pt"))
        print(f"[Rank 0] Checkpoints written successfully to: {run_dir}")

    if is_ddp:
        dist.destroy_process_group()


# =============================================================================
# MAIN INVOCATION ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Perform a short verification run")
    parser.add_argument("--restrict-to", nargs="+", help="Restrict active topologies to subset", default=None)
    parser.add_argument("--total-timesteps", type=int, default=100, help="Total online training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for updates")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Replay buffer max size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--tau", type=float, default=0.5, help="Temperature for advantage weight exponent")
    parser.add_argument("--ddp-port", type=str, default="29500", help="DDP communication port")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for distributed DDP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sizing", type=str, default="log", choices=["log", "linear"], help="Scaling mode for action parameters")
    parser.add_argument("--actor", type=str, default="cfm", choices=["ddpm", "cfm"], help="Actor type: ddpm (original) or cfm (Conditional Flow Matching)")
    parser.add_argument("--encoder", type=str, default="gin",
                        choices=["gin", "circuit", "circuit-typed", "circuit-typed-param"],
                        help="Topology encoder: gin (paper), circuit (KCL/KVL), "
                             "circuit-typed (terminal-role R-GCN + RWSE), "
                             "or circuit-typed-param (typed + R3 parameter nodes)")
    parser.add_argument("--action-space", type=str, default="slot", choices=["slot", "device"],
                        help=f"Action parameterization: slot ({SLOT_ACTION_DIM}-dim) or device (per-device CFM)")
    parser.add_argument("--fc-mode", type=str, default="spec", choices=["fixed28", "spec"],
                        help="SPICE measurement frequency: spec-tracking (default) or fixed28 (paper ablation)")
    parser.add_argument("--bounds", type=str, default="electrical",
                        choices=["legacy", "electrical", "sky130"],
                        help="Action-to-params bounds: electrical fc-centered (default), "
                             "legacy (paper), or sky130 (PDK geometry)")
    parser.add_argument("--switch-model", type=str, default="ideal",
                        choices=["ideal", "realistic"],
                        help="Switch parasitics: ideal (R_off=10k) or realistic (R_off_eff from C_off)")
    parser.add_argument("--expert-bonus-scale", type=float, default=0.0,
                        help="Scale for heuristic topology bonus (0=off/default, 1=legacy ablation)")
    parser.add_argument("--coupled-actions", action="store_true",
                        help="D1a: role+bounds-conditioned coupled action head "
                             "(always on for --encoder circuit-typed-param)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Optional explicit checkpoint/log directory")
    parser.add_argument("--sim", type=str, default="spice", choices=["spice", "mna"],
                        help="Rollout scorer: spice (ngspice) or mna (matched fast ablation)")
    parser.add_argument("--skip-prior", action="store_true",
                        help="Disable ABCD/physics prior gate (Phase 6 on/off ablation)")
    args = parser.parse_args()
    # If running with multiple GPUs, spawn distributed processes
    if args.gpus > 1:
        print(f"Spawning DDP training across {args.gpus} GPUs...")
        mp.spawn(train, nprocs=args.gpus, args=(args.gpus, args), join=True)
    else:
        train(0, 1, args)
