"""Switch parasitics: tech constants, not action dimensions.

A designer does not size R_on / R_off — they buy a switch (or sit in a process)
and the parasitics come with it. Keeping them in the action space is the same
bug species as §9.2 (G_I/G_Q > 1 buying active gain).

Two models:

  ideal      — R_on=3 Ω, R_off=10 kΩ frequency-independent (templates / published
               MLCAD numbers). Isolation is unrealistically good at mmWave.
  realistic  — R_off_eff(tech, fc) = 1 / (2 π f C_off(tech)). Same resistive
               form as the rest of the stack; magnitude matches a capacitive
               off-state at the carrier. At 28 GHz with C_off=20 fF this is
               284 Ω — the T0.1 measurement that collapses series-switch
               branch-select topologies.

`tech` is reinterpreted as switch technology (PIN / GaAs pHEMT / SOI SPDT),
not CMOS/SiGe/GaAs process.
"""
from __future__ import annotations

import math
from typing import Literal

SwitchModel = Literal["ideal", "realistic"]

# tech index -> (name, R_on_ohm, C_off_F)
# C_off values are board-level SPDT-class; calibrate against a real BOM before
# publishing absolute numbers. Ideal model ignores C_off.
TECH_SWITCH: dict[int, tuple[str, float, float]] = {
    0: ("PIN_diode", 3.0, 20e-15),      # ~284 Ω @ 28 GHz
    1: ("GaAs_pHEMT", 2.5, 20e-15),
    2: ("SOI_SPDT", 2.0, 25e-15),      # slightly higher C_off
}

IDEAL_R_ON = 3.0
IDEAL_R_OFF = 1.0e4


def r_off_eff(c_off_f: float, fc_ghz: float) -> float:
    """Equivalent off resistance of C_off at carrier: 1/(2 π f C)."""
    f_hz = max(float(fc_ghz), 1e-6) * 1e9
    return 1.0 / (2.0 * math.pi * f_hz * max(c_off_f, 1e-21))


def switch_params(
    tech: int = 0,
    fc_ghz: float = 28.0,
    model: SwitchModel = "ideal",
) -> dict[str, float]:
    """Return {R_on, R_off} for injection into params_dict / netlist."""
    if model == "ideal":
        return {"R_on": IDEAL_R_ON, "R_off": IDEAL_R_OFF}
    name, r_on, c_off = TECH_SWITCH.get(int(tech), TECH_SWITCH[0])
    _ = name  # available for logging
    return {"R_on": float(r_on), "R_off": float(r_off_eff(c_off, fc_ghz))}


def format_switch_params(params: dict[str, float]) -> dict[str, str]:
    """SPICE-string form matching clamp_spice_value conventions."""
    return {
        "R_on": f"{params['R_on']:.4e}",
        "R_off": f"{params['R_off']:.4e}",
    }
