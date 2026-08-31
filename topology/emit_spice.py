"""Netlist dict → ngspice ``.sp`` template emitter."""
from __future__ import annotations

import os
from typing import Optional

from env.netlist_graph import Dev, TOPOLOGY_NETLIST


VP_EXPR = "1.897e8"
FC_DEFAULT = "28e9"


def _dev_line(dname: str, dev: Dev) -> list[str]:
    """Device dict keys are SPICE element names (R*/C*/L*/T*/E*)."""
    lines = []
    pins = dev.nets
    if dev.dtype == "R_switch":
        val = "{" + (dev.switch_param or "R_on") + "}"
        lines.append(f"{dname} {pins[0]} {pins[1]} {val}")
    elif dev.dtype == "R_fixed":
        lines.append(f"{dname} {pins[0]} {pins[1]} 50")
    elif dev.dtype == "C":
        key = dev.sizes or "C"
        lines.append(f"{dname} {pins[0]} {pins[1]} {{{key}*1e-12}}")
    elif dev.dtype == "L":
        key = dev.sizes or "L"
        lines.append(f"{dname} {pins[0]} {pins[1]} {{{key}*1e-9}}")
    elif dev.dtype == "TLine":
        z0 = dev.aux_sizes[0] if dev.aux_sizes else "Z0"
        lmm = dev.sizes or "L_mm"
        lines.append(
            f"{dname} {pins[0]} 0 {pins[1]} 0 "
            f"Z0={{{z0}}} TD={{{lmm}*1e-3 / {VP_EXPR}}}"
        )
    elif dev.dtype == "VCVS":
        gain = dev.sizes or "1"
        lines.append(f"{dname} {pins[0]} {pins[1]} {pins[2]} {pins[3]} {{{gain}}}")
    return lines


def _collect_switch_params(netlist: dict[str, Dev]) -> list[str]:
    params = []
    seen = set()
    for dev in netlist.values():
        if dev.switch_param and dev.switch_param not in seen:
            seen.add(dev.switch_param)
            params.append(dev.switch_param)
    return sorted(params)


def _state_table_lines(netlist: dict[str, Dev], ideal_step: float,
                       bits: int = 1) -> list[str]:
    sw_params = _collect_switch_params(netlist)
    lines = [
        "* === STATE_TABLE ===",
        f"* bits={bits}",
        f"* ideal_step_deg={ideal_step}",
    ]
    # State 0: switches with on_in_states containing 0 are R_on, else R_off
    s0_parts = []
    s1_parts = []
    for sp in sw_params:
        # Find representative device for this param
        on0 = any(d.switch_param == sp and 0 in d.on_in_states
                  for d in netlist.values())
        on1 = any(d.switch_param == sp and 1 in d.on_in_states
                  for d in netlist.values())
        s0_parts.append(f"{sp}={'{R_on}' if on0 else '{R_off}'}")
        s1_parts.append(f"{sp}={'{R_on}' if on1 else '{R_off}'}")
    lines.append("* state_0: " + " ".join(s0_parts))
    lines.append("* state_1: " + " ".join(s1_parts))
    lines.append("* === END_STATE_TABLE ===")
    return lines


def _param_defaults(netlist: dict[str, Dev]) -> list[str]:
    keys = []
    seen = set()
    for dev in netlist.values():
        for k in ([dev.sizes] if dev.sizes else []) + list(dev.aux_sizes):
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
    lines = [".PARAM R_on=3 R_off=10k", f".PARAM fc={FC_DEFAULT}"]
    for k in keys:
        if k.startswith("Z0"):
            lines.append(f".PARAM {k}=50")
        elif k.endswith("_mm"):
            lines.append(f".PARAM {k}=1.69")
        elif k.endswith("_pf"):
            lines.append(f".PARAM {k}=0.1")
        elif k.endswith("_nh"):
            lines.append(f".PARAM {k}=0.3")
        elif k.endswith("_br_pf"):
            lines.append(f".PARAM {k}=0.1")
        elif k.endswith("_c_pf"):
            lines.append(f".PARAM {k}=0.05")
        else:
            lines.append(f".PARAM {k}=1.0")
    # FRAMEWORK_CONTROLLED switch params
    for sp in _collect_switch_params(netlist):
        lines.append(f".PARAM {sp}=3      $ FRAMEWORK_CONTROLLED")
    return lines


def emit_spice(
    topology_name: str,
    netlist: dict[str, Dev],
    ideal_step_deg: float = -90.0,
    out_path: Optional[str] = None,
) -> str:
    """Emit a full ngspice netlist string; optionally write to ``out_path``."""
    header = [
        f"* {topology_name} — composed phase shifter, 1-bit",
        "",
    ]
    header.extend(_state_table_lines(netlist, ideal_step_deg))
    header.append("")
    header.extend(_param_defaults(netlist))
    header.extend([
        "",
        ".option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500",
        "",
        "Vsrc src 0 DC 0.5 AC 1",
        "Rsrc src in 50",
        "",
    ])
    body = []
    for dname in sorted(netlist.keys()):
        body.extend(_dev_line(dname, netlist[dname]))
    footer = [
        "",
        "Rload out 0 50",
        "",
        ".control",
        "op",
        "ac lin 201 24G 32G",
        "",
        "let s11 = (v(in) - 0.5) / 0.5",
        "let s21 = (v(out) / 0.5)",
        "let s21_mag_db = db(s21)",
        "let s11_mag_db = db(s11)",
        "let s21_phase  = 180/pi * cph(s21)",
        "",
        "meas ac il_db_at_fc  FIND s21_mag_db AT=28e9",
        "meas ac phase_at_fc  FIND s21_phase  AT=28e9",
        "meas ac rl_db_at_fc  FIND s11_mag_db AT=28e9",
        "",
        "let phase_deg = phase_at_fc",
        "let il_db     = -1 * il_db_at_fc",
        "let rl_db     = -1 * rl_db_at_fc",
        "let gain_err_db = 0.0",
        "print phase_deg il_db rl_db gain_err_db",
        "",
        "quit",
        ".endc",
        "",
        ".end",
    ]
    text = "\n".join(header + body + footer) + "\n"
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(text)
    return text


def emit_registered(topology_name: str, ideal_step_deg: float,
                    out_path: str) -> str:
    nl = TOPOLOGY_NETLIST[topology_name]
    return emit_spice(topology_name, nl, ideal_step_deg, out_path)
