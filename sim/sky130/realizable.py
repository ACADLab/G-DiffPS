"""Decode a PDK-aware CircuitGraph / Dev netlist into legal SKY130 SPICE.

Targets the ``topology/emit_spice.emit_spice`` path for structural emission,
but replaces ideal R_switch / lumped R/C instances with PDK primitives when
``Dev.pdk_model`` is set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from env.netlist_graph import Dev, TOPOLOGY_NETLIST, _normalize_name, PORT_NETS
from sim.sky130 import load_pin, render_header

VP_EXPR = "1.897e8"


@dataclass
class DecodeResult:
    spice: str
    devices_emitted: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _area_params(w: float) -> str:
    return (
        f"ad='{w}*0.29' as='{w}*0.29' "
        f"pd='2*({w}+0.29)' ps='2*({w}+0.29)'"
    )


def _geo(dev: Dev, params: dict, key: str, default: float) -> float:
    if key in params:
        try:
            return float(params[key])
        except (TypeError, ValueError):
            pass
    if key in dev.geometry:
        try:
            return float(dev.geometry[key])
        except (TypeError, ValueError):
            pass
    # Prefixed by device name, e.g. R_in_path_W_um
    prefixed = f"{key}"
    for cand in (f"{dev.sizes}_{key}" if dev.sizes else None, prefixed):
        if cand and cand in params:
            try:
                return float(params[cand])
            except (TypeError, ValueError):
                pass
    return float(default)


def device_to_spice(
    dname: str,
    dev: Dev,
    params: Optional[dict] = None,
    *,
    state: int = 0,
) -> tuple[str, Optional[str]]:
    """Return (spice_line, error_or_None) for one device."""
    params = params or {}
    pins = list(dev.nets)

    # Ideal / non-PDK path (legacy templates).
    if not dev.pdk_model:
        if dev.dtype == "R_switch":
            val = "{" + (dev.switch_param or "R_on") + "}"
            return f"{dname} {pins[0]} {pins[1]} {val}", None
        if dev.dtype == "R_fixed":
            return f"{dname} {pins[0]} {pins[1]} 50", None
        if dev.dtype == "C":
            key = dev.sizes or "C"
            return f"{dname} {pins[0]} {pins[1]} {{{key}*1e-12}}", None
        if dev.dtype == "L":
            key = dev.sizes or "L"
            return f"{dname} {pins[0]} {pins[1]} {{{key}*1e-9}}", None
        if dev.dtype == "TLine":
            z0 = dev.aux_sizes[0] if dev.aux_sizes else "Z0"
            lmm = dev.sizes or "L_mm"
            return (
                f"{dname} {pins[0]} 0 {pins[1]} 0 "
                f"Z0={{{z0}}} TD={{{lmm}*1e-3 / {VP_EXPR}}}",
                None,
            )
        if dev.dtype == "VCVS":
            gain = dev.sizes or "1"
            return (
                f"{dname} {pins[0]} {pins[1]} {pins[2]} {pins[3]} {{{gain}}}",
                None,
            )
        if dev.dtype == "Port":
            return f"{dname} {pins[0]} {pins[1]} 50", None
        return "", f"unsupported ideal dtype {dev.dtype!r} for {dname}"

    model = dev.pdk_model
    pin = load_pin()
    # MOS switch / transistor
    if "nfet" in model or "pfet" in model:
        if len(pins) < 2:
            return "", f"{dname}: MOS needs ≥2 signal pins"
        drain, source = pins[0], pins[1]
        gate = dev.control.get("gate", f"{dname}_g")
        body = dev.control.get("body", "0")
        w = _geo(dev, params, "W_um", pin["devices"]["nmos"]["geometry"]["W_um"][0])
        l = _geo(dev, params, "L_um", pin["devices"]["nmos"]["geometry"]["L_um"][0])
        nf = int(_geo(dev, params, "nf", 1))
        # Gate bias: on/off from state table when this is a switch.
        lines = []
        if dev.is_switch or dev.switch_param:
            v_on = float(dev.control.get("vgate_on", pin["operating_limits"]["vdd_v"]["nom"]))
            v_off = float(dev.control.get("vgate_off", 0.0))
            on = int(state) in (dev.on_in_states or ())
            vgate = v_on if on else v_off
            if "pfet" in model:
                # PMOS: on when gate is low.
                vgate = v_off if on else v_on
            lines.append(f"V{dname}_g {gate} 0 DC {vgate}")
        lines.append(
            f"X{dname} {drain} {gate} {source} {body} {model} "
            f"L={l} W={w} nf={nf} {_area_params(w)}"
        )
        return "\n".join(lines), None

    if "res_" in model:
        if len(pins) < 2:
            return "", f"{dname}: resistor needs 2 pins"
        body = pins[2] if len(pins) > 2 else "0"
        length = _geo(dev, params, "L_um", 10.0)
        mult = int(_geo(dev, params, "mult", 1))
        return (
            f"X{dname} {pins[0]} {pins[1]} {body} {model} L={length} mult={mult}",
            None,
        )

    if "cap_" in model:
        if len(pins) < 2:
            return "", f"{dname}: capacitor needs 2 pins"
        w = _geo(dev, params, "W_um", 5.0)
        l = _geo(dev, params, "L_um", 5.0)
        mf = int(_geo(dev, params, "mf", 1))
        return (
            f"X{dname} {pins[0]} {pins[1]} {model} W={w} L={l} mf={mf}",
            None,
        )

    return "", f"{dname}: unknown pdk_model {model!r}"


def _control_block(fc_hz: float, n_points: int = 201, bw_pct: float = 20.0) -> list[str]:
    f1 = fc_hz * (1.0 - bw_pct / 200.0)
    f2 = fc_hz * (1.0 + bw_pct / 200.0)
    return [
        ".control",
        "op",
        f"ac lin {n_points} {f1:g} {f2:g}",
        "let s11 = (v(in) - 0.5) / 0.5",
        "let s21 = v(out) / 0.5",
        "let s21_mag_db = db(s21)",
        "let s11_mag_db = db(s11)",
        "let s21_phase  = 180/pi * cph(s21)",
        f"meas ac il_db_at_fc  FIND s21_mag_db AT={fc_hz:g}",
        f"meas ac phase_at_fc  FIND s21_phase  AT={fc_hz:g}",
        f"meas ac rl_db_at_fc  FIND s11_mag_db AT={fc_hz:g}",
        "let phase_deg = phase_at_fc",
        "let il_db     = -1 * il_db_at_fc",
        "let rl_db     = -1 * rl_db_at_fc",
        "let gain_err_db = 0.0",
        "print phase_deg il_db rl_db gain_err_db",
        "quit",
        ".endc",
    ]


def circuit_to_spice(
    netlist: dict[str, Dev],
    *,
    topology_name: str = "custom",
    params: Optional[dict] = None,
    spec: Optional[dict] = None,
    state: int = 0,
    include_pdk_header: bool = True,
    corner: Optional[str] = None,
) -> DecodeResult:
    """Build a full SKY130-capable deck from a Dev netlist dict."""
    params = dict(params or {})
    spec = dict(spec or {})
    fc_ghz = float(spec.get("fc_ghz", 1.0))
    # SKY130 explore band is ≤3 GHz; default C0 smoke uses 1 GHz.
    fc_hz = fc_ghz * 1e9
    bw_pct = float(spec.get("bw_pct", 20.0))
    result = DecodeResult(spice="")

    # Ensure port nets exist.
    try:
        port_in, port_out = PORT_NETS.get(_normalize_name(topology_name), ("in", "out"))
    except Exception:
        port_in, port_out = "in", "out"

    lines: list[str] = []
    if include_pdk_header:
        lines.append(render_header(corner=corner).rstrip())
        lines.append("")

    lines.extend([
        f"* {topology_name} — SKY130-realizable phase shifter (state={state})",
        f".PARAM fc={fc_hz:g}",
        ".option reltol=1e-3 abstol=1e-12 itl1=500 itl2=500",
        "",
        "Vsrc src 0 DC 0.5 AC 1",
        f"Rsrc src {port_in} 50",
        "",
    ])

    # Pass through non-geometry .PARAM values for TLine / ideal leftovers.
    for k, v in params.items():
        if k.endswith(("_um",)) or k in ("nf", "mult", "mf", "W_um", "L_um"):
            continue
        try:
            float(str(v).rstrip("kmunp"))
            lines.append(f".PARAM {k}={v}")
        except Exception:
            lines.append(f".PARAM {k}={v}")

    lines.append("")
    for dname, dev in netlist.items():
        text, err = device_to_spice(dname, dev, params, state=state)
        if err:
            result.errors.append(err)
            continue
        if text:
            lines.append(text)
            result.devices_emitted.append(dname)

    lines.extend([
        "",
        f"Rload {port_out} 0 50",
        "",
    ])
    lines.extend(_control_block(fc_hz, bw_pct=bw_pct))
    lines.append("")
    lines.append(".end")
    result.spice = "\n".join(lines) + "\n"
    return result


def loaded_line_sky130_graph(
    *,
    w_um: float = 5.0,
    l_um: float = 0.15,
    c_load_pf: float = 0.5,
    z0: float = 50.0,
    l_quarter_mm: float = 47.43,  # λ/4 at 1 GHz
) -> tuple[dict[str, Dev], dict[str, Any]]:
    """Manual C0 Loaded_Line with real SKY130 NFET shunt switches.

    Ideal R_switch devices are replaced by nfet_01v8 series elements whose
    gates are biased on/off per state. Cap and TL remain ideal for this first
    smoke (passives can be swapped to MIM/poly later).
    """
    pin = load_pin()
    nmos = pin["devices"]["nmos"]["model"]
    netlist = {
        "T_main": Dev(
            "TLine", ("in", "out"), sizes="L_quarter_mm",
            aux_sizes=("Z0_line",), param_role="length", pdk="sky130",
        ),
        "M_in_path": Dev(
            "R_switch", ("in", "n_in"),
            switch_param="R_path_in", on_in_states=(1,),
            pdk_model=nmos,
            geometry={"W_um": w_um, "L_um": l_um, "nf": 1},
            control={"gate": "g_in", "body": "0",
                     "vgate_on": 1.8, "vgate_off": 0.0},
            is_switch=True, param_role="switch_width", pdk="sky130",
        ),
        "C_in_load": Dev(
            "C", ("n_in", "0"), sizes="C_load_pf",
            param_role="capacitance", pdk="sky130",
        ),
        "M_out_path": Dev(
            "R_switch", ("out", "n_out"),
            switch_param="R_path_out", on_in_states=(1,),
            pdk_model=nmos,
            geometry={"W_um": w_um, "L_um": l_um, "nf": 1},
            control={"gate": "g_out", "body": "0",
                     "vgate_on": 1.8, "vgate_off": 0.0},
            is_switch=True, param_role="switch_width", pdk="sky130",
        ),
        "C_out_load": Dev(
            "C", ("n_out", "0"), sizes="C_load_pf",
            param_role="capacitance", pdk="sky130",
        ),
    }
    params = {
        "Z0_line": z0,
        "L_quarter_mm": l_quarter_mm,
        "C_load_pf": c_load_pf,
        "W_um": w_um,
        "L_um": l_um,
        "nf": 1,
    }
    return netlist, params
