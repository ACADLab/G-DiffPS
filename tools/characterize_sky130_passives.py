#!/usr/bin/env python3
"""Characterize SKY130 passives + switch corners (Milestone B continuation).

Sweeps:
  * res_xhigh_po_0p35 resistance vs L
  * cap_mim_m3_1 capacitance vs W×L
  * nfet switch Ron @ W=5 um across corners tt/ss/ff

Usage:
  export PDK_ROOT="$(pwd)/pdk"
  python3 tools/characterize_sky130_passives.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim.sky130 import load_pin, run_deck, write_deck


def res_body(length_um: float) -> str:
    return f"""
Vforce a 0 0.1
Xb a 0 0 sky130_fd_pr__res_xhigh_po_0p35 L={length_um} mult=1
.control
  op
  let i_force = abs(i(Vforce))
  let r_meas = 0.1 / i_force
  echo i_force = $&i_force
  echo r_meas = $&r_meas
  quit
.endc
.end
"""


def cap_body(w: float, l: float) -> str:
    return f"""
Vac a 0 DC 0 AC 0.01
XC a 0 sky130_fd_pr__cap_mim_m3_1 W={w} L={l} mf=1
.control
  ac lin 1 1e6 1e6
  let iac = abs(i(Vac))
  let c_est = iac / (6.283185307179586 * 1.0e6 * 0.01)
  echo iac = $&iac
  echo c_est = $&c_est
  quit
.endc
.end
"""


def ron_body(w: float = 5.0) -> str:
    return f"""
.param W={w}
Vds d 0 0.05
Vg  g 0 1.8
Vs  s 0 0
Vb  b 0 0
XMn d g s b sky130_fd_pr__nfet_01v8 L=0.15 W={{W}} nf=1 ad='{w}*0.29' as='{w}*0.29' pd='2*({w}+0.29)' ps='2*({w}+0.29)'
.control
  op
  let id = abs(i(Vds))
  let r_ds = 0.05 / id
  echo id = $&id
  echo r_ds = $&r_ds
  quit
.endc
.end
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results/sky130/passive_char.json"))
    args = ap.parse_args()
    pin = load_pin()
    work = REPO_ROOT / "results/sky130/runs/passive_char"
    work.mkdir(parents=True, exist_ok=True)

    res_rows = []
    for L in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        deck = write_deck(res_body(L), workdir=work / f"res_L{L}", name="res.spice")
        r = run_deck(deck, metric_keys=["i_force", "r_meas"], pin=pin)
        res_rows.append({"L_um": L, "ok": r.ok, "metrics": r.metrics})
        print(f"R  L={L:>5.1f} um  R={r.metrics.get('r_meas', float('nan')):.3g} Ohm  ok={r.ok}")

    cap_rows = []
    for w, l in [(2, 2), (5, 5), (10, 10), (20, 20), (5, 10), (10, 5)]:
        deck = write_deck(cap_body(w, l), workdir=work / f"cap_{w}x{l}", name="cap.spice")
        r = run_deck(deck, metric_keys=["iac", "c_est"], pin=pin)
        c_fF = (r.metrics.get("c_est") or 0) * 1e15
        cap_rows.append({"W_um": w, "L_um": l, "ok": r.ok, "metrics": r.metrics, "c_fF": c_fF})
        print(f"C  {w}x{l} um  C={c_fF:.2f} fF  ok={r.ok}")

    corner_rows = []
    for corner in pin["pdk"]["corners"]:
        deck = write_deck(ron_body(5.0), corner=corner, workdir=work / f"ron_{corner}", name="ron.spice")
        r = run_deck(deck, metric_keys=["id", "r_ds"], pin=pin)
        corner_rows.append({"corner": corner, "W_um": 5.0, "ok": r.ok, "metrics": r.metrics})
        print(f"Ron W=5 corner={corner}  Ron={r.metrics.get('r_ds', float('nan')):.3g} Ohm  ok={r.ok}")

    out = {
        "resistor": {"model": pin["devices"]["resistor"]["model"], "rows": res_rows},
        "capacitor": {"model": pin["devices"]["capacitor"]["model"], "rows": cap_rows},
        "switch_ron_corners": {"model": pin["devices"]["nmos"]["model"], "rows": corner_rows},
        "notes": [
            "Resistor: DC small-signal R vs drawn length.",
            "Capacitor: AC |I|/(2πfVac) estimate at 1 MHz.",
            "Ron corners: Vgs=1.8 V, Vds=50 mV, W=5 um, L=0.15 um.",
        ],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {path}")
    all_ok = all(r["ok"] for r in res_rows + cap_rows + corner_rows)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
