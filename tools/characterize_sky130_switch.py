#!/usr/bin/env python3
"""Characterize SKY130 nfet_01v8 as a series switch (Milestone B start).

Measures:
  - On-state resistance at Vgs=VDD, small Vds
  - Off-state resistance at Vgs=0, small Vds
  - Off-state AC feedthrough magnitude at selected frequencies

Usage:
  export PDK_ROOT="$(pwd)/pdk"
  python3 tools/characterize_sky130_switch.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim.sky130 import load_pin, run_deck, write_deck

W_UM = [0.42, 1.0, 2.0, 5.0, 10.0]
FREQ_HZ = [1e6, 10e6, 100e6, 1e9]
L_UM = 0.15
VDD = 1.8
VDS_PROBE = 0.05


def _area_params(w: float) -> str:
    return f"ad='{w}*0.29' as='{w}*0.29' pd='2*({w}+0.29)' ps='2*({w}+0.29)'"


def ron_roff_body(w: float, vgs: float) -> str:
    return f"""
.param W={w}
.param VGS={vgs}
.param VDS={VDS_PROBE}

Vds d 0 {{VDS}}
Vg  g 0 {{VGS}}
Vs  s 0 0
Vb  b 0 0

XMn d g s b sky130_fd_pr__nfet_01v8 L={L_UM} W={{W}} nf=1 {_area_params(w)}

.control
  op
  let id = abs(i(Vds))
  let r_ds = {VDS_PROBE} / id
  echo id = $&id
  echo r_ds = $&r_ds
  quit
.endc

.end
"""


def feedthrough_body(w: float, freq: float) -> str:
    # Series NFET between in and out, load 50 ohm to gnd, gate off.
    return f"""
.param W={w}
.param FREQ={freq}

Vin in 0 DC 0 AC 1
Vg  g 0 0
Vb  b 0 0
RL  out 0 50

XMn out g in 0 sky130_fd_pr__nfet_01v8 L={L_UM} W={{W}} nf=1 {_area_params(w)}

.control
  ac lin 1 {freq:g} {freq:g}
  let vout_mag = abs(v(out))
  let att_db = db(v(out))
  echo vout_mag = $&vout_mag
  echo att_db = $&att_db
  quit
.endc

.end
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results/sky130/switch_char.json"))
    ap.add_argument("--corner", default="tt")
    args = ap.parse_args()

    pin = load_pin()
    work = REPO_ROOT / "results/sky130/runs/switch_char"
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for w in W_UM:
        # On
        deck = write_deck(ron_roff_body(w, VDD), corner=args.corner, workdir=work / f"on_w{w}", name="on.spice")
        on = run_deck(deck, metric_keys=["id", "r_ds"], pin=pin)
        # Off
        deck = write_deck(ron_roff_body(w, 0.0), corner=args.corner, workdir=work / f"off_w{w}", name="off.spice")
        off = run_deck(deck, metric_keys=["id", "r_ds"], pin=pin)

        ft = []
        for f in FREQ_HZ:
            deck = write_deck(
                feedthrough_body(w, f),
                corner=args.corner,
                workdir=work / f"ft_w{w}_f{f:g}",
                name="ft.spice",
            )
            res = run_deck(deck, metric_keys=["vout_mag", "att_db"], pin=pin)
            ft.append({"freq_hz": f, "ok": res.ok, "metrics": res.metrics, "error": res.error})

        row = {
            "W_um": w,
            "L_um": L_UM,
            "vdd_v": VDD,
            "on": {"ok": on.ok, "metrics": on.metrics, "error": on.error},
            "off": {"ok": off.ok, "metrics": off.metrics, "error": off.error},
            "feedthrough_off": ft,
        }
        rows.append(row)
        ron = on.metrics.get("r_ds", float("nan"))
        roff = off.metrics.get("r_ds", float("nan"))
        ratio = (roff / ron) if on.ok and off.ok and ron else float("nan")
        print(f"W={w:>5.2f} um  Ron={ron:.3g} Ohm  Roff={roff:.3g} Ohm  Roff/Ron={ratio:.3g}")

    # Suggest an initial band where off isolation is still useful (att_db < -20 at that W=5)
    suggestion = {
        "criterion": "prefer frequencies where |att_db| >= 20 for W>=5 um off switch into 50 Ohm",
        "candidates_hz": [],
    }
    for f in FREQ_HZ:
        ok_ws = []
        for row in rows:
            for ft in row["feedthrough_off"]:
                if ft["freq_hz"] == f and ft["ok"] and row["W_um"] >= 5.0:
                    att = ft["metrics"].get("att_db", 0.0)
                    ok_ws.append(att <= -20.0)
        if ok_ws and all(ok_ws):
            suggestion["candidates_hz"].append(f)

    out = {
        "corner": args.corner,
        "device": pin["devices"]["nmos"]["model"],
        "probe": {"vds_v": VDS_PROBE, "load_ohm": 50},
        "rows": rows,
        "frequency_suggestion": suggestion,
        "notes": [
            "Ron/Roff use DC small-Vds probes; not RF s-parameter characterization.",
            "Feedthrough is |Vout/Vin| with gate off and 50 Ohm load — first-pass isolation map.",
            "Model validity for RF above ~few hundred MHz needs foundry/docs cross-check (plan §2).",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    print("Frequency candidates with >=20 dB off isolation @ W>=5 um:", suggestion["candidates_hz"])


if __name__ == "__main__":
    main()
