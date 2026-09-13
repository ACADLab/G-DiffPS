#!/usr/bin/env python3
"""Batch-run the pinned SKY130 device / OTA integration decks.

Usage:
  export PDK_ROOT="$(pwd)/pdk"
  python tools/run_sky130_harness.py
  python tools/run_sky130_harness.py --deck nfet_01v8_dc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.sky130 import dump_environment_report, load_pin, run_deck, write_deck

TB_DIR = REPO_ROOT / "sim" / "sky130" / "testbenches"

DECKS = {
    "nfet_01v8_dc": {
        "file": "nfet_01v8_dc.spice",
        "metrics": ["id_op", "vgs_op", "vds_op", "id_max"],
    },
    "pfet_01v8_dc": {
        "file": "pfet_01v8_dc.spice",
        "metrics": ["id_op", "vsg_op", "vsd_op"],
    },
    "res_xhigh_po_0p35_dc": {
        "file": "res_xhigh_po_0p35_dc.spice",
        "metrics": ["i_force", "r_meas"],
    },
    "cap_mim_m3_1_ac": {
        "file": "cap_mim_m3_1_ac.spice",
        "metrics": ["iac", "c_est"],
    },
    "ota_miller_dc_ac_tran": {
        "file": "ota_miller_dc_ac_tran.spice",
        "metrics": ["vout_op", "vtail", "vnbias", "gain_db_1k"],
    },
    "ota_miller_tran": {
        "file": "ota_miller_tran.spice",
        "metrics": ["vout_tran_end", "vout_tran_mid"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", choices=list(DECKS) + ["all"], default="all")
    ap.add_argument("--corner", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sky130" / "harness_results.json"))
    args = ap.parse_args()

    pin = load_pin()
    env = dump_environment_report()
    print(json.dumps({"environment": env}, indent=2))
    if not env["ngspice_on_path"] or not env["ngspice_lib_ok"]:
        print("SKY130 environment incomplete — aborting deck runs.", file=sys.stderr)
        sys.exit(2)

    names = list(DECKS) if args.deck == "all" else [args.deck]
    results = {}
    work = REPO_ROOT / "results" / "sky130" / "runs"
    work.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for name in names:
        meta = DECKS[name]
        body = (TB_DIR / meta["file"]).read_text()
        # Strip any accidental leading .lib — harness injects the pinned include.
        body_lines = [ln for ln in body.splitlines() if not ln.strip().lower().startswith(".lib")]
        body = "\n".join(body_lines) + "\n"
        deck_path = write_deck(body, corner=args.corner, workdir=work / name, name=meta["file"])
        res = run_deck(deck_path, metric_keys=meta["metrics"], pin=pin)
        results[name] = {
            "ok": res.ok,
            "returncode": res.returncode,
            "metrics": res.metrics,
            "error": res.error,
            "deck": str(deck_path),
            "log": res.raw_path,
        }
        status = "PASS" if res.ok else "FAIL"
        print(f"[{status}] {name}  metrics={res.metrics}  err={res.error}")
        all_ok = all_ok and res.ok

    out = {
        "corner": args.corner or pin["pdk"]["default_corner"],
        "environment": env,
        "results": results,
        "all_ok": all_ok,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
