"""SKY130 harness smoke tests (skip if PDK or ngspice missing)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim.sky130 import environment_report, load_pin, run_deck, write_deck


@pytest.fixture(scope="module")
def env():
    os.environ.setdefault("PDK_ROOT", str(REPO_ROOT / "pdk"))
    return environment_report()


def test_pin_file_loads():
    pin = load_pin()
    assert pin["pdk"]["variant"] == "sky130A"
    assert pin["pdk"]["volare_hash"]
    assert "nfet_01v8" in pin["devices"]["nmos"]["model"]


def test_environment_or_skip(env):
    if not env["ngspice_on_path"]:
        pytest.skip("ngspice not on PATH")
    if not env["ngspice_lib_ok"]:
        pytest.skip("SKY130 PDK not enabled under PDK_ROOT")
    assert env["ngspice_version"]


def test_nfet_deck_or_skip(env):
    if not env["ngspice_on_path"] or not env["ngspice_lib_ok"]:
        pytest.skip("SKY130 environment incomplete")
    body = (REPO_ROOT / "sim/sky130/testbenches/nfet_01v8_dc.spice").read_text()
    body = "\n".join(ln for ln in body.splitlines() if not ln.strip().lower().startswith(".lib")) + "\n"
    deck = write_deck(body, workdir=REPO_ROOT / "results/sky130/runs/pytest_nfet", name="nfet.spice")
    res = run_deck(deck, metric_keys=["id_op", "vgs_op", "vds_op"])
    assert res.ok, res.error
    assert res.metrics["id_op"] > 0
