"""Test that --fc-mode spec rewrites the AC sweep and meas AT= lines."""
from __future__ import annotations

import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_diffusion import make_spice_netlist, rewrite_control_block
from env.graph_utils import TOPOLOGY_PARAMS


def test_fixed28_unchanged():
    for topo in TOPOLOGY_PARAMS:
        path = os.path.join(REPO_ROOT, f"specset/templates/{topo.lower()}.sp")
        with open(path) as fh:
            raw = fh.read()
        out = rewrite_control_block(raw, fc_ghz=2.4, bw_pct=30.0, fc_mode="fixed28")
        assert "AT=28e9" in out or "AT=28E9" in out.upper().replace(" ", "")
        assert "24G" in out


def test_spec_mode_rewrites_all_templates():
    for topo in TOPOLOGY_PARAMS:
        params = {k: "1.0" for k in TOPOLOGY_PARAMS[topo]}
        spec = {"fc_ghz": 2.4, "bw_pct": 40.0}
        path = make_spice_netlist(topo, params, spec_dict=spec, fc_mode="spec")
        try:
            with open(path) as fh:
                text = fh.read()
            assert "2.4G" in text or "2.40000G" in text or "2400000000" in text, text
            # Measurement point must not still be hardcoded 28e9
            assert "AT=28e9" not in text
            assert "AT=28E9" not in text
            # Sweep should not still be 24G-32G
            assert "24G 32G" not in text
        finally:
            os.remove(path)


def test_spec_mode_38ghz():
    params = {k: "1.0" for k in TOPOLOGY_PARAMS["Loaded_Line"]}
    path = make_spice_netlist(
        "Loaded_Line", params,
        spec_dict={"fc_ghz": 38.0, "bw_pct": 20.0},
        fc_mode="spec",
    )
    try:
        text = open(path).read()
        assert "38G" in text or "38.0G" in text
    finally:
        os.remove(path)


if __name__ == "__main__":
    test_fixed28_unchanged()
    print("OK fixed28")
    test_spec_mode_rewrites_all_templates()
    print("OK spec mode all templates")
    test_spec_mode_38ghz()
    print("OK 38 GHz")
    print("ALL FC-HARNESS TESTS PASSED")
