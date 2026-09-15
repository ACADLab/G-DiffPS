"""Hand-defined RF phase-shifter motif vocabulary.

These are textbook building blocks, not mined from six data points:

  switched-line     SPDT pair selecting between two line lengths
  loaded-line       series TLine with switched shunt reactive loads
  reflection-type   3 dB branchline hybrid plus a pair of reflective loads
  switched-filter   switch-selected high-pass / low-pass sections
  vector-modulator  I/Q split plus vector combine
  all-pass          bridged-T all-pass section (also the compensation
                    network that shows up inside switched-line designs)

Level-3 motif nodes attach to member devices. Probe-accuracy claims over
this six-class set are not reported; the vocabulary is a design prior.
"""
from __future__ import annotations

from dataclasses import dataclass


MOTIF_TYPES = (
    "spdt",
    "tline",
    "shunt_load",
    "hybrid",
    "reflective_load",
    "hpf",
    "lpf",
    "iq_split",
    "iq_combine",
    "allpass",
)
MOTIF_TYPE_IDX = {t: i for i, t in enumerate(MOTIF_TYPES)}
N_MOTIF_TYPES = len(MOTIF_TYPES)


@dataclass(frozen=True)
class MotifSpec:
    name: str
    mtype: str
    members: tuple[str, ...]
    min_members: int = 1


# Device names match TOPOLOGY_NETLIST (not synthetic P_in/P_out).
TOPOLOGY_MOTIFS: dict[str, tuple[MotifSpec, ...]] = {
    "Loaded_Line": (
        MotifSpec("line", "tline", ("T_main",)),
        MotifSpec("load_in", "shunt_load", ("R_in_path", "C_in_load"), 2),
        MotifSpec("load_out", "shunt_load", ("R_out_path", "C_out_load"), 2),
    ),
    "Switched_Line": (
        MotifSpec("spdt_in", "spdt", ("R_in_short", "R_in_long"), 2),
        MotifSpec("arm_short", "tline", ("T_short",)),
        MotifSpec("arm_long", "tline", ("T_long",)),
        MotifSpec("spdt_out", "spdt", ("R_out_short", "R_out_long"), 2),
    ),
    "Reflection_Type": (
        MotifSpec("hybrid", "hybrid",
                  ("T_top", "T_bottom", "T_left", "T_right"), 4),
        MotifSpec("load_A", "reflective_load",
                  ("C_baseA", "C_tuneA", "R_pathA"), 2),
        MotifSpec("load_B", "reflective_load",
                  ("C_baseB", "C_tuneB", "R_pathB"), 2),
    ),
    "Switched_Filter": (
        MotifSpec("spdt_in", "spdt", ("R_in_hpf", "R_in_lpf"), 2),
        MotifSpec("hpf", "hpf",
                  ("R_in_hpf", "Lp_hpf_in", "C_hpf_ser", "Lp_hpf_out", "R_out_hpf"), 3),
        MotifSpec("lpf", "lpf",
                  ("R_in_lpf", "Cp_lpf_in", "L_lpf_ser", "Cp_lpf_out", "R_out_lpf"), 3),
        MotifSpec("spdt_out", "spdt", ("R_out_hpf", "R_out_lpf"), 2),
    ),
    "Vector_Modulator": (
        MotifSpec("iq_split", "iq_split", ("T_quad", "R_q_term"), 1),
        MotifSpec("iq_combine", "iq_combine", ("E_I", "E_Q", "R_drv_out"), 2),
    ),
    "All_Pass": (
        MotifSpec("spdt_in", "spdt", ("R_in_apA", "R_in_apB"), 2),
        MotifSpec("section_A", "allpass",
                  ("L_apA_ser1", "L_apA_ser2", "C_brA_brg", "C_cA_shnt"), 3),
        MotifSpec("section_B", "allpass",
                  ("L_apB_ser1", "L_apB_ser2", "C_brB_brg", "C_cB_shnt"), 3),
        MotifSpec("spdt_out", "spdt", ("R_out_apA", "R_out_apB"), 2),
    ),
}


def motif_instances(topology: str, device_names: list[str]) -> list[dict]:
    """Instantiate motifs whose member devices are present in ``device_names``."""
    specs = TOPOLOGY_MOTIFS.get(topology, ())
    present = set(device_names)
    out = []
    for spec in specs:
        members = [d for d in spec.members if d in present]
        if len(members) < spec.min_members:
            continue
        out.append({
            "name": spec.name,
            "mtype": spec.mtype,
            "members": members,
        })
    return out
