"""Heuristic topology scoring for phase-shifter design.

Each topology gets a score given a target spec; the highest-scoring topology
is the expert-recommended choice. Used both to label the SpecSet-PhaseShifter
benchmark and to provide reward shaping during DQN training.

Topology selection rules grounded in standard RF design tradeoffs (see e.g.
Razavi, RF Microelectronics, 2nd ed., Ch. 13). Tune empirically against
simulation results once template skeletons are validated.
"""

TOPOLOGY_LABELS = [
    "Switched_Line",
    "Loaded_Line",
    "Reflection_Type",
    "Switched_Filter",
    "Vector_Modulator",
    "All_Pass",
]


def score_topology(topology: str, spec: dict) -> float:
    score = 10.0  # neutral starting score; bonuses/penalties applied below

    fc = spec["fc_ghz"]
    bw_pct = spec["bw_pct"]
    bits = spec["phase_bits"]            # 0 = analog continuous; else 3,4,5,6
    pwr_mw = spec["pmax_mw"]
    coverage = spec["phase_coverage_deg"]
    tech = spec["tech"]                  # 0=CMOS, 1=SiGe, 2=GaAs

    if topology == "Switched_Line":
        # Digital, simple control, but transmission lines get large at low fc
        if bits >= 4:
            score += 5
        if fc > 20:
            score -= 3                   # mmWave area still large
        if fc < 5:
            score -= 5                   # too bulky on-chip below ~5 GHz
        if bw_pct > 30:
            score += 2

    elif topology == "Loaded_Line":
        # Analog continuous control via varactors; narrowband
        if bits == 0:
            score += 4
        if bw_pct > 25:
            score -= 5
        if fc < 10:
            score += 2
        if coverage > 180:
            score -= 4                   # struggles to reach full 360 deg

    elif topology == "Reflection_Type":
        # 3-dB coupler + varactors; broadband, mmWave-friendly
        if fc > 15:
            score += 5
        if bw_pct > 30:
            score += 4
        if pwr_mw < 5:
            score -= 3                   # coupler footprint costs DC

    elif topology == "Switched_Filter":
        # High-pass / low-pass swap; broadband digital
        if bits >= 4 and bw_pct > 40:
            score += 8
        if fc < 6:
            score += 3
        if fc > 20:
            score -= 2

    elif topology == "Vector_Modulator":
        # I/Q sum; full 360 deg coverage, fine resolution, power-hungry
        if bits >= 5:
            score += 6
        if coverage >= 360:
            score += 3
        if pwr_mw < 10:
            score -= 5
        if bw_pct > 30:
            score += 3

    elif topology == "All_Pass":
        # Schiffman-style; broadband fixed phase difference
        if bw_pct > 50:
            score += 6
        if bits == 0:
            score += 2
        if coverage > 180:
            score -= 3

    # Technology preference
    if tech == 2 and topology in ("Switched_Line", "Reflection_Type"):
        score += 1                       # GaAs MMIC favors distributed

    return score