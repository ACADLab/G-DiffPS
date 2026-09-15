"""Register a first SKY130-realizable Loaded_Line into the topology tables."""
from __future__ import annotations

from env.topology_registry import register_topology
from sim.sky130.realizable import loaded_line_sky130_graph


def register_sky130_loaded_line(overwrite: bool = True) -> str:
    netlist, _params = loaded_line_sky130_graph()
    name = "Loaded_Line_SKY130"
    register_topology(
        name,
        netlist,
        port_nets=("in", "out"),
        n_states=2,
        param_keys=["Z0_line", "L_quarter_mm", "C_load_pf", "W_um", "L_um"],
        ideal_step=-22.5,
        overwrite=overwrite,
    )
    return name


if __name__ == "__main__":
    n = register_sky130_loaded_line()
    print("registered", n)
