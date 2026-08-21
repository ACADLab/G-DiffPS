import torch
from torch_geometric.data import Data

NODE_TYPES = {
    "TLine":  [1.0, 0.0, 0.0, 0.0, 0.0],
    "Switch": [0.0, 1.0, 0.0, 0.0, 0.0],
    "Cap":    [0.0, 0.0, 1.0, 0.0, 0.0],
    "Ind":    [0.0, 0.0, 0.0, 1.0, 0.0],
    "Res":    [0.0, 0.0, 0.0, 0.0, 1.0],
}

TOPOLOGY_PARAMS = {
    "Loaded_Line": ["Z0_line", "L_quarter_mm", "C_load_pf", "R_on", "R_off"],
    "Switched_Line": ["Z0_line", "L_short_mm", "L_long_mm", "R_on", "R_off"],
    "Reflection_Type": ["Z0_main", "Z0_branch", "L_quarter_mm", "C_base_pf", "C_tune_pf", "R_on", "R_off"],
    "Switched_Filter": ["C_hpf_pf", "L_hpf_nh", "L_lpf_nh", "C_lpf_pf", "R_on", "R_off"],
    "Vector_Modulator": ["Z0_line", "L_quarter_mm", "G_I_scale", "G_Q_scale", "R_on", "R_off"],
    "All_Pass": ["L_apA_nh", "C_brA_pf", "C_cA_pf", "L_apB_nh", "C_brB_pf", "C_cB_pf", "R_on", "R_off"],
}

def get_topology_graph(topology_name: str) -> Data:
    """Build PyG Data object for the requested topology."""
    # Handle topology capitalization variants
    name = topology_name.lower().replace("_", "")

    if name == "loadedline":
        # Node 0: main lambda/4 transmission line
        # Node 1: input shunt loading capacitor
        # Node 2: input switch R_on/R_off
        # Node 3: output shunt loading capacitor
        # Node 4: output switch R_on/R_off
        x = torch.tensor([
            NODE_TYPES["TLine"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Switch"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Switch"],
        ], dtype=torch.float)
        
        # Bidirectional connections:
        # TLine to Cap input (0-1), Cap input to Switch input (1-2), TLine to Switch input (0-2)
        # TLine to Cap output (0-3), Cap output to Switch output (3-4), TLine to Switch output (0-4)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 2, 0, 3, 0, 4, 3, 4],
            [1, 0, 2, 0, 2, 1, 3, 0, 4, 0, 4, 3]
        ], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    elif name == "switchedline":
        # Node 0: Switch short input (R_in_short)
        # Node 1: Switch long input (R_in_long)
        # Node 2: TLine short (T_short)
        # Node 3: TLine long (T_long)
        # Node 4: Switch short output (R_out_short)
        # Node 5: Switch long output (R_out_long)
        x = torch.tensor([
            NODE_TYPES["Switch"],
            NODE_TYPES["Switch"],
            NODE_TYPES["TLine"],
            NODE_TYPES["TLine"],
            NODE_TYPES["Switch"],
            NODE_TYPES["Switch"],
        ], dtype=torch.float)
        
        edge_index = torch.tensor([
            [0, 2, 2, 4, 1, 3, 3, 5],
            [2, 0, 4, 2, 3, 1, 5, 3]
        ], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    elif name == "reflectiontype":
        # Node 0: Coupler top TLine (T_top)
        # Node 1: Coupler bottom TLine (T_bottom)
        # Node 2: Coupler left TLine (T_left)
        # Node 3: Coupler right TLine (T_right)
        # Node 4: Cap (C_base A)
        # Node 5: Cap (C_tune A)
        # Node 6: Switch (R_path A)
        # Node 7: Cap (C_base B)
        # Node 8: Cap (C_tune B)
        # Node 9: Switch (R_path B)
        x = torch.tensor([
            NODE_TYPES["TLine"],
            NODE_TYPES["TLine"],
            NODE_TYPES["TLine"],
            NODE_TYPES["TLine"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Switch"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Switch"],
        ], dtype=torch.float)
        
        edge_index = torch.tensor([
            [0, 2, 0, 3, 1, 2, 1, 3, 0, 4, 3, 4, 0, 5, 3, 5, 5, 6, 1, 7, 2, 7, 1, 8, 2, 8, 8, 9],
            [2, 0, 3, 0, 2, 1, 3, 1, 4, 0, 4, 3, 5, 0, 5, 3, 6, 5, 7, 1, 7, 2, 8, 1, 8, 2, 9, 8]
        ], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    elif name == "switchedfilter":
        # Node 0: Switch in HPF (R_in_hpf)
        # Node 1: Ind shunt in HPF (Lp_hpf_in)
        # Node 2: Cap series HPF (C_hpf_ser)
        # Node 3: Ind shunt out HPF (Lp_hpf_out)
        # Node 4: Switch out HPF (R_out_hpf)
        # Node 5: Switch in LPF (R_in_lpf)
        # Node 6: Cap shunt in LPF (Cp_lpf_in)
        # Node 7: Ind series LPF (L_lpf_ser)
        # Node 8: Cap shunt out LPF (Cp_lpf_out)
        # Node 9: Switch out LPF (R_out_lpf)
        x = torch.tensor([
            NODE_TYPES["Switch"],
            NODE_TYPES["Ind"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Ind"],
            NODE_TYPES["Switch"],
            NODE_TYPES["Switch"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Ind"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Switch"],
        ], dtype=torch.float)
        
        edge_index = torch.tensor([
            [0, 1, 0, 2, 2, 3, 2, 4, 5, 6, 5, 7, 7, 8, 7, 9],
            [1, 0, 2, 0, 3, 2, 4, 2, 6, 5, 7, 5, 8, 7, 9, 7]
        ], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    elif name == "vectormodulator":
        # Node 0: TLine quad (T_quad)
        # Node 1: Res quad term (R_q_term)
        # Node 2: Res active VCVS (E_I)
        # Node 3: Res active VCVS (E_Q)
        # Node 4: Res output driver (R_drv_out)
        x = torch.tensor([
            NODE_TYPES["TLine"],
            NODE_TYPES["Res"],
            NODE_TYPES["Res"],
            NODE_TYPES["Res"],
            NODE_TYPES["Res"],
        ], dtype=torch.float)
        
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 2, 3, 2, 4, 3, 4],
            [1, 0, 2, 0, 3, 0, 3, 2, 4, 2, 4, 3]
        ], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    elif name == "allpass":
        # Node 0: Switch in apA (R_in_apA)
        # Node 1: Ind series apA (L_apA)
        # Node 2: Cap bridge apA (C_brA)
        # Node 3: Cap center shunt apA (C_cA)
        # Node 4: Switch out apA (R_out_apA)
        # Node 5: Switch in apB (R_in_apB)
        # Node 6: Ind series apB (L_apB)
        # Node 7: Cap bridge apB (C_brB)
        # Node 8: Cap center shunt apB (C_cB)
        # Node 9: Switch out apB (R_out_apB)
        x = torch.tensor([
            NODE_TYPES["Switch"],
            NODE_TYPES["Ind"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Switch"],
            NODE_TYPES["Switch"],
            NODE_TYPES["Ind"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Cap"],
            NODE_TYPES["Switch"],
        ], dtype=torch.float)
        
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 3, 1, 2, 2, 4, 1, 4, 5, 6, 5, 7, 6, 8, 6, 7, 7, 9, 6, 9],
            [1, 0, 2, 0, 3, 1, 2, 1, 4, 2, 4, 1, 6, 5, 7, 5, 8, 6, 7, 6, 9, 7, 9, 6]
        ], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    else:
        raise ValueError(f"Unknown topology name: {topology_name}")
