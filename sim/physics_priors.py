import torch
import numpy as np

def compute_tline_abcd(Z0, L_mm, freq_hz):
    """
    Calculates lossless ABCD transmission line matrices.
    
    Args:
        Z0: Characteristic impedance (Ohm)
        L_mm: Length of the line (mm)
        freq_hz: Frequency (Hz)
        
    Returns:
        A, B, C, D: Complex ABCD matrix components
    """
    c_speed = 3e8
    vp = c_speed / 2.0  # Assumes eps_eff = 2.5/4.0 average. Baseline eps_eff=2.5.
    beta = 2 * np.pi * freq_hz / vp
    theta = beta * (L_mm * 1e-3)
    
    # Using complex type
    A = torch.cos(torch.tensor(theta)) + 0j
    B = 1j * Z0 * torch.sin(torch.tensor(theta))
    C = 1j * (1.0 / Z0) * torch.sin(torch.tensor(theta))
    D = torch.cos(torch.tensor(theta)) + 0j
    return A, B, C, D

def abcd_to_y(A, B, C, D):
    """Convert ABCD matrix to Y admittance matrix."""
    Y11 = D / B
    Y12 = (B * C - A * D) / B
    Y21 = -1.0 / B
    Y22 = A / B
    return Y11, Y12, Y21, Y22

def y_to_abcd(Y11, Y12, Y21, Y22):
    """Convert Y admittance matrix to ABCD matrix."""
    A = -Y22 / Y21
    B = -1.0 / Y21
    C = Y12 - Y11 * Y22 / Y21
    D = -Y11 / Y21
    return A, B, C, D

def compute_loaded_line_s_params(Z0_line, L_quarter_mm, C_load_pf, R_on, R_off, freq_hz, state):
    """
    Analytic S-parameters of Loaded-Line phase shifter.
    
    State 0: switches off (R_off).
    State 1: switches on (R_on).
    """
    omega = 2 * np.pi * freq_hz
    R = R_on if state == 1 else R_off
    C = C_load_pf * 1e-12
    
    # Shunt branch impedance: Z = R + 1/(j*omega*C)
    Z_branch = R + 1.0 / (1j * omega * C)
    Y_shunt = 1.0 / Z_branch
    
    # Shunt branch ABCD matrix
    # [1, 0; Y_shunt, 1]
    A_sh, B_sh, C_sh, D_sh = 1.0 + 0j, 0.0 + 0j, Y_shunt, 1.0 + 0j
    
    # Transmission line ABCD matrix
    A_tl, B_tl, C_tl, D_tl = compute_tline_abcd(Z0_line, L_quarter_mm, freq_hz)
    
    # Cascade: Shunt A * TL * Shunt B
    # M1 = Shunt * TL
    A1 = A_sh * A_tl + B_sh * C_tl
    B1 = A_sh * B_tl + B_sh * D_tl
    C1 = C_sh * A_tl + D_sh * C_tl
    D1 = C_sh * B_tl + D_sh * D_tl
    
    # M_total = M1 * Shunt
    A = A1 * A_sh + B1 * C_sh
    B = A1 * B_sh + B1 * D_sh
    C = C1 * A_sh + D1 * C_sh
    D = C1 * B_sh + D1 * D_sh
    
    # Convert ABCD to S-parameters (50-ohm reference)
    Z0_ref = 50.0
    denom = A + B / Z0_ref + C * Z0_ref + D
    s11 = (A + B / Z0_ref - C * Z0_ref - D) / denom
    s21 = 2.0 / denom
    
    return s11, s21

def compute_switched_line_s_params(Z0_line, L_short_mm, L_long_mm, R_on, R_off, freq_hz, state):
    """
    Analytic S-parameters of Switched-Line phase shifter using parallel combination.
    """
    R_sh = R_on if state == 0 else R_off
    R_lo = R_off if state == 0 else R_on
    
    # Short path branch ABCD matrix: Switch * TL_short * Switch
    A_tls, B_tls, C_tls, D_tls = compute_tline_abcd(Z0_line, L_short_mm, freq_hz)
    # [1, R_sh; 0, 1] * TL * [1, R_sh; 0, 1]
    # M1 = Switch * TL
    A1 = A_tls + R_sh * C_tls
    B1 = B_tls + R_sh * D_tls
    C1 = C_tls
    D1 = D_tls
    # M_short = M1 * Switch
    A_s = A1
    B_s = A1 * R_sh + B1
    C_s = C1
    D_s = C1 * R_sh + D1
    
    # Long path branch ABCD matrix: Switch * TL_long * Switch
    A_tll, B_tll, C_tll, D_tll = compute_tline_abcd(Z0_line, L_long_mm, freq_hz)
    # M1_l = Switch * TL
    A1_l = A_tll + R_lo * C_tll
    B1_l = B_tll + R_lo * D_tll
    C1_l = C_tll
    D1_l = D_tll
    # M_long = M1_l * Switch
    A_l = A1_l
    B_l = A1_l * R_lo + B1_l
    C_l = C1_l
    D_l = C1_l * R_lo + D1_l
    
    # Convert both paths to Admittance Y-parameters to add them in parallel
    Y11_s, Y12_s, Y21_s, Y22_s = abcd_to_y(A_s, B_s, C_s, D_s)
    Y11_l, Y12_l, Y21_l, Y22_l = abcd_to_y(A_l, B_l, C_l, D_l)
    
    # Sum parallel admittances
    Y11 = Y11_s + Y11_l
    Y12 = Y12_s + Y12_l
    Y21 = Y21_s + Y21_l
    Y22 = Y22_s + Y22_l
    
    # Convert summed Y matrix back to ABCD
    A, B, C, D = y_to_abcd(Y11, Y12, Y21, Y22)
    
    # S-parameters
    Z0_ref = 50.0
    denom = A + B / Z0_ref + C * Z0_ref + D
    s11 = (A + B / Z0_ref - C * Z0_ref - D) / denom
    s21 = 2.0 / denom
    
    return s11, s21

def check_physics_priors(topology_name: str, params_dict: dict, fc_ghz: float) -> bool:
    """
    Microwave physics pre-filter.
    Returns True if parameters yield physically sane performance (s11 and s21),
    Returns False if they are wildly non-resonant.
    """
    name = topology_name.lower().replace("_", "")
    freq_hz = fc_ghz * 1e9
    
    # Parse parameters
    try:
        def get_val(k, default):
            v = params_dict.get(k)
            if v is None:
                return default
            # Clean and parse SPICE suffix
            v = str(v).strip().lower()
            if v.endswith('meg'): return float(v[:-3]) * 1e6
            if v.endswith('u'): return float(v[:-1]) * 1e-6
            if v.endswith('n'): return float(v[:-1]) * 1e-9
            if v.endswith('p'): return float(v[:-1]) * 1e-12
            if v.endswith('k'): return float(v[:-1]) * 1e3
            if v.endswith('m'): return float(v[:-1]) * 1e-3
            return float(v)
            
        r_on = get_val("R_on", 3.0)
        r_off = get_val("R_off", 10000.0)
        
        if name == "loadedline":
            z0 = get_val("Z0_line", 50.0)
            l_quarter = get_val("L_quarter_mm", 1.69)
            c_load = get_val("C_load_pf", 0.04)
            
            s11_0, s21_0 = compute_loaded_line_s_params(z0, l_quarter, c_load, r_on, r_off, freq_hz, state=0)
            s11_1, s21_1 = compute_loaded_line_s_params(z0, l_quarter, c_load, r_on, r_off, freq_hz, state=1)
            
        elif name == "switchedline":
            z0 = get_val("Z0_line", 50.0)
            l_short = get_val("L_short_mm", 2.68)
            l_long = get_val("L_long_mm", 5.36)
            
            s11_0, s21_0 = compute_switched_line_s_params(z0, l_short, l_long, r_on, r_off, freq_hz, state=0)
            s11_1, s21_1 = compute_switched_line_s_params(z0, l_short, l_long, r_on, r_off, freq_hz, state=1)
            
        elif name == "reflectiontype":
            z0_main   = get_val("Z0_main",   50.0)
            z0_branch = get_val("Z0_branch",  35.35)
            c_base    = get_val("C_base_pf",  0.10) * 1e-12
            c_tune    = get_val("C_tune_pf",  0.20) * 1e-12
            omega     = 2 * np.pi * freq_hz

            # 1. Z0_branch within [0.60, 0.85]×Z0_main (ideal = 1/√2 ≈ 0.707, ±15%)
            ratio = z0_branch / z0_main
            if not (0.60 <= ratio <= 0.85):
                return False

            # 2. Switched-cap phase shift must be detectable (5°) but not resonant (90°)
            phi_base = -2.0 * np.arctan(omega * c_base * z0_main)
            phi_both = -2.0 * np.arctan(omega * (c_base + c_tune) * z0_main)
            delta_phi_deg = abs(np.degrees(phi_both - phi_base))
            if not (5.0 < delta_phi_deg < 90.0):
                return False

            return True

        elif name == "switchedfilter":
            c_hpf = get_val("C_hpf_pf", 0.114) * 1e-12
            l_hpf = get_val("L_hpf_nh", 0.284) * 1e-9
            c_lpf = get_val("C_lpf_pf", 0.114) * 1e-12
            l_lpf = get_val("L_lpf_nh", 0.284) * 1e-9
            omega  = 2 * np.pi * freq_hz

            # 1. Characteristic impedance of each LC section: Z=sqrt(L/C) in [10, 200] Ω
            z0_hpf = np.sqrt(l_hpf / c_hpf)
            z0_lpf = np.sqrt(l_lpf / c_lpf)
            if not (10.0 <= z0_hpf <= 200.0):
                return False
            if not (10.0 <= z0_lpf <= 200.0):
                return False

            # 2. LC resonance must be within [0.1, 10]×ωfc
            if not (0.1 * omega <= 1.0 / np.sqrt(l_hpf * c_hpf) <= 10.0 * omega):
                return False
            if not (0.1 * omega <= 1.0 / np.sqrt(l_lpf * c_lpf) <= 10.0 * omega):
                return False

            return True

        elif name == "vectormodulator":
            l_quarter = get_val("L_quarter_mm", 1.69)
            g_i       = get_val("G_I_scale",    1.0)
            g_q       = get_val("G_Q_scale",    1.0)
            lam4 = 47.43 / fc_ghz   # λ/4 in mm, eps_eff=2.5; fc_ghz is function arg

            # 1. TL length must be within [0.1, 3.0]×λ/4
            if not (0.1 * lam4 <= l_quarter <= 3.0 * lam4):
                return False

            # 2. Scale > 1.0 produces active gain (IL = -20log10(scale) < 0)
            if g_i > 1.0 or g_q > 1.0:
                return False

            # 3. I/Q gain imbalance < 40% to prevent vector collapse
            if abs(g_i - g_q) >= 0.4:
                return False

            return True

        elif name == "allpass":
            l_apA = get_val("L_apA_nh", 0.100) * 1e-9
            c_brA = get_val("C_brA_pf", 0.025) * 1e-12
            c_cA  = get_val("C_cA_pf",  0.080) * 1e-12
            l_apB = get_val("L_apB_nh", 0.200) * 1e-9
            c_brB = get_val("C_brB_pf", 0.060) * 1e-12
            c_cB  = get_val("C_cB_pf",  0.200) * 1e-12
            Z0_sq = 50.0 ** 2   # 50-Ω reference system
            omega = 2 * np.pi * freq_hz

            # Bridged-T balance: C_br ≈ L_ap/Z0² (factor-10 tolerance)
            # C_c/C_br ratio is now guaranteed ≥ 1.2 by action_to_params
            # reparameterization (C_c = k×C_br, k ∈ [1.2, 4.0]), so no
            # ratio check needed here. Resonance check retained.
            # Resonance: LC resonant frequency must be within [0.2, 5]×ωfc
            for l_ap, c_br, c_c in [(l_apA, c_brA, c_cA), (l_apB, c_brB, c_cB)]:
                ideal_cbr = l_ap / Z0_sq
                if not (0.1 * ideal_cbr <= c_br <= 10.0 * ideal_cbr):
                    return False
                omega_res = 1.0 / np.sqrt(l_ap * c_br)
                if not (0.2 * omega <= omega_res <= 5.0 * omega):
                    return False

            return True

        else:
            # Unknown topology — pass through rather than silently block
            return True
            
        # S-parameter validation thresholds:
        # 1. Return loss must be better than 5 dB: |S11| < 0.56
        # 2. Insertion loss must be better than 15 dB: |S21| > 0.17
        max_s11_mag = 0.56  # 5 dB return loss
        min_s21_mag = 0.17  # 15 dB insertion loss
        
        s11_0_mag = torch.abs(s11_0).item()
        s11_1_mag = torch.abs(s11_1).item()
        s21_0_mag = torch.abs(s21_0).item()
        s21_1_mag = torch.abs(s21_1).item()
        
        # Check if they are physically sane
        if np.isnan(s11_0_mag) or np.isnan(s11_1_mag) or np.isnan(s21_0_mag) or np.isnan(s21_1_mag):
            return False
            
        if s11_0_mag > max_s11_mag or s11_1_mag > max_s11_mag:
            return False
            
        if s21_0_mag < min_s21_mag or s21_1_mag < min_s21_mag:
            return False
            
        return True
    except Exception as e:
        # Fallback to True if parsing fails to avoid breaking simulations
        print(f"[Physics Priors Filter Warning] {e}")
        return True
