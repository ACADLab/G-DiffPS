import os
import sys
import time
import json
import numpy as np
from scipy.optimize import minimize

# Ensure imports work from project root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker
from sim.physics_priors import check_physics_priors
from test_spec_generalization import check_sane_design_success

def run_baseline_optimization():
    print("="*80)
    print(" BASELINE ITERATIVE OPTIMIZATION (NELDER-MEAD SPICE IN-THE-LOOP)")
    print("="*80)
    
    env = PhaseShifterEnv()
    
    # Same 10 diverse target specification scenarios spanning 2.4 GHz to 38 GHz
    scenarios = [
        {
            "name": "Scenario 1: 28 GHz mm-Wave Spec (Loaded Line)",
            "topo": "Loaded_Line",
            "spec": {
                "fc_ghz": 28.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0,
                "phase_bits": 5, "rms_phase_err_deg": 3.0, "rms_gain_err_db": 1.0,
                "max_il_db": 2.0, "min_rl_db": 18.0, "vdd": 1.8, "pmax_mw": 15.0,
                "tech": 0, "app": 2,
            }
        },
        {
            "name": "Scenario 2: 38 GHz Ka-band Spec (Loaded Line)",
            "topo": "Loaded_Line",
            "spec": {
                "fc_ghz": 38.0, "bw_pct": 15.0, "phase_coverage_deg": 180.0,
                "phase_bits": 5, "rms_phase_err_deg": 3.5, "rms_gain_err_db": 1.0,
                "max_il_db": 2.2, "min_rl_db": 15.0, "vdd": 1.8, "pmax_mw": 15.0,
                "tech": 0, "app": 2,
            }
        },
        {
            "name": "Scenario 3: 14 GHz Ku-band Spec (Switched Line)",
            "topo": "Switched_Line",
            "spec": {
                "fc_ghz": 14.0, "bw_pct": 30.0, "phase_coverage_deg": 180.0,
                "phase_bits": 4, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.5,
                "max_il_db": 2.5, "min_rl_db": 15.0, "vdd": 2.5, "pmax_mw": 25.0,
                "tech": 1, "app": 3,
            }
        },
        {
            "name": "Scenario 4: 24 GHz High Freq Spec (Switched Line)",
            "topo": "Switched_Line",
            "spec": {
                "fc_ghz": 24.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0,
                "phase_bits": 4, "rms_phase_err_deg": 4.5, "rms_gain_err_db": 1.0,
                "max_il_db": 2.0, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0,
                "tech": 0, "app": 2,
            }
        },
        {
            "name": "Scenario 5: 5 GHz Sub-6 GHz Spec (Vector Modulator)",
            "topo": "Vector_Modulator",
            "spec": {
                "fc_ghz": 5.0, "bw_pct": 40.0, "phase_coverage_deg": 360.0,
                "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0,
                "max_il_db": 1.5, "min_rl_db": 20.0, "vdd": 3.3, "pmax_mw": 35.0,
                "tech": 2, "app": 0,
            }
        },
        {
            "name": "Scenario 6: 8 GHz X-band Spec (Vector Modulator)",
            "topo": "Vector_Modulator",
            "spec": {
                "fc_ghz": 8.0, "bw_pct": 30.0, "phase_coverage_deg": 360.0,
                "phase_bits": 5, "rms_phase_err_deg": 4.0, "rms_gain_err_db": 1.0,
                "max_il_db": 1.8, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 35.0,
                "tech": 2, "app": 3,
            }
        },
        {
            "name": "Scenario 7: 28 GHz mm-Wave Spec (Reflection Type)",
            "topo": "Reflection_Type",
            "spec": {
                "fc_ghz": 28.0, "bw_pct": 20.0, "phase_coverage_deg": 90.0,
                "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0,
                "max_il_db": 2.0, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 15.0,
                "tech": 0, "app": 2,
            }
        },
        {
            "name": "Scenario 8: 10 GHz Medium Freq Spec (Reflection Type)",
            "topo": "Reflection_Type",
            "spec": {
                "fc_ghz": 10.0, "bw_pct": 25.0, "phase_coverage_deg": 180.0,
                "phase_bits": 4, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.5,
                "max_il_db": 2.0, "min_rl_db": 15.0, "vdd": 2.5, "pmax_mw": 25.0,
                "tech": 1, "app": 3,
            }
        },
        {
            "name": "Scenario 9: 2.4 GHz ISM band Spec (All Pass)",
            "topo": "All_Pass",
            "spec": {
                "fc_ghz": 2.4, "bw_pct": 10.0, "phase_coverage_deg": 180.0,
                "phase_bits": 3, "rms_phase_err_deg": 8.0, "rms_gain_err_db": 2.0,
                "max_il_db": 1.5, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 40.0,
                "tech": 2, "app": 0,
            }
        },
        {
            "name": "Scenario 10: 18 GHz Ku-band Spec (Switched Filter)",
            "topo": "Switched_Filter",
            "spec": {
                "fc_ghz": 18.0, "bw_pct": 20.0, "phase_coverage_deg": 90.0,
                "phase_bits": 4, "rms_phase_err_deg": 6.0, "rms_gain_err_db": 2.0,
                "max_il_db": 2.5, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0,
                "tech": 0, "app": 3,
            }
        }
    ]
    
    results = []
    
    # To run Nelder-Mead quickly, we limit the maximum SPICE simulations to 50
    # reflecting a fast comparative study budget
    max_spice_calls = 50
    
    for idx, sc in enumerate(scenarios):
        print("\n" + "-"*80)
        print(f"[*] Optimizing {sc['name']}")
        print(f"    - Budget: {max_spice_calls} SPICE simulations")
        print("-"*80)
        
        sim_count = 0
        best_reward = -999.0
        best_metrics = None
        
        # Initial guess in center of [0, 1] normalized bounds
        x0 = np.full(9, 0.5)
        
        start_time = time.time()
        
        def objective(x):
            nonlocal sim_count, best_reward, best_metrics
            
            # Clip actions to bounds
            x_clipped = np.clip(x, 0.0, 1.0)
            params = action_to_params(x_clipped, sc["topo"])
            
            # 1. Physics pre-filter check
            passed_prior = check_physics_priors(sc["topo"], params, sc["spec"]["fc_ghz"])
            expert_bonus = env.compute_expert_bonus(sc["topo"], sc["spec"])
            
            if not passed_prior:
                # Assign static loss penalty to bypass simulation
                return 10.0 - expert_bonus
                
            sim_count += 1
            if sim_count > max_spice_calls:
                return -best_reward
                
            # 2. Compile and run ngspice
            netlist_path = make_spice_netlist(sc["topo"], params)
            reward, agg_metrics, _ = parallel_eval_worker(
                (netlist_path, sc["spec"], sc["topo"], expert_bonus, None)
            )
            
            if reward > best_reward:
                best_reward = reward
                best_metrics = agg_metrics
                
            # Minimize loss = -reward
            return -reward
            
        # Run optimization
        res = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxfev': max_spice_calls, 'xatol': 1e-3, 'fatol': 1e-3}
        )
        
        elapsed_sec = time.time() - start_time
        success = check_sane_design_success(sc["topo"], best_metrics, sc["spec"])
        
        print(f"  [+] Optimization Complete in: {elapsed_sec:.2f} s")
        print(f"  [+] SPICE Simulations Executed: {sim_count}")
        
        if best_metrics is None:
            phase_err = 360.0
            il_val = -20.0
            rl_val = 0.0
            print("      [!] Optimization Failed to synthesize any viable circuit.")
        else:
            phase_err = best_metrics['rms_phase_err_deg']
            il_val = best_metrics['il_db']
            rl_val = best_metrics['rl_db']
            print(f"      * Measured RMS Phase Error: {phase_err:.3f}°")
            print(f"      * Measured Average Insertion Loss: {il_val:.3f} dB")
            print(f"      * Measured Average Return Loss: {rl_val:.3f} dB")
            print(f"      * Strict Physical Compliance: {'PASS' if success else 'FAIL'}")
            
        results.append({
            "scenario": idx + 1,
            "topo": sc["topo"],
            "freq": sc["spec"]["fc_ghz"],
            "time_sec": elapsed_sec,
            "sim_count": sim_count,
            "phase_err": phase_err,
            "il": il_val,
            "rl": rl_val,
            "success": success
        })
        
    print("\n" + "="*80)
    print(" SUMMARY OF NELDER-MEAD OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Scenario':<10} | {'Topology':<18} | {'Freq (GHz)':<10} | {'Time (s)':<10} | {'Sims':<6} | {'Phase Err':<10} | {'IL (dB)':<8} | {'RL (dB)':<8} | {'Status':<6}")
    print("-"*95)
    for r in results:
        status_str = "PASS" if r["success"] else "FAIL"
        print(f"{r['scenario']:<10} | {r['topo']:<18} | {r['freq']:<10.1f} | {r['time_sec']:<10.2f} | {r['sim_count']:<6} | {r['phase_err']:<10.2f} | {r['il']:<8.2f} | {r['rl']:<8.2f} | {status_str:<6}")
    print("="*80)

if __name__ == '__main__':
    run_baseline_optimization()
