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

def run_pareto_matching():
    print("="*80)
    print(" G-DiffPS VS. ITERATIVE OPTIMIZATION EQUAL-QUALITY PARETO MATCHING STUDY")
    print("="*80)
    
    env = PhaseShifterEnv()
    
    # Pre-logged high-fidelity phase errors achieved by G-DiffPS (T=10 denoising)
    scenarios = [
        {
            "name": "Scenario 1: 28 GHz mm-Wave Spec (Loaded Line)",
            "topo": "Loaded_Line",
            "gdiffps_phase_err": 9.91,
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
            "gdiffps_phase_err": 10.97,
            "spec": {
                "fc_ghz": 38.0, "bw_pct": 15.0, "phase_coverage_deg": 180.0,
                "phase_bits": 5, "rms_phase_err_deg": 3.5, "rms_gain_err_db": 1.0,
                "max_il_db": 2.2, "min_rl_db": 15.0, "vdd": 1.8, "pmax_mw": 15.0,
                "tech": 0, "app": 2,
            }
        },
        {
            "name": "Scenario 3: 14 GHz Ku-band Spec (Switched Line)",
            "gdiffps_phase_err": 66.23,
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
            "gdiffps_phase_err": 7.98,
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
            "gdiffps_phase_err": 74.27,
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
            "gdiffps_phase_err": 77.61,
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
            "gdiffps_phase_err": 23.89,
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
            "gdiffps_phase_err": 20.95,
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
            "gdiffps_phase_err": 63.65,
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
            "gdiffps_phase_err": 64.65,
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
    
    # We increase the max budget to 600 SPICE simulations to allow true convergence matching
    max_spice_calls = 600
    
    for idx, sc in enumerate(scenarios):
        print("\n" + "-"*80)
        print(f"[*] Optimizing {sc['name']}")
        print(f"    - Target G-DiffPS Phase Error: {sc['gdiffps_phase_err']}°")
        print(f"    - Convergence Limit: {max_spice_calls} SPICE simulations")
        print("-"*80)
        
        sim_count = 0
        best_reward = -999.0
        best_metrics = None
        target_matched = False
        matching_sim_count = 0
        matching_time = 0.0
        
        x0 = np.full(9, 0.5)
        start_time = time.time()
        
        def objective(x):
            nonlocal sim_count, best_reward, best_metrics, target_matched, matching_sim_count, matching_time
            
            # Clip actions to bounds
            x_clipped = np.clip(x, 0.0, 1.0)
            params = action_to_params(x_clipped, sc["topo"])
            
            passed_prior = check_physics_priors(sc["topo"], params, sc["spec"]["fc_ghz"])
            expert_bonus = env.compute_expert_bonus(sc["topo"], sc["spec"])
            
            if not passed_prior:
                return 10.0 - expert_bonus
                
            sim_count += 1
            if sim_count > max_spice_calls:
                return -best_reward
                
            netlist_path = make_spice_netlist(sc["topo"], params)
            reward, agg_metrics, _ = parallel_eval_worker(
                (netlist_path, sc["spec"], sc["topo"], expert_bonus, None)
            )
            
            if reward > best_reward:
                best_reward = reward
                best_metrics = agg_metrics
                
            # Check if this SPICE run matches or beats the target G-DiffPS phase error
            if agg_metrics is not None and not target_matched:
                measured_err = agg_metrics['rms_phase_err_deg']
                if measured_err <= sc["gdiffps_phase_err"]:
                    target_matched = True
                    matching_sim_count = sim_count
                    matching_time = time.time() - start_time
                    print(f"    [!] Quality Matched at SIM {sim_count} | Phase Error: {measured_err:.3f}°")
                    
            return -reward
            
        # Run Nelder-Mead optimization to converge on target quality
        res = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxfev': max_spice_calls, 'xatol': 1e-3, 'fatol': 1e-3}
        )
        
        elapsed_sec = time.time() - start_time
        success = check_sane_design_success(sc["topo"], best_metrics, sc["spec"])
        
        if not target_matched:
            # If Nelder-Mead never matched the quality, report the full budget
            matching_sim_count = sim_count
            matching_time = elapsed_sec
            
        print(f"  [+] Complete. Matching Time: {matching_time:.2f} s | Matching SPICE Sims: {matching_sim_count}")
        
        if best_metrics is None:
            phase_err = 360.0
            il_val = -20.0
            rl_val = 0.0
        else:
            phase_err = best_metrics['rms_phase_err_deg']
            il_val = best_metrics['il_db']
            rl_val = best_metrics['rl_db']
            
        results.append({
            "scenario": idx + 1,
            "topo": sc["topo"],
            "freq": sc["spec"]["fc_ghz"],
            "target_err": sc["gdiffps_phase_err"],
            "match_sims": matching_sim_count,
            "match_time": matching_time,
            "final_err": phase_err,
            "matched": target_matched
        })
        
    print("\n" + "="*90)
    print(" SUMMARY OF EQUAL-QUALITY PARETO MATCHING BASES")
    print("="*90)
    print(f"{'Scenario':<10} | {'Topology':<18} | {'Freq (GHz)':<10} | {'G-DiffPS Err':<12} | {'NM Match Sims':<13} | {'NM Match Time':<13} | {'Matched':<8}")
    print("-"*90)
    for r in results:
        matched_str = "YES" if r["matched"] else "NO"
        print(f"{r['scenario']:<10} | {r['topo']:<18} | {r['freq']:<10.1f} | {r['target_err']:<12.2f} | {r['match_sims']:<13} | {r['match_time']:<13.2f} | {matched_str:<8}")
    print("="*90)

if __name__ == '__main__':
    run_pareto_matching()
