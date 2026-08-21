"""
J8 — LLM-as-sizer baseline for App D.

Zero-shot framing (Framing 1):
  Prompt: SPICE template description + target spec → JSON params → SPICE eval.
  No SPICE feedback to the LLM. Each LLM call = 1 candidate design.

Compares: Anthropic Claude (sonnet-4-6), OpenAI GPT-4o, DeepSeek.
Outputs: results/baselines/llm_eval/<provider>_<scenario>_seed<s>.json

Usage:
    python3 baselines/llm_sizer.py --provider anthropic --n-samples 5
    python3 baselines/llm_sizer.py --provider all      --n-samples 5
"""

import argparse
import json
import os
import re
import sys
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import TOPOLOGY_PARAMS
from train_diffusion import make_spice_netlist, parallel_eval_worker
from sim.physics_priors import check_physics_priors


# Load API keys from PowerShell-format key file
def load_keys(path=os.path.join(REPO_ROOT, "mykeys.txt")):
    keys = {}
    if not os.path.exists(path):
        return keys
    for line in open(path):
        m = re.match(r"\$env:(\w+)\s*=\s*['\"]?([^'\"\s]+)['\"]?", line.strip())
        if m:
            keys[m.group(1)] = m.group(2)
    return keys


SCENARIOS = [
    {"name": "S1_LL_28",  "topo": "Loaded_Line",      "spec": {"fc_ghz": 28.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0, "phase_bits": 5, "rms_phase_err_deg": 3.0, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 18.0, "vdd": 1.8, "pmax_mw": 15.0, "tech": 0, "app": 2}},
    {"name": "S2_LL_38",  "topo": "Loaded_Line",      "spec": {"fc_ghz": 38.0, "bw_pct": 15.0, "phase_coverage_deg": 180.0, "phase_bits": 5, "rms_phase_err_deg": 3.5, "rms_gain_err_db": 1.0, "max_il_db": 2.2, "min_rl_db": 15.0, "vdd": 1.8, "pmax_mw": 15.0, "tech": 0, "app": 2}},
    {"name": "S3_SL_14",  "topo": "Switched_Line",    "spec": {"fc_ghz": 14.0, "bw_pct": 30.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.5, "max_il_db": 2.5, "min_rl_db": 15.0, "vdd": 2.5, "pmax_mw": 25.0, "tech": 1, "app": 3}},
    {"name": "S4_SL_24",  "topo": "Switched_Line",    "spec": {"fc_ghz": 24.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 4.5, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0, "tech": 0, "app": 2}},
    {"name": "S5_VM_5",   "topo": "Vector_Modulator", "spec": {"fc_ghz": 5.0,  "bw_pct": 40.0, "phase_coverage_deg": 360.0, "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0, "max_il_db": 1.5, "min_rl_db": 20.0, "vdd": 3.3, "pmax_mw": 35.0, "tech": 2, "app": 0}},
    {"name": "S6_VM_8",   "topo": "Vector_Modulator", "spec": {"fc_ghz": 8.0,  "bw_pct": 30.0, "phase_coverage_deg": 360.0, "phase_bits": 5, "rms_phase_err_deg": 4.0, "rms_gain_err_db": 1.0, "max_il_db": 1.8, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 35.0, "tech": 2, "app": 3}},
    {"name": "S7_RT_28",  "topo": "Reflection_Type",  "spec": {"fc_ghz": 28.0, "bw_pct": 20.0, "phase_coverage_deg": 90.0,  "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 15.0, "tech": 0, "app": 2}},
    {"name": "S8_RT_10",  "topo": "Reflection_Type",  "spec": {"fc_ghz": 10.0, "bw_pct": 25.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.5, "max_il_db": 2.0, "min_rl_db": 15.0, "vdd": 2.5, "pmax_mw": 25.0, "tech": 1, "app": 3}},
    {"name": "S9_AP_2.4", "topo": "All_Pass",         "spec": {"fc_ghz": 2.4,  "bw_pct": 10.0, "phase_coverage_deg": 180.0, "phase_bits": 3, "rms_phase_err_deg": 8.0, "rms_gain_err_db": 2.0, "max_il_db": 1.5, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 40.0, "tech": 2, "app": 0}},
    {"name": "S10_SF_18", "topo": "Switched_Filter",  "spec": {"fc_ghz": 18.0, "bw_pct": 20.0, "phase_coverage_deg": 90.0,  "phase_bits": 4, "rms_phase_err_deg": 6.0, "rms_gain_err_db": 2.0, "max_il_db": 2.5, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0, "tech": 0, "app": 3}},
]


# Physical ranges per parameter, matching action_to_params
PARAM_RANGES = {
    "Z0_line":       ("Ω",  25.0, 75.0,    "transmission line characteristic impedance"),
    "Z0_main":       ("Ω",  40.0, 60.0,    "main line characteristic impedance"),
    "Z0_branch":     ("Ω",  25.0, 45.0,    "branch line impedance (≈Z0_main/√2 ideal)"),
    "L_quarter_mm":  ("mm", 0.5,  20.0,    "λ/4 transmission line length (λ/4 = 47.43/fc_GHz mm)"),
    "L_short_mm":    ("mm", 0.3,  10.0,    "short-path TL length (must be < L_long)"),
    "L_long_mm":     ("mm", 1.0,  20.0,    "long-path TL length"),
    "C_load_pf":     ("pF", 0.005, 0.5,    "shunt loading capacitor (sub-pF for mmWave resonance)"),
    "C_base_pf":     ("pF", 0.01, 1.0,     "base capacitance"),
    "C_tune_pf":     ("pF", 0.01, 1.0,     "tuning capacitance"),
    "C_hpf_pf":      ("pF", 0.005, 2.0,    "HPF capacitance; design ≈ 1/(2π·fc·Z0)"),
    "C_lpf_pf":      ("pF", 0.005, 2.0,    "LPF capacitance"),
    "L_hpf_nh":      ("nH", 0.05, 5.0,     "HPF inductance; design ≈ Z0/(2π·fc)"),
    "L_lpf_nh":      ("nH", 0.05, 5.0,     "LPF inductance"),
    "G_I_scale":     ("",   0.7,  1.0,     "I-path VCVS gain (≤ 1.0 for passivity)"),
    "G_Q_scale":     ("",   0.7,  1.0,     "Q-path VCVS gain"),
    "L_apA_nh":      ("nH", 0.05, 5.0,     "All-pass section A inductance"),
    "C_brA_pf":      ("pF", 0.005, 0.5,    "All-pass section A bridge cap"),
    "C_cA_pf":       ("pF", 0.006, 2.0,    "All-pass section A coupling cap (≥ 1.2 × C_brA)"),
    "L_apB_nh":      ("nH", 0.05, 5.0,     "All-pass section B inductance"),
    "C_brB_pf":      ("pF", 0.005, 0.5,    "All-pass section B bridge cap"),
    "C_cB_pf":       ("pF", 0.006, 2.0,    "All-pass section B coupling cap (≥ 1.2 × C_brB)"),
    "R_on":          ("Ω",  0.5,  10.0,    "switch on-resistance"),
    "R_off":         ("Ω",  1e3,  1e6,     "switch off-resistance (log scale)"),
}

TOPO_DESCRIPTIONS = {
    "Loaded_Line":      "Two cascaded λ/4 transmission lines with switched shunt loading capacitors. Phase state set by toggling C_load via R_on/R_off.",
    "Switched_Line":    "Two parallel TL paths of different electrical lengths, switched via R_on/R_off. Phase = β·(L_long − L_short).",
    "Reflection_Type":  "Quadrature 3-dB hybrid feeding two reflective terminations made of TL + tunable C. Reflection phase = -2·atan(ω·C·Z₀).",
    "Switched_Filter":  "Switchable LPF/HPF pi-sections. Phase shift = arg(H_HPF) − arg(H_LPF) at fc.",
    "Vector_Modulator": "Two orthogonal IQ paths (quad coupler + VCVS gain blocks). Output phase = atan2(G_Q, G_I).",
    "All_Pass":         "Two cascaded bridged-T sections with L, C_br, C_c. Requires C_c ≈ 2·C_br for low IL; ω_res = 1/√(LC) ≈ ω_fc.",
}


def build_prompt(scenario):
    topo = scenario["topo"]
    spec = scenario["spec"]
    keys = TOPOLOGY_PARAMS[topo]

    param_lines = []
    for k in keys:
        unit, lo, hi, descr = PARAM_RANGES[k]
        param_lines.append(f"  - {k}  [{lo:g} .. {hi:g} {unit}]  : {descr}")

    return f"""You are an expert RF circuit designer. Size the following passive RF phase shifter.

TOPOLOGY: {topo}
{TOPO_DESCRIPTIONS[topo]}

TARGET SPECIFICATION:
  center frequency      fc        = {spec['fc_ghz']:.2f} GHz
  bandwidth             bw        = {spec['bw_pct']:.1f} %
  phase coverage        Δφ        = {spec['phase_coverage_deg']:.1f} deg
  bits                  N         = {spec['phase_bits']}
  max RMS phase error   ε_φ       ≤ {spec['rms_phase_err_deg']:.1f} deg
  max RMS gain error    ε_g       ≤ {spec['rms_gain_err_db']:.1f} dB
  max insertion loss    IL        ≤ {spec['max_il_db']:.1f} dB
  min return loss       RL        ≥ {spec['min_rl_db']:.1f} dB

CONSTRAINTS:
  - Each parameter must lie strictly within its bracketed range.
  - Use physics: λ/4 = 47.43 / fc_GHz mm. For mmWave, capacitors are sub-pF.
  - R_off should be > 10 kΩ for off-state isolation; R_on < 5 Ω for low loss.

PARAMETERS TO SIZE:
{chr(10).join(param_lines)}

Respond with ONLY a JSON object mapping parameter name to value (no units, no explanation, no markdown fences). Example:
{{"Z0_line": 50.0, "L_quarter_mm": 1.7, "C_load_pf": 0.05, "R_on": 1.0, "R_off": 50000.0}}
"""


def call_anthropic(prompt, api_key, model="claude-sonnet-4-5"):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai(prompt, api_key, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def call_deepseek(prompt, api_key, model="deepseek-chat"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


PROVIDERS = {
    "anthropic": (call_anthropic, "ANTHROPIC_API_KEY"),
    "openai":    (call_openai,    "OPENAI_API_KEY"),
    "deepseek":  (call_deepseek,  "DEEPSEEK_API_KEY"),
}


def parse_json(text):
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    # Greedy match for outer JSON object
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(m.group(0))


def coerce_params(parsed, topo):
    """Validate + clip parsed dict against TOPOLOGY_PARAMS schema."""
    keys = TOPOLOGY_PARAMS[topo]
    params = {}
    for k in keys:
        if k not in parsed:
            raise KeyError(f"missing key {k}")
        v = float(parsed[k])
        _, lo, hi, _ = PARAM_RANGES[k]
        v = max(lo, min(hi, v))
        # SPICE wants string representations (matches action_to_params output)
        params[k] = f"{v:.6g}"
    return params


def eval_llm_design(scenario, provider, api_key, env):
    prompt = build_prompt(scenario)
    fn, _ = PROVIDERS[provider]

    t0 = time.time()
    try:
        raw = fn(prompt, api_key)
        parsed = parse_json(raw)
        params = coerce_params(parsed, scenario["topo"])
        parse_ok = True
        err = None
    except Exception as e:
        return {
            "reward": -999.0, "metrics": None, "params": None,
            "passed_prior": False, "parse_ok": False, "error": str(e),
            "raw_response": locals().get("raw", "")[:500],
            "elapsed_s": time.time() - t0,
        }

    topo = scenario["topo"]
    spec = scenario["spec"]
    passed = check_physics_priors(topo, params, spec["fc_ghz"])
    if not passed:
        return {
            "reward": -5.0, "metrics": None, "params": params,
            "passed_prior": False, "parse_ok": parse_ok, "error": err,
            "raw_response": raw[:500],
            "elapsed_s": time.time() - t0,
        }

    netlist = make_spice_netlist(topo, params)
    reward, metrics, _ = parallel_eval_worker((netlist, spec, topo, 0.0, None))
    return {
        "reward": float(reward), "metrics": metrics, "params": params,
        "passed_prior": True, "parse_ok": parse_ok, "error": err,
        "raw_response": raw[:500],
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic",
                    choices=list(PROVIDERS.keys()) + ["all"])
    ap.add_argument("--n-samples", type=int, default=3,
                    help="LLM samples per scenario (best-of-N)")
    ap.add_argument("--out-dir", default="results/baselines/llm_eval")
    args = ap.parse_args()

    keys = load_keys()
    print(f"[LLM] Found {len(keys)} API keys")

    providers = [args.provider] if args.provider != "all" else list(PROVIDERS.keys())
    os.makedirs(args.out_dir, exist_ok=True)
    env = PhaseShifterEnv()

    for prov in providers:
        _, key_name = PROVIDERS[prov]
        if key_name not in keys:
            print(f"[LLM] {prov}: no key, skipping")
            continue
        api_key = keys[key_name]

        for sc in SCENARIOS:
            best = {"reward": -999.0}
            samples = []
            print(f"\n[{prov}] {sc['name']}  topo={sc['topo']}  fc={sc['spec']['fc_ghz']} GHz")
            for k in range(args.n_samples):
                res = eval_llm_design(sc, prov, api_key, env)
                samples.append(res)
                if res["reward"] > best["reward"]:
                    best = res
                marker = " *" if res["reward"] == best["reward"] else ""
                print(f"  sample {k}: reward={res['reward']:+.3f}  passed_prior={res['passed_prior']}  parse_ok={res['parse_ok']}  {res['elapsed_s']:.1f}s{marker}")

            out_path = os.path.join(args.out_dir, f"{prov}_{sc['name']}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "method": f"llm_{prov}",
                    "scenario": sc["name"],
                    "topology": sc["topo"],
                    "fc_ghz": sc["spec"]["fc_ghz"],
                    "n_samples": args.n_samples,
                    "best_reward": best["reward"],
                    "best_metrics": best.get("metrics"),
                    "best_params": best.get("params"),
                    "best_passed_prior": best.get("passed_prior", False),
                    "spice_calls": sum(1 for s in samples if s["passed_prior"]),
                    "samples": samples,
                }, f, indent=2)
            print(f"  → {out_path}")


if __name__ == "__main__":
    main()
