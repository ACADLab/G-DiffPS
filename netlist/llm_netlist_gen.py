import json
import os
import re
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- Backend Selection ----
# Priority: OLLAMA (local, fast) > OpenRouter (cloud, fallback)
BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "openrouter"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# Lazy-init client only when needed
_openai_client = None
def _get_openrouter_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Set it with: export OPENROUTER_API_KEY='your-key-here'"
            )
        _openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _openai_client


def _call_llm(system_prompt: str, user_prompt: str,
              model_override: str = None,
              temperature: float = 0.2) -> str:
    """Unified LLM caller — routes to Ollama or OpenRouter.

    Args:
        temperature: sampling temperature, passed through to the backend.
            Default 0.2 preserves session-2 baseline behavior. Higher values
            (used by exploration mechanism #36a) increase output diversity;
            empirically T>1.0 causes qwen2.5:7b to emit prose around the
            .PARAM line, so callers should cap at 1.0.
    """
    if BACKEND == "ollama":
        import requests
        model = model_override or OLLAMA_MODEL
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": 200}
                },
                timeout=120
            )
            return resp.json().get("response", "")
        except Exception as e:
            print(f"[Ollama Error] {e}")
            # Fallback empty .PARAM; downstream backfill will fill from
            # skeleton defaults so simulation does not hang.
            return ".PARAM (llm-fallback)"
    else:
        client = _get_openrouter_client()
        model = model_override or OPENROUTER_MODEL
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[OpenRouter Error] {e}")
            # Fallback empty .PARAM; downstream backfill will fill from
            # skeleton defaults so simulation does not hang.
            return ".PARAM (llm-fallback)"


def spec_to_vec(s):
    """Convert a phase-shifter spec dict to a numeric vector for cosine similarity in RAG."""
    return [
        s["fc_ghz"], s["bw_pct"], s["phase_coverage_deg"],
        float(s["phase_bits"]), s["max_il_db"], s["min_rl_db"],
        s["vdd"], s["pmax_mw"],
    ]


def parse_spice_val(val_str):
    """Convert SPICE value strings to float.
    Handles plain floats (1.5e-13), SI suffixes (2u, 500n, 1p), and combos.
    """
    s = val_str.strip()
    if not s:
        return None
    # First try: plain float (handles scientific notation directly)
    try:
        return float(s)
    except ValueError:
        pass
    # Fall back: number + SI suffix
    match = re.match(r"([+\-]?[0-9]*\.?[0-9]+)([a-zA-Z]+)", s)
    if not match:
        return None
    try:
        num_part = float(match.group(1))
    except ValueError:
        return None
    suffix = match.group(2).lower()
    if suffix.startswith('meg'): return num_part * 1e6
    if suffix.startswith('u'):   return num_part * 1e-6
    if suffix.startswith('n'):   return num_part * 1e-9
    if suffix.startswith('p'):   return num_part * 1e-12
    if suffix.startswith('m'):   return num_part * 1e-3
    if suffix.startswith('k'):   return num_part * 1e3
    return num_part  # unknown suffix: treat as bare number


def clamp_spice_value(key, val_str):
    """Physical constraint layer -- clamp values to physically reasonable bounds.

    Handles phase-shifter parameters by their unit suffix:
      *_mm    transmission-line lengths (mm)
      *_pf    capacitors (pF)
      *_nh    inductors (nH)
      *_scale dimensionless mismatch multipliers
      r_on/r_off, z0   switches and characteristic impedance (ohms)
    Unmatched keys fall through to a scientific-notation echo with no clamp.
    """
    val = parse_spice_val(val_str)
    if val is None:
        return val_str

    key_lower = key.lower()

    # --- Phase-shifter parameters ---
    # Transmission line lengths: keys like L_short_mm, L_long_mm, L_quarter_mm
    # Stored in mm; bound from 0.1 mm (very short) to 50 mm (impractical on-chip)
    if "_mm" in key_lower:
        val = max(0.1, min(50.0, val))
        return f"{val:.3f}"

    # Characteristic impedance: typically 25–100 ohm for RF lines
    if key_lower.startswith("z0") or key_lower == "z0_line":
        val = max(25.0, min(100.0, val))
        return f"{val:.1f}"

    # Switch on-resistance: 0.5 ohm (very good FET) to 10 ohm (small device)
    if key_lower in ("r_on", "ron"):
        val = max(0.5, min(10.0, val))
        return f"{val:.2f}"

    # Switch off-resistance: 1k ohm (poor isolation) to 1Meg ohm (very good)
    if key_lower in ("r_off", "roff"):
        val = max(1e3, min(1e6, val))
        return f"{val:.2e}"

    # Generic R_* parameters (e.g. R_short_in, R_long_out): could be either
    # on- or off-resistance, so use a wider envelope
    if key_lower.startswith("r_") or key_lower.startswith("r"):
        if any(tok in key_lower for tok in ("short", "long", "on", "off", "load", "src", "term")):
            val = max(0.5, min(1e6, val))
            return f"{val:.3e}"

    # Capacitors named *_pf carry their value in picofarads (mirrors the *_mm
    # convention for TL lengths). Bounded to a sane RF range: 5 fF–10 pF.
    if key_lower.endswith("_pf"):
        val = max(0.005, min(10.0, val))
        return f"{val:.4f}"

    # Inductors named *_nh carry their value in nanohenries. Bounded to a
    # sane RF range: 5 pH–10 nH. Mirrors the *_pf / *_mm unit-in-suffix
    # convention used elsewhere in the phase-shifter parameter set.
    if key_lower.endswith("_nh"):
        val = max(0.005, min(10.0, val))
        return f"{val:.4f}"

    # Dimensionless gain-mismatch scale factors on vector-modulator I/Q
    # VCVS branches. Ideal = 1.0; LLM tunes to compensate amplitude
    # imbalance. Range 0.7-1.3 matches _PARAM_INFO and represents realistic
    # +/-30% I/Q mismatch envelopes seen in mmWave vector modulators.
    if key_lower.endswith("_scale"):
        val = max(0.7, min(1.3, val))
        return f"{val:.4f}"

    # Catch-all: scientific notation, no clamp. Reached only if a param key
    # does not match any of the unit-suffix patterns above.
    return f"{val:.4e}"


# ==========================================================================
# Item #27: parameter priors for the LLM prompt
# --------------------------------------------------------------------------
# Each entry: (clamp_range_min, clamp_range_max, units, brief_description)
# Ranges must match clamp_spice_value above. Descriptions are one-liners
# the LLM uses to choose physically sensible values rather than rails.
# ==========================================================================

_PARAM_INFO = {
    # Transmission line geometry
    "Z0_line": (25.0, 100.0, "ohm",
                "characteristic impedance of main line; 50 is standard for RF"),
    "L_short_mm": (0.1, 50.0, "mm",
                   "short transmission line path length (sets reference phase)"),
    "L_long_mm": (0.1, 50.0, "mm",
                  "long transmission line path length (longer = more phase delay)"),
    "L_quarter_mm": (0.1, 50.0, "mm",
                     "approx quarter-wavelength main line at fc; ~1.7 mm at 28 GHz"),

    # Switch resistances
    "R_on": (0.5, 10.0, "ohm",
             "switch ON resistance; smaller = lower IL, typical 2-5 ohm"),
    "R_off": (1e3, 1e6, "ohm",
              "switch OFF resistance; larger = better isolation, typical 5k-50k"),

    # Loading capacitors
    "C_load_pf": (0.005, 10.0, "pF",
                  "shunt loading capacitance; small (0.02-0.1 pF) for good RL at 28 GHz"),
    "C_var_pf": (0.005, 10.0, "pF",
                 "varactor capacitance"),

    # Reflection-type branchline hybrid parameters
    "Z0_main": (25.0, 100.0, "ohm",
                "shunt-arm characteristic impedance of branchline hybrid; "
                "50 is standard, match to port impedance"),
    "Z0_branch": (25.0, 80.0, "ohm",
                  "series-arm impedance of branchline hybrid; "
                  "ideal is Z0_main/sqrt(2), so ~35 ohm for 50-ohm system"),
    "C_base_pf": (0.02, 1.0, "pF",
                  "base capacitance always loading the reflective port; "
                  "sets the reference reflection phase. Small values "
                  "(0.05-0.2 pF) preferred at 28 GHz"),
    "C_tune_pf": (0.02, 1.0, "pF",
                  "switched (added) capacitance loading the reflective "
                  "port in state 1; larger -> more phase shift but worse "
                  "IL. ~0.2 pF gives ~22.5 deg step at 28 GHz"),

    # Switched-filter HPF/LPF pi-section elements. Designed so that
    # L = Z0/(2*pi*fc) and C = 1/(2*pi*fc*Z0) at the design frequency,
    # giving a 50-ohm match and ~+/-45 deg phase per section at fc.
    "C_hpf_pf": (0.005, 10.0, "pF",
                 "HPF series capacitor in the switched-filter path; "
                 "~0.11 pF at 28 GHz for a 50-ohm match"),
    "L_hpf_nh": (0.005, 10.0, "nH",
                 "HPF shunt inductor in the switched-filter path; "
                 "~0.28 nH at 28 GHz for a 50-ohm match"),
    "C_lpf_pf": (0.005, 10.0, "pF",
                 "LPF shunt capacitor in the switched-filter path; "
                 "~0.11 pF at 28 GHz for a 50-ohm match"),
    "L_lpf_nh": (0.005, 10.0, "nH",
                 "LPF series inductor in the switched-filter path; "
                 "~0.28 nH at 28 GHz for a 50-ohm match"),

    # Bridged-T LC all-pass section elements. Two sections (A, B) per
    # template; each section is independently sized so that the
    # series reactance and the bridge susceptance are conjugate when
    # terminated in Z0, yielding flat magnitude and a phase set by
    # the section's time constant tau. Differential phase between
    # sections sets the bit step.
    "L_apA_nh": (0.005, 10.0, "nH",
                 "all-pass section A series inductor; smaller tau gives "
                 "less phase delay. ~0.12 nH at 28 GHz for ~-45 deg"),
    "C_brA_pf": (0.005, 10.0, "pF",
                 "all-pass section A bridge capacitor; sized so "
                 "L_apA / C_brA ~ Z0^2 for unity magnitude"),
    "C_cA_pf":  (0.005, 10.0, "pF",
                 "all-pass section A center-shunt capacitor; sized as "
                 "C_cA ~ 2 * C_brA for proper bridged-T balance"),
    "L_apB_nh": (0.005, 10.0, "nH",
                 "all-pass section B series inductor; larger tau gives "
                 "more phase delay. ~0.69 nH at 28 GHz for ~-135 deg"),
    "C_brB_pf": (0.005, 10.0, "pF",
                 "all-pass section B bridge capacitor; "
                 "L_apB / C_brB ~ Z0^2 for unity magnitude"),
    "C_cB_pf":  (0.005, 10.0, "pF",
                 "all-pass section B center-shunt capacitor; "
                 "C_cB ~ 2 * C_brB for proper bridged-T balance"),

    # Vector-modulator I/Q mismatch scale factors. Model the I/Q
    # amplitude imbalance present in any real vector modulator (VGA
    # core mismatch, DAC quantization, process variation). Ideal
    # calibration is 1.0 on both paths; the LLM tunes within roughly
    # +/-30% to compensate. Pure scalar multipliers on the VCVS
    # gains -- dimensionless, no unit conversion.
    "G_I_scale": (0.7, 1.3, "",
                  "vector-modulator I-path gain scale (mismatch "
                  "compensation); ideal=1.0, range ~0.7-1.3"),
    "G_Q_scale": (0.7, 1.3, "",
                  "vector-modulator Q-path gain scale (mismatch "
                  "compensation); ideal=1.0, range ~0.7-1.3"),
}

# Per-topology one-line context. Keys match TOPOLOGY_LABELS.
_TOPOLOGY_CONTEXT = {
    "Switched_Line":
        "Two transmission line paths (short, long) selected by switch resistors. "
        "Phase delta comes from path length difference. Low IL when ON-resistance is small.",
    "Loaded_Line":
        "Single main line loaded by shunt caps at input and output. Loading state "
        "selected by series switch resistance. Small caps keep RL high; larger caps "
        "give more phase but worsen RL. Symmetric loading is critical.",
    "Reflection_Type":
        "Quadrature coupler with reflective loads. Phase from load reactance variation. "
        "Broadband, low IL, mmWave-friendly.",
    "Switched_Filter":
        "Switched HPF/LPF pair. Each filter contributes ~90 deg leading/lagging phase. "
        "Broadband, good for digital phase shifting.",
    "Vector_Modulator":
        "Sum of I/Q paths with variable weights. Continuous phase via amplitude control. "
        "Requires careful amplitude/phase balance.",
    "All_Pass":
        "All-pass LC network. Unity gain magnitude with frequency-dependent phase. "
        "Narrowband; phase set by L/C resonance.",
}


def _extract_param_defaults(skeleton_content: str) -> dict:
    """Pull {name: default_value_string} from non-tagged .PARAM lines.

    Mirrors the existing name-extractor in generate() but also captures the
    RHS so we can show the LLM what the original template designers chose.
    Tagged ($ FRAMEWORK_CONTROLLED or $ DERIVED) lines are skipped.
    """
    defaults = {}
    # Match each `key=value` on a .PARAM line. The value is everything up to
    # the next whitespace or comment.
    pat = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s\$]+)")
    for line in skeleton_content.split('\n'):
        s = line.strip()
        if not s.upper().startswith('.PARAM'):
            continue
        upper = s.upper()
        if 'DERIVED' in upper or 'FRAMEWORK_CONTROLLED' in upper:
            continue
        for m in pat.finditer(s):
            key = m.group(1)
            if key.upper() == 'PARAM':
                continue
            defaults[key] = m.group(2)
    return defaults


def _build_param_table(allowed_keys: list, defaults: dict) -> str:
    """Build a multi-line per-parameter table for the LLM prompt.

    Each row: NAME  default=X  range=[lo, hi] UNITS  -- description
    Falls back gracefully for params without _PARAM_INFO entries (uses
    a wider envelope so the LLM at least knows the parameter exists).
    """
    lines = []
    for key in allowed_keys:
        info = _PARAM_INFO.get(key)
        default_str = defaults.get(key, "?")
        if info is not None:
            lo, hi, units, desc = info
            # Format range with appropriate precision
            if hi >= 1000:
                rng = f"[{lo:g}, {hi:g}]"
            elif hi >= 1:
                rng = f"[{lo:g}, {hi:g}]"
            else:
                rng = f"[{lo:g}, {hi:g}]"
            lines.append(f"  {key:<14s} default={default_str:<10s} "
                         f"range={rng} {units:<5s} -- {desc}")
        else:
            lines.append(f"  {key:<14s} default={default_str:<10s} "
                         f"(no range info)")
    return "\n".join(lines)


# Max parameter keys rendered per memory row. Keeps the prompt within
# token budget when params dicts are wide. The 6 chosen keys are
# topology-dominant; aux parameters drop off the table. See deferred #48.
_MEMORY_MAX_PARAMS_PER_ROW = 6


def _format_param_for_memory(val) -> str:
    """Compact representation of a param value for the memory table.

    SPICE-suffixed strings pass through unchanged ("1.7", "10k", "0.04").
    Bare floats get 3-sig-fig formatting. Anything else is str()-ified.
    """
    if isinstance(val, (int, float)):
        return f"{val:.3g}"
    return str(val)


def _build_memory_block(memory_hits: list[dict]) -> str:
    """Render top-K past attempts as a markdown table for the prompt.

    Columns: rank | fc | bits | rms_err | reward | params (truncated).
    The "rank" column matches retrieval order (1 = most similar).

    Returns "" if memory_hits is None or empty, so callers can
    unconditionally concatenate the result into the user prompt.
    """
    if not memory_hits:
        return ""

    rows = ["Recent past attempts for similar specs (most similar first):",
            "rank | fc_GHz | bits | rms_err_deg | reward | params"]
    for i, hit in enumerate(memory_hits, start=1):
        s = hit.get("spec", {})
        metrics = hit.get("metrics") or {}
        params = hit.get("params") or {}

        fc = s.get("fc_ghz", float("nan"))
        bits = s.get("phase_bits", "?")
        rms = metrics.get("rms_phase_err_deg")
        rms_str = f"{rms:.1f}" if isinstance(rms, (int, float)) else "n/a"
        reward = hit.get("reward", float("nan"))

        # Truncate params dict to the first N keys to bound prompt size.
        param_items = list(params.items())[:_MEMORY_MAX_PARAMS_PER_ROW]
        params_str = " ".join(
            f"{k}={_format_param_for_memory(v)}" for k, v in param_items
        )
        if len(params) > _MEMORY_MAX_PARAMS_PER_ROW:
            params_str += " ..."

        rows.append(
            f"  {i}  | {fc:.1f}  | {bits}    | {rms_str:>9s}   "
            f"| {reward:+.2f}  | {params_str}"
        )
    rows.append(
        "Use these as priors. If a high-reward past attempt is close to "
        "this spec, lean toward its parameters; if a past attempt scored "
        "poorly, avoid its parameter region."
    )
    return "\n".join(rows) + "\n\n"


class LLMNetlistGen:
    def __init__(self, dataset_path=None, model=None):
        # An optional reference dataset can be passed for cosine-similarity
        # few-shot retrieval. Phase-shifter sizing_hints in the bundled specset
        # are static placeholders, so this is unused on the default path; we
        # silently start with an empty store when no path is given.
        self.dataset = []
        if dataset_path is not None:
            try:
                with open(dataset_path, "r") as f:
                    self.dataset = json.load(f)
            except Exception as e:
                print(f"[Warning] LLMNetlistGen failed to load {dataset_path}: {e}")
                self.dataset = []
        self.model = model

    def set_model(self, model: str):
        self.model = model

    def generate(self, spec: dict, topology: str, temperature: float = 0.2,
                 memory_hits: list[dict] | None = None) -> tuple[str, dict]:
        # Optional RAG: find closest reference designs from an injected dataset.
        # Phase-shifter sizing_hints in the specset are static placeholders, so
        # phase-shifter generation does not consume few-shot matches; this is
        # only used when an explicit reference dataset is passed at construction.
        matches = [d for d in self.dataset if d["topology"] == topology]

        if not matches:
            few_shot = "No reference examples available."
        else:
            target_vec = np.array(spec_to_vec(spec)).reshape(1, -1)
            match_vecs = np.array([spec_to_vec(m["spec"]) for m in matches])
            sim = cosine_similarity(target_vec, match_vecs)[0]
            top_idx = sim.argsort()[-3:][::-1]

            few_shot = ""
            for idx in top_idx:
                m = matches[idx]
                few_shot += f"Spec: {m['spec']} -> Params: {m['sizing_hints']}\n"

        # Load skeleton template
        skeleton_path = os.path.join(os.path.dirname(__file__), f"../specset/templates/{topology.lower()}.sp")
        try:
            with open(skeleton_path, "r") as f:
                skeleton_content = f.read()
        except Exception as e:
            skeleton_content = f"* {topology} Template\n*.PARAM placeholder\n* Missing {skeleton_path}"

        # Extract existing .PARAM key names from skeleton so LLM knows what to override
        skeleton_param_keys = []
        for line in skeleton_content.split('\n'):
            if line.strip().upper().startswith('.PARAM'):
                if 'DERIVED' in line.upper() or 'FRAMEWORK_CONTROLLED' in line.upper():
                    continue
                for km in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*=', line):
                    skeleton_param_keys.append(km.group(1))

        # Also extract default values for the prompt (#27)
        skeleton_defaults = _extract_param_defaults(skeleton_content)

        allowed_keys_str = ', '.join(skeleton_param_keys) if skeleton_param_keys else 'Z0_line, L_short_mm, L_long_mm, R_on, R_off'

        is_phase_shifter = "fc_ghz" in spec
        if not is_phase_shifter:
            raise ValueError(
                "LLMNetlistGen.generate expects a phase-shifter spec "
                "(containing 'fc_ghz')."
            )

        # Build the per-parameter info table the LLM will see
        param_table = _build_param_table(skeleton_param_keys, skeleton_defaults)
        topo_context = _TOPOLOGY_CONTEXT.get(
            topology,
            "RF phase-shifter topology. Tune parameters for low phase error, "
            "low IL, high RL."
        )

        # Memory block (#36b). Empty string if memory_hits is None/empty,
        # which is the empty-on-no-match contract. The block is placed
        # before the param table so the LLM reads "here are past outcomes"
        # before "here are the knobs."
        memory_block = _build_memory_block(memory_hits)

        sys_prompt = (
            "You are an RF circuit designer sizing a phase-shifter at mmWave/sub-6 GHz. "
            "Your job: produce ONE .PARAM line with sensible starting values that an "
            "experienced RF engineer would try first. Stay within the listed ranges. "
            "Stay close to the defaults unless the spec gives a clear reason to deviate. "
            "Do NOT just pick rail values; rail values are physically extreme and "
            "almost always wrong. "
            f"Use ONLY these parameter names: {allowed_keys_str}. "
            "You MUST include every single one of these parameter names in your output. "
            "Missing any name will break the simulation. "
            "Use SPICE suffixes naturally: u=micro, n=nano, p=pico, k=kilo, meg=mega. "
            "For *_mm and *_pf parameters, write the bare number (units are implicit "
            "in the name). Output format example: "
            ".PARAM Z0_line=50 L_quarter_mm=1.7 C_load_pf=0.04 R_on=3 R_off=10k"
        )

        user_prompt = (
            f"Topology: {topology}\n"
            f"Context: {topo_context}\n\n"
            f"Target spec:\n"
            f"  fc            = {spec['fc_ghz']:.1f} GHz\n"
            f"  bandwidth     = {spec['bw_pct']:.0f}% (relative)\n"
            f"  coverage      = {spec['phase_coverage_deg']:.0f} deg\n"
            f"  phase_bits    = {spec['phase_bits']}\n"
            f"  max RMS phase err = {spec['rms_phase_err_deg']:.1f} deg\n"
            f"  max IL        = {spec['max_il_db']:.1f} dB\n"
            f"  min RL        = {spec['min_rl_db']:.1f} dB\n"
            f"  VDD           = {spec['vdd']:.1f} V\n"
            f"  max power     = {spec['pmax_mw']:.1f} mW\n\n"
            f"{memory_block}"
            f"Tunable parameters (with template defaults and physical ranges):\n"
            f"{param_table}\n\n"
            f"Output the single .PARAM line now:"
        )

        llm_out = _call_llm(sys_prompt, user_prompt,
                            model_override=self.model,
                            temperature=temperature)

        # Parse key=value pairs from LLM output
        param_matches = re.finditer(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9eE\.\-]+[a-zA-Z]*)",
            llm_out
        )
        params_dict = {}
        for m in param_matches:
            key = m.group(1)
            if skeleton_param_keys and key not in skeleton_param_keys:
                continue
            val = m.group(2).strip()
            val = re.sub(r"[,;.]+$", "", val)
            clamped_val = clamp_spice_value(key, val)
            params_dict[key] = clamped_val

        # Backfill missing keys from skeleton defaults.
        # The LLM does not always emit every skeleton-declared key. If the
        # generator strips the original .PARAM lines and emits a partial
        # replacement, references to the missing key in the schematic
        # become undefined and ngspice hangs at AC-sweep time. The fix
        # is to merge: LLM-emitted values take precedence; defaults fill
        # any omissions.
        if skeleton_param_keys and skeleton_defaults:
            for key in skeleton_param_keys:
                if key not in params_dict and key in skeleton_defaults:
                    params_dict[key] = clamp_spice_value(
                        key, skeleton_defaults[key]
                    )

        # Fallback: use sensible defaults if LLM produced nothing usable.
        # Prefer the skeleton's own defaults -- they reflect the template
        # author's calibration, not a guess.
        if not params_dict:
            if skeleton_defaults:
                params_dict = dict(skeleton_defaults)
            else:
                # Truly nothing usable -- last-resort generic phase-shifter
                params_dict = {
                    "Z0_line": "50",
                    "L_short_mm": "2.68",
                    "L_long_mm": "5.36",
                    "R_on": "3",
                    "R_off": "10k",
                }

        param_str = ".PARAM " + " ".join([f"{k}={v}" for k, v in params_dict.items()]) + "\n"

        # Remove original .PARAM line from skeleton, keep title line FIRST
        lines = skeleton_content.split('\n')
        new_lines = []
        for l in lines:
            stripped = l.strip().upper()
            if stripped.startswith(".PARAM"):
                if 'DERIVED' in stripped or 'FRAMEWORK_CONTROLLED' in stripped:
                    new_lines.append(l)
            else:
                new_lines.append(l)

        # SPICE requires title as first line. Insert .PARAM after it.
        if new_lines:
            final_netlist = new_lines[0] + '\n' + param_str + '\n'.join(new_lines[1:])
        else:
            final_netlist = f"* {topology}\n" + param_str

        fd, path = tempfile.mkstemp(suffix=".sp", prefix=f"llm_{topology}_")
        with os.fdopen(fd, 'w') as f:
            f.write(final_netlist)

        backend_label = BACKEND.upper()
        model_label = (self.model or (OLLAMA_MODEL if BACKEND == "ollama" else OPENROUTER_MODEL)).split("/")[-1]
        print(f"[{backend_label}_{model_label}] {len(params_dict)} params -> {path}")
        # Return both the netlist path and the parsed params dict. The
        # params dict is needed by the env's memory writer (#36b) so the
        # selected best-of-K attempt can be persisted as a (spec, params,
        # metrics, reward) tuple. params_dict values are SPICE value strings
        # (post-clamp), not floats; consumers that need numerics should
        # parse via parse_spice_val.
        return path, params_dict


# Module-level singleton
_gen_singleton = None

def generate(spec: dict, topology: str, temperature: float = 0.2,
             memory_hits: list[dict] | None = None) -> tuple[str, dict]:
    """Generate a netlist for (spec, topology).

    Args:
        spec: target spec dict.
        topology: template name (e.g. "loaded_line").
        temperature: LLM sampling temperature, passed through to the backend.
        memory_hits: optional list of past attempts (from Memory.read_top_k)
            to inject into the prompt. Each entry must have keys spec,
            params, metrics, reward. None or empty list -> no memory block
            in the prompt (session-3 baseline behavior).

    Returns:
        (netlist_path, params_dict) tuple. params_dict maps parameter name
        to SPICE value string. See LLMNetlistGen.generate.
    """
    global _gen_singleton
    if _gen_singleton is None:
        _gen_singleton = LLMNetlistGen()
    return _gen_singleton.generate(
        spec, topology, temperature=temperature, memory_hits=memory_hits
    )

def set_model(model: str):
    global _gen_singleton
    if _gen_singleton is None:
        _gen_singleton = LLMNetlistGen(model=model)
    else:
        _gen_singleton.set_model(model)
