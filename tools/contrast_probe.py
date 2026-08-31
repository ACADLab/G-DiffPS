"""Does the state-contrast block of z_tau carry the phase shift?

The claim under test: a phase shifter is an indexed family of circuits, one
per switch state, and the specified quantity is a difference between two
members of that family. If that is right, the contrast block

    z_contrast = pool_d mean_{s<s'} | h_s[d] - h_s'[d] |

should predict the realised phase step, and the symmetric block

    z_mean     = pool_d mean_s h_s[d]

should not -- pooling over states destroys the contrast that is the device.

Ground truth comes from the MNA scorer (same incidence graph, same 50-ohm port
convention as the SPICE templates), so no ngspice calls are needed. Since the
phase step wraps, the regression targets are cos(dphi) and sin(dphi).

Both blocks are read from an *untrained* encoder by default: this measures what
the representation makes available, not what a trained model has learned. Pass
--run to load a checkpoint instead.

Usage:
    python tools/contrast_probe.py --n 400 --out results/graphs/contrast_probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import (
    N_STATES, TOPOLOGY_NETLIST, nominal_params, _normalize_name,
)
from models.circuit_encoder import CircuitEncoder
from sim.mna_scorer import solve_sparams
from train_diffusion import TOPOLOGY_PARAMS, action_to_params


def _wrap_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def contrast_states(topology: str) -> tuple[int, int]:
    """The state pair whose difference the spec names: one phase step.

    This is (0, 1) for every topology, including the 16-state Vector
    Modulator, whose `ideal_step_deg` is 22.5. The wide pair (0, n//2) is not
    usable as ground truth: it negates the VM's I/Q drive exactly, so the step
    is 180 deg for every sizing and the target becomes a constant.
    """
    _ = N_STATES[_normalize_name(topology)]
    return (0, 1)


def measure_dphi(topology: str, params: dict, fc_ghz: float) -> float | None:
    """Realised phase step between the two contrasted states, in degrees."""
    s_a, s_b = contrast_states(topology)
    try:
        _, s21_a = solve_sparams(topology, params, fc_ghz, state=s_a)
        _, s21_b = solve_sparams(topology, params, fc_ghz, state=s_b)
    except Exception:
        return None
    if not (np.isfinite(abs(s21_a)) and np.isfinite(abs(s21_b))):
        return None
    # Reject states that carry no signal: the phase of a null is meaningless.
    if abs(s21_a) < 1e-9 or abs(s21_b) < 1e-9:
        return None
    pa = math.degrees(math.atan2(s21_a.imag, s21_a.real))
    pb = math.degrees(math.atan2(s21_b.imag, s21_b.real))
    return _wrap_180(pb - pa)


def ridge_cv_r2(X: np.ndarray, Y: np.ndarray, n_folds: int = 5,
                alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
                seed: int = 0) -> float:
    """Cross-validated R^2 of ridge regression from X to Y (multi-output).

    Standardises X on each training fold and picks alpha by an inner split, so
    the reported number is held-out and not tuned on the test fold.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    if n < n_folds * 3:
        return float("nan")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ss_res, ss_tot = 0.0, 0.0
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        # Inner split to choose alpha.
        cut = max(4, int(0.8 * len(tr)))
        best_a, best_err = alphas[0], np.inf
        for a in alphas:
            m = Ridge(alpha=a).fit(Xtr[:cut], Y[tr][:cut])
            err = float(np.mean((m.predict(Xtr[cut:]) - Y[tr][cut:]) ** 2))
            if err < best_err:
                best_err, best_a = err, a
        model = Ridge(alpha=best_a).fit(Xtr, Y[tr])
        pred = model.predict(Xte)
        ss_res += float(np.sum((Y[te] - pred) ** 2))
        ss_tot += float(np.sum((Y[te] - Y[tr].mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def collect(encoder: CircuitEncoder, topology: str, n: int, rng: np.random.Generator,
            fc_lo: float, fc_hi: float, bounds: str, z_sizing: str = "nominal",
            dphi_sizing: str = "sampled"):
    """Sample sizings, encode them, and measure the resulting phase step.

    `z_sizing` controls what the *encoder* sees. In deployment it is always
    `nominal`: the encoder runs before the actor picks values, so it can only
    be given the bounds midpoint. Passing `sampled` measures the capacity of
    the representation given values it does not actually have at run time, and
    will overstate what the deployed encoder knows.
    """
    keys = TOPOLOGY_PARAMS[topology]
    Zm, Zc, D = [], [], []
    attempts = 0
    while len(D) < n and attempts < n * 6:
        attempts += 1
        fc = float(rng.uniform(fc_lo, fc_hi))
        action = rng.random(len(keys)).astype(np.float32)
        params = action_to_params(action, topology, {"fc_ghz": fc},
                                  sizing="log", bounds=bounds)
        nominal = nominal_params(topology, {"fc_ghz": fc}, bounds=bounds)
        dphi = measure_dphi(topology, params if dphi_sizing == "sampled" else nominal, fc)
        if dphi is None:
            continue
        z_params = params if z_sizing == "sampled" else nominal
        with torch.no_grad():
            _, parts = encoder(topology, {"fc_ghz": fc}, params=z_params,
                               bounds=bounds, return_parts=True)
        Zm.append(parts["z_mean"].squeeze(0).cpu().numpy())
        Zc.append(parts["z_contrast"].squeeze(0).cpu().numpy())
        D.append(dphi)
    if not D:
        return None
    d = np.asarray(D, dtype=np.float64)
    Y = np.stack([np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))], axis=1)
    return np.asarray(Zm), np.asarray(Zc), Y, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="samples per topology")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fc-lo", type=float, default=2.0)
    ap.add_argument("--fc-hi", type=float, default=40.0)
    ap.add_argument("--bounds", type=str, default="electrical",
                    choices=["legacy", "electrical"])
    ap.add_argument("--pool-states", type=str, default="endpoints",
                    choices=["endpoints", "all"])
    ap.add_argument("--z-sizing", type=str, default="nominal",
                    choices=["nominal", "sampled"],
                    help="sizing the encoder sees; 'nominal' matches deployment")
    ap.add_argument("--dphi-sizing", type=str, default="sampled",
                    choices=["nominal", "sampled"],
                    help="sizing the phase step is measured on")
    ap.add_argument("--run", type=str, default=None,
                    help="checkpoint dir with gnn_encoder.pt (default: untrained)")
    ap.add_argument("--out", type=str,
                    default="results/graphs/contrast_probe.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    encoder = CircuitEncoder(pool_states=args.pool_states)
    if args.run:
        ckpt = os.path.join(args.run, "gnn_encoder.pt")
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"loaded {ckpt}")
    encoder.eval()

    rng = np.random.default_rng(args.seed)
    report = {
        "n_per_topology": args.n,
        "encoder": "trained" if args.run else "untrained",
        "run": args.run,
        "bounds": args.bounds,
        "pool_states": args.pool_states,
        "z_sizing": args.z_sizing,
        "dphi_sizing": args.dphi_sizing,
        "fc_range_ghz": [args.fc_lo, args.fc_hi],
        "target": "cos/sin of measured dphi between the two contrasted states",
        "per_topology": {},
    }

    hdr = f"{'topology':<18} {'n':>5} {'contrast':>9} {'mean':>9} {'both':>9} {'sd(dphi)':>9}"
    print(hdr)
    print("-" * len(hdr))

    all_zc, all_zm, all_y = [], [], []
    for topo in TOPOLOGY_NETLIST:
        got = collect(encoder, topo, args.n, rng, args.fc_lo, args.fc_hi, args.bounds,
                      z_sizing=args.z_sizing, dphi_sizing=args.dphi_sizing)
        if got is None:
            print(f"{topo:<18} {'--':>5}  no usable samples")
            continue
        Zm, Zc, Y, d = got
        if float(np.mean(np.var(Y, axis=0))) < 1e-8:
            # A constant target makes R^2 meaningless (and trivially 1.0).
            print(f"{topo:<18} {len(d):>5}  target is constant -- skipped")
            report["per_topology"][topo] = {
                "n": int(len(d)), "skipped": "constant dphi",
                "dphi_std_deg": float(np.std(d)),
            }
            continue
        r2_c = ridge_cv_r2(Zc, Y, seed=args.seed)
        r2_m = ridge_cv_r2(Zm, Y, seed=args.seed)
        r2_b = ridge_cv_r2(np.concatenate([Zm, Zc], axis=1), Y, seed=args.seed)
        sd = float(np.std(d))
        report["per_topology"][topo] = {
            "n": int(len(d)), "r2_contrast": r2_c, "r2_mean": r2_m,
            "r2_both": r2_b, "dphi_std_deg": sd,
        }
        print(f"{topo:<18} {len(d):>5} {r2_c:>9.3f} {r2_m:>9.3f} {r2_b:>9.3f} {sd:>9.1f}")
        all_zc.append(Zc)
        all_zm.append(Zm)
        all_y.append(Y)

    if all_y:
        Zc = np.concatenate(all_zc)
        Zm = np.concatenate(all_zm)
        Y = np.concatenate(all_y)
        pooled = {
            "n": int(Y.shape[0]),
            "r2_contrast": ridge_cv_r2(Zc, Y, seed=args.seed),
            "r2_mean": ridge_cv_r2(Zm, Y, seed=args.seed),
            "r2_both": ridge_cv_r2(np.concatenate([Zm, Zc], axis=1), Y, seed=args.seed),
        }
        report["pooled"] = pooled
        print("-" * len(hdr))
        print(f"{'POOLED':<18} {pooled['n']:>5} {pooled['r2_contrast']:>9.3f} "
              f"{pooled['r2_mean']:>9.3f} {pooled['r2_both']:>9.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
