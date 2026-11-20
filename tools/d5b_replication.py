"""Frozen D5b replication: three preselected role/holdout pairs, three seeds.

Thread limits affect execution speed only. Training/data/loss match D5b.
No RL and no hyperparameter selection from these holdouts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import d5b_role_transfer as pilot

CASES = [('impedance', 'Vector_Modulator'), ('impedance', 'Switched_Line'),
         ('length', 'Vector_Modulator')]
VARIANTS = ['action', 'gin', 'circuit-typed', 'd1a', 'r3']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/graphs/d5b_replication_bce_fix',
                    help='Fresh dir required (protocol SHA must match this code)')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(cases=CASES, variants=VARIANTS, seeds=args.seeds,
                    n=args.n, epochs=args.epochs, prior=False,
                    pilot_sha256=hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest(),
                    uncertainty='Within-run sample bootstrap; seed range reported separately. Not pooled independent trials.')
    manifest = root/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError('Output directory has a different frozen protocol')
    manifest.write_text(json.dumps(protocol, indent=2))
    results = []
    for role, holdout in CASES:
        for seed in map(int, args.seeds.split(',')):
            dest = root/f'{role}_{holdout}_seed{seed}.json'
            rec = json.loads(dest.read_text()) if dest.exists() else dict(role=role, holdout=holdout, seed=seed, variants={})
            if len(rec['variants']) < len(VARIANTS):
                by_topo = {t: pilot.build_role_samples(t, role, args.n,
                    np.random.default_rng(seed*1000+pilot.TOPO_SEED[t])) for t in pilot.ROLE_TOPOS[role]}
                if any(len(rows) != args.n for rows in by_topo.values()):
                    raise RuntimeError('Incomplete dataset')
                train = [r for t, rows in by_topo.items() if t != holdout for r in rows]
                test = by_topo[holdout]
                for variant in VARIANTS:
                    if variant in rec['variants']:
                        continue
                    print(f'{role} -> {holdout}, seed={seed}, {variant}', flush=True)
                    rec['variants'][variant] = pilot.train_eval(variant, train, test, args.epochs, seed,
                        ckpt_dir=str(root/'checkpoints'/role/holdout/str(seed)))
                    temporary = dest.with_suffix('.tmp')
                    temporary.write_text(json.dumps(rec, indent=2))
                    temporary.replace(dest)
                    print(f"  gate={rec['variants'][variant]['mean_balanced_sign_gate']}", flush=True)
            results.append(rec)
    summary = []
    for role, holdout in CASES:
        for variant in VARIANTS:
            runs = [r['variants'][variant] for r in results if r['role']==role and r['holdout']==holdout]
            metrics = {}
            for metric in pilot.METRIC_KEYS:
                vals = [r['uncertainty'][metric]['balanced_sign_accuracy'] for r in runs]
                valid = [v for v in vals if v is not None]
                metrics[metric] = dict(balanced_sign_by_seed=vals,
                    mean_balanced_sign=float(np.mean(valid)) if valid else None,
                    seed_range=[min(valid),max(valid)] if valid else None,
                    mae_by_seed=[r['errors']['mae'][metric] for r in runs],
                    counts_by_seed=[r['uncertainty'][metric]['counts'] for r in runs])
            summary.append(dict(role=role, holdout=holdout, variant=variant, metrics=metrics))
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    print('Replication complete', flush=True)

if __name__ == '__main__':
    main()
