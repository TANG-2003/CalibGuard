#!/usr/bin/env python3
"""Empirical category-conditioned score thresholds, shrinkage, and routing policy.
NOT independent split-conformal inference: validation selected the checkpoint.
FPR control is empirical on labeled SAFE validation examples, not a guaranteed
category-conditional claim (SAFE samples have no ground-truth risk category).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def quantile_higher(values, q):
    if len(values) == 0:
        raise ValueError('Cannot take quantile of empty array')
    return float(np.quantile(np.asarray(values, dtype=float), q, method='higher'))


def calc(cal, alpha=0.01, kappa=100.0, review_width=0.05, min_pos=20):
    if not 0 < alpha < 0.5 or kappa <= 0 or review_width < 0:
        raise ValueError('Bad alpha/kappa/review-width')
    assert cal['prompt_hash'].is_unique
    y = cal['unsafe_label'].to_numpy(int)
    s = cal['unsafe_score'].to_numpy(float)
    safe = s[y == 0]
    harmful = s[y == 1]
    if len(safe) < 1 or len(harmful) < 1 or not np.isfinite(s).all():
        raise ValueError('Calibration requires finite scores and both classes')
    # For strict '>' on score, benign FPR on calibration = fraction safe above threshold.
    # This is empirical, since the same validation data selected checkpoints.
    global_high = quantile_higher(safe, 1 - alpha)
    global_low = quantile_higher(harmful, alpha)
    rows = []
    for j in range(22):
        col = f'risk_true_{j:02d}'
        target = (cal[col].to_numpy(int) == 1) & (y == 1) & cal['risk_mask'].to_numpy(bool)
        pos = s[target]
        n = len(pos)
        independent_low = quantile_higher(pos, alpha) if n else None
        gamma = n / (n + kappa)
        # Shrink the category-specific UNSAFE lower-tail threshold toward global.
        # SAFE FPR is controlled by the global upper score threshold, not this low threshold.
        shrunk_low = gamma * independent_low + (1-gamma) * global_low if n else global_low
        # Expand REVIEW toward both sides. Do NOT reorder to force a non-empty ALLOW region.
        low = max(0.0, min(shrunk_low, global_high) - review_width)
        high = min(1.0, max(global_high, low) + review_width)
        rows.append({'risk_index': j, 'unsafe_support': n, 'gamma':gamma,
                     'independent_low':independent_low, 'global_low':global_low,
                     'shrunk_low':shrunk_low, 'tau_low':low, 'tau_high':high,
                     'global_safe_upper_quantile':global_high,
                     'short_tail_under_min_pos':bool(n < min_pos)})
    return pd.DataFrame(rows), {'alpha':alpha, 'kappa':kappa, 'review_width': review_width,
                                  'min_pos':min_pos, 'n_safe':int(len(safe)),
                                  'n_unsafe':int(len(harmful)),
                                  'global_low':global_low,'global_high':global_high,
                                  'calibration_scope':'empirical reuse of model-selection validation set',
                                  'guarantee':'no finite-sample conformal or per-category FPR guarantee'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True,choices=[42,43,44])
    parser.add_argument('--cache-root',default='results/stage34_cache')
    parser.add_argument('--out-root',default='results/stage3')
    parser.add_argument('--alpha',type=float,default=0.01)
    parser.add_argument('--kappa',type=float,default=100)
    parser.add_argument('--review-width',type=float,default=0.05)
    a=parser.parse_args()
    cache=Path(a.cache_root)/f'seed{a.seed}'/'validation.csv.gz'
    cal=pd.read_csv(cache)
    thresholds,meta=calc(cal,a.alpha,a.kappa,a.review_width)
    out=Path(a.out_root)/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    tab=out/'thresholds.csv'
    if tab.exists(): raise FileExistsError(tab)
    thresholds.to_csv(tab,index=False)
    meta['seed']=a.seed
    (out/'policy.json').write_text(json.dumps({'metadata':meta,'thresholds':thresholds.replace({np.nan:None}).to_dict(orient='records')},indent=2,ensure_ascii=False))
    print('OK',tab,'global upper',meta['global_high'])

if __name__=='__main__':main()
