#!/usr/bin/env python3
"""Compare actual externally-generated classifier predictions with a cached stage2 split.
Required external CSV: prompt_hash, unsafe_score OR predicted_unsafe (0/1).
No invented LlamaGuard or raw target-model results. Baseline producer/model/protocol
must be documented alongside file; all rows must match cache, exactly once.
"""
import argparse, json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    ap.add_argument('--split',required=True)
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--baseline-name',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    cache=pd.read_csv(f'results/stage34_cache/seed{a.seed}/{a.split}.csv.gz',dtype={'prompt_hash':str})
    ext=pd.read_csv(a.predictions,dtype={'prompt_hash':str})
    if ext.prompt_hash.duplicated().any():raise ValueError('External duplicate hashes')
    if not ('unsafe_score' in ext or 'predicted_unsafe' in ext):
        raise ValueError('Need actual unsafe_score or predicted_unsafe')
    cols=['prompt_hash','unsafe_score'] if 'unsafe_score' in ext else ['prompt_hash','predicted_unsafe']
    merged=cache[['prompt_hash','unsafe_label']].merge(ext[cols],on='prompt_hash',how='left',validate='one_to_one',indicator=True)
    if len(merged)!=len(cache) or not (merged['_merge']=='both').all():
        raise ValueError('Missing external predictions')
    y=merged.unsafe_label.to_numpy(int)
    if 'unsafe_score' in merged:
        prob=merged.unsafe_score.to_numpy(float)
        if not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any():raise ValueError('Bad probabilities')
        pred=prob>=.5
    else:
        pred=merged.predicted_unsafe.to_numpy(float)
        if not np.isin(pred,[0,1]).all():raise ValueError('Bad labels')
        pred=pred.astype(bool)
        prob=None
    results={'baseline':a.baseline_name,'seed':a.seed,'split':a.split,'n':len(y),
             'accuracy':float(np.mean(pred==y)),
             'safe_fpr':float(np.mean(pred[y==0])) if (y==0).any() else None,
             'unsafe_recall':float(np.mean(pred[y==1])) if (y==1).any() else None}
    if prob is not None:
        results['brier']=float(brier_score_loss(y,prob))
        results['auroc']=float(roc_auc_score(y,prob)) if len(set(y))==2 else None
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('OK',out,results)
if __name__=='__main__':main()
