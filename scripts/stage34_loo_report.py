#!/usr/bin/env python3
"""Summarize actual strict LOO fold model evaluation, ECE and Brier; report missing folds."""
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import brier_score_loss

def ece(y,p,bins=15):
    edges=np.linspace(0,1,bins+1)
    ids=np.clip(np.digitize(p,edges,right=False)-1,0,bins-1)
    total=0.
    for i in range(bins):
        mask=ids==i
        if mask.any():total+=float(mask.mean())*abs(float(y[mask].mean())-float(p[mask].mean()))
    return total

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-config',default='results/stage2_full_seed42_6ep_v1/run_config.json')
    ap.add_argument('--root',default='results/stage4/strict_loo')
    ap.add_argument('--out',default='results/stage4/strict_loo_summary.csv')
    a=ap.parse_args()
    cfg=json.loads(Path(a.reference_config).read_text())
    rows=[];missing=[]
    for axis in ('domain','risk'):
        for label in cfg['domain_vocab' if axis=='domain' else 'risk_vocab']:
            slug=re.sub('[^a-zA-Z0-9_-]+','_',label).strip('_').lower()
            for seed in (42,43,44):
                folder=Path(a.root)/axis/slug/f'seed{seed}'
                if not (folder/'COMPLETE.json').exists():
                    missing.append((axis,label,seed));continue
                info=json.loads((folder/'test_metrics.json').read_text())
                pred=pd.read_csv(folder/'fold_predictions.csv.gz')
                y=pred.unsafe_label.to_numpy(int);p=pred.unsafe_score.to_numpy(float)
                rec={'axis':axis,'heldout_label':label,'seed':seed,'test_n':len(y),
                     'best_epoch':info['best_epoch'],'unsafe_f1':info['metrics']['unsafe_f1'],
                     'accuracy':info['metrics']['accuracy'],'brier':float(brier_score_loss(y,p)),
                     'ece15':ece(y,p),'safe_n':int((y==0).sum()),'unsafe_n':int((y==1).sum())}
                rec['safe_fpr']=float((p[y==0]>=.5).mean()) if (y==0).any() else None
                rec['unsafe_recall']=float((p[y==1]>=.5).mean()) if (y==1).any() else None
                rows.append(rec)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('ACTUAL RESULTS',len(rows),'MISSING',len(missing),'output',out)
    if missing:
        print('INCOMPLETE STRICT LOO: cannot report 41x3 completed. Examples:',missing[:8])

if __name__=='__main__':main()
