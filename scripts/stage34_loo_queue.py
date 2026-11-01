#!/usr/bin/env python3
"""Deterministic, resumable strict fold training queue. No GPU work without --execute.
A full queue is 41 folds * 3 seeds = 123 independent model fits; expensive.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-config',default='results/stage2_full_seed42_6ep_v1/run_config.json')
    p.add_argument('--axes',nargs='+',choices=['risk','domain'],default=['domain','risk'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--max-runs',type=int,default=0,help='0=all; max NEW runs, skipping completed')
    p.add_argument('--out-root',default='results/stage4/strict_loo')
    a=p.parse_args()
    cfg=json.loads(Path(a.reference_config).read_text(encoding='utf-8'))
    todo=[];completed=0
    for axis in a.axes:
        for label in cfg['domain_vocab' if axis=='domain' else 'risk_vocab']:
            slug=re.sub('[^a-zA-Z0-9_-]+','_',label).strip('_').lower()
            for seed in a.seeds:
                dest=Path(a.out_root)/axis/slug/f'seed{seed}'
                if (dest/'COMPLETE.json').exists():
                    completed+=1;continue
                todo.append((axis,label,seed))
    print('STRICT LOO queue:', 'completed',completed,'pending',len(todo),'run now',a.max_runs or len(todo))
    if not a.execute:
        print('DRY RUN ONLY: add --execute to train. This is a costly independent retraining matrix.')
        for axis,label,seed in todo[:12]: print('PENDING',axis,label,seed)
        return
    for axis,label,seed in (todo[:a.max_runs] if a.max_runs>0 else todo):
        cmd=[sys.executable,'-u','scripts/stage34_loo_train.py','--axis',axis,'--label',label,
             '--seed',str(seed),'--epochs',str(a.epochs),'--reference-config',a.reference_config,
             '--out-root',a.out_root,'--execute']
        print('RUN',cmd,flush=True)
        subprocess.run(cmd,check=True)
    print('Queue slice complete; rerun command to resume remaining folds.')

if __name__=='__main__':main()
