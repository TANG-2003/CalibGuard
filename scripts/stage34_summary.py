#!/usr/bin/env python3
"""Aggregate saved stage3/4 empirical policies and stage4 routing rows, not reviewer outcomes."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/stage4/offline_routing_summary.csv')
    a=p.parse_args()
    rows=[]
    for seed in (42,43,44):
        policy=Path(f'results/stage3/seed{seed}/policy.json')
        if not policy.exists():continue
        meta=json.loads(policy.read_text())['metadata']
        for split in ('test_id','test_orbench_hard','test_xstest','test_harmful'):
            f=Path(f'results/stage4/seed{seed}/{split}_routing_curve.csv')
            if not f.exists():continue
            t=pd.read_csv(f)
            for _,r in t.iterrows():
                rows.append({'seed':seed,'split':split,'alpha':meta['alpha'],
                             'global_high':meta['global_high'],**r.to_dict()})
    if not rows:raise RuntimeError('No saved routing results found')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('Saved',out,'rows',len(rows),'(REVIEW unresolved; not final system accuracy)')
if __name__=='__main__':main()
