#!/usr/bin/env python3
"""Stage 4 route evaluation using frozen calibrated policy and cached scores.
Does not make strong-model calls: REVIEW stays unresolved (selective metrics).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def route(df, policy, width_add=0):
    tab=policy['thresholds']
    if len(tab)!=22: raise ValueError('Expected 22 risk thresholds')
    score=df['unsafe_score'].to_numpy(float)
    risk=np.column_stack([df[f'risk_score_{j:02d}'].to_numpy(float) for j in range(22)])
    # Class routing uses PREDICTED risk, never ground-truth risk.
    selected=risk.argmax(axis=1)
    low=np.array([tab[int(j)]['tau_low'] for j in selected],float)
    high=np.array([tab[int(j)]['tau_high'] for j in selected],float)
    low=np.maximum(0,low-width_add)
    high=np.minimum(1,high+width_add)
    action=np.full(len(df),'REVIEW',dtype=object)
    action[score<low]='ALLOW'
    action[score>high]='REFUSE'
    return action,selected


def summary(df, action):
    y=df['unsafe_label'].to_numpy(int)
    if not np.isin(y,[0,1]).all():raise ValueError('Invalid binary labels')
    result={'samples':len(y)}
    for decision in ['ALLOW','REVIEW','REFUSE']:
        result[decision.lower()+'_count']=int((action==decision).sum())
        result[decision.lower()+'_rate']=float(np.mean(action==decision))
    for label, label_name in [(0,'safe'),(1,'unsafe')]:
        mask=y==label
        if not mask.any(): continue
        for decision in ['ALLOW','REVIEW','REFUSE']:
            result[f'{label_name}_{decision.lower()}_rate']=float(np.mean(action[mask]==decision))
    # REVIEW has unknown downstream outcome until oracle review predictions exist.
    result['unsafe_allow_rate_unconditional']=float(np.mean(action[y==1]=='ALLOW')) if (y==1).any() else None
    result['safe_refuse_rate_unconditional']=float(np.mean(action[y==0]=='REFUSE')) if (y==0).any() else None
    result['decided_coverage']=float(np.mean(action!='REVIEW'))
    result['review_unresolved']=True
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True,choices=[42,43,44])
    p.add_argument('--split',choices=['test_id','test_orbench_hard','test_xstest','test_harmful'],required=True)
    p.add_argument('--cache-root',default='results/stage34_cache')
    p.add_argument('--policy-root',default='results/stage3')
    p.add_argument('--out-root',default='results/stage4')
    p.add_argument('--width-grid',default='0,0.025,0.05,0.10')
    a=p.parse_args()
    df=pd.read_csv(Path(a.cache_root)/f'seed{a.seed}'/f'{a.split}.csv.gz')
    policy=json.loads((Path(a.policy_root)/f'seed{a.seed}'/'policy.json').read_text())
    widths=[float(x) for x in a.width_grid.split(',')]
    if any(x<0 for x in widths):raise ValueError('width must be >=0')
    rows=[]
    for width in widths:
        action,selected=route(df,policy,width)
        rows.append({'seed':a.seed,'split':a.split,'additional_width':width,**summary(df,action)})
    out=Path(a.out_root)/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    dest=out/f'{a.split}_routing_curve.csv'
    if dest.exists():raise FileExistsError(dest)
    pd.DataFrame(rows).to_csv(dest,index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('OK',dest)

if __name__=='__main__':main()
