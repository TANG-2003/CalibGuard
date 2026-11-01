#!/usr/bin/env python3
"""Stage 4 REVIEW cost calculation: actual adjudications required for final decisions.
Input external decisions CSV must have prompt_hash,review_decision (ALLOW/REFUSE),
and optionally latency_ms,cost_usd. Does NOT invent API latency or costs.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from stage34_evaluate import route, summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True,choices=[42,43,44])
    p.add_argument('--split',default='test_id')
    p.add_argument('--cache-root',default='results/stage34_cache')
    p.add_argument('--policy-root',default='results/stage3')
    p.add_argument('--review-decisions',required=True,help='Actual API adjudications by hash; never simulated')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    df=pd.read_csv(Path(a.cache_root)/f'seed{a.seed}'/f'{a.split}.csv.gz',dtype={'prompt_hash':str})
    policy=json.loads((Path(a.policy_root)/f'seed{a.seed}'/'policy.json').read_text())
    action,_=route(df,policy)
    reviews=df.loc[action=='REVIEW',['prompt_hash']].copy()
    decisions=pd.read_csv(a.review_decisions,dtype={'prompt_hash':str})
    if decisions['prompt_hash'].duplicated().any():raise ValueError('Duplicate review hash')
    if not {'prompt_hash','review_decision'}.issubset(decisions.columns):raise ValueError('Missing required columns')
    m=reviews.merge(decisions,on='prompt_hash',how='left',validate='one_to_one')
    if len(m)!=len(reviews) or m['review_decision'].isna().any():
        raise ValueError('Missing actual adjudications for REVIEW examples')
    if not m['review_decision'].isin(['ALLOW','REFUSE']).all():raise ValueError('Invalid decision')
    final=action.copy()
    final[action=='REVIEW']=m['review_decision'].to_numpy()
    result={'seed':a.seed,'split':a.split,'review_count':len(m),
            'review_rate':len(m)/len(df),'final':summary(df,final)}
    if 'latency_ms' in m:
        result['mean_review_latency_ms']=float(pd.to_numeric(m['latency_ms']).mean())
        result['review_latency_ms_per_input']=float(pd.to_numeric(m['latency_ms']).sum()/len(df))
    if 'cost_usd' in m:
        result['review_cost_usd_per_1000_inputs']=float(pd.to_numeric(m['cost_usd']).sum()*1000/len(df))
    result['final']['review_unresolved']=False
    dest=Path(a.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('OK',dest,result)

if __name__=='__main__':main()
