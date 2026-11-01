#!/usr/bin/env python3
"""Generate truly held-out fold manifests. No fake OOD results from a pre-trained full model.
Multilabel rows containing the held-out label are removed from fit/calibration.
This needs separate model retraining per fold before claiming strict LODO/LORO.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-path',default='data/splits/calibguard_split_v2.csv')
    p.add_argument('--axis',choices=['domain','risk'],required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    df=pd.read_csv(a.data_path,low_memory=False)
    from sys import path
    path.insert(0,str(Path(__file__).resolve().parent))
    import formal_train_deberta_multitask as base
    col='domains' if a.axis=='domain' else 'risk_categories'
    df['_labels']=df[col].map(base.parse_label_list)
    train=df.loc[df['split']=='train'].copy()
    test=df.loc[df['split']=='test_id'].copy()
    vocab=sorted(set(x for lst in train['_labels'] for x in lst if x!='No Risk'))
    rows=[]
    for label in vocab:
        train_has=train['_labels'].map(lambda labels:label in labels)
        test_has=test['_labels'].map(lambda labels:label in labels)
        rows.append({'axis':a.axis,'heldout_label':label,
                     'fit_pool_count_excluding_heldout':int((~train_has).sum()),
                     'train_rows_excluded':int(train_has.sum()),
                     'heldout_id_test_count':int(test_has.sum()),
                     'requires_retraining':True,
                     'note':'A valid strict fold cannot reuse a model trained on this held-out label.'})
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('OK',out,'folds=',len(rows))

if __name__=='__main__':main()
