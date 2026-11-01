import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage34_calibrate import calc
from stage34_evaluate import route, summary

class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        n=300
        y=np.array([0]*150+[1]*150)
        s=np.r_[rng.uniform(0,0.4,150),rng.uniform(0.5,1,150)]
        d={'prompt_hash':[f'h{i}' for i in range(n)],'unsafe_label':y,
           'unsafe_score':s,'risk_mask':np.ones(n,bool)}
        for j in range(22):
            d[f'risk_true_{j:02d}']=(np.arange(n)%22==j).astype(int)
            d[f'risk_score_{j:02d}']=rng.uniform(size=n)
        self.df=pd.DataFrame(d)

    def test_calibration_bounds(self):
        tab,meta=calc(self.df)
        self.assertEqual(len(tab),22)
        self.assertTrue((tab['tau_low']<=tab['tau_high']).all())
        self.assertTrue(((tab['gamma']>=0)&(tab['gamma']<=1)).all())
        self.assertEqual(meta['n_safe'],150)

    def test_route_counts(self):
        tab,meta=calc(self.df)
        policy={'thresholds':tab.where(pd.notnull(tab),None).to_dict(orient='records')}
        act,_=route(self.df,policy)
        out=summary(self.df,act)
        self.assertEqual(sum(out[f'{a}_count'] for a in ('allow','review','refuse')),300)

if __name__=='__main__':unittest.main()
