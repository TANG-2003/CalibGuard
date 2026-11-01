import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage34_review_api import parse_decision,review_subset
from stage34_baselines import frozen_global_threshold,basic_metrics,selective_curve

class ExtraTests(unittest.TestCase):
    def test_decision_validation(self):
        self.assertEqual(parse_decision('allow\n'),'ALLOW')
        self.assertEqual(parse_decision('REFUSE'),'REFUSE')
        with self.assertRaises(ValueError):parse_decision('ALLOW because safe')
    def test_threshold_and_curve(self):
        df=pd.DataFrame({'unsafe_label':[0,0,0,1,1], 'unsafe_score':[.1,.2,.3,.8,.9]})
        q=frozen_global_threshold(df,.1)
        self.assertGreaterEqual(q,.3)
        self.assertEqual(basic_metrics(df,q)['safe_fpr'],0.)
        curve=selective_curve(df,[0,.1])
        self.assertEqual(len(curve),2)
        self.assertTrue((curve['coverage']<=1).all())
    def test_review_only(self):
        p={'thresholds':[{'tau_low':.4,'tau_high':.6} for _ in range(22)]}
        d={'prompt_hash':['a','b','c'],'unsafe_score':[.1,.5,.9]}
        for j in range(22):d[f'risk_score_{j:02d}']=[1.0,1.0,1.0]
        review=review_subset(pd.DataFrame(d),p)
        self.assertEqual(review['prompt_hash'].tolist(),['b'])
if __name__=='__main__':unittest.main()
