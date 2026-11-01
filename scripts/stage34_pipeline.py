#!/usr/bin/env python3
"""Resume-aware Stage3/4 score-cache pipeline. Existing output is never overwritten.
By default runs 3 seeds x 5 splits once, then calibration and offline analyses.
External API calls and strict 123 fit LOO are deliberately separate commands.
"""
import argparse,subprocess,sys
from pathlib import Path

def run(args,out):
    if out.exists():
        print('SKIP',out,flush=True);return
    print('RUN',' '.join(args),flush=True)
    subprocess.run(args,check=True)
    if not out.exists():raise RuntimeError(f'Expected output absent after command: {out}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    ap.add_argument('--calibration-only',action='store_true',help='Use pre-existing prediction caches; no GPU')
    a=ap.parse_args()
    splits=['validation','test_id','test_orbench_hard','test_xstest','test_harmful']
    for seed in a.seeds:
        if not a.calibration_only:
            for split in splits:
                path=Path(f'results/stage34_cache/seed{seed}/{split}.csv.gz')
                run([sys.executable,'-u','scripts/stage34_export.py','--seed',str(seed),'--split',split],path)
        policy=Path(f'results/stage3/seed{seed}/policy.json')
        run([sys.executable,'scripts/stage34_calibrate.py','--seed',str(seed)],policy)
        for split in splits[1:]:
            route_path=Path(f'results/stage4/seed{seed}/{split}_routing_curve.csv')
            run([sys.executable,'scripts/stage34_evaluate.py','--seed',str(seed),'--split',split],route_path)
            base_path=Path(f'results/stage4/baselines/seed{seed}/{split}_global_fixed.csv')
            run([sys.executable,'scripts/stage34_baselines.py','--seed',str(seed),'--split',split],base_path)
    print('CORE PIPELINE COMPLETE. Actual API REVIEW and strict LOO are independent follow-up runs.')

if __name__=='__main__':main()
