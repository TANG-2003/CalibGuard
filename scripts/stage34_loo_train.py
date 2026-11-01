#!/usr/bin/env python3
"""STRICT fold-specific leave-one-label-out Full DeBERTa re-training.

Held-out label NEVER appears in fold fit OR model-selection validation, including
multilabel rows carrying the held-out label. Uses frozen test_id only for fold
out-of-distribution evaluation. These are NEW models, not Stage-2 checkpoints.

No training occurs without --execute. One fold per invocation; checkpoint_best only.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
sys.path.insert(0, str(Path(__file__).resolve().parent))
import formal_train_deberta_multitask as base


def fold_data(df, axis, label, val_ratio=.1):
    col = 'domains' if axis == 'domain' else 'risk_categories'
    heldout = df[col].map(base.parse_label_list).map(lambda x: label in x)
    train = df.loc[(df.split == 'train') & ~heldout].copy()
    # Held-out evaluation truly absent from fit and validation. Risk folds include
    # ID-test SAFE controls so that false positive rates remain measurable.
    test = df.loc[(df.split == 'test_id') & (heldout | ((df.is_safe == 1) & (axis == 'risk')))].copy()
    if not len(train) or not len(test) or not ((df.split=='test_id') & heldout).any():
        raise ValueError('Empty fold; check label and source data')
    if (train.prompt_hash.astype(str).isin(test.prompt_hash.astype(str))).any():
        raise ValueError('Overlapping fit and test hashes')
    return train, test


def build(args):
    if args.epochs < 1 or not 0 < args.val_ratio < .5:
        raise ValueError('Invalid training settings')
    conf = json.loads(Path(args.reference_config).read_text(encoding='utf-8'))
    vocab = conf['domain_vocab'] if args.axis == 'domain' else conf['risk_vocab']
    if args.label not in vocab:
        raise ValueError(f'Unknown label: {args.label}')
    df = pd.read_csv(args.data_path, low_memory=False)
    train, test = fold_data(df, args.axis, args.label)
    y = pd.to_numeric(train.is_safe, errors='raise').astype(int).to_numpy()
    fit_idx, val_idx = train_test_split(np.arange(len(train)), test_size=args.val_ratio,
                                         random_state=42, stratify=y)
    fit, val = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()
    for p in (fit,val,test):
        p['risk_categories_parsed'] = p.risk_categories.map(base.parse_label_list)
        p['domains_parsed'] = p.domains.map(base.parse_label_list)
    col = 'domains_parsed' if args.axis == 'domain' else 'risk_categories_parsed'
    for p in (fit,val):
        assert not p[col].map(lambda ls: args.label in ls).any()
    cfg = dict(axis=args.axis, label=args.label, seed=args.seed, epochs=args.epochs,
               model_id=args.model_id, original_split='V2 train/test 70/30',
               fold_fit=len(fit), fold_validation=len(val), fold_test=len(test),
               heldout_positive_test=int(test[col].map(lambda ls:args.label in ls).sum()),
               fold_selection='validation unsafe F1', risk_vocab=conf['risk_vocab'],
               domain_vocab=conf['domain_vocab'],
               split_train_sha256=hashlib.sha256('\n'.join(fit.prompt_hash.astype(str)).encode()).hexdigest(),
               split_val_sha256=hashlib.sha256('\n'.join(val.prompt_hash.astype(str)).encode()).hexdigest())
    return fit, val, test, cfg


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--axis',choices=['risk','domain'],required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    ap.add_argument('--data-path',default='data/splits/calibguard_split_v2.csv')
    ap.add_argument('--reference-config',default='results/stage2_full_seed42_6ep_v1/run_config.json')
    ap.add_argument('--model-id',default='models/deberta-v3-base')
    ap.add_argument('--out-root',default='results/stage4/strict_loo')
    ap.add_argument('--epochs',type=int,default=6)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--eval-batch-size',type=int,default=32)
    ap.add_argument('--max-length',type=int,default=128)
    ap.add_argument('--val-ratio',type=float,default=.1)
    ap.add_argument('--learning-rate',type=float,default=2e-5)
    ap.add_argument('--weight-decay',type=float,default=.01)
    ap.add_argument('--warmup-ratio',type=float,default=.06)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    fit,val,test,cfg=build(a)
    import re
    slug=re.sub('[^a-zA-Z0-9_-]+','_',a.label).strip('_').lower()
    out=Path(a.out_root)/a.axis/slug/f'seed{a.seed}'
    if not a.execute:
        print(json.dumps({'dry_run':True,'output':str(out),**cfg},ensure_ascii=False,indent=2));return
    if out.exists() and any(out.iterdir()):
        if (out/'COMPLETE.json').exists():
            print('SKIP COMPLETE',out);return
        raise FileExistsError(f'Incomplete folder (inspect before retry): {out}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out.mkdir(parents=True,exist_ok=True)
    (out/'fold_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    base.seed_all(a.seed)
    for p in (fit,val,test):
        if p.prompt_hash.duplicated().any():raise ValueError('duplicate hashes')
    rv,dv=cfg['risk_vocab'],cfg['domain_vocab']
    tokenizer=AutoTokenizer.from_pretrained(a.model_id,use_fast=False)
    fit_ds=base.MultiTaskDataset(fit,rv,dv)
    val_ds=base.MultiTaskDataset(val,rv,dv)
    test_ds=base.MultiTaskDataset(test,rv,dv)
    collate=base.make_collator(tokenizer,a.max_length)
    def loader(ds,batch,shuffle=False):
        return DataLoader(ds,batch_size=batch,shuffle=shuffle,collate_fn=collate,
                          num_workers=0,pin_memory=True)
    train_loader=loader(fit_ds,a.batch_size,True)
    val_loader=loader(val_ds,a.eval_batch_size)
    test_loader=loader(test_ds,a.eval_batch_size)
    model=base.DebertaMultiTaskModel(a.model_id,len(rv),len(dv),dropout=.1).cuda()
    device=torch.device('cuda')
    opt=AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    steps=a.epochs*len(train_loader)
    scheduler=get_linear_schedule_with_warmup(opt,int(steps*a.warmup_ratio),steps)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    best=-1.;best_epoch=None;history=[]
    start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();loss_sum=0.
        for batch in train_loader:
            batch=base.move_batch(batch,device)
            y=batch.pop('binary_labels')
            rt,rm=batch.pop('risk_targets'),batch.pop('risk_mask')
            dt,dm=batch.pop('domain_targets'),batch.pop('domain_mask')
            with base.autocast_ctx(device,True):
                logits=model(**batch)
                lb=nn.functional.cross_entropy(logits['binary_logits'].float(),y)
                lr=base.masked_multilabel_loss(logits['risk_logits'],rt,rm)
                ld=base.masked_multilabel_loss(logits['domain_logits'],dt,dm)
                loss=lb+.3*lr+.3*ld
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);scheduler.step()
            loss_sum+=float(loss.item())
        vm,_,_,_=base.evaluate(model,val_loader,device,True,rv,dv)
        history.append({'epoch':epoch,'val_unsafe_f1':vm['unsafe_f1'],
                        'val_safe_fpr':vm['safe_false_positive_rate_over_refusal_proxy'],
                        'mean_train_loss':loss_sum/len(train_loader)})
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print('FOLD',a.axis,a.label,a.seed,'epoch',epoch,'val_F1',vm['unsafe_f1'],flush=True)
        if vm['unsafe_f1']>best:
            best=vm['unsafe_f1'];best_epoch=epoch
            ckpt=out/'checkpoint_best';ckpt.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),ckpt/'model_state.pt')
            tokenizer.save_pretrained(ckpt)
            (ckpt/'model_config.json').write_text(json.dumps(dict(epoch=epoch,dropout=.1,
                    risk_vocab=rv,domain_vocab=dv),indent=2,ensure_ascii=False),encoding='utf-8')
    # Reload selected checkpoint, not terminal epoch.
    model.load_state_dict(torch.load(out/'checkpoint_best/model_state.pt',map_location='cpu',weights_only=True))
    tm,y,pred,prob=base.evaluate(model,test_loader,device,True,rv,dv)
    result={'axis':a.axis,'label':a.label,'seed':a.seed,'best_epoch':best_epoch,
            'best_val_f1':best,'metrics':tm,'test_n':len(test),
            'heldout_positive_n':cfg['heldout_positive_test'],
            'elapsed_sec':time.time()-start,
            'interpretation':'new heldout-label model, not frozen Stage-2 checkpoint'}
    (out/'test_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    # Produce fold-specific probability caches and empirical routing policy.
    # Calibration uses ONLY the fold's training-excluding-heldout validation.
    from stage34_calibrate import calc
    from stage34_evaluate import route, summary
    def cache_for(part, ds, ld):
        model.eval();bs=[];rs=[];dscores=[]
        with torch.inference_mode():
            for batch in ld:
                batch=base.move_batch(batch,device)
                for key in ('binary_labels','risk_targets','risk_mask','domain_targets','domain_mask'):
                    batch.pop(key)
                with base.autocast_ctx(device,True):
                    outputs=model(**batch)
                bs.extend(torch.softmax(outputs['binary_logits'].float(),dim=-1)[:,1].cpu().tolist())
                rs.extend(torch.sigmoid(outputs['risk_logits'].float()).cpu().numpy())
                dscores.extend(torch.sigmoid(outputs['domain_logits'].float()).cpu().numpy())
        result=pd.DataFrame({'prompt_hash':part.prompt_hash.astype(str).to_numpy(),
                            'unsafe_label':ds.y_binary.astype(int), 'unsafe_score':bs,
                            'risk_mask':ds.risk_mask.astype(bool),
                            'domain_mask':ds.domain_mask.astype(bool)})
        risk=np.asarray(rs);domain=np.asarray(dscores)
        for j in range(len(rv)):
            result[f'risk_score_{j:02d}']=risk[:,j]
            result[f'risk_true_{j:02d}']=ds.risk_targets[:,j].astype(int)
        for j in range(len(dv)):
            result[f'domain_score_{j:02d}']=domain[:,j]
            result[f'domain_true_{j:02d}']=ds.domain_targets[:,j].astype(int)
        return result
    val_cache=cache_for(val,val_ds,val_loader)
    test_cache=cache_for(test,test_ds,test_loader)
    val_cache.to_csv(out/'fold_validation_cache.csv.gz',index=False,compression='gzip')
    test_cache.to_csv(out/'fold_test_cache.csv.gz',index=False,compression='gzip')
    thresholds,meta=calc(val_cache,alpha=.01,kappa=100.,review_width=.05)
    policy={'metadata':{**meta,'heldout_axis':a.axis,'heldout_label':a.label},
            'thresholds':thresholds.replace({np.nan:None}).to_dict(orient='records')}
    (out/'fold_policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2))
    action,_=route(test_cache,policy)
    (out/'fold_routing_metrics.json').write_text(json.dumps(
        {'axis':a.axis,'label':a.label,'seed':a.seed,**summary(test_cache,action)},indent=2))
    # The test fold is either heldout-domain rows or heldout-risk unsafe rows plus safe controls.
    # Save each selected test score for ECE/Brier/restricted group metrics, without raw text.
    pd.DataFrame({'prompt_hash':test.prompt_hash.astype(str).to_numpy(),
                  'unsafe_label':test_ds.y_binary.astype(int),
                  'unsafe_score':prob}).to_csv(out/'fold_predictions.csv.gz',index=False,compression='gzip')
    (out/'COMPLETE.json').write_text(json.dumps({'completed':True,'best_epoch':best_epoch},indent=2))
    print('FOLD COMPLETE',out,flush=True)

if __name__=='__main__':main()
