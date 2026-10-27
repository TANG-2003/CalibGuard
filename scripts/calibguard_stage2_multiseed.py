#!/usr/bin/env python3
"""CalibGuard Stage-2: common 9-epoch trajectory, snapshots and task ablations.

Frozen V2 test_id is NEVER evaluated in this exploration script.
Uses the same stratified 10% train-internal validation split as earlystop V1
when --seed=42 and --val_ratio=0.10. Run variants in separate output dirs.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formal_train_deberta_multitask as base


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path', default='data/splits/calibguard_split_v2.csv')
    p.add_argument('--model_id', default='models/deberta-v3-base')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--variant', choices=['binary','risk','domain','full'], default='full')
    p.add_argument('--epochs', type=int, default=9)
    p.add_argument('--save_epochs', default='3,6,9')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.06)
    p.add_argument('--risk_loss_weight', type=float, default=0.3)
    p.add_argument('--domain_loss_weight', type=float, default=0.3)
    p.add_argument('--val_ratio', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--max_train_batches', type=int, default=0, help='Smoke only; 0=all')
    p.add_argument('--max_val_batches', type=int, default=0, help='Smoke only; 0=all')
    return p.parse_args()


def save_checkpoint(model, tokenizer, dest, epoch, vocab_risk, vocab_domain, val_metrics):
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest / 'model_state.pt')
    tokenizer.save_pretrained(dest)
    (dest / 'model_config.json').write_text(json.dumps({
        'epoch': epoch, 'risk_vocab': vocab_risk, 'domain_vocab': vocab_domain,
        'validation_unsafe_f1': val_metrics['unsafe_f1'], 'dropout': 0.1,
    }, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    a = parse_args()
    if a.epochs <= 0 or a.batch_size <= 0 or a.eval_batch_size <= 0 or a.max_length <= 0:
        raise ValueError('Epochs, batch sizes and max length must be positive')
    if not 0 < a.val_ratio < 0.5 or not 0 <= a.warmup_ratio < 1:
        raise ValueError('Invalid validation or warmup ratio')
    if min(a.risk_loss_weight, a.domain_loss_weight) < 0:
        raise ValueError('Loss weights must be nonnegative')
    save_epochs = sorted(set(int(v.strip()) for v in a.save_epochs.split(',') if v.strip()))
    if not save_epochs or min(save_epochs) < 1 or max(save_epochs) > a.epochs:
        raise ValueError('--save_epochs must be within 1..epochs')
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output already exists, refusing overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    base.seed_all(a.seed)
    df = pd.read_csv(a.data_path, low_memory=False)
    required = {'split','text','is_safe','risk_categories','domains','prompt_hash'}
    if missing := required - set(df.columns):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    train = df.loc[df['split'] == 'train'].copy()
    test_rows = int((df['split'] == 'test_id').sum())
    if len(train) != base.EXPECTED_TRAIN or test_rows != base.EXPECTED_TEST:
        raise ValueError('Frozen V2 row count mismatch')
    if train['text'].isna().any() or train['prompt_hash'].isna().any():
        raise ValueError('Missing text or hash')
    safe = pd.to_numeric(train.is_safe, errors='raise').astype(int).to_numpy()
    if not np.isin(safe, [0,1]).all():
        raise ValueError('Invalid safety labels')
    tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=a.val_ratio,
                                      random_state=a.split_seed, stratify=safe)
    fit = train.iloc[tr_idx].copy()
    val = train.iloc[va_idx].copy()
    for part in (fit, val):
        part['risk_categories_parsed'] = part.risk_categories.map(base.parse_label_list)
        part['domains_parsed'] = part.domains.map(base.parse_label_list)
    # The same vocabulary derived from fit only across all variants.
    risk_vocab = base.make_vocab(fit['risk_categories_parsed'], exclude_no_risk=True)
    domain_vocab = base.make_vocab(fit['domains_parsed'])
    if (len(risk_vocab), len(domain_vocab)) != (22,19):
        raise ValueError(f'Unexpected vocab sizes: {len(risk_vocab)}, {len(domain_vocab)}')
    fit_ds = base.MultiTaskDataset(fit, risk_vocab, domain_vocab)
    val_ds = base.MultiTaskDataset(val, risk_vocab, domain_vocab)
    tokenizer = AutoTokenizer.from_pretrained(a.model_id, use_fast=False)
    model = base.DebertaMultiTaskModel(a.model_id, len(risk_vocab), len(domain_vocab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Formal training requires CUDA')
    model.to(device)
    collate = base.make_collator(tokenizer, a.max_length)
    fit_loader = DataLoader(fit_ds, batch_size=a.batch_size, shuffle=True,
                            collate_fn=collate, num_workers=0, pin_memory=True,
                            generator=torch.Generator().manual_seed(a.seed))
    val_loader = DataLoader(val_ds, batch_size=a.eval_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    steps = len(fit_loader) * a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*a.warmup_ratio), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    risk_w = a.risk_loss_weight if a.variant in ('risk','full') else 0.0
    domain_w = a.domain_loss_weight if a.variant in ('domain','full') else 0.0
    smoke = bool(a.max_train_batches or a.max_val_batches)
    indices = pd.DataFrame({'source_row_index': val.index.to_numpy(),
                            'prompt_hash': val['prompt_hash'].to_numpy()})
    indices.to_csv(out/'validation_indices.csv', index=False)
    split_digest = hashlib.sha256('\n'.join(indices.prompt_hash.astype(str)).encode()).hexdigest()
    conf = vars(a).copy()
    conf.update({'train_fit_rows':len(fit),'validation_rows':len(val),
                 'test_id_rows_untouched':test_rows,'risk_vocab':risk_vocab,
                 'domain_vocab':domain_vocab,'effective_risk_weight':risk_w,
                 'effective_domain_weight':domain_w,'is_smoke_test_run':smoke,
                 'validation_hash_sha256':split_digest,'selection_metric':'validation_unsafe_f1',
                 'test_used_for_selection':False,'test_evaluated':False,
                 'scheduler_total_epochs':a.epochs})
    (out/'run_config.json').write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'CALIBGUARD STAGE2 | variant={a.variant} | epochs={a.epochs} | snapshots={save_epochs}',flush=True)
    print(f'fit={len(fit)}, validation={len(val)}, untouched_test={test_rows}; GPU={torch.cuda.get_device_name(0)}',flush=True)
    print(f'weights: risk={risk_w}, domain={domain_w}; smoke={smoke}; validation_hash={split_digest}',flush=True)
    history=[]
    best_score=-1.0
    best_epoch=None
    start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train()
        totals=np.zeros(4,dtype=np.float64)
        seen=0
        for bi,batch in enumerate(fit_loader):
            if a.max_train_batches and bi >= a.max_train_batches:
                break
            batch=base.move_batch(batch,device)
            y=batch.pop('binary_labels')
            rt, rm=batch.pop('risk_targets'),batch.pop('risk_mask')
            dt, dm=batch.pop('domain_targets'),batch.pop('domain_mask')
            with base.autocast_ctx(device,True):
                pred=model(**batch)
                loss_bin=nn.functional.cross_entropy(pred['binary_logits'].float(),y)
                loss_risk=base.masked_multilabel_loss(pred['risk_logits'],rt,rm) if risk_w else loss_bin.new_zeros(())
                loss_domain=base.masked_multilabel_loss(pred['domain_logits'],dt,dm) if domain_w else loss_bin.new_zeros(())
                loss=loss_bin+risk_w*loss_risk+domain_w*loss_domain
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite loss at epoch={epoch} batch={bi+1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            totals += [float(x.item()) for x in (loss,loss_bin,loss_risk,loss_domain)]
            seen+=1
            if seen%500==0:
                print(f'epoch={epoch} batch={seen}/{len(fit_loader)} train_loss={totals[0]/seen:.6f}',flush=True)
        if not seen:
            raise RuntimeError('No training batches')
        vm,_,_,_=base.evaluate(model,val_loader,device,True,risk_vocab,domain_vocab,a.max_val_batches)
        if vm['evaluated_samples'] != len(val) and not a.max_val_batches:
            raise RuntimeError('Validation incomplete')
        row={'epoch':epoch,'train_batches':seen,'mean_train_total_loss':totals[0]/seen,
             'mean_train_binary_loss':totals[1]/seen,
             'mean_train_risk_loss':totals[2]/seen,
             'mean_train_domain_loss':totals[3]/seen,
             'validation_unsafe_f1':vm['unsafe_f1'],
             'validation_unsafe_recall':vm['unsafe_recall'],
             'validation_safe_fpr':vm['safe_false_positive_rate_over_refusal_proxy'],
             'validation_accuracy':vm['accuracy'],
             'validation_evaluated_samples':vm['evaluated_samples']}
        history.append(row)
        pd.DataFrame(history).to_csv(out/'epoch_validation_log.csv',index=False)
        if vm['unsafe_f1'] > best_score:
            best_score, best_epoch=vm['unsafe_f1'],epoch
            save_checkpoint(model,tokenizer,out/'checkpoint_best',epoch,risk_vocab,domain_vocab,vm)
        if epoch in save_epochs:
            save_checkpoint(model,tokenizer,out/f'checkpoint_epoch_{epoch}',epoch,risk_vocab,domain_vocab,vm)
        print(f"Epoch {epoch}: val_f1={vm['unsafe_f1']:.6f} val_recall={vm['unsafe_recall']:.6f} "
              f"val_safe_fpr={vm['safe_false_positive_rate_over_refusal_proxy']:.6f} best_epoch={best_epoch}",flush=True)
    summary={'variant':a.variant,'best_epoch':best_epoch,'best_validation_unsafe_f1':best_score,
             'epochs_completed':len(history),'snapshots':save_epochs,
             'total_runtime_seconds':time.time()-start,'is_smoke_test_run':smoke,
             'id_test_was_evaluated':False,'validation_hash_sha256':split_digest}
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('Run finished. ID TEST NOT EVALUATED.',flush=True)


if __name__ == '__main__':
    main()
