#!/usr/bin/env python3
"""CalibGuard multi-task training with TRAIN-internal validation and early stopping.

Maintains frozen V2 test_id; selects epoch solely from train-internal validation.
Depends on formal_train_deberta_multitask.py in same scripts/ directory.
"""
import argparse
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


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='data/splits/calibguard_split_v2.csv')
    p.add_argument('--model_id', default='models/deberta-v3-base')
    p.add_argument('--output_dir', default='results/multitask_earlystop_v1')
    p.add_argument('--epochs', type=int, default=8, help='Maximum epochs')
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--min_delta', type=float, default=0.0001)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.06)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--risk_loss_weight', type=float, default=0.3)
    p.add_argument('--domain_loss_weight', type=float, default=0.3)
    p.add_argument('--max_train_batches', type=int, default=0, help='Smoke-test only')
    p.add_argument('--max_val_batches', type=int, default=0, help='Smoke-test only')
    p.add_argument('--max_test_batches', type=int, default=0, help='Smoke-test only')
    return p.parse_args()


def main():
    a = get_args()
    if a.epochs < 1 or a.patience < 1 or a.batch_size < 1 or a.eval_batch_size < 1 or a.max_length < 1:
        raise ValueError('epochs, patience, batch sizes and max_length must be positive')
    if not (0 < a.val_ratio < 0.5):
        raise ValueError('val_ratio must be between 0 and 0.5')
    if not (0 <= a.warmup_ratio < 1) or a.min_delta < 0:
        raise ValueError('Invalid warmup_ratio or min_delta')
    if a.risk_loss_weight < 0 or a.domain_loss_weight < 0:
        raise ValueError('Auxiliary loss weights cannot be negative')
    base.seed_all(a.seed)
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty; will not overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(a.data_path, low_memory=False)
    required = {'split', 'text', 'is_safe', 'risk_categories', 'domains'}
    if not required.issubset(data.columns):
        raise ValueError(f'Missing data fields: {required - set(data.columns)}')
    tr = data.loc[data.split == 'train'].copy()
    te = data.loc[data.split == 'test_id'].copy()
    if len(tr) != base.EXPECTED_TRAIN or len(te) != base.EXPECTED_TEST:
        raise ValueError('Frozen V2 row counts do not match')
    if tr.text.isna().any() or te.text.isna().any():
        raise ValueError('Missing prompt text')
    tr['risk_categories_parsed'] = tr.risk_categories.map(base.parse_label_list)
    tr['domains_parsed'] = tr.domains.map(base.parse_label_list)
    te['risk_categories_parsed'] = te.risk_categories.map(base.parse_label_list)
    te['domains_parsed'] = te.domains.map(base.parse_label_list)
    safe = pd.to_numeric(tr.is_safe, errors='raise').astype(int)
    if not safe.isin([0, 1]).all():
        raise ValueError('Invalid training binary labels')
    tr_idx, val_idx = train_test_split(
        np.arange(len(tr)), test_size=a.val_ratio, random_state=a.seed, stratify=safe.to_numpy()
    )
    # Preserve deterministic mapping, and persist exact row indices to reproduce split.
    tr_fit = tr.iloc[tr_idx].copy()
    va = tr.iloc[val_idx].copy()
    pd.DataFrame({'source_row_index': tr.index[val_idx], 'prompt_hash': tr.iloc[val_idx]['prompt_hash']}).to_csv(
        out / 'validation_indices.csv', index=False
    )
    risk_vocab = base.make_vocab(tr_fit['risk_categories_parsed'], exclude_no_risk=True)
    domain_vocab = base.make_vocab(tr_fit['domains_parsed'])
    if len(risk_vocab) != 22 or len(domain_vocab) != 19:
        raise ValueError(f'Unexpected auxiliary vocab sizes: {len(risk_vocab)} / {len(domain_vocab)}')
    tr_ds = base.MultiTaskDataset(tr_fit, risk_vocab, domain_vocab)
    va_ds = base.MultiTaskDataset(va, risk_vocab, domain_vocab)
    te_ds = base.MultiTaskDataset(te, risk_vocab, domain_vocab)
    tokenizer = AutoTokenizer.from_pretrained(a.model_id, use_fast=False)
    model = base.DebertaMultiTaskModel(a.model_id, len(risk_vocab), len(domain_vocab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('This formal run requires CUDA')
    model.to(device)
    collate = base.make_collator(tokenizer, a.max_length)
    tr_loader = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                           num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(a.seed))
    va_loader = DataLoader(va_ds, batch_size=a.eval_batch_size, shuffle=False,
                           collate_fn=collate, num_workers=0, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=a.eval_batch_size, shuffle=False,
                           collate_fn=collate, num_workers=0, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    total_steps = len(tr_loader) * a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * a.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    smoke = bool(a.max_train_batches or a.max_val_batches or a.max_test_batches)
    config = vars(a).copy()
    config.update({'train_fit_rows': len(tr_fit), 'validation_rows': len(va), 'test_id_rows': len(te),
                   'risk_vocab': risk_vocab, 'domain_vocab': domain_vocab, 'is_smoke_test_run': smoke,
                   'selection_metric': 'validation_unsafe_f1', 'selection_threshold': 0.5,
                   'validation_split': 'stratified random 10% (or val_ratio) from frozen V2 train only',
                   'test_used_for_selection': False, 'max_epochs_scheduler': a.epochs})
    (out / 'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    best_dir = out / 'checkpoint_best'
    best_dir.mkdir(exist_ok=True)
    print('CALIBGUARD MULTITASK EARLY STOPPING', flush=True)
    print(f'Train fit={len(tr_fit)}; Validation={len(va)}; untouched Test ID={len(te)}', flush=True)
    print(f'GPU={torch.cuda.get_device_name(0)}; max_epochs={a.epochs}; patience={a.patience}; seed={a.seed}', flush=True)
    print(f'Risk labeled fit={int(tr_ds.risk_mask.sum())}; domain labeled fit={int(tr_ds.domain_mask.sum())}', flush=True)
    print(f'Smoke={smoke}; output={out}', flush=True)
    best_score = -1.0
    best_epoch = None
    stale = 0
    history = []
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        processed = 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tr_loader):
            if a.max_train_batches and i >= a.max_train_batches:
                break
            batch = base.move_batch(batch, device)
            labels = batch.pop('binary_labels')
            rt, rm = batch.pop('risk_targets'), batch.pop('risk_mask')
            dt, dm = batch.pop('domain_targets'), batch.pop('domain_mask')
            with base.autocast_ctx(device, True):
                pred = model(**batch)
                binary_loss = nn.functional.cross_entropy(pred['binary_logits'].float(), labels)
                risk_loss = base.masked_multilabel_loss(pred['risk_logits'], rt, rm)
                domain_loss = base.masked_multilabel_loss(pred['domain_logits'], dt, dm)
                loss = binary_loss + a.risk_loss_weight * risk_loss + a.domain_loss_weight * domain_loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {i+1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            totals += [float(x.item()) for x in (loss, binary_loss, risk_loss, domain_loss)]
            processed += 1
            if processed % 500 == 0:
                print(f'epoch={epoch} batch={processed}/{len(tr_loader)} mean_total_loss={totals[0]/processed:.5f}', flush=True)
        if processed == 0:
            raise RuntimeError('Zero training batches processed')
        val_metrics, _, _, _ = base.evaluate(model, va_loader, device, True, risk_vocab, domain_vocab, a.max_val_batches)
        score = val_metrics['unsafe_f1']
        improved = score > best_score + a.min_delta
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), best_dir / 'model_state.pt')
            tokenizer.save_pretrained(best_dir)
            (best_dir / 'model_config.json').write_text(json.dumps({
                'risk_vocab': risk_vocab, 'domain_vocab': domain_vocab, 'dropout': 0.1,
                'best_epoch': epoch, 'validation_unsafe_f1': best_score,
            }, indent=2, ensure_ascii=False), encoding='utf-8')
        else:
            stale += 1
        row = {'epoch': epoch, 'train_batches': processed,
               'mean_train_total_loss': totals[0] / processed,
               'mean_train_binary_loss': totals[1] / processed,
               'mean_train_risk_loss': totals[2] / processed,
               'mean_train_domain_loss': totals[3] / processed,
               'validation_unsafe_f1': score,
               'validation_unsafe_recall': val_metrics['unsafe_recall'],
               'validation_safe_false_positive_rate': val_metrics['safe_false_positive_rate_over_refusal_proxy'],
               'validation_evaluated_samples': val_metrics['evaluated_samples'],
               'best_epoch_so_far': best_epoch, 'best_unsafe_f1_so_far': best_score,
               'patience_counter': stale}
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'epoch_validation_log.csv', index=False)
        print(f"Epoch {epoch} done: train_loss={row['mean_train_total_loss']:.6f}; "
              f"val_unsafe_f1={score:.6f}; val_unsafe_recall={row['validation_unsafe_recall']:.6f}; "
              f"val_safe_fpr={row['validation_safe_false_positive_rate']:.6f}; "
              f"best_epoch={best_epoch}; stale={stale}/{a.patience}", flush=True)
        if stale >= a.patience:
            print(f'Early stopping after epoch {epoch}', flush=True)
            break
    # Load exactly the validation-selected weights before touching frozen Test.
    model.load_state_dict(torch.load(best_dir / 'model_state.pt', map_location=device, weights_only=True))
    print(f'Final untouched ID Test evaluation with validation-selected epoch {best_epoch}', flush=True)
    test_metrics, y, pred, prob = base.evaluate(model, te_loader, device, True, risk_vocab, domain_vocab, a.max_test_batches)
    test_metrics.update({'best_epoch': best_epoch, 'best_validation_unsafe_f1': best_score,
                         'selection_metric': 'validation_unsafe_f1', 'is_smoke_test_run': smoke,
                         'test_protocol': 'limited smoke test' if a.max_test_batches else 'full ID test'})
    (out / 'id_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    predictions = te.iloc[:len(y)].copy()
    predictions['label_is_unsafe'] = y
    predictions['pred_is_unsafe'] = pred
    predictions['prob_unsafe'] = prob
    predictions['correct'] = (y == pred)
    predictions.to_csv(out / 'id_test_predictions.csv', index=False)
    (out / 'run_summary.json').write_text(json.dumps({
        'best_epoch': best_epoch, 'epochs_completed': len(history), 'best_validation_unsafe_f1': best_score,
        'total_runtime_seconds': time.time()-start, 'is_smoke_test_run': smoke,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False), flush=True)
    print('Run finished.', flush=True)


if __name__ == '__main__':
    main()
