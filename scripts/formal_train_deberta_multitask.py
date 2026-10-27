#!/usr/bin/env python3
"""CalibGuard Stage 2: DeBERTa multi-task safety classification.

Uses the frozen V2 train/test_id split. No validation split is created.
Primary task: SAFE/UNSAFE binary classification.
Auxiliary tasks: multi-label risk categories (excluding 'No Risk') and domains.
Auxiliary losses are computed only for rows with non-empty valid annotations.
"""
import argparse
import ast
import json
import math
import random
import time
from contextlib import nullcontext
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

DEFAULT_MODEL = "models/deberta-v3-base"
DEFAULT_DATA = "data/splits/calibguard_split_v2.csv"
DEFAULT_OUT = "results/multitask_deberta_v1"
EXPECTED_TRAIN, EXPECTED_TEST = 73365, 31444


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default=DEFAULT_DATA)
    p.add_argument('--model_id', default=DEFAULT_MODEL)
    p.add_argument('--output_dir', default=DEFAULT_OUT)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.06)
    p.add_argument('--grad_accum_steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--risk_loss_weight', type=float, default=0.3)
    p.add_argument('--domain_loss_weight', type=float, default=0.3)
    p.add_argument('--max_train_batches', type=int, default=0, help='Smoke test only; 0=all')
    p.add_argument('--max_test_batches', type=int, default=0, help='Smoke test only; 0=all')
    p.add_argument('--allow_nonstandard_row_counts', action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'not-installed'


def parse_label_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        result = value
    else:
        s = str(value).strip()
        if not s or s.lower() in {'nan', 'none', 'null'}:
            return []
        try:
            result = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            try:
                result = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                raise ValueError(f'Cannot parse label list: {s[:160]!r}')
    if not isinstance(result, list):
        raise ValueError(f'Expected a list label, got {type(result).__name__}')
    # Normalize only known spelling variation observed in the audit.
    normalized = []
    for x in result:
        label = str(x).strip()
        if not label:
            continue
        if label.casefold() == 'dataprivacy':
            label = 'Data Privacy'
        normalized.append(label)
    return normalized


def make_vocab(series, exclude_no_risk=False):
    values = set()
    for labels in series:
        for label in labels:
            if exclude_no_risk and label.casefold().replace('_', ' ') == 'no risk':
                continue
            values.add(label)
    return sorted(values, key=lambda x: x.casefold())


class MultiTaskDataset(Dataset):
    def __init__(self, frame, risk_vocab, domain_vocab):
        self.frame = frame.reset_index(drop=True)
        self.texts = self.frame['text'].astype(str).tolist()
        safe = pd.to_numeric(self.frame['is_safe'], errors='raise').astype(int).to_numpy()
        if not np.isin(safe, [0, 1]).all():
            raise ValueError('is_safe must contain only 0/1')
        self.y_binary = (1 - safe).astype(np.int64)  # 0 SAFE, 1 UNSAFE
        self.risk_vocab = risk_vocab
        self.domain_vocab = domain_vocab
        self.risk_index = {x: i for i, x in enumerate(risk_vocab)}
        self.domain_index = {x: i for i, x in enumerate(domain_vocab)}
        self.risk_targets = []
        self.risk_mask = []
        self.domain_targets = []
        self.domain_mask = []
        unknown_risk = 0
        unknown_domain = 0
        for y, raw_risk, raw_domain in zip(
            self.y_binary, self.frame['risk_categories_parsed'], self.frame['domains_parsed']
        ):
            risk_labels = [x for x in raw_risk if x.casefold().replace('_', ' ') != 'no risk']
            risk_vec = np.zeros(len(risk_vocab), dtype=np.float32)
            recognized_risk = 0
            for label in risk_labels:
                if label in self.risk_index:
                    risk_vec[self.risk_index[label]] = 1.0
                    recognized_risk += 1
                else:
                    unknown_risk += 1
            # The risk task is defined only for explicitly labeled UNSAFE rows.
            rmask = bool(y == 1 and recognized_risk > 0)
            self.risk_targets.append(risk_vec)
            self.risk_mask.append(rmask)

            domain_vec = np.zeros(len(domain_vocab), dtype=np.float32)
            recognized_domain = 0
            for label in raw_domain:
                if label in self.domain_index:
                    domain_vec[self.domain_index[label]] = 1.0
                    recognized_domain += 1
                else:
                    unknown_domain += 1
            self.domain_targets.append(domain_vec)
            self.domain_mask.append(bool(recognized_domain > 0))
        self.risk_targets = np.stack(self.risk_targets)
        self.risk_mask = np.asarray(self.risk_mask, dtype=bool)
        self.domain_targets = np.stack(self.domain_targets)
        self.domain_mask = np.asarray(self.domain_mask, dtype=bool)
        self.unknown_risk = unknown_risk
        self.unknown_domain = unknown_domain

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {
            'text': self.texts[i],
            'binary_label': int(self.y_binary[i]),
            'risk_target': self.risk_targets[i],
            'risk_mask': bool(self.risk_mask[i]),
            'domain_target': self.domain_targets[i],
            'domain_mask': bool(self.domain_mask[i]),
        }


def make_collator(tokenizer, max_length):
    def collate(items):
        encoded = tokenizer(
            [x['text'] for x in items], padding=True, truncation=True,
            max_length=max_length, return_tensors='pt'
        )
        encoded['binary_labels'] = torch.tensor([x['binary_label'] for x in items], dtype=torch.long)
        encoded['risk_targets'] = torch.tensor(np.stack([x['risk_target'] for x in items]), dtype=torch.float32)
        encoded['risk_mask'] = torch.tensor([x['risk_mask'] for x in items], dtype=torch.bool)
        encoded['domain_targets'] = torch.tensor(np.stack([x['domain_target'] for x in items]), dtype=torch.float32)
        encoded['domain_mask'] = torch.tensor([x['domain_mask'] for x in items], dtype=torch.bool)
        return encoded
    return collate


def autocast_ctx(device, enabled):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=enabled)
    return nullcontext()


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class DebertaMultiTaskModel(nn.Module):
    def __init__(self, model_id, n_risk, n_domain, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden = self.encoder.config.hidden_size
        self.pooler = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden, 2)
        self.risk_head = nn.Linear(hidden, n_risk)
        self.domain_head = nn.Linear(hidden, n_domain)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        outputs = self.encoder(**kwargs)
        pooled = torch.tanh(self.pooler(outputs.last_hidden_state[:, 0, :]))
        pooled = self.dropout(pooled)
        return {
            'binary_logits': self.binary_head(pooled),
            'risk_logits': self.risk_head(pooled),
            'domain_logits': self.domain_head(pooled),
        }


def masked_multilabel_loss(logits, targets, mask):
    if not mask.any():
        return logits.sum() * 0.0
    per_element = nn.functional.binary_cross_entropy_with_logits(
        logits[mask].float(), targets[mask].float(), reduction='none'
    )
    # Mean over labels per annotated example, then over annotated examples.
    return per_element.mean(dim=1).mean()


def evaluate(model, loader, device, use_fp16, risk_vocab, domain_vocab, max_batches=0):
    model.eval()
    ys, preds, probs, hashes = [], [], [], []
    risk_true, risk_prob = [], []
    domain_true, domain_prob = [], []
    binary_loss_sum = 0.0
    n_batches = 0
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            batch = move_batch(batch, device)
            binary_labels = batch.pop('binary_labels')
            risk_targets = batch.pop('risk_targets')
            risk_mask = batch.pop('risk_mask')
            domain_targets = batch.pop('domain_targets')
            domain_mask = batch.pop('domain_mask')
            with autocast_ctx(device, use_fp16):
                out = model(**batch)
                loss = nn.functional.cross_entropy(out['binary_logits'].float(), binary_labels)
            logits = out['binary_logits'].float()
            ys.extend(binary_labels.cpu().tolist())
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            binary_loss_sum += float(loss.item())
            n_batches += 1
            if risk_mask.any():
                risk_true.extend(risk_targets[risk_mask].cpu().numpy().tolist())
                risk_prob.extend(torch.sigmoid(out['risk_logits'][risk_mask].float()).cpu().numpy().tolist())
            if domain_mask.any():
                domain_true.extend(domain_targets[domain_mask].cpu().numpy().tolist())
                domain_prob.extend(torch.sigmoid(out['domain_logits'][domain_mask].float()).cpu().numpy().tolist())
    if not ys:
        raise RuntimeError('Evaluation processed zero batches')
    y = np.asarray(ys, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    prob = np.asarray(probs, dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        'evaluated_samples': int(len(y)), 'safe_samples': int((y == 0).sum()),
        'unsafe_samples': int((y == 1).sum()), 'threshold': 0.5,
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'unsafe_precision': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'unsafe_recall': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'unsafe_f1': float(f1_score(y, pred, pos_label=1, zero_division=0)),
        'specificity_safe_recall': float(tn / (tn + fp)) if tn + fp else None,
        'safe_false_positive_rate_over_refusal_proxy': float(fp / (tn + fp)) if tn + fp else None,
        'roc_auc': float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        'average_precision_unsafe': float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else None,
        'confusion_matrix': {'tn_safe_correct': int(tn), 'fp_safe_predicted_unsafe': int(fp),
                             'fn_unsafe_predicted_safe': int(fn), 'tp_unsafe_correct': int(tp)},
        'mean_batch_binary_loss': float(binary_loss_sum / n_batches),
    }
    if risk_true:
        rt, rp = np.asarray(risk_true), np.asarray(risk_prob)
        metrics['risk_auxiliary'] = {
            'labeled_samples': int(len(rt)),
            'micro_f1_at_0.5': float(f1_score(rt, rp >= 0.5, average='micro', zero_division=0)),
            'macro_f1_at_0.5': float(f1_score(rt, rp >= 0.5, average='macro', zero_division=0)),
            'micro_average_precision': float(average_precision_score(rt, rp, average='micro')),
            'label_count': len(risk_vocab),
        }
    if domain_true:
        dt, dp = np.asarray(domain_true), np.asarray(domain_prob)
        metrics['domain_auxiliary'] = {
            'labeled_samples': int(len(dt)),
            'micro_f1_at_0.5': float(f1_score(dt, dp >= 0.5, average='micro', zero_division=0)),
            'macro_f1_at_0.5': float(f1_score(dt, dp >= 0.5, average='macro', zero_division=0)),
            'micro_average_precision': float(average_precision_score(dt, dp, average='micro')),
            'label_count': len(domain_vocab),
        }
    return metrics, y, pred, prob


def main():
    a = parse_args()
    if min(a.epochs, a.batch_size, a.eval_batch_size, a.grad_accum_steps, a.max_length) < 1:
        raise ValueError('epochs, batch sizes, grad accumulation, and max_length must be positive')
    if not 0 <= a.warmup_ratio < 1:
        raise ValueError('warmup_ratio must be in [0,1)')
    if a.risk_loss_weight < 0 or a.domain_loss_weight < 0:
        raise ValueError('Auxiliary loss weights must be non-negative')
    seed_all(a.seed)
    data_path, out = Path(a.data_path), Path(a.output_dir)
    ckroot = out / 'checkpoints'
    out.mkdir(parents=True, exist_ok=True)
    ckroot.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    df = pd.read_csv(data_path, low_memory=False)
    required = {'split', 'text', 'is_safe', 'risk_categories', 'domains'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df['text'].isna().any():
        raise ValueError('Missing text values found')
    df['risk_categories_parsed'] = df['risk_categories'].map(parse_label_list)
    df['domains_parsed'] = df['domains'].map(parse_label_list)
    tr = df.loc[df['split'] == 'train'].copy()
    te = df.loc[df['split'] == 'test_id'].copy()
    if not a.allow_nonstandard_row_counts:
        if len(tr) != EXPECTED_TRAIN or len(te) != EXPECTED_TEST:
            raise ValueError(f'Unexpected V2 row counts: train={len(tr)}, test_id={len(te)}')
    risk_vocab = make_vocab(tr['risk_categories_parsed'], exclude_no_risk=True)
    domain_vocab = make_vocab(tr['domains_parsed'])
    if not risk_vocab or not domain_vocab:
        raise ValueError('Risk or domain vocabulary is empty')
    train_ds = MultiTaskDataset(tr, risk_vocab, domain_vocab)
    test_ds = MultiTaskDataset(te, risk_vocab, domain_vocab)
    print('=' * 72)
    print('CALIBGUARD - DeBERTa MULTI-TASK SAFETY CLASSIFICATION')
    print(f'Data={data_path.resolve()}')
    print(f'Output={out.resolve()}')
    print(f'Model={a.model_id}; seed={a.seed}; epochs={a.epochs}; max_length={a.max_length}')
    print('Primary labels: is_safe=1 -> SAFE(0), is_safe=0 -> UNSAFE(1)')
    print(f'Train={len(tr):,}; ID Test={len(te):,}; risk_labels={len(risk_vocab)}; domain_labels={len(domain_vocab)}')
    print(f'Train annotated risk rows={int(train_ds.risk_mask.sum()):,}; domain rows={int(train_ds.domain_mask.sum()):,}')
    print(f'Test annotated risk rows={int(test_ds.risk_mask.sum()):,}; domain rows={int(test_ds.domain_mask.sum()):,}')
    print(f'Unknown train labels ignored: risk={train_ds.unknown_risk}, domains={train_ds.unknown_domain}')

    tokenizer = AutoTokenizer.from_pretrained(a.model_id, use_fast=False)
    model = DebertaMultiTaskModel(a.model_id, len(risk_vocab), len(domain_vocab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA unavailable; run formal training on GPU server')
    model.to(device)
    use_fp16 = True
    print(f'Device={device}; GPU={torch.cuda.get_device_name(0)}; FP16={use_fp16}')

    collate = make_collator(tokenizer, a.max_length)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(a.seed))
    test_loader = DataLoader(test_ds, batch_size=a.eval_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=collate)
    optimizer = AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / a.grad_accum_steps)
    total_steps = steps_per_epoch * a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * a.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    smoke = bool(a.max_train_batches or a.max_test_batches)
    config = {
        'model_id': a.model_id, 'data_path': str(data_path), 'seed': a.seed,
        'epochs': a.epochs, 'batch_size': a.batch_size, 'eval_batch_size': a.eval_batch_size,
        'max_length': a.max_length, 'learning_rate': a.learning_rate, 'weight_decay': a.weight_decay,
        'warmup_ratio': a.warmup_ratio, 'grad_accum_steps': a.grad_accum_steps,
        'mixed_precision': 'fp16', 'primary_label_definition': 'label_is_unsafe = 1 - is_safe',
        'risk_vocab': risk_vocab, 'domain_vocab': domain_vocab,
        'risk_loss_weight': a.risk_loss_weight, 'domain_loss_weight': a.domain_loss_weight,
        'train_rows': len(tr), 'test_id_rows': len(te), 'uses_validation_split': False,
        'is_smoke_test_run': smoke, 'torch_version': torch.__version__,
        'transformers_version': pkg_version('transformers'), 'pandas_version': pkg_version('pandas'),
        'sklearn_version': pkg_version('scikit-learn'), 'gpu': torch.cuda.get_device_name(0),
        'max_train_batches': a.max_train_batches, 'max_test_batches': a.max_test_batches,
    }
    (out / 'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    epoch_logs = []
    started = time.time()
    binary_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, a.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum = binary_sum = risk_sum = domain_sum = 0.0
        processed = 0
        optimizer.zero_grad(set_to_none=True)
        print(f'\n===== Epoch {epoch}/{a.epochs} =====', flush=True)
        for bi, batch in enumerate(train_loader):
            if a.max_train_batches and bi >= a.max_train_batches:
                break
            batch = move_batch(batch, device)
            binary_labels = batch.pop('binary_labels')
            risk_targets, risk_mask = batch.pop('risk_targets'), batch.pop('risk_mask')
            domain_targets, domain_mask = batch.pop('domain_targets'), batch.pop('domain_mask')
            with autocast_ctx(device, use_fp16):
                output = model(**batch)
                loss_binary = binary_criterion(output['binary_logits'].float(), binary_labels)
                loss_risk = masked_multilabel_loss(output['risk_logits'], risk_targets, risk_mask)
                loss_domain = masked_multilabel_loss(output['domain_logits'], domain_targets, domain_mask)
                loss = loss_binary + a.risk_loss_weight * loss_risk + a.domain_loss_weight * loss_domain
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {bi + 1}')
            scaler.scale(loss / a.grad_accum_steps).backward()
            do_step = ((bi + 1) % a.grad_accum_steps == 0 or
                       (a.max_train_batches and bi + 1 == a.max_train_batches) or
                       (not a.max_train_batches and bi + 1 == len(train_loader)))
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += float(loss.item())
            binary_sum += float(loss_binary.item())
            risk_sum += float(loss_risk.item())
            domain_sum += float(loss_domain.item())
            processed += 1
            if processed % 100 == 0:
                print(f'epoch={epoch} batch={processed}/{len(train_loader)} '
                      f'mean_loss={loss_sum/processed:.5f} binary={binary_sum/processed:.5f} '
                      f'risk={risk_sum/processed:.5f} domain={domain_sum/processed:.5f}', flush=True)
        if not processed:
            raise RuntimeError('No training batches processed')
        record = {
            'epoch': epoch, 'train_batches': processed, 'mean_train_loss': loss_sum / processed,
            'mean_binary_loss': binary_sum / processed, 'mean_risk_loss': risk_sum / processed,
            'mean_domain_loss': domain_sum / processed, 'epoch_seconds': time.time() - epoch_start,
        }
        epoch_logs.append(record)
        pd.DataFrame(epoch_logs).to_csv(out / 'train_log.csv', index=False)
        checkpoint = ckroot / f'epoch_{epoch}'
        checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint / 'model_state.pt')
        tokenizer.save_pretrained(checkpoint)
        (checkpoint / 'model_config.json').write_text(json.dumps({
            'hidden_size': model.encoder.config.hidden_size,
            'risk_vocab': risk_vocab, 'domain_vocab': domain_vocab,
            'dropout': 0.1, 'seed': a.seed, 'epoch': epoch,
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        (checkpoint / 'training_state.json').write_text(json.dumps({'epoch': epoch, **record, 'config': config}, indent=2), encoding='utf-8')
        print(f"Epoch {epoch} complete: loss={record['mean_train_loss']:.6f}; "
              f"binary={record['mean_binary_loss']:.6f}; risk={record['mean_risk_loss']:.6f}; "
              f"domain={record['mean_domain_loss']:.6f}; seconds={record['epoch_seconds']:.1f}; "
              f"checkpoint={checkpoint}", flush=True)
        torch.cuda.empty_cache()

    print('\n===== Final ID Test evaluation =====', flush=True)
    metrics, y, pred, prob = evaluate(model, test_loader, device, use_fp16, risk_vocab, domain_vocab, a.max_test_batches)
    metrics.update({
        'is_smoke_test_run': smoke, 'model_id': a.model_id, 'checkpoint': str(ckroot / f'epoch_{a.epochs}'),
        'seed': a.seed, 'max_length': a.max_length,
        'test_protocol': 'limited batches for smoke test' if a.max_test_batches else 'full ID test',
    })
    (out / 'id_test_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    eval_frame = te.iloc[:len(y)].copy().reset_index(drop=True)
    eval_frame['label_is_unsafe'] = y
    eval_frame['pred_is_unsafe'] = pred
    eval_frame['prob_unsafe'] = prob
    eval_frame['correct'] = (y == pred)
    eval_frame.to_csv(out / 'id_test_predictions.csv', index=False)
    summary = {
        'total_runtime_seconds': time.time() - started, 'epoch_logs': epoch_logs,
        'metrics_file': str(out / 'id_test_metrics.json'),
        'predictions_file': str(out / 'id_test_predictions.csv'),
        'checkpoint_dir': str(ckroot / f'epoch_{a.epochs}'), 'is_smoke_test_run': smoke,
    }
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"\nTrain log: {out/'train_log.csv'}\nMetrics: {out/'id_test_metrics.json'}\n"
          f"Predictions: {out/'id_test_predictions.csv'}\nCheckpoint: {ckroot/f'epoch_{a.epochs}'}", flush=True)
    print('Run finished.', flush=True)


if __name__ == '__main__':
    main()
