#!/usr/bin/env python3
"""Frozen Stage-2 checkpoint -> probability-only Stage-3/4 cache.
Run from the CalibGuard project root. Never trains or changes checkpoints.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formal_train_deberta_multitask as base

SPLITS = ('validation', 'test_id', 'test_orbench_hard', 'test_xstest', 'test_harmful')
EXPECTED = {'validation': 7337, 'test_id': 31444, 'test_orbench_hard': 1319,
            'test_xstest': 448, 'test_harmful': 630}

def sha256_hashes(hashes):
    return hashlib.sha256('\n'.join(hashes.astype(str)).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    p.add_argument('--split', choices=SPLITS, required=True)
    p.add_argument('--out-root', default='results/stage34_cache')
    args = p.parse_args()

    run = Path(f'results/stage2_full_seed{args.seed}_6ep_v1')
    ckpt = run / 'checkpoint_best'
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    mc = json.loads((ckpt / 'model_config.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'run_summary.json').read_text(encoding='utf-8'))
    assert config['variant'] == 'full' and config['seed'] == args.seed
    assert config['split_seed'] == 42 and summary['epochs_completed'] == 6
    assert int(mc['epoch']) == int(summary['best_epoch'])
    assert mc['risk_vocab'] == config['risk_vocab'] and mc['domain_vocab'] == config['domain_vocab']
    assert len(mc['risk_vocab']) == 22 and len(mc['domain_vocab']) == 19

    df = pd.read_csv(config['data_path'], low_memory=False)
    if args.split == 'validation':
        train = df.loc[df['split'] == 'train']
        labels = pd.to_numeric(train['is_safe'], errors='raise').astype(int).to_numpy()
        _, val_idx = train_test_split(np.arange(len(train)), test_size=config['val_ratio'],
                                      random_state=config['split_seed'], stratify=labels)
        part = train.iloc[val_idx].copy()
        assert sha256_hashes(part['prompt_hash']) == config['validation_hash_sha256']
    else:
        part = df.loc[df['split'] == args.split].copy()
    assert len(part) == EXPECTED[args.split], (args.split, len(part))
    assert part['prompt_hash'].is_unique
    part['risk_categories_parsed'] = part['risk_categories'].map(base.parse_label_list)
    part['domains_parsed'] = part['domains'].map(base.parse_label_list)

    dataset = base.MultiTaskDataset(part, mc['risk_vocab'], mc['domain_vocab'])
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=False)
    loader = DataLoader(dataset, batch_size=config['eval_batch_size'], shuffle=False,
                        collate_fn=base.make_collator(tokenizer, config['max_length']), num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base.DebertaMultiTaskModel(config['model_id'], len(mc['risk_vocab']),
                                       len(mc['domain_vocab']), dropout=mc['dropout'])
    state = torch.load(ckpt / 'model_state.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    binary_probs, risk_probs, domain_probs = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = base.move_batch(batch, device)
            for k in ('binary_labels', 'risk_targets', 'risk_mask', 'domain_targets', 'domain_mask'):
                batch.pop(k)
            with base.autocast_ctx(device, device.type == 'cuda'):
                output = model(**batch)
            binary_probs.append(torch.softmax(output['binary_logits'].float(), dim=-1)[:, 1].cpu().numpy())
            risk_probs.append(torch.sigmoid(output['risk_logits'].float()).cpu().numpy())
            domain_probs.append(torch.sigmoid(output['domain_logits'].float()).cpu().numpy())
    bp = np.concatenate(binary_probs)
    rp = np.concatenate(risk_probs)
    dp = np.concatenate(domain_probs)
    assert len(bp) == len(part) and np.isfinite(bp).all()
    assert rp.shape == (len(part),22) and dp.shape == (len(part),19)
    assert np.isfinite(rp).all() and np.isfinite(dp).all()
    out = pd.DataFrame({'prompt_hash': part['prompt_hash'].astype(str).to_numpy(),
                        'unsafe_label': dataset.y_binary.astype(int), 'unsafe_score': bp,
                        'risk_mask': dataset.risk_mask.astype(bool),
                        'domain_mask': dataset.domain_mask.astype(bool)})
    for i in range(22):
        out[f'risk_score_{i:02d}'] = rp[:, i]
        out[f'risk_true_{i:02d}'] = dataset.risk_targets[:, i].astype(int)
    for i in range(19):
        out[f'domain_score_{i:02d}'] = dp[:, i]
        out[f'domain_true_{i:02d}'] = dataset.domain_targets[:, i].astype(int)
    out_dir = Path(args.out_root) / f'seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{args.split}.csv.gz'
    if path.exists():
        raise FileExistsError(f'Cache exists; will not overwrite: {path}')
    out.to_csv(path, index=False, compression='gzip')
    meta = {'seed': args.seed, 'split': args.split, 'count': len(part),
            'hash_sha256': sha256_hashes(part['prompt_hash']),
            'validation_was_used_for_checkpoint_selection': args.split == 'validation',
            'checkpoint_best_epoch': mc['epoch'], 'risk_vocab': mc['risk_vocab'],
            'domain_vocab': mc['domain_vocab'], 'score': 'unsafe_score=softmax(binary_logits)[:,1]',
            'output': str(path), 'not_independent_conformal_calibration': True}
    path.with_suffix('.metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print('OK', path, 'n=', len(out), 'best_epoch=', mc['epoch'], flush=True)

if __name__ == '__main__':
    main()
