#!/usr/bin/env python
"""CalibGuard DeBERTa-v3-base binary safety baseline.

Frozen V2 protocol: train=73,365; test_id=31,444. No validation split.
Dataset label: is_safe=1 safe, is_safe=0 unsafe.
Model target: label_is_unsafe = 1 - is_safe (0=safe, 1=unsafe).
"""
import argparse, json, math, random, time
from contextlib import nullcontext
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL_ID = "microsoft/deberta-v3-base"
DATA_PATH = "data/splits/calibguard_split_v2.csv"
OUTPUT_DIR = "results/baseline_deberta_v1"
EXPECTED_TRAIN, EXPECTED_TEST = 73365, 31444


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default=DATA_PATH)
    p.add_argument('--model_id', default=MODEL_ID)
    p.add_argument('--output_dir', default=OUTPUT_DIR)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.06)
    p.add_argument('--grad_accum_steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_train_batches', type=int, default=0, help='Smoke test only; 0=all batches')
    p.add_argument('--max_test_batches', type=int, default=0, help='Smoke test only; 0=all batches')
    p.add_argument('--allow_nonstandard_row_counts', action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def pkg_version(name):
    try: return version(name)
    except PackageNotFoundError: return 'not-installed'


class PromptDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame['text'].astype(str).tolist()
        safe = pd.to_numeric(frame['is_safe'], errors='raise').astype(int).to_numpy()
        if not np.isin(safe, [0, 1]).all():
            raise ValueError('is_safe must contain only 0/1 values')
        self.labels = (1 - safe).astype(np.int64)  # positive class = unsafe
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], int(self.labels[i])


def collator_for(tokenizer, max_length):
    def collate(items):
        texts, labels = zip(*items)
        batch = tokenizer(list(texts), padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt')
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch
    return collate


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast_ctx(device, enabled):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=enabled)
    return nullcontext()


def evaluate(model, loader, device, use_fp16, max_batches=0):
    model.eval(); ys=[]; preds=[]; probs=[]; loss_total=0.0; n_batches=0
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches: break
            batch = move_batch(batch, device); labels = batch.pop('labels')
            with autocast_ctx(device, use_fp16): outputs = model(**batch, labels=labels)
            logits = outputs.logits.float()
            ys.extend(labels.cpu().tolist())
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            loss_total += float(outputs.loss.item()); n_batches += 1
    if not ys: raise RuntimeError('Evaluation processed zero batches')
    y=np.asarray(ys, dtype=np.int64); pred=np.asarray(preds, dtype=np.int64); prob=np.asarray(probs)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    metrics = {
        'evaluated_samples': int(len(y)), 'safe_samples': int((y==0).sum()),
        'unsafe_samples': int((y==1).sum()), 'threshold': 0.5,
        'accuracy': float(accuracy_score(y,pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
        'unsafe_precision': float(precision_score(y,pred,pos_label=1,zero_division=0)),
        'unsafe_recall': float(recall_score(y,pred,pos_label=1,zero_division=0)),
        'unsafe_f1': float(f1_score(y,pred,pos_label=1,zero_division=0)),
        'specificity_safe_recall': float(tn/(tn+fp)) if tn+fp else None,
        'safe_false_positive_rate_over_refusal_proxy': float(fp/(tn+fp)) if tn+fp else None,
        'roc_auc': float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None,
        'average_precision_unsafe': float(average_precision_score(y,prob)) if len(np.unique(y))==2 else None,
        'confusion_matrix': {'tn_safe_correct':int(tn),'fp_safe_predicted_unsafe':int(fp),
                             'fn_unsafe_predicted_safe':int(fn),'tp_unsafe_correct':int(tp)},
        'mean_batch_loss': float(loss_total/n_batches),
    }
    return metrics, y, pred, prob


def main():
    a=args_parse()
    if min(a.epochs,a.batch_size,a.eval_batch_size,a.grad_accum_steps,a.max_length) < 1:
        raise ValueError('epochs, batch sizes, grad accumulation, and max_length must be positive')
    if not 0 <= a.warmup_ratio < 1: raise ValueError('warmup_ratio must be in [0,1)')
    seed_all(a.seed)
    data_path=Path(a.data_path); out=Path(a.output_dir); ckroot=out/'checkpoints'
    out.mkdir(parents=True,exist_ok=True); ckroot.mkdir(parents=True,exist_ok=True)
    print('='*70+'\nCALIBGUARD - DeBERTa BINARY BASELINE\n'+'='*70)
    print(f'Data: {data_path.resolve()}'); print(f'Output: {out.resolve()}')
    print(f'Model={a.model_id}; seed={a.seed}; epochs={a.epochs}; max_length={a.max_length}')
    print('Label mapping: is_safe=1 -> model label 0 (SAFE); is_safe=0 -> model label 1 (UNSAFE)')
    print('No validation split; ID test is evaluated only after the final epoch.')
    if not data_path.is_file(): raise FileNotFoundError(data_path)
    df=pd.read_csv(data_path,low_memory=False)
    missing={'split','text','is_safe'}-set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df['text'].isna().any(): raise ValueError('Missing text values found')
    tr=df.loc[df['split']=='train'].copy(); te=df.loc[df['split']=='test_id'].copy()
    if not a.allow_nonstandard_row_counts:
        if len(tr)!=EXPECTED_TRAIN: raise ValueError(f'Expected {EXPECTED_TRAIN} train rows; found {len(tr)}. Check V2 split.')
        if len(te)!=EXPECTED_TEST: raise ValueError(f'Expected {EXPECTED_TEST} test_id rows; found {len(te)}. Check V2 split.')
    for name, frame in [('train',tr),('test_id',te)]:
        if frame.empty or not pd.to_numeric(frame['is_safe'],errors='raise').isin([0,1]).all():
            raise ValueError(f'{name} is empty or has invalid is_safe labels')
    trsafe=int((pd.to_numeric(tr.is_safe)==1).sum()); trunsafe=int((pd.to_numeric(tr.is_safe)==0).sum())
    tesafe=int((pd.to_numeric(te.is_safe)==1).sum()); teunsafe=int((pd.to_numeric(te.is_safe)==0).sum())
    print(f'Train={len(tr):,} (safe={trsafe:,}, unsafe={trunsafe:,})')
    print(f'ID Test={len(te):,} (safe={tesafe:,}, unsafe={teunsafe:,})')

    tokenizer=AutoTokenizer.from_pretrained(
    a.model_id,
    use_fast=False
)
    model=AutoModelForSequenceClassification.from_pretrained(
        a.model_id,num_labels=2,id2label={0:'SAFE',1:'UNSAFE'},label2id={'SAFE':0,'UNSAFE':1})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('CUDA unavailable; run this formal baseline on the GPU server')
    use_fp16=True; model.to(device)
    print(f'Device={device}; GPU={torch.cuda.get_device_name(0)}; VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB; FP16={use_fp16}')

    collate=collator_for(tokenizer,a.max_length)
    train_loader=DataLoader(PromptDataset(tr),batch_size=a.batch_size,shuffle=True,num_workers=0,
                            pin_memory=True,collate_fn=collate,generator=torch.Generator().manual_seed(a.seed))
    test_loader=DataLoader(PromptDataset(te),batch_size=a.eval_batch_size,shuffle=False,num_workers=0,
                           pin_memory=True,collate_fn=collate)
    optimizer=AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    steps_per_epoch=math.ceil(len(train_loader)/a.grad_accum_steps)
    total_steps=steps_per_epoch*a.epochs; warmup=int(total_steps*a.warmup_ratio)
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup,total_steps)
    scaler=torch.amp.GradScaler('cuda',enabled=use_fp16)
    smoke=bool(a.max_train_batches or a.max_test_batches)
    config={
        'model_id':a.model_id,'data_path':str(data_path),'seed':a.seed,'epochs':a.epochs,
        'batch_size':a.batch_size,'eval_batch_size':a.eval_batch_size,'max_length':a.max_length,
        'learning_rate':a.learning_rate,'weight_decay':a.weight_decay,'warmup_ratio':a.warmup_ratio,
        'grad_accum_steps':a.grad_accum_steps,'mixed_precision':'fp16','label_definition':'label_is_unsafe = 1 - is_safe',
        'train_rows':len(tr),'test_id_rows':len(te),'uses_validation_split':False,'is_smoke_test_run':smoke,
        'torch_version':torch.__version__,'transformers_version':pkg_version('transformers'),
        'pandas_version':pkg_version('pandas'),'sklearn_version':pkg_version('scikit-learn'),
        'gpu':torch.cuda.get_device_name(0),'max_train_batches':a.max_train_batches,'max_test_batches':a.max_test_batches}
    (out/'run_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')

    epoch_logs=[]; start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train(); t0=time.time(); loss_sum=0.0; processed=0; optimizer.zero_grad(set_to_none=True)
        print(f'\n===== Epoch {epoch}/{a.epochs} =====')
        for bi,batch in enumerate(train_loader):
            if a.max_train_batches and bi>=a.max_train_batches: break
            batch=move_batch(batch,device); labels=batch.pop('labels')
            with autocast_ctx(device,use_fp16): outputs=model(**batch,labels=labels); loss=outputs.loss
            if not torch.isfinite(loss).item(): raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {bi+1}')
            scaler.scale(loss/a.grad_accum_steps).backward()
            do_step=((bi+1)%a.grad_accum_steps==0 or
                     (a.max_train_batches and bi+1==a.max_train_batches) or
                     (not a.max_train_batches and bi+1==len(train_loader)))
            if do_step:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            loss_sum+=float(loss.item()); processed+=1
            if processed%100==0: print(f'epoch={epoch} batch={processed}/{len(train_loader)} mean_loss={loss_sum/processed:.5f}')
        if not processed: raise RuntimeError('No training batches processed')
        record={'epoch':epoch,'train_batches':processed,'mean_train_loss':loss_sum/processed,'epoch_seconds':time.time()-t0}
        epoch_logs.append(record); (out/'train_log.csv')
        pd.DataFrame(epoch_logs).to_csv(out/'train_log.csv',index=False)
        checkpoint=ckroot/f'epoch_{epoch}'; checkpoint.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(checkpoint,safe_serialization=True); tokenizer.save_pretrained(checkpoint)
        (checkpoint/'training_state.json').write_text(json.dumps({'epoch':epoch,'mean_train_loss':record['mean_train_loss'],'seed':a.seed,'config':config},indent=2),encoding='utf-8')
        print(f"Epoch {epoch} complete: mean_train_loss={record['mean_train_loss']:.6f}; seconds={record['epoch_seconds']:.1f}; checkpoint={checkpoint}")
        torch.cuda.empty_cache()

    print('\n===== Final ID Test evaluation =====')
    metrics,y,pred,prob=evaluate(model,test_loader,device,use_fp16,a.max_test_batches)
    metrics.update({'is_smoke_test_run':smoke,'model_id':a.model_id,'checkpoint':str(ckroot/f'epoch_{a.epochs}'),
                    'seed':a.seed,'max_length':a.max_length,'test_protocol':'limited batches for smoke test' if a.max_test_batches else 'full ID test'})
    (out/'id_test_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    n=len(y); eval_frame=te.iloc[:n].copy(); eval_frame.insert(0,'source_row_index',eval_frame.index)
    eval_frame['label_is_unsafe']=y; eval_frame['pred_is_unsafe']=pred; eval_frame['prob_unsafe']=prob; eval_frame['correct']=(y==pred)
    eval_frame.to_csv(out/'id_test_predictions.csv',index=False)
    summary={'total_runtime_seconds':time.time()-start,'epoch_logs':epoch_logs,'metrics_file':str(out/'id_test_metrics.json'),
             'predictions_file':str(out/'id_test_predictions.csv'),'checkpoint_dir':str(ckroot/f'epoch_{a.epochs}'),'is_smoke_test_run':smoke}
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metrics,indent=2,ensure_ascii=False))
    print(f'\nTrain log: {out/"train_log.csv"}\nMetrics: {out/"id_test_metrics.json"}\nPredictions: {out/"id_test_predictions.csv"}\nCheckpoint: {ckroot/f"epoch_{a.epochs}"}')
    print('Run finished.')

if __name__=='__main__': main()
