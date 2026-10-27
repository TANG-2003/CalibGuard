#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/CalibGuard_GPU

echo "Waiting for experiment A..."
while ps -p 4234 > /dev/null 2>&1; do
    sleep 30
done

echo "Checking experiment A..."
grep -q "Run finished." logs/multitask_deberta_v1_*.log
test -s results/multitask_deberta_v1/id_test_metrics.json

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting experiment B smoke test..."
python -u scripts/formal_train_deberta_multitask_earlystop.py \
  --output_dir results/smoke_earlystop_v1 \
  --epochs 2 \
  --max_train_batches 2 \
  --max_val_batches 2 \
  --max_test_batches 2

echo "Starting formal experiment B..."
python -u scripts/formal_train_deberta_multitask_earlystop.py \
  --output_dir results/multitask_earlystop_v1 \
  --epochs 8 \
  --patience 2 \
  --val_ratio 0.10 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_length 128 \
  --seed 42 \
  --risk_loss_weight 0.3 \
  --domain_loss_weight 0.3

echo "EXPERIMENT B FINISHED SUCCESSFULLY"
