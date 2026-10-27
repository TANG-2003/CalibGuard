#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/shared-nvme/CalibGuard_GPU
mkdir -p logs results
EXPECTED_HASH='78dc4064cc37f938ef115a97b3456260eee1e71b4563f6495cf02b8bc145ee96'
SCRIPT='scripts/calibguard_stage2_multiseed.py'
REF='results/stage2_full_9ep_v1/run_config.json'
QUEUE_LOG_PREFIX='STAGE2 MULTISEED'
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { log "STOP: $*"; exit 1; }
trap 'rc=$?; log "ERROR: exit $rc at line $LINENO"; exit "$rc"' ERR
[[ -f "$SCRIPT" && -f "$REF" ]] || fail 'Missing training script or reference config'
# Prevent overlap with earlier or other stage2 experiments.
if pgrep -af 'python .*scripts/calibguard_stage2_(epoch_ablation|multiseed)\.py' >/dev/null; then
  pgrep -af 'python .*scripts/calibguard_stage2_(epoch_ablation|multiseed)\.py' || true
  fail 'Another stage2 training process is running'
fi
# Take every shared hyperparameter from the actual seed42 full run; do not silently use defaults.
mapfile -t COMMON < <(python - "$REF" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f: d=json.load(f)
keys=('batch_size','eval_batch_size','max_length','learning_rate','weight_decay','warmup_ratio','val_ratio','risk_loss_weight','domain_loss_weight')
missing=[k for k in keys if k not in d]
if missing:
    raise SystemExit('Reference config missing keys: '+str(missing))
for k in keys:
    print('--'+k)
    print(d[k])
PY
)
[[ ${#COMMON[@]} -eq 18 ]] || fail 'Could not read all common hyperparameters from Full run_config.json'
log "$QUEUE_LOG_PREFIX: starting 8 sequential runs; checkpoint_best + epoch_6 only for each new seed"
for seed in 43 44; do
  for variant in binary risk domain full; do
    out="results/stage2_${variant}_seed${seed}_9ep_v1"
    logfile="logs/stage2_${variant}_seed${seed}_9ep_v1_$(date +%Y%m%d_%H%M%S).log"
    [[ ! -e "$out" ]] || fail "Existing output directory $out; inspect before restarting"
    avail=$(df -Pk /root/shared-nvme | awk 'NR==2{print $4}')
    (( avail > 4194304 )) || fail 'Less than 4 GiB of space available before new run'
    log "START seed=$seed variant=$variant log=$logfile"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u "$SCRIPT" \
      --data_path data/splits/calibguard_split_v2.csv \
      --model_id models/deberta-v3-base \
      --output_dir "$out" \
      --variant "$variant" \
      --epochs 9 --save_epochs 6 \
      --split_seed 42 --seed "$seed" \
      --max_train_batches 0 --max_val_batches 0 \
      "${COMMON[@]}" > "$logfile" 2>&1 || { tail -n 40 "$logfile"; fail "Training failed: seed=$seed variant=$variant"; }
    grep -q 'Run finished. ID TEST NOT EVALUATED.' "$logfile" || fail "Missing successful-completion marker: $logfile"
    python - "$out" "$seed" "$variant" "$EXPECTED_HASH" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); seed=int(sys.argv[2]); var=sys.argv[3]; expected_hash=sys.argv[4]
with (out/'run_summary.json').open(encoding='utf-8') as f: r=json.load(f)
with (out/'run_config.json').open(encoding='utf-8') as f: c=json.load(f)
checks={
  'variant':r.get('variant')==var,
  'epochs_completed':r.get('epochs_completed')==9,
  'snapshot_6':6 in r.get('snapshots',[]),
  'non_smoke':r.get('is_smoke_test_run') is False,
  'id_test_untouched':r.get('id_test_was_evaluated') is False,
  'validation_hash':r.get('validation_hash_sha256')==expected_hash,
  'train_seed':int(c.get('seed',-1))==seed,
  'checkpoint_best':(out/'checkpoint_best'/'model_state.pt').is_file(),
  'checkpoint_epoch_6':(out/'checkpoint_epoch_6'/'model_state.pt').is_file(),
}
# Some versions of the training script may not persist the new CLI field.
if 'split_seed' in c: checks['split_seed']=int(c['split_seed'])==42
bad=[k for k,ok in checks.items() if not ok]
if bad: raise SystemExit('CHECK FAILED: '+', '.join(bad))
print(f"VERIFIED seed={seed} variant={var}: best_epoch={r.get('best_epoch')}, F1={r.get('best_validation_unsafe_f1'):.6f}")
PY
    log "COMPLETE seed=$seed variant=$variant"
  done
done
log 'QUEUE COMPLETE: all 8 seed43/44 experiments successful'
