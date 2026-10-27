#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/shared-nvme/CalibGuard_GPU
mkdir -p logs results

OLD_LOG="logs/stage2_multiseed_queue.log"
SCRIPT="scripts/calibguard_stage2_multiseed.py"
REF="results/stage2_full_9ep_v1/run_config.json"
EXPECTED_HASH="78dc4064cc37f938ef115a97b3456260eee1e71b4563f6495cf02b8bc145ee96"

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "STOP: $*"; exit 1; }

trap 'rc=$?; log "ERROR: exit code $rc at line $LINENO"; exit "$rc"' ERR

[[ -f "$SCRIPT" && -f "$REF" ]] || fail "Missing script/config"

# Prevent duplicate 6-epoch queue instances.
mkdir logs/stage2_6ep_queue.lock 2>/dev/null ||
    fail "Another 6-epoch queue may already exist"
trap 'rmdir logs/stage2_6ep_queue.lock 2>/dev/null || true' EXIT

# Wait for previous 9-epoch queue to finish successfully.
log "Waiting for previous 9-epoch queue"

for i in $(seq 1 240); do
    if grep -q 'QUEUE COMPLETE: all 8 seed43/44 experiments successful' "$OLD_LOG"; then
        log "Previous queue completed successfully"
        break
    fi
    if grep -Eq 'STOP:|ERROR: exit' "$OLD_LOG"; then
        fail "Previous queue reported failure"
    fi
    if (( i == 240 )); then
        fail "Timed out waiting for previous queue"
    fi
    sleep 60
done

# Check no previous stage2 Python training is active.
if pgrep -af 'python .*scripts/calibguard_stage2_(epoch_ablation|multiseed)\.py' >/dev/null; then
    fail "Another stage2 Python training process is active"
fi

# Load identical hyperparameters from the seed42 reference run.
mapfile -t COMMON < <(python - "$REF" <<'PY'
import json, sys

with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)

keys = (
    "batch_size", "eval_batch_size", "max_length",
    "learning_rate", "weight_decay", "warmup_ratio",
    "val_ratio", "risk_loss_weight", "domain_loss_weight"
)

for k in keys:
    if k not in d:
        raise SystemExit(f"Missing configuration: {k}")
    print("--" + k)
    print(d[k])
PY
)

[[ ${#COMMON[@]} -eq 18 ]] || fail "Invalid reference parameters"

# Estimate storage needed for 12 best checkpoints.
REF_WEIGHT="results/stage2_full_9ep_v1/checkpoint_best/model_state.pt"
[[ -f "$REF_WEIGHT" ]] || fail "Reference checkpoint missing"

WEIGHT_KB=$(du -k "$REF_WEIGHT" | awk '{print $1}')
AVAILABLE_KB=$(df -Pk /root/shared-nvme | awk 'NR==2{print $4}')
REQUIRED_KB=$((WEIGHT_KB * 12 + 5 * 1024 * 1024))

log "Free space: $((AVAILABLE_KB / 1024 / 1024)) GiB"
log "Estimated storage requirement including safety reserve: $((REQUIRED_KB / 1024 / 1024)) GiB"

(( AVAILABLE_KB > REQUIRED_KB )) ||
    fail "Insufficient projected storage space"

log "START: 12 experiments, 6 epochs each"

for seed in 42 43 44; do
    for variant in binary risk domain full; do

        OUT="results/stage2_${variant}_seed${seed}_6ep_v1"
        RUNLOG="logs/stage2_${variant}_seed${seed}_6ep_v1.log"

        [[ ! -e "$OUT" ]] || fail "Output already exists: $OUT"

        FREE=$(df -Pk /root/shared-nvme | awk 'NR==2{print $4}')
        (( FREE > 5 * 1024 * 1024 )) ||
            fail "Less than 5 GiB free before starting next run"

        log "START seed=$seed variant=$variant"

        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python -u "$SCRIPT" \
          --data_path data/splits/calibguard_split_v2.csv \
          --model_id models/deberta-v3-base \
          --output_dir "$OUT" \
          --variant "$variant" \
          --epochs 6 \
          --save_epochs 6 \
          --split_seed 42 \
          --seed "$seed" \
          --max_train_batches 0 \
          --max_val_batches 0 \
          "${COMMON[@]}" > "$RUNLOG" 2>&1 || {
              tail -n 30 "$RUNLOG"
              fail "Training failed seed=$seed variant=$variant"
          }

        grep -q 'Run finished. ID TEST NOT EVALUATED.' "$RUNLOG" ||
            fail "Completion marker missing"

        python - "$OUT" "$seed" "$variant" "$EXPECTED_HASH" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
seed = int(sys.argv[2])
variant = sys.argv[3]
expected_hash = sys.argv[4]

with (out / "run_summary.json").open() as f:
    r = json.load(f)

with (out / "run_config.json").open() as f:
    c = json.load(f)

checks = {
    "variant": r.get("variant") == variant,
    "epochs": r.get("epochs_completed") == 6,
    "snapshot": 6 in r.get("snapshots", []),
    "smoke": r.get("is_smoke_test_run") is False,
    "test": r.get("id_test_was_evaluated") is False,
    "validation_hash": r.get("validation_hash_sha256") == expected_hash,
    "seed": int(c.get("seed", -1)) == seed,
    "best_checkpoint": (out / "checkpoint_best" / "model_state.pt").is_file(),
    "epoch6_checkpoint": (out / "checkpoint_epoch_6" / "model_state.pt").is_file(),
    "epoch_log": (out / "epoch_validation_log.csv").is_file(),
}

if "split_seed" in c:
    checks["split_seed"] = int(c["split_seed"]) == 42

bad = [k for k, v in checks.items() if not v]

if bad:
    raise SystemExit("VERIFICATION FAILED: " + ", ".join(bad))

print(
    f"VERIFIED seed={seed} variant={variant} "
    f"best_epoch={r['best_epoch']} "
    f"F1={r['best_validation_unsafe_f1']:.6f}",
    flush=True
)
PY

        # Preserve all validation metrics and best model.
        # Remove redundant epoch-6 weights to save storage.
        rm -rf -- "$OUT/checkpoint_epoch_6"

        log "COMPLETE seed=$seed variant=$variant"
    done
done

log "QUEUE COMPLETE: all 12 six-epoch experiments successful"
