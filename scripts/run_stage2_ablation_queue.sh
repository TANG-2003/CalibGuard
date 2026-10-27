#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/shared-nvme/CalibGuard_GPU"
cd "$ROOT"

DATA="data/splits/calibguard_split_v2.csv"
MODEL="/root/shared-nvme/CalibGuard_GPU/models/deberta-v3-base"
TRAIN_SCRIPT="scripts/calibguard_stage2_epoch_ablation.py"
FULL_DIR="results/stage2_full_9ep_v1"
EXPECTED_HASH="78dc4064cc37f938ef115a97b3456260eee1e71b4563f6495cf02b8bc145ee96"

mkdir -p logs

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

on_error() {
    rc=$?
    log "ERROR: queue stopped at line ${BASH_LINENO[0]} (exit=${rc}). Check the latest experiment log."
    exit "$rc"
}
trap on_error ERR

verify_summary() {
    local outdir="$1"
    local expected_variant="$2"
    python - "$outdir/run_summary.json" "$expected_variant" "$EXPECTED_HASH" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
expected_variant = sys.argv[2]
expected_hash = sys.argv[3]

if not path.is_file():
    raise SystemExit(f"Missing run summary: {path}")

with path.open(encoding="utf-8") as f:
    s = json.load(f)

if s.get("variant") != expected_variant:
    raise SystemExit(f"Wrong variant in {path}: {s.get('variant')!r}")
if int(s.get("epochs_completed", 0)) != 9:
    raise SystemExit(f"Expected 9 completed epochs, got {s.get('epochs_completed')!r}")
if s.get("is_smoke_test_run") is not False:
    raise SystemExit("Run is marked as smoke test or lacks the expected full-run flag.")
if s.get("id_test_was_evaluated") is not False:
    raise SystemExit("ID test was evaluated or the summary does not confirm it was untouched.")
if not all(e in s.get("snapshots", []) for e in (3, 6, 9)):
    raise SystemExit(f"Missing expected snapshots 3/6/9: {s.get('snapshots')!r}")
if s.get("validation_hash_sha256") != expected_hash:
    raise SystemExit(
        "Validation split hash differs from the Full Multi-task run: "
        f"{s.get('validation_hash_sha256')!r}"
    )

print(
    f"Verified {expected_variant}: epochs=9, snapshots=3/6/9, "
    f"validation hash={expected_hash}, ID test untouched."
)
PY
}

# Confirm the completed Full Multi-task run is the comparison reference.
log "Checking the completed Full Multi-task reference."
verify_summary "$FULL_DIR" "full"

# Load the same hyperparameters recorded for the Full Multi-task run.
# The ablation script's variant logic disables the auxiliary task(s) not
# included in each variant.
mapfile -t SHARED_ARGS < <(python - "$FULL_DIR/run_config.json" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
with p.open(encoding="utf-8") as f:
    c = json.load(f)

defaults = {
    "batch_size": 16,
    "eval_batch_size": 32,
    "max_length": 128,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "risk_loss_weight": 0.5,
    "domain_loss_weight": 0.5,
    "val_ratio": 0.10,
    "seed": 42,
}
for k, default in defaults.items():
    v = c.get(k, default)
    print(f"--{k}")
    print(str(v))
PY
)

# Wait for the already-running Binary Only experiment; never launch a duplicate.
BINARY_PIDFILE="logs/stage2_binary_9ep_v1.pid"
BINARY_OUT="results/stage2_binary_9ep_v1"

if [[ ! -f "$BINARY_PIDFILE" ]]; then
    log "STOP: Cannot find $BINARY_PIDFILE. Refusing to guess whether Binary Only is running."
    exit 2
fi

BINARY_PID="$(tr -d '[:space:]' < "$BINARY_PIDFILE")"
if [[ ! "$BINARY_PID" =~ ^[0-9]+$ ]]; then
    log "STOP: Invalid PID in $BINARY_PIDFILE."
    exit 2
fi

log "Waiting for existing Binary Only process PID=$BINARY_PID to finish."
while kill -0 "$BINARY_PID" 2>/dev/null; do
    sleep 30
done

log "Binary Only process has stopped; validating its saved result."
BINARY_LOG="$(ls -t logs/stage2_binary_9ep_v1_*.log 2>/dev/null | head -n 1 || true)"
if [[ -z "$BINARY_LOG" ]] || ! grep -q "Run finished. ID TEST NOT EVALUATED." "$BINARY_LOG"; then
    log "STOP: Binary Only log lacks the expected successful completion marker."
    exit 3
fi
verify_summary "$BINARY_OUT" "binary"

run_variant() {
    local variant="$1"
    local outdir="results/stage2_${variant}_9ep_v1"
    local logfile="logs/stage2_${variant}_9ep_v1_$(date +%Y%m%d_%H%M%S).log"

    if [[ -e "$outdir" ]]; then
        log "STOP: Output directory already exists: $outdir. Inspect it before retrying."
        exit 4
    fi

    log "Starting variant=$variant; output=$outdir; log=$logfile"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python -u "$TRAIN_SCRIPT" \
        --data_path "$DATA" \
        --model_id "$MODEL" \
        --output_dir "$outdir" \
        --variant "$variant" \
        --epochs 9 \
        --save_epochs 3,6,9 \
        --max_train_batches 0 \
        --max_val_batches 0 \
        "${SHARED_ARGS[@]}" \
        > "$logfile" 2>&1

    if ! grep -q "Run finished. ID TEST NOT EVALUATED." "$logfile"; then
        log "STOP: $variant log lacks the expected successful completion marker."
        tail -n 40 "$logfile" || true
        exit 5
    fi

    verify_summary "$outdir" "$variant"
    log "Completed variant=$variant successfully."
}

run_variant "risk"
run_variant "domain"

log "QUEUE COMPLETE: Binary Only, Binary+Risk, Binary+Domain and Full Multi-task results are ready for Stage-2 comparison."
