#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# This requires 123 independent full model training runs; no implicit launch.
python -u scripts/stage34_loo_queue.py --execute "$@"
python scripts/stage34_loo_report.py
