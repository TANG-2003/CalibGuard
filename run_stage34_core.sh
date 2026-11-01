#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u scripts/stage34_pipeline.py
python scripts/stage34_summary.py
python scripts/stage34_loo_manifest.py --axis domain --out results/stage4/lodo_manifest.csv || test -f results/stage4/lodo_manifest.csv
python scripts/stage34_loo_manifest.py --axis risk --out results/stage4/loro_manifest.csv || test -f results/stage4/loro_manifest.csv
echo 'STAGE3/4 OFFLINE CORE FINISHED. Strict LOO and API still require explicit runs.'
