# CalibGuard Stage 3 & 4 — consolidated RTX 5090 deployment bundle

**This is the consolidated replacement for v0.1 / v0.2. Do not stack multiple old bundles.** It contains new stage34 scripts only; preserve all stage2 files, checkpoints, dataset V2 and previous test results.

## Frozen research decisions

- V2 7:3 train/test, Fit 66,028, validation 7,337, ID Test 31,444. **No repartitioning** for frozen Stage2 Full model.
- Three checkpoint_best: seeds 42, 43, 44. Model selection uses original validation Unsafe F1, 6 total epochs.
- Stage3 calibration intentionally reuses the model-selection validation. It is **empirical post-selection calibration**, **NOT** a standard independent split-conformal finite-sample bound. `alpha=0.01` is a target/quantile parameter, NOT a proven population safety constraint.
- Stage3 class-conditioned lower thresholds use unsafe labeled risk groups and shrink `gamma=n/(n+100)` to global unsafe threshold. Upper REFUSE boundary estimated globally from safe examples. **No per-risk SAFE FPR guarantee**, because safe examples have no risk ground truth. The double thresholds are policy boundaries, NOT statistical confidence interval endpoints.
- Same test sets are already observed in Stage2. Avoid further tuning using test results; offline policy configurations are predeclared.

## Project installation (Windows / local)

Extract this ZIP, and copy `scripts/stage34_*.py` to `<CalibGuard>/scripts/`, `configs/stage34_protocol.json` to `<CalibGuard>/configs/`, and `run_stage34_*.sh` to the project root. Do **not** overwrite `formal_train_deberta_multitask.py` or other Stage2 source code. Place `README_DEPLOY.md` in `docs/`.

## GPU environment prerequisites

- Linux Python (same dependencies as Stage2: torch, transformers, numpy, pandas, sklearn; plus `openai` **only if** actual REVIEW API calls), CUDA driver/PyTorch compatible with RTX5090, source `models/deberta-v3-base`, `data/splits/calibguard_split_v2.csv`.
- For **each seed**: `results/stage2_full_seed${seed}_6ep_v1/checkpoint_best/` with `model_state.pt`, `model_config.json`, tokenizer files; plus `run_config.json`, `run_summary.json` at run root.
- The 3-checkpoint backup contains the `checkpoint_best` folders but **not** `run_config.json` and `run_summary.json`: copy these from `experiment_records/stage2_formal/...`. The project must also have model base files in `models/deberta-v3-base` and V2 CSV from trusted storage.
- Files are read from **project root**; do not run scripts from `scripts/` working directory.
- Do NOT place API keys in scripts or GitHub.

## One GPU inference pass per frozen model, resumable

```bash
bash run_stage34_core.sh
```

Sequence: 3 seeds × [validation, ID Test, OR-Bench Hard, XSTest, Harmful] -> 15 cached model inferences total; 3 policies; 12 routed test summaries; global fixed-threshold baseline for each test/seed; summary CSV; 2 LOO manifests. Skip existing cache/policy/metrics for resume. **Important**: existing cache must belong to same checkpoint, data and protocol; don't reuse unverified output directories.

## Strict LODO / LORO (separate costly matrix)

```bash
# Show entire 19 domain + 22 risk × 3 seed plan WITHOUT training:
python scripts/stage34_loo_queue.py

# Run ONE fold as GPU integration check; no repeated data audit:
python -u scripts/stage34_loo_queue.py --execute --max-runs 1

# Resume ALL remaining folds only if budget/storage time is acceptable:
bash run_stage34_strict_loo.sh
```

This is **123 separate finetunings**, not 123 forward passes. Each fold removes all multilabel training rows containing its heldout label, including model-selection validation. Risk fold test uses heldout risk test items plus existing benign ID-test controls. Domain fold tests heldout domain items. For each fold, a new model is trained, best epoch selected on that fold's validation, and actual fold ID-test scores, postselection empirical threshold policy, and route results saved. Outputs are large (~123 checkpoints); **do not run full queue blindly on shared quota**. Queue resumes `COMPLETE.json` folds but incomplete folders halt safely.

## Actual REVIEW API

```bash
# dry-run; prints REVIEW count and does not transmit data:
python scripts/stage34_review_api.py --seed 42 --split test_id --max-calls 10

# Only when owner permits sending raw benchmark prompts to an external provider,
# confirmed cost cap and configured OPENAI_API_KEY, run explicitly:
# python scripts/stage34_review_api.py --seed 42 --split test_id --max-calls 10 \
#   --execute --approved-data-sharing --model gpt-4o
```

The API script logs decisions / tokens / latency, supports re-run to resume, and does **not** auto-launch. `--max-calls` is per execution not a total lifetime quota; estimate full costs and enforce budget before using it repeatedly. `stage34_review_cost.py` refuses to pretend unreviewed items were correctly resolved.

## External baselines (Llama Guard 3, Raw Target, strong API)

**Not included as synthetic model outputs.** Run the official model/target under its own conditions then create a CSV with `prompt_hash` and actual `unsafe_score` or `predicted_unsafe` (0/1), one row per cache item. Use:

```bash
python scripts/stage34_external_baseline.py --seed 42 --split test_id \
  --baseline-name llama_guard_3 --predictions <ACTUAL_BASELINE.csv> \
  --out results/stage4/external/llamaguard3_seed42_test_id.json
```

This validates exact hash alignment and scores comparable metrics. **It does not run Llama Guard 3 or the target model itself.** Model/license/access are external prerequisites. You cannot claim those baselines finished without real predictions.

## What is vs is NOT complete

Executable code: frozen cache exporter, empirical quantile + shrinkage, three-way routing, global threshold baseline, empirical selective risk-coverage, optional real REVIEW API, cost merger, strict LOO trainer/queue/reporter, external-baseline import, offline summary.

**NOT an automatically completed research matrix**: You must actually run GPU fits (123), obtain actual external model predictions, complete all REVIEW judgments and confirm API permissions, as well as document targets and parameters. 'Pareto' in current code is a **selective routing proxy**, not a proven complete end-to-end security/utility frontier. Don't report LODO/LORO ASR as actual downstream jailbreak success (it's classifier unsafe acceptance proxy).

## Data, experimental fairness & reliability cautions

- Folds can have extreme class imbalance or some heldout domain test might have only one binary class; report undefined metrics rather than invent them.
- 123 fold models with 6 epochs each is expensive in compute and disk; checkpoints should be backed up selectively and pruning should be done only after verified summaries.
- Full review may cost money and transmit prompts. Do not use private data without permission.
- Model versions, timestamp, model-specific prompts, latency machine specs, token prices, data hashes and seed must be logged for reproducibility.
- Python syntax verification is not a GPU hardware compatibility check; setup newer CUDA compatible PyTorch for RTX5090.
