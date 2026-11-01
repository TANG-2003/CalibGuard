# CalibGuard Stage 3/4 — implementation starter v0.1

This bundle **contains new scripts only**. Copy `scripts/*.py` into the existing project `scripts/`; do not replace Stage-2 code. Tested for syntax and synthetic scoring behavior. GPU inference must be validated in the actual RTX 5090 environment.

## Frozen research protocol

- Fit 66,028, validation 7,337, ID Test 31,444; **no repartitioning**.
- Frozen Full checkpoints: `results/stage2_full_seed{42,43,44}_6ep_v1/checkpoint_best`.
- Reconstruct original validation with the same `train_test_split`, `split_seed=42`, `val_ratio=.1` and validate its hash.
- Validation was already used to choose the best checkpoint. Reusing it for Stage-3 calibration is an **explicit user choice**. We call resulting thresholds **empirical validation-calibrated thresholds**, and do not claim independent split-conformal coverage.
- `is_safe=1` implies binary UNSAFE label 0; `is_safe=0` implies binary UNSAFE label 1.
- Risk labels are supervised **only on labeled unsafe examples**. SAFE examples have no ground-truth risk category. Consequently category-specific SAFE false-positive guarantees are **not identifiable** from existing labels.

## Implemented method (prototype, not claimed to be teacher's uniquely specified formula)

- Scalar route score is **Binary Head Unsafe Score**; Risk Head determines a **predicted** risk class for threshold lookup (`argmax`), never oracle truth at deployment.
- Upper boundary uses the empirical 99th percentile of scores on SAFE validation examples, with `score > high` interpreted as REFUSE. It estimates **global** safe FPR; category-conditional safe FPR is not claimed.
- Lower boundary uses the empirical 1st percentile of Unsafe Score among each risk category's labeled unsafe examples. `gamma=n/(n+kappa)` shrinks class lower bound toward global unsafe 1st percentile. For missing classes use the global quantile. This is a transparent heuristic; its statistical validity must be empirically evaluated.
- The REVIEW width expands the zone around lower / upper thresholds. A REFUSE outcome is a classifier blocking decision, **not** downstream target-model refusal.
- No forced reordering masquerading as confidence interval; `tau_low`/`tau_high` are **routing decision boundaries**, not mathematically derived confidence interval endpoints.
- `alpha=0.01`, `kappa=100`, `review_width=0.05` are **pre-specified exploratory parameters**, not learned from test. Ablations can use separate values without overwriting saved results.

## Usage (project root, after checkpoint/model/data availability)

```bash
python -m unittest discover -s tests -p 'test_stage34.py' -v
python -m py_compile scripts/stage34_*.py

for seed in 42 43 44; do
  for split in validation test_id test_orbench_hard test_xstest test_harmful; do
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/stage34_export.py --seed "$seed" --split "$split" || exit 1
  done
  python scripts/stage34_calibrate.py --seed "$seed" --alpha 0.01 --kappa 100 --review-width 0.05 || exit 1
  for split in test_id test_orbench_hard test_xstest test_harmful; do
    python scripts/stage34_evaluate.py --seed "$seed" --split "$split" || exit 1
  done
done
```

Caches are `results/stage34_cache/seed*/{validation,test_id,...}.csv.gz`. They have no raw prompt text but **hashes can still be sensitive identifiers**. They include no API keys and no model weights; decide what may be published before `git add`.

## Stage 4 remaining implementation / experiments

1. **Strong reviewer**: use an actual API reviewer with versioned prompt, model, date and logged `ALLOW/REFUSE` labels, timings and billed usage. Feed reviewed decisions to `stage34_review_cost.py`. Until actual adjudications exist, REVIEW is unresolved and **must not be assigned correct outcomes**.
2. **Strict leave-one-domain-out / leave-one-risk-out**: requires training models whose fit and calibration exclude the entire held-out category. `stage34_loo_manifest.py` only prepares row counts; it does **not** claim to implement the required 19+22 retrainings. Multilabel category leakage requires special care.
3. **Baselines**: Llama Guard 3, strong model, Global Temperature Scaling and raw target-model refusals need their own actual run outputs. Do not report synthetic comparisons.
4. **Pareto**: the width-grid curve is **descriptive selective routing**; actual security–utility Pareto frontier requires fully adjudicated REVIEW outcomes and a defined cost/utility measure.
5. **Safety statistical caveat**: 99th percentile on the same checkpoint-selection validation set does not guarantee population FPR ≤1%. Rare classes with n<99 cannot substantiate alpha=.01 class coverage.

## Artifacts / provenance

This package was authored from uploaded Stage-2 source scripts and `run_config.json` + `model_config.json` (seed42). Seed43 and seed44 metadata are read dynamically at execution, never inferred from seed42.

## v0.2 additions (install alongside existing v0.1 files)

- `scripts/stage34_baselines.py`: validation-frozen global fixed threshold, safe FPR, unsafe recall and a **descriptive selective** risk-coverage curve. This is not yet a full safety-utility Pareto frontier.
- `scripts/stage34_review_api.py`: optional GPT-4o (or specified API model) adjudication for actual REVIEW samples, with an explicit **dry-run default**, `--execute --approved-data-sharing`, per-run call cap, resume-on-restart, actual token/latency logging and optional *user-supplied* pricing. Requires `pip install openai` only if calling the API. **Raw prompt text is transmitted externally when executed**. Obtain dataset permission first; never upload private review decisions or raw prompts publicly.
- `tests/test_stage34_extra.py`: extra synthetic tests.

Examples once cache and policy exist:

```bash
python scripts/stage34_baselines.py --seed 42 --split test_id
python scripts/stage34_review_api.py --seed 42 --split test_id --max-calls 10
# Only after permission, budget review, and setting OPENAI_API_KEY:
# python scripts/stage34_review_api.py --seed 42 --split test_id --max-calls 10 --execute --approved-data-sharing
# After all REVIEW rows have actual decisions:
# python scripts/stage34_review_cost.py --seed 42 --split test_id --review-decisions results/stage4/review_api/seed42/test_id_gpt-4o_decisions.csv --out results/stage4/seed42/test_id_review_final.json
```

**Important**: strict 19-fold LODO and 22-fold LORO remain intentionally unimplemented, so **do not claim the experimental matrix is complete**. They need fold-specific fit/calibration and ideally separate checkpoints, meaning extra GPU work. The score-only cached evaluation is not a substitute. The user-selected existing Validation reused for checkpoint selection remains **empirical calibration**, without formal finite-sample conformal FPR guarantees. No API model or prices are assumed available; validate availability/terms before real calls.
