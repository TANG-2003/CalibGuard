
import json
from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/calibguard_merged_raw.csv")
OUTPUT = Path("data/processed/calibguard_canonical.csv")
SUMMARY = Path(
    "results/e0_full_redbench/canonical_summary.csv"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)

print("Loading merged data...")

df = pd.read_csv(INPUT, keep_default_na=False)
df["is_safe"] = pd.to_numeric(
    df["is_safe"], errors="coerce"
).astype("Int64")

required = {
    "text", "prompt_hash", "is_safe",
    "risk_category", "domain",
    "dataset_source", "config",
    "original_category",
    "reserved_hard_eval",
    "reserved_xstest_eval",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

def unique_values(series):
    return sorted({
        str(v).strip()
        for v in series
        if str(v).strip().lower()
        not in {"", "none", "nan", "null"}
    })

def pack(series):
    return json.dumps(
        unique_values(series),
        ensure_ascii=False
    )

records = []

print("Deduplicating by prompt_hash...")

for prompt_hash, group in df.groupby(
    "prompt_hash", sort=True
):
    labels = set(
        group["is_safe"].dropna().astype(int)
    )

    if len(labels) > 1:
        label = pd.NA
        status = "needs_review"
    elif len(labels) == 0:
        label = pd.NA
        status = "unlabeled"
    else:
        label = next(iter(labels))
        status = "provisional"

    # Prefer a RedBench text copy if available.
    red = group[
        group["dataset_source"] == "RedBench"
    ]
    representative = (
        red.iloc[0] if not red.empty
        else group.iloc[0]
    )

    records.append({
        "prompt_hash": prompt_hash,
        "text": representative["text"],
        "is_safe": label,
        "label_status": status,
        "risk_categories": pack(group["risk_category"]),
        "domains": pack(group["domain"]),
        "dataset_sources": pack(
            group["dataset_source"]
        ),
        "configs": pack(group["config"]),
        "original_categories": pack(
            group["original_category"]
        ),
        "source_record_count": len(group),
        "reserved_hard_eval": bool(
            group["reserved_hard_eval"].astype(bool).any()
        ),
        "reserved_xstest_eval": bool(
            group["reserved_xstest_eval"].astype(bool).any()
        ),
        "split": "unassigned",
    })

result = pd.DataFrame(records)
result["is_safe"] = result["is_safe"].astype("Int64")

# Integrity assertions
assert result["prompt_hash"].is_unique
assert len(result) == df["prompt_hash"].nunique()
assert (
    result["source_record_count"].sum() == len(df)
)
assert result["split"].eq("unassigned").all()

result.to_csv(
    OUTPUT, index=False, encoding="utf-8-sig"
)

summary = {
    "raw_rows": len(df),
    "canonical_prompts": len(result),
    "provisional_safe": int(
        ((result["is_safe"] == 1)
         & (result["label_status"] == "provisional")).sum()
    ),
    "provisional_unsafe": int(
        ((result["is_safe"] == 0)
         & (result["label_status"] == "provisional")).sum()
    ),
    "needs_review_prompts": int(
        (result["label_status"] == "needs_review").sum()
    ),
    "unlabeled_prompts": int(
        (result["label_status"] == "unlabeled").sum()
    ),
    "hard_reserved": int(
        result["reserved_hard_eval"].sum()
    ),
    "xstest_reserved": int(
        result["reserved_xstest_eval"].sum()
    ),
    "any_eval_reserved": int(
        (
            result["reserved_hard_eval"]
            | result["reserved_xstest_eval"]
        ).sum()
    ),
}

pd.DataFrame([summary]).to_csv(
    SUMMARY, index=False, encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("CANONICAL DATASET SUMMARY")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved:", OUTPUT)
print("All integrity checks passed.")
