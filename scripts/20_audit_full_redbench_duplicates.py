
from collections import defaultdict
from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/redbench_full_mapped.csv")
OUTPUT_DIR = Path("results/e0_full_redbench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).casefold()


df = pd.read_csv(
    INPUT,
    keep_default_na=False,
)

df["normalized_prompt"] = df["text"].apply(normalize)

groups = defaultdict(list)

for row in df.itertuples(index=False):
    groups[row.normalized_prompt].append({
        "config": row.config,
        "is_safe": str(row.is_safe),
        "risk_category": row.risk_category,
        "domain": row.domain,
    })

duplicates = {
    prompt: items
    for prompt, items in groups.items()
    if len(items) > 1
}

cross_config = {
    prompt: items
    for prompt, items in duplicates.items()
    if len({item["config"] for item in items}) > 1
}


def valid_binary_label(value):
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    return None


conflicts = {}

for prompt, items in duplicates.items():
    labels = {
        valid_binary_label(item["is_safe"])
        for item in items
    }

    labels.discard(None)

    if len(labels) > 1:
        conflicts[prompt] = items


summary = {
    "total_rows": len(df),
    "unique_prompts": len(groups),
    "duplicate_rows": len(df) - len(groups),
    "duplicate_groups": len(duplicates),
    "cross_config_duplicate_groups": len(cross_config),
    "binary_conflict_groups": len(conflicts),
}

print("\nFULL REDBENCH DUPLICATE AUDIT")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

pd.DataFrame([summary]).to_csv(
    OUTPUT_DIR / "full_redbench_duplicate_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved duplicate summary.")
print("Audit complete.")
