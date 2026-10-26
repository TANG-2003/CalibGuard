
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import (
    get_dataset_config_names,
    load_dataset,
)

DATASET_ID = "knoveleng/redbench"
OUTPUT_DIR = Path("results/e0_full_redbench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

configs = get_dataset_config_names(DATASET_ID)

category_counts = Counter()
domain_counts = Counter()
config_records = []

total_rows = 0

for index, config in enumerate(configs, start=1):
    print(f"[{index:02d}/{len(configs)}] Loading {config}...")

    dataset = load_dataset(
        DATASET_ID,
        config,
        split="train",
    )

    df = dataset.to_pandas()

    if not {"prompt", "category", "domain"}.issubset(
        df.columns
    ):
        raise ValueError(
            f"Missing required columns in {config}"
        )

    categories = (
        df["category"]
        .fillna("[MISSING]")
        .astype(str)
        .str.strip()
    )

    domains = (
        df["domain"]
        .fillna("[MISSING]")
        .astype(str)
        .str.strip()
    )

    category_counts.update(categories)
    domain_counts.update(domains)

    no_risk_count = int((categories == "No Risk").sum())

    config_records.append({
        "config": config,
        "rows": len(df),
        "no_risk": no_risk_count,
        "other_categories": len(df) - no_risk_count,
        "missing_category": int(
            categories.isin(["", "[MISSING]"]).sum()
        ),
        "missing_domain": int(
            domains.isin(["", "[MISSING]"]).sum()
        ),
    })

    total_rows += len(df)

summary = pd.DataFrame(config_records)

summary.to_csv(
    OUTPUT_DIR / "full_config_label_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    category_counts.items(),
    columns=["category", "count"],
).sort_values(
    "count",
    ascending=False,
).to_csv(
    OUTPUT_DIR / "full_category_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    domain_counts.items(),
    columns=["domain", "count"],
).sort_values(
    "count",
    ascending=False,
).to_csv(
    OUTPUT_DIR / "full_domain_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 65)
print("FULL REDBENCH LABEL AUDIT")
print("=" * 65)

print(f"Total loaded rows: {total_rows}")
print(f"Unique category values: {len(category_counts)}")
print(f"Unique domain values: {len(domain_counts)}")

print("\nPer-config label summary:")
print(summary.to_string(index=False))

print("\nAll category values:")
for name, count in category_counts.most_common():
    print(f"  {name}: {count}")

print("\nAll domain values:")
for name, count in domain_counts.most_common():
    print(f"  {name}: {count}")

print("\nSaved results to:", OUTPUT_DIR)
