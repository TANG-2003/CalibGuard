
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

DATASET_ID = "knoveleng/redbench"
OUTPUT = Path("data/processed/redbench_full_mapped.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

MISSING = {"", "none", "null", "nan", "[missing]"}


def clean_value(value):
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in MISSING:
        return None

    return text


records = []
configs = get_dataset_config_names(DATASET_ID)

for index, config in enumerate(configs, start=1):
    print(f"[{index:02d}/{len(configs)}] {config}")

    ds = load_dataset(DATASET_ID, config, split="train")

    for row in ds:
        category = clean_value(row.get("category"))
        domain = clean_value(row.get("domain"))
        prompt = clean_value(row.get("prompt"))

        if category == "No Risk":
            is_safe = 1
        elif category is None:
            is_safe = None
        else:
            is_safe = 0

        records.append({
            "text": prompt,
            "is_safe": is_safe,
            "risk_category": category,
            "domain": domain,
            "config": config,
            "source": clean_value(row.get("source")),
            "original_category": row.get("category"),
            "original_domain": row.get("domain"),
            "label_status": (
                "unlabeled"
                if category is None
                else "provisional"
            ),
        })

df = pd.DataFrame(records)

# Nullable integer type preserves missing binary labels.
df["is_safe"] = df["is_safe"].astype("Int64")

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 60)
print("FULL REDBENCH MAPPING SUMMARY")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Safe (provisional): {(df['is_safe'] == 1).sum()}")
print(f"Unsafe (provisional): {(df['is_safe'] == 0).sum()}")
print(f"Unlabeled: {df['is_safe'].isna().sum()}")
print(f"Missing prompt: {df['text'].isna().sum()}")
print(f"Missing domain: {df['domain'].isna().sum()}")
print(f"Risk categories: {df['risk_category'].dropna().nunique()}")
print(f"Domains: {df['domain'].dropna().nunique()}")

print(f"\nSaved: {OUTPUT}")
