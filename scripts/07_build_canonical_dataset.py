
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "knoveleng/redbench"

CONFIGS = [
    "ORBench",
    "XSTest",
    "CoCoNot",
    "JBBBehaviours",
    "HarmBench",
    "MedSafetyBench",
]

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

groups = defaultdict(list)

for config in CONFIGS:
    print(f"Loading {config}...")
    ds = load_dataset(DATASET_ID, config, split="train")

    for row in ds:
        prompt = str(row["prompt"])
        normalized = prompt.strip().casefold()

        if not normalized:
            continue

        category = str(row["category"]).strip()

        groups[normalized].append({
            "prompt": prompt,
            "config": config,
            "category": category,
            "domain": row["domain"],
            "source": row["source"],
            "label": int(category != "No Risk"),
        })

records = []
conflicts = []

for normalized, items in groups.items():
    labels = {item["label"] for item in items}

    prompt_id = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    if len(labels) > 1:
        conflicts.append(prompt_id)
        continue

    representative = items[0]

    records.append({
        "prompt_id": prompt_id,
        "prompt": representative["prompt"],
        "label": representative["label"],
        "category": representative["category"],
        "domain": representative["domain"],
        "source": representative["source"],
        "configs": "|".join(
            sorted({item["config"] for item in items})
        ),
        "categories": "|".join(
            sorted({item["category"] for item in items})
        ),
        "duplicate_count": len(items),
    })

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_DIR / "redbench_canonical.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nCanonical dataset summary")
print("=" * 50)
print(f"Unique prompts: {len(df)}")
print(f"Benign: {(df['label'] == 0).sum()}")
print(f"Harmful: {(df['label'] == 1).sum()}")
print(f"Binary conflicts excluded: {len(conflicts)}")

print("\nSource configuration counts:")
print(df["configs"].value_counts().to_string())

print("\nSaved: data/processed/redbench_canonical.csv")
