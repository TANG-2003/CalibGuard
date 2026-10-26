
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "bench-llm/or-bench"
REDBENCH_FILE = Path(
    "data/processed/redbench_full_mapped.csv"
)
OUTPUT_DIR = Path("results/e0_full_redbench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(prompt):
    if prompt is None:
        return ""
    return " ".join(str(prompt).split()).casefold()


print("Loading OR-Bench 80K...")
dataset = load_dataset(
    DATASET_ID,
    "or-bench-80k",
    split="train",
)

print("Loading OR-Bench Hard...")
hard_dataset = load_dataset(
    DATASET_ID,
    "or-bench-hard-1k",
    split="train",
)

print("Reading local RedBench mapping...")
redbench = pd.read_csv(
    REDBENCH_FILE,
    usecols=["text"],
    keep_default_na=False,
)

prompts_80k = [
    normalize(row["prompt"])
    for row in dataset
]

set_80k = set(prompts_80k)

set_hard = {
    normalize(row["prompt"])
    for row in hard_dataset
}

set_redbench = {
    normalize(value)
    for value in redbench["text"]
}

category_counts = Counter(
    str(row["category"])
    for row in dataset
)

print("\n" + "=" * 60)
print("OR-BENCH 80K AUDIT")
print("=" * 60)

print("Total rows:", len(dataset))
print("Unique prompts:", len(set_80k))
print(
    "Internal duplicate rows:",
    len(dataset) - len(set_80k)
)

print("\nCategory distribution:")
for category, count in category_counts.most_common():
    print(f"  {category}: {count}")

print("\nOverlap analysis:")
print("80K vs Hard:", len(set_80k & set_hard))
print("80K vs full RedBench:", len(set_80k & set_redbench))
print(
    "80K prompts not in RedBench:",
    len(set_80k - set_redbench)
)

summary = pd.DataFrame([
    {
        "dataset": "or-bench-80k",
        "total_rows": len(dataset),
        "unique_prompts": len(set_80k),
        "internal_duplicates": len(dataset) - len(set_80k),
        "overlap_with_hard": len(set_80k & set_hard),
        "overlap_with_redbench": len(
            set_80k & set_redbench
        ),
        "new_vs_redbench": len(
            set_80k - set_redbench
        ),
    }
])

output = OUTPUT_DIR / "orbench_80k_audit.csv"

summary.to_csv(
    output,
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved:", output)
print("Audit complete.")
