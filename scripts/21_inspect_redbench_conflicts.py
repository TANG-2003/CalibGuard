
from pathlib import Path
from hashlib import sha256
from collections import defaultdict

import pandas as pd

INPUT = Path("data/processed/redbench_full_mapped.csv")
OUTPUT_DIR = Path("results/e0_full_redbench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return " ".join(str(text).split()).casefold()


def parse_label(value):
    text = str(value).strip()

    if text in {"1", "1.0"}:
        return 1
    if text in {"0", "0.0"}:
        return 0

    return None


df = pd.read_csv(INPUT, keep_default_na=False)

groups = defaultdict(list)

for _, row in df.iterrows():
    prompt = normalize(row["text"])
    groups[prompt].append(row.to_dict())

conflict_records = []
mixed_status_records = []

for prompt, records in groups.items():
    if len(records) < 2:
        continue

    labels = {
        parse_label(row["is_safe"])
        for row in records
    }

    has_unlabeled = None in labels
    valid_labels = labels - {None}

    prompt_hash = sha256(
        prompt.encode("utf-8")
    ).hexdigest()[:16]

    if len(valid_labels) > 1:
        for row in records:
            conflict_records.append({
                "prompt_hash": prompt_hash,
                "text": row["text"],
                "config": row["config"],
                "source": row["source"],
                "is_safe": row["is_safe"],
                "risk_category": row["risk_category"],
                "domain": row["domain"],
            })

    if has_unlabeled and valid_labels:
        for row in records:
            mixed_status_records.append({
                "prompt_hash": prompt_hash,
                "text": row["text"],
                "config": row["config"],
                "is_safe": row["is_safe"],
                "risk_category": row["risk_category"],
            })

conflict_df = pd.DataFrame(conflict_records)
mixed_df = pd.DataFrame(mixed_status_records)

conflict_path = OUTPUT_DIR / "binary_label_conflicts.csv"
mixed_path = OUTPUT_DIR / "labeled_unlabeled_overlaps.csv"

conflict_df.to_csv(
    conflict_path,
    index=False,
    encoding="utf-8-sig",
)

mixed_df.to_csv(
    mixed_path,
    index=False,
    encoding="utf-8-sig",
)

print("\nREDBENCH CONFLICT INSPECTION")
print("=" * 60)

if conflict_df.empty:
    print("No binary label conflicts.")
else:
    print(
        "Binary conflict groups:",
        conflict_df["prompt_hash"].nunique()
    )

    for prompt_hash, group in conflict_df.groupby("prompt_hash"):
        print(f"\nPrompt hash: {prompt_hash}")

        for _, row in group.iterrows():
            print(
                f"  config={row['config']}, "
                f"label={row['is_safe']}, "
                f"category={row['risk_category']}, "
                f"domain={row['domain']}"
            )

print("\n" + "=" * 60)

if mixed_df.empty:
    print("Labeled-unlabeled overlap groups: 0")
else:
    print(
        "Labeled-unlabeled overlap groups:",
        mixed_df["prompt_hash"].nunique()
    )

print("\nSaved:")
print(conflict_path)
print(mixed_path)
print("\nInspection complete.")
