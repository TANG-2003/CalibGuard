
from pathlib import Path
from hashlib import sha256

import pandas as pd

INPUT = Path("data/processed/redbench_full_mapped.csv")
OUTPUT_DIR = Path("results/e0_full_redbench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return " ".join(str(text).split()).casefold()


df = pd.read_csv(
    INPUT,
    keep_default_na=False,
)

df["normalized_prompt"] = df["text"].apply(normalize)

df["prompt_hash"] = df["normalized_prompt"].apply(
    lambda value: sha256(
        value.encode("utf-8")
    ).hexdigest()
)

def parse_label(value):
    text = str(value).strip()

    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1

    return None


df["parsed_label"] = df["is_safe"].apply(parse_label)

# Detect contradictory binary labels for the same prompt.
conflict_hashes = set()

for prompt_hash, group in df.groupby("prompt_hash"):
    labels = set(group["parsed_label"].dropna())

    if len(labels) > 1:
        conflict_hashes.add(prompt_hash)

df["review_status"] = "provisional"

df.loc[
    df["parsed_label"].isna(),
    "review_status",
] = "unlabeled"

df.loc[
    df["prompt_hash"].isin(conflict_hashes),
    "review_status",
] = "needs_review"

summary = df.groupby("review_status").size()

print("\nLABEL REVIEW PREPARATION")
print("=" * 60)
print(summary.to_string())

print(
    "\nConflict prompt groups:",
    len(conflict_hashes)
)

review_df = df[
    df["review_status"] != "provisional"
].copy()

review_df.to_csv(
    OUTPUT_DIR / "label_review_queue.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    "\nSaved:",
    OUTPUT_DIR / "label_review_queue.csv"
)

print("\nPreparation complete.")
