
from pathlib import Path
from hashlib import sha256

import pandas as pd
from datasets import load_dataset

ROOT = Path("data/processed")
ROOT.mkdir(parents=True, exist_ok=True)

OUTPUT = ROOT / "calibguard_merged_raw.csv"
AUDIT_DIR = Path("results/e0_full_redbench")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return " ".join(str(text).split()).casefold()


def make_hash(text):
    return sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


def clean(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in {"", "none", "null", "nan"}:
        return None

    return value


records = []

# 1. RedBench
print("[1/3] Loading local RedBench...")

redbench = pd.read_csv(
    ROOT / "redbench_full_mapped.csv",
    keep_default_na=False,
)

for row in redbench.itertuples(index=False):
    label = clean(row.is_safe)

    records.append({
        "text": row.text,
        "is_safe": (
            int(float(label))
            if label is not None else None
        ),
        "risk_category": clean(row.risk_category),
        "domain": clean(row.domain),
        "dataset_source": "RedBench",
        "config": row.config,
        "original_category": clean(row.original_category),
        "label_origin": "redbench_category_proxy",
    })

# 2. OR-Bench 80K
print("[2/3] Loading OR-Bench 80K...")

orbench = load_dataset(
    "bench-llm/or-bench",
    "or-bench-80k",
    split="train",
)

for row in orbench:
    records.append({
        "text": row["prompt"],
        "is_safe": 1,
        "risk_category": None,
        "domain": None,
        "dataset_source": "OR-Bench",
        "config": "or-bench-80k",
        "original_category": clean(row["category"]),
        "label_origin": "orbench_benign_subset",
    })

# 3. JBB Benign
print("[3/3] Loading JBB Benign...")

jbb = load_dataset(
    "JailbreakBench/JBB-Behaviors",
    "behaviors",
    split="benign",
)

for row in jbb:
    records.append({
        "text": row["Goal"],
        "is_safe": 1,
        "risk_category": None,
        "domain": None,
        "dataset_source": "JBB",
        "config": "benign",
        "original_category": clean(row["Category"]),
        "label_origin": "jbb_benign_split",
    })

# Build merged table
df = pd.DataFrame(records)

df["prompt_hash"] = df["text"].apply(make_hash)
df["is_safe"] = df["is_safe"].astype("Int64")

# Group by normalized prompt hash
groups = df.groupby("prompt_hash", sort=False)

group_sizes = groups.size()

label_counts = groups["is_safe"].nunique(dropna=True)

conflict_hashes = set(
    label_counts[label_counts > 1].index
)

df["review_status"] = "provisional"

df.loc[
    df["is_safe"].isna(),
    "review_status",
] = "unlabeled"

df.loc[
    df["prompt_hash"].isin(conflict_hashes),
    "review_status",
] = "needs_review"

# Include full official OR-Bench Hard in evaluation reservation
hard = load_dataset(
    "bench-llm/or-bench",
    "or-bench-hard-1k",
    split="train",
)

hard_hashes = {
    make_hash(row["prompt"])
    for row in hard
}

df["reserved_hard_eval"] = (
    df["prompt_hash"].isin(hard_hashes)
)

# XSTest is an additional evaluation candidate.
# All duplicates of its prompts must receive the same flag.
xstest_hashes = set(
    df.loc[
        (df["dataset_source"] == "RedBench")
        & (df["config"] == "XSTest"),
        "prompt_hash",
    ]
)

df["reserved_xstest_eval"] = (
    df["prompt_hash"].isin(xstest_hashes)
)

# Save merged, NOT deduplicated data
df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "total_rows": len(df),
    "unique_prompts": df["prompt_hash"].nunique(),
    "duplicate_extra_rows": (
        len(df) - df["prompt_hash"].nunique()
    ),
    "duplicate_groups": int((group_sizes > 1).sum()),
    "binary_conflict_groups": len(conflict_hashes),
    "unlabeled_rows": int(df["is_safe"].isna().sum()),
    "needs_review_rows": int(
        (df["review_status"] == "needs_review").sum()
    ),
    "reserved_hard_unique": len(
        set(df.loc[df["reserved_hard_eval"], "prompt_hash"])
    ),
    "reserved_xstest_unique": len(
        set(df.loc[df["reserved_xstest_eval"], "prompt_hash"])
    ),
}

print("\n" + "=" * 60)
print("CALIBGUARD MERGED DATA AUDIT")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSource counts:")
print(df["dataset_source"].value_counts().to_string())

pd.DataFrame([summary]).to_csv(
    AUDIT_DIR / "merged_sources_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"\nSaved: {OUTPUT}")
print("Merge and audit complete.")
