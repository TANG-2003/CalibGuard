
from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/calibguard_canonical.csv")
OUTPUT = Path("results/e0_full_redbench/split_eligibility.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)

def parse_bool(series):
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(["true", "false"]).all():
        raise ValueError("Unexpected evaluation flag values")
    return values.eq("true")

df["reserved_hard_eval"] = parse_bool(
    df["reserved_hard_eval"]
)
df["reserved_xstest_eval"] = parse_bool(
    df["reserved_xstest_eval"]
)

df["is_safe"] = pd.to_numeric(
    df["is_safe"], errors="coerce"
)

# Define mutually exclusive groups for counting.
hard = df["reserved_hard_eval"]
xstest = df["reserved_xstest_eval"]

df["partition_candidate"] = "development_candidate"
df.loc[hard, "partition_candidate"] = "hard_eval"
df.loc[xstest & ~hard, "partition_candidate"] = "xstest_eval"
df.loc[hard & xstest, "partition_candidate"] = "both_eval"

# Unlabeled and conflicting prompts remain excluded.
df.loc[
    df["label_status"] == "unlabeled",
    "partition_candidate"
] = "unlabeled_holdout"

df.loc[
    df["label_status"] == "needs_review",
    "partition_candidate"
] = "conflict_holdout"

summary = pd.crosstab(
    df["partition_candidate"],
    df["is_safe"],
    dropna=False
)

summary.columns = [
    "unlabeled" if pd.isna(c)
    else "safe" if c == 1
    else "unsafe" if c == 0
    else str(c)
    for c in summary.columns
]

summary["total"] = summary.sum(axis=1)

print("\nCALIBGUARD SPLIT ELIGIBILITY AUDIT")
print("=" * 65)
print(summary.to_string())

print("\nChecks:")
print("Unique prompts:", df["prompt_hash"].is_unique)
print("Unassigned splits:", df["split"].eq("unassigned").all())
print("Hard and XSTest overlap:", int((hard & xstest).sum()))
print(
    "Reserved prompts:",
    int((hard | xstest).sum())
)

summary.to_csv(OUTPUT, encoding="utf-8-sig")

print("\nSaved:", OUTPUT)
print("Audit complete.")
