
from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/redbench_canonical.csv")
OUTPUT = Path("results/e0_label_audit/domain_label_distribution.csv")

df = pd.read_csv(INPUT)

print("\nDataset summary")
print("=" * 60)
print(f"Total: {len(df)}")
print(f"Benign: {(df['label'] == 0).sum()}")
print(f"Harmful: {(df['label'] == 1).sum()}")

# Examine whether each domain contains both binary classes.
df["domain_clean"] = (
    df["domain"]
    .fillna("[MISSING]")
    .astype(str)
    .str.strip()
)

counts = pd.crosstab(
    df["domain_clean"],
    df["label"]
)

counts = counts.reindex(columns=[0, 1], fill_value=0)
counts.columns = ["benign", "harmful"]

counts["total"] = counts["benign"] + counts["harmful"]
counts["both_labels"] = (
    (counts["benign"] > 0)
    & (counts["harmful"] > 0)
)

counts = counts.sort_values(
    "total",
    ascending=False
)

print("\nDomain × Label distribution")
print("=" * 60)
print(counts.to_string())

print("\nSummary")
print("=" * 60)
print(f"Domains: {len(counts)}")
print(f"Domains with both labels: {counts['both_labels'].sum()}")
print(
    "Domains with only benign: "
    f"{((counts['benign'] > 0) & (counts['harmful'] == 0)).sum()}"
)
print(
    "Domains with only harmful: "
    f"{((counts['harmful'] > 0) & (counts['benign'] == 0)).sum()}"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
counts.to_csv(OUTPUT, encoding="utf-8-sig")

print(f"\nSaved: {OUTPUT}")
