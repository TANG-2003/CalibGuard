
from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/redbench_canonical.csv")
OUTPUT = Path("results/e0_label_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Use the first configuration for the seven cross-dataset
# duplicate prompts. Also retain the complete configs field.
df["primary_config"] = df["configs"].str.split("|", regex=False).str[0]

print("\nSource x Binary Label")
print("=" * 70)

source_label = pd.crosstab(
    df["primary_config"],
    df["label"]
).reindex(columns=[0, 1], fill_value=0)

source_label.columns = ["benign", "harmful"]
print(source_label.to_string())

print("\nSource x Domain (Top 10 domains)")
print("=" * 70)

top_domains = df["domain"].value_counts().head(10).index

source_domain = pd.crosstab(
    df["primary_config"],
    df["domain"]
)

print(source_domain.reindex(
    columns=top_domains,
    fill_value=0
).to_string())

print("\nHealthcare source distribution")
print("=" * 70)

healthcare = df[df["domain"] == "Healthcare"]

healthcare_counts = pd.crosstab(
    healthcare["primary_config"],
    healthcare["label"]
).reindex(columns=[0, 1], fill_value=0)

healthcare_counts.columns = ["benign", "harmful"]
print(healthcare_counts.to_string())

source_label.to_csv(
    OUTPUT / "source_label.csv",
    encoding="utf-8-sig"
)

source_domain.to_csv(
    OUTPUT / "source_domain.csv",
    encoding="utf-8-sig"
)

healthcare_counts.to_csv(
    OUTPUT / "healthcare_source_label.csv",
    encoding="utf-8-sig"
)

print("\nAudit complete.")
