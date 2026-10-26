
import json
from collections import Counter

import pandas as pd

INPUT = "data/splits/calibguard_split_v2.csv"

df = pd.read_csv(INPUT, low_memory=False)
train = df[df["split"] == "train"].copy()


def parse_list(value):
    if pd.isna(value):
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


INVALID = {"", "None", "No Risk", "null", "nan"}


def clean_labels(value):
    labels = parse_list(value)
    return sorted({
        str(x).strip()
        for x in labels
        if x is not None
        and str(x).strip() not in INVALID
    })


train["risk_clean"] = train["risk_categories"].apply(
    clean_labels
)
train["domain_clean"] = train["domains"].apply(
    clean_labels
)

train["n_risk"] = train["risk_clean"].apply(len)
train["n_domain"] = train["domain_clean"].apply(len)

risk_counter = Counter(
    x for labels in train["risk_clean"] for x in labels
)
domain_counter = Counter(
    x for labels in train["domain_clean"] for x in labels
)

print("=" * 65)
print("CALIBGUARD AUXILIARY LABEL INSPECTION")
print("=" * 65)

print("\nUnique valid risk categories:", len(risk_counter))
print("Unique valid domains:", len(domain_counter))

print("\nRisk label counts:")
print(train["n_risk"].value_counts().sort_index())

print("\nDomain label counts:")
print(train["n_domain"].value_counts().sort_index())

print("\nRisk categories:")
for name, count in sorted(risk_counter.items()):
    print(f"  {name}: {count}")

print("\nDomains:")
for name, count in sorted(domain_counter.items()):
    print(f"  {name}: {count}")

print("\nExamples with multiple risk categories:")
for _, row in train[train["n_risk"] > 1].head(10).iterrows():
    print(
        row["prompt_hash"],
        row["risk_clean"],
        parse_list(row["configs"])
    )

print("\nExamples with multiple domains:")
for _, row in train[train["n_domain"] > 1].head(10).iterrows():
    print(
        row["prompt_hash"],
        row["domain_clean"],
        parse_list(row["configs"])
    )

print("\nInspection complete.")
