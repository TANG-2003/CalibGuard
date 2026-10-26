
import json
from pathlib import Path

import pandas as pd

INPUT = Path("data/splits/calibguard_split_v2.csv")


def parse_list(value):
    if pd.isna(value):
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (TypeError, ValueError):
        return []


df = pd.read_csv(INPUT, low_memory=False)
train = df[df["split"] == "train"].copy()

train["risk_list"] = train["risk_categories"].apply(parse_list)
train["domain_list"] = train["domains"].apply(parse_list)

train["has_risk_label"] = train["risk_list"].apply(
    lambda values: any(x != "No Risk" for x in values)
)

train["has_domain_label"] = train["domain_list"].apply(
    lambda values: len(values) > 0
)

train["multiple_risk_labels"] = train["risk_list"].apply(
    lambda values: len(
        [x for x in values if x != "No Risk"]
    ) > 1
)

train["multiple_domain_labels"] = train["domain_list"].apply(
    lambda values: len(values) > 1
)

print("\nCALIBGUARD TRAINING LABEL AUDIT")
print("=" * 60)

print("Train rows:", len(train))

print("\nBinary safety labels:")
print(train["is_safe"].value_counts().to_string())

print("\nAuxiliary label availability:")
print("Has risk category:", int(train["has_risk_label"].sum()))
print("Has domain:", int(train["has_domain_label"].sum()))
print(
    "Multiple risk categories:",
    int(train["multiple_risk_labels"].sum())
)
print(
    "Multiple domains:",
    int(train["multiple_domain_labels"].sum())
)

print("\nBy binary label:")
print(
    train.groupby("is_safe")[
        ["has_risk_label", "has_domain_label"]
    ].sum().to_string()
)

print("\nAudit complete.")
