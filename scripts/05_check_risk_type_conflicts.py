
import json
from pathlib import Path

import pandas as pd

INPUT = Path("results/e0_label_audit/disagreements.csv")

df = pd.read_csv(INPUT)

results = []

for config, group in df.groupby("config"):
    only_risk = 0
    mixed_no_risk = 0
    empty_types = 0
    multiple_risks = 0

    for value in group["predicted_risk_types"]:
        types = json.loads(value)

        normalized = {
            str(t).strip().lower()
            for t in types
            if str(t).strip()
        }

        has_no_risk = "no risk" in normalized
        risk_types = normalized - {"no risk", "none"}

        if not normalized:
            empty_types += 1
        elif has_no_risk and risk_types:
            mixed_no_risk += 1
        elif risk_types:
            only_risk += 1

        if len(risk_types) > 1:
            multiple_risks += 1

    results.append({
        "config": config,
        "disagreements": len(group),
        "risk_only": only_risk,
        "mixed_no_risk": mixed_no_risk,
        "empty_types": empty_types,
        "multiple_risks": multiple_risks,
    })

result = pd.DataFrame(results)

print("\nRisk type conflict audit")
print("=" * 65)
print(result.to_string(index=False))

output = Path(
    "results/e0_label_audit/risk_type_conflicts.csv"
)

result.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved: {output}")
