
import json
from collections import Counter
from pathlib import Path

import pandas as pd

INPUT = Path("results/e0_label_audit/disagreements.csv")
OUTPUT = Path("results/e0_label_audit/risk_type_summary.csv")

df = pd.read_csv(INPUT)

records = []

for config, group in df.groupby("config"):
    counter = Counter()

    for value in group["predicted_risk_types"]:
        types = json.loads(value)
        counter.update(set(types))

    print(f"\n{'=' * 55}")
    print(f"Dataset: {config}")
    print(f"Disagreement samples: {len(group)}")
    print("=" * 55)

    for risk_type, count in counter.most_common(15):
        print(f"{risk_type}: {count}")

    for risk_type, count in counter.items():
        records.append({
            "config": config,
            "risk_type": risk_type,
            "count": count,
            "disagreement_samples": len(group),
            "percentage": round(
                100 * count / len(group), 2
            ),
        })

result = pd.DataFrame(records)
result = result.sort_values(
    ["config", "count"],
    ascending=[True, False],
)

result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nSaved: {OUTPUT}")
