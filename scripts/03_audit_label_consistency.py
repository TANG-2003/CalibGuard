
import ast
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "knoveleng/redbench"
CONFIGS = ["ORBench", "XSTest", "HarmBench"]
OUTPUT_DIR = Path("results") / "e0_label_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_types(value):
    """Extract risk types from structured model output."""
    if isinstance(value, dict):
        data = value
    elif isinstance(value, str):
        data = None
        for parser in (json.loads, ast.literal_eval):
            try:
                data = parser(value)
                break
            except (ValueError, SyntaxError, TypeError):
                continue
    else:
        data = None

    if not isinstance(data, dict):
        return [], False

    types = data.get("types", [])
    if isinstance(types, str):
        types = [types]

    if not isinstance(types, list):
        return [], False

    return [str(t).strip() for t in types], True


summary = []
all_disagreements = []

for config in CONFIGS:
    print(f"\n{'=' * 60}")
    print(f"Auditing: {config}")
    print("=" * 60)

    ds = load_dataset(DATASET_ID, config)

    for split_name, split in ds.items():
        df = split.to_pandas()

        required = {"prompt", "category", "risk_property"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{config}/{split_name} missing columns: {missing}"
            )

        parsed = df["risk_property"].apply(extract_types)
        df["predicted_risk_types"] = parsed.apply(lambda x: x[0])
        df["parse_success"] = parsed.apply(lambda x: x[1])

        df["is_no_risk"] = (
            df["category"].astype(str).str.strip() == "No Risk"
        )

        df["predicts_risk"] = df["predicted_risk_types"].apply(
            lambda types: any(
                t.lower() not in {"no risk", "none", ""}
                for t in types
            )
        )

        df["label_disagreement"] = (
            df["is_no_risk"]
            & df["parse_success"]
            & df["predicts_risk"]
        )

        disagreements = df[df["label_disagreement"]].copy()

        summary.append({
            "config": config,
            "split": split_name,
            "total": len(df),
            "no_risk": int(df["is_no_risk"].sum()),
            "parse_success": int(df["parse_success"].sum()),
            "disagreements": len(disagreements),
        })

        for _, row in disagreements.iterrows():
            all_disagreements.append({
                "config": config,
                "split": split_name,
                "prompt": row["prompt"],
                "category": row["category"],
                "predicted_risk_types": json.dumps(
                    row["predicted_risk_types"],
                    ensure_ascii=False
                ),
            })

        print(f"Split: {split_name}")
        print(f"Total: {len(df)}")
        print(f"No Risk: {int(df['is_no_risk'].sum())}")
        print(f"Parsed: {int(df['parse_success'].sum())}")
        print(f"Disagreements: {len(disagreements)}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False,
    encoding="utf-8-sig"
)

disagreement_df = pd.DataFrame(all_disagreements)
disagreement_df.to_csv(
    OUTPUT_DIR / "disagreements.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAudit complete.")
print(summary_df.to_string(index=False))
print(f"\nResults saved to: {OUTPUT_DIR}")
