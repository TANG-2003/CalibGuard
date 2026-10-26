
from collections import Counter
from datasets import load_dataset
import pandas as pd
from pathlib import Path

DATASET_ID = "knoveleng/redbench"

CONFIGS = [
    "ORBench",
    "XSTest",
    "CoCoNot",
    "JBBBehaviours",
    "HarmBench",
    "MedSafetyBench",
]

OUTPUT_DIR = Path("results/e0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_stats = []
all_frames = []

for config in CONFIGS:
    print(f"\n{'=' * 60}")
    print(f"Loading: {config}")
    print("=" * 60)

    try:
        ds = load_dataset(
            DATASET_ID,
            name=config,
            split="train",
        )
        df = ds.to_pandas()

        print("Rows:", len(df))
        print("Columns:", list(df.columns))

        for field in [
            "category",
            "domain",
            "task",
            "subtask",
            "risk_response",
            "risk_property",
            "domain_response",
            "domain_property",
        ]:
            if field in df.columns:
                print(f"\n{field} distribution:")
                print(
                    df[field]
                    .fillna("<MISSING>")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .to_string()
                )

        if "prompt" in df.columns:
            print("\nMissing prompts:",
                  df["prompt"].isna().sum())
            print("Duplicate prompts:",
                  df["prompt"].duplicated().sum())

        # Only print limited text to inspect actual schema.
        for i, row in df.head(2).iterrows():
            print(f"\nSample {i}:")
            for field in [
                "prompt", "category", "domain",
                "task", "subtask", "source",
                "risk_response", "risk_property",
            ]:
                value = row.get(field)
                print(f"  {field}: {str(value)[:160]}")

        stats = {
            "config": config,
            "rows": len(df),
            "unique_prompts": (
                df["prompt"].nunique()
                if "prompt" in df.columns else None
            ),
            "missing_category": df["category"].isna().sum(),
            "missing_domain": df["domain"].isna().sum(),
        }
        all_stats.append(stats)

        df["source_config"] = config
        all_frames.append(df)

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")

if all_stats:
    pd.DataFrame(all_stats).to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

if all_frames:
    combined = pd.concat(
        all_frames, ignore_index=True
    )
    print("\n" + "=" * 60)
    print("COMBINED AUDIT")
    print("=" * 60)
    print("Total rows:", len(combined))
    print(
        "Unique prompts:",
        combined["prompt"].nunique()
    )
    print(
        "Duplicate prompt rows:",
        combined["prompt"].duplicated().sum()
    )

print("\nAudit finished.")
