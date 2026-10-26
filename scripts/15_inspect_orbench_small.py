
from collections import Counter
from datasets import load_dataset

DATASET_ID = "bench-llm/or-bench"
CONFIGS = ["or-bench-hard-1k", "or-bench-toxic"]

for config in CONFIGS:
    print("\n" + "=" * 60)
    print(f"Inspecting: {config}")
    print("=" * 60)

    try:
        ds = load_dataset(DATASET_ID, config)

        print("Splits:", list(ds.keys()))

        for split_name, split in ds.items():
            print(f"\nSplit: {split_name}")
            print(f"Rows: {len(split)}")
            print(f"Columns: {split.column_names}")

            counts = Counter(
                str(row["category"])
                for row in split
            )

            print("\nCategory distribution:")
            for category, count in counts.most_common():
                print(f"  {category}: {count}")

            normalized = [
                str(row["prompt"]).strip().casefold()
                for row in split
            ]

            print(
                "Unique normalized prompts:",
                len(set(normalized))
            )
            print(
                "Duplicate rows:",
                len(normalized) - len(set(normalized))
            )

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}"
        )

print("\nInspection complete.")
