
import csv
from pathlib import Path

from datasets import (
    get_dataset_config_names,
    load_dataset_builder,
)

DATASET_ID = "knoveleng/redbench"
OUTPUT_DIR = Path("results/e0_full_redbench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CalibGuard - Step 1.1: Full RedBench Metadata Audit")
print("=" * 65)

configs = get_dataset_config_names(DATASET_ID)
print(f"\nTotal configurations: {len(configs)}")

records = []
known_total = 0
unknown_count = 0

for index, config in enumerate(configs, start=1):
    try:
        builder = load_dataset_builder(
            DATASET_ID,
            name=config,
        )

        splits = builder.info.splits

        if not splits:
            records.append({
                "config": config,
                "split": "",
                "num_examples": "",
                "status": "unknown",
            })
            unknown_count += 1
            print(
                f"[{index:02d}/{len(configs)}] "
                f"{config}: split metadata unavailable"
            )
            continue

        for split_name, split_info in splits.items():
            count = split_info.num_examples

            if count is None:
                unknown_count += 1
                status = "unknown"
            else:
                known_total += count
                status = "ok"

            records.append({
                "config": config,
                "split": split_name,
                "num_examples": (
                    count if count is not None else ""
                ),
                "status": status,
            })

            print(
                f"[{index:02d}/{len(configs)}] "
                f"{config} / {split_name}: "
                f"{count if count is not None else 'UNKNOWN'}"
            )

    except Exception as exc:
        records.append({
            "config": config,
            "split": "",
            "num_examples": "",
            "status": "error",
        })

        print(
            f"[{index:02d}/{len(configs)}] "
            f"{config}: ERROR - "
            f"{type(exc).__name__}: {exc}"
        )

output_path = OUTPUT_DIR / "redbench_metadata_audit.csv"

with output_path.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "config",
            "split",
            "num_examples",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(records)

errors = sum(
    record["status"] == "error"
    for record in records
)

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Configurations: {len(configs)}")
print(f"Known example count: {known_total}")
print(f"Unknown split counts: {unknown_count}")
print(f"Configuration errors: {errors}")

if unknown_count == 0 and errors == 0:
    print("Metadata total is complete.")
else:
    print(
        "Metadata total is incomplete. "
        "Do not treat known count as dataset total."
    )

print(f"\nSaved: {output_path}")
