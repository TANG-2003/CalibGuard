
from datasets import (
    get_dataset_config_names,
    load_dataset_builder,
)

DATASET_ID = "bench-llm/or-bench"

print("=" * 60)
print("CalibGuard - Step 1.3.1: OR-Bench Metadata Audit")
print("=" * 60)

configs = get_dataset_config_names(DATASET_ID)

print(f"\nConfigurations: {len(configs)}")

total = 0
complete = True

for config in configs:
    builder = load_dataset_builder(
        DATASET_ID,
        name=config,
    )

    print(f"\nConfig: {config}")
    print("Features:", list(builder.info.features))

    if not builder.info.splits:
        print("Split metadata unavailable")
        complete = False
        continue

    for split_name, info in builder.info.splits.items():
        count = info.num_examples
        print(f"  {split_name}: {count}")

        if count is None:
            complete = False
        else:
            total += count

print("\n" + "=" * 60)
print("Metadata complete:", complete)

if complete:
    print("Total records:", total)
else:
    print("Total records: not fully known")

print("\nAudit complete.")
