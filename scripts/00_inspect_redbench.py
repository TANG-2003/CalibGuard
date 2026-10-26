
from datasets import get_dataset_config_names

DATASET_ID = "knoveleng/redbench"

print("=" * 60)
print("CalibGuard E0 - RedBench Dataset Inspection")
print("=" * 60)

try:
    configs = get_dataset_config_names(DATASET_ID)

    print(f"\nDataset: {DATASET_ID}")
    print(f"Number of configurations: {len(configs)}")

    for i, config in enumerate(configs, start=1):
        print(f"{i:02d}. {config}")

except Exception as exc:
    print(f"\nInspection failed: {type(exc).__name__}")
    print(str(exc))
    raise
