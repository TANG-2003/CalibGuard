
from datasets import load_dataset_builder

DATASET_ID = "knoveleng/redbench"

CONFIGS = [
    "ORBench",
    "XSTest",
    "CoCoNot",
    "JBBBehaviours",
    "HarmBench",
    "MedSafetyBench",
]

print("=" * 65)
print("CalibGuard E0 - Dataset Configuration Audit")
print("=" * 65)

for config in CONFIGS:
    print(f"\n{'=' * 65}")
    print(f"CONFIG: {config}")
    print("=" * 65)

    try:
        builder = load_dataset_builder(
            DATASET_ID,
            name=config,
        )

        print("\nFeatures:")
        print(builder.info.features)

        print("\nSplits:")
        print(builder.info.splits)

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")

print("\nInspection finished.")
