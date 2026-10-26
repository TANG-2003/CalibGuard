
from collections import Counter

from datasets import load_dataset

DATASET_ID = "knoveleng/redbench"
CONFIG = "SGXSTest"

print("=" * 60)
print("Step 1.2.1 - SGXSTest Label Inspection")
print("=" * 60)

ds = load_dataset(
    DATASET_ID,
    CONFIG,
    split="train",
)

print(f"\nTotal samples: {len(ds)}")
print(f"Fields: {ds.column_names}")

for field in [
    "category",
    "domain",
    "source",
    "task",
    "subtask",
]:
    if field not in ds.column_names:
        continue

    counts = Counter(
        (
            type(row[field]).__name__,
            repr(row[field])
        )
        for row in ds
    )

    print(f"\n{field}:")
    for (value_type, value_repr), count in counts.most_common(10):
        print(
            f"  type={value_type}, "
            f"value={value_repr}, "
            f"count={count}"
        )

print("\nFirst three records (metadata only):")

for index in range(min(3, len(ds))):
    row = ds[index]
    print(f"\nRecord {index + 1}")

    for field in [
        "category",
        "domain",
        "source",
        "task",
        "subtask",
    ]:
        print(f"  {field}: {repr(row.get(field))}")

print("\nInspection complete.")
