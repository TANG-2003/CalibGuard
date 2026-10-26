
from collections import defaultdict
from hashlib import sha256

from datasets import load_dataset

CONFIGS = [
    "ORBench",
    "XSTest",
    "CoCoNot",
    "JBBBehaviours",
    "HarmBench",
    "MedSafetyBench",
]

DATASET_ID = "knoveleng/redbench"
groups = defaultdict(list)

for config in CONFIGS:
    ds = load_dataset(DATASET_ID, config, split="train")

    for row in ds:
        prompt = str(row["prompt"]).strip().casefold()
        label = str(row["category"]).strip()

        groups[prompt].append({
            "config": config,
            "label": label,
        })

print("\nCross-dataset overlap audit")
print("=" * 55)

duplicate_groups = {
    prompt: records
    for prompt, records in groups.items()
    if len(records) > 1
}

cross_dataset = {
    prompt: records
    for prompt, records in duplicate_groups.items()
    if len({r["config"] for r in records}) > 1
}

binary_conflicts = {
    prompt: records
    for prompt, records in duplicate_groups.items()
    if len({
        r["label"] == "No Risk"
        for r in records
    }) > 1
}

print(f"Unique normalized prompts: {len(groups)}")
print(f"Duplicate prompt groups: {len(duplicate_groups)}")
print(f"Cross-dataset overlap groups: {len(cross_dataset)}")
print(f"Binary label conflict groups: {len(binary_conflicts)}")

print("\nCross-dataset overlap details:")

for prompt, records in cross_dataset.items():
    digest = sha256(prompt.encode("utf-8")).hexdigest()[:12]
    print(f"\nPrompt hash: {digest}")

    for record in records:
        print(
            f"  {record['config']}: "
            f"{record['label']}"
        )

print("\nAudit complete.")
