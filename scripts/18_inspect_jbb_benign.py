
from datasets import load_dataset
from collections import Counter

DATASET_ID = "JailbreakBench/JBB-Behaviors"

print("=" * 60)
print("CalibGuard - Step 1.4.1: JBB Benign Audit")
print("=" * 60)

# Correct structure: config=behaviors, split=benign
benign = load_dataset(
    DATASET_ID,
    "behaviors",
    split="benign",
)

harmful = load_dataset(
    DATASET_ID,
    "behaviors",
    split="harmful",
)

print("\nBenign dataset")
print("-" * 50)
print("Rows:", len(benign))
print("Columns:", benign.column_names)

print("\nHarmful dataset")
print("-" * 50)
print("Rows:", len(harmful))
print("Columns:", harmful.column_names)

def normalize(text):
    return " ".join(str(text).split()).casefold()

def extract_prompts(ds):
    if "Goal" in ds.column_names:
        field = "Goal"
    elif "goal" in ds.column_names:
        field = "goal"
    else:
        print("Cannot identify prompt field.")
        return set()

    return {
        normalize(row[field])
        for row in ds
    }

benign_prompts = extract_prompts(benign)
harmful_prompts = extract_prompts(harmful)

print("\nOverlap check")
print("-" * 50)
print("Unique benign prompts:", len(benign_prompts))
print("Unique harmful prompts:", len(harmful_prompts))
print(
    "Benign-Harmful overlap:",
    len(benign_prompts & harmful_prompts)
)

for name, ds in [
    ("Benign", benign),
    ("Harmful", harmful),
]:
    if "Category" in ds.column_names:
        field = "Category"
    elif "category" in ds.column_names:
        field = "category"
    else:
        continue

    print(f"\n{name} category distribution:")

    counts = Counter(str(row[field]) for row in ds)

    for category, count in counts.most_common():
        print(f"  {category}: {count}")

print("\nAudit complete.")
