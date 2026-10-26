
from pathlib import Path
from datasets import load_dataset
import pandas as pd

DATASET_ID = "JailbreakBench/JBB-Behaviors"

REDBENCH_FILE = Path(
    "data/processed/redbench_full_mapped.csv"
)

OUTPUT = Path(
    "results/e0_full_redbench/jbb_overlap_summary.csv"
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def normalize(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).casefold()


print("Loading JBB Benign and Harmful...")

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

jbb_benign = {
    normalize(row["Goal"])
    for row in benign
}

jbb_harmful = {
    normalize(row["Goal"])
    for row in harmful
}

print("Reading local RedBench mapping...")

redbench = pd.read_csv(
    REDBENCH_FILE,
    usecols=["text", "is_safe", "config"],
)

redbench["normalized"] = (
    redbench["text"].apply(normalize)
)

red_all = set(redbench["normalized"])

red_benign = set(
    redbench.loc[
        redbench["is_safe"] == 1,
        "normalized",
    ]
)

red_harmful = set(
    redbench.loc[
        redbench["is_safe"] == 0,
        "normalized",
    ]
)

red_jbb = set(
    redbench.loc[
        redbench["config"] == "JBBBehaviours",
        "normalized",
    ]
)

print("Loading OR-Bench 80K...")

orbench = load_dataset(
    "bench-llm/or-bench",
    "or-bench-80k",
    split="train",
)

or80k = {
    normalize(row["prompt"])
    for row in orbench
}

print("\n" + "=" * 60)
print("JBB CROSS-DATASET OVERLAP AUDIT")
print("=" * 60)

results = {
    "jbb_benign_count": len(jbb_benign),
    "jbb_harmful_count": len(jbb_harmful),
    "benign_vs_redbench": len(jbb_benign & red_all),
    "harmful_vs_redbench": len(jbb_harmful & red_all),
    "harmful_vs_redbench_jbb": len(
        jbb_harmful & red_jbb
    ),
    "benign_vs_orbench_80k": len(
        jbb_benign & or80k
    ),
    "harmful_vs_orbench_80k": len(
        jbb_harmful & or80k
    ),
    "benign_vs_redbench_harmful": len(
        jbb_benign & red_harmful
    ),
    "harmful_vs_redbench_benign": len(
        jbb_harmful & red_benign
    ),
    "benign_vs_jbb_harmful": len(
        jbb_benign & jbb_harmful
    ),
}

for key, value in results.items():
    print(f"{key}: {value}")

pd.DataFrame([results]).to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nSaved: {OUTPUT}")
print("Audit complete.")
