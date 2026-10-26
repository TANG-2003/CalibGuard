
from pathlib import Path
import json

import pandas as pd

INPUT = Path("data/processed/calibguard_canonical.csv")
OUTPUT_DIR = Path("results/e0_full_redbench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)

df["is_safe"] = pd.to_numeric(
    df["is_safe"], errors="coerce"
)

def parse_configs(value):
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []

df["config_list"] = df["configs"].apply(parse_configs)

candidates = ["HarmBench", "StrongREJECT"]
candidate_sets = {}

print("\nHARMFUL EVALUATION CANDIDATE AUDIT")
print("=" * 65)

for config in candidates:
    mask = df["config_list"].apply(
        lambda items: config in items
    )

    selected = df[mask]
    candidate_sets[config] = set(selected["prompt_hash"])

    print(f"\nConfig: {config}")
    print("Unique prompts:", len(selected))
    print("Candidate unsafe:", int(
        (selected["is_safe"] == 0).sum()
    ))
    print("Candidate safe:", int(
        (selected["is_safe"] == 1).sum()
    ))
    print("Unlabeled/conflict:", int(
        selected["is_safe"].isna().sum()
    ))

    print("Already reserved for benign eval:", int(
        (
            selected["reserved_hard_eval"].astype(str)
            .str.lower().eq("true")
            |
            selected["reserved_xstest_eval"].astype(str)
            .str.lower().eq("true")
        ).sum()
    ))

    print("Label status:")
    print(selected["label_status"].value_counts().to_string())

overlap = (
    candidate_sets["HarmBench"]
    & candidate_sets["StrongREJECT"]
)

print("\n" + "=" * 65)
print("Overlap between harmful candidates:", len(overlap))
print(
    "Combined unique prompts:",
    len(candidate_sets["HarmBench"] |
        candidate_sets["StrongREJECT"])
)

summary = pd.DataFrame([
    {
        "config": config,
        "unique_prompts": len(candidate_sets[config]),
    }
    for config in candidates
])

summary.to_csv(
    OUTPUT_DIR / "harmful_eval_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAudit complete.")
