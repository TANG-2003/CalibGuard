
import json
from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/calibguard_canonical.csv")
OUTPUT = Path(
    "results/e0_full_redbench/harmful_eval_leakage.csv"
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)

def parse_json_list(value):
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []

df["config_list"] = df["configs"].apply(parse_json_list)

EVAL_CONFIGS = {"HarmBench", "StrongREJECT"}

def has_eval_config(items):
    return bool(EVAL_CONFIGS.intersection(items))

selected = df[
    df["config_list"].apply(has_eval_config)
].copy()

selected["other_configs"] = selected["config_list"].apply(
    lambda items: sorted(set(items) - EVAL_CONFIGS)
)

selected["has_other_config"] = (
    selected["other_configs"].apply(bool)
)

selected["reserved_benign"] = (
    selected["reserved_hard_eval"].astype(str)
    .str.lower().eq("true")
    |
    selected["reserved_xstest_eval"].astype(str)
    .str.lower().eq("true")
)

print("\nHARMFUL EVAL LEAKAGE AUDIT")
print("=" * 60)

print("Candidate harmful prompts:", len(selected))
print(
    "Also present in other configs:",
    int(selected["has_other_config"].sum())
)
print(
    "Overlapping benign eval:",
    int(selected["reserved_benign"].sum())
)
print(
    "Non-provisional labels:",
    int((selected["label_status"] != "provisional").sum())
)

print("\nOther configuration frequencies:")

from collections import Counter

counter = Counter(
    config
    for configs in selected["other_configs"]
    for config in configs
)

for config, count in counter.most_common():
    print(f"  {config}: {count}")

# Keep only metadata, not original prompt text.
selected[
    [
        "prompt_hash",
        "configs",
        "label_status",
        "is_safe",
        "has_other_config",
        "reserved_benign",
    ]
].to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", OUTPUT)
print("Audit complete.")
