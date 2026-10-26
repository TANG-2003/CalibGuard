
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

SEED = 42
INPUT = Path("data/processed/calibguard_canonical.csv")
OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)


def parse_bool(series):
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false"}).all():
        raise ValueError("Unexpected evaluation flag values")
    return values.eq("true")


def parse_list(value):
    if pd.isna(value):
        return []
    try:
        result = json.loads(value)
    except (ValueError, TypeError):
        return []
    return result if isinstance(result, list) else []


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("Loading canonical dataset...")

df = pd.read_csv(INPUT, low_memory=False)

required = {
    "prompt_hash", "text", "is_safe", "label_status",
    "configs", "reserved_hard_eval",
    "reserved_xstest_eval"
}
assert required.issubset(df.columns)
assert df["prompt_hash"].notna().all()
assert df["prompt_hash"].is_unique
assert len(df) == 107307

df["is_safe"] = pd.to_numeric(
    df["is_safe"], errors="coerce"
)

hard = parse_bool(df["reserved_hard_eval"])
xstest = parse_bool(df["reserved_xstest_eval"])

configs = df["configs"].apply(parse_list)
harmful = configs.apply(
    lambda x: bool({"HarmBench", "StrongREJECT"} & set(x))
)

# Every prompt receives exactly one partition.
# Isolation rules take priority over evaluation reservations.
df["split"] = "development_candidate"
df.loc[hard, "split"] = "test_orbench_hard"
df.loc[xstest, "split"] = "test_xstest"
df.loc[harmful, "split"] = "test_harmful"

df.loc[
    df["label_status"].eq("unlabeled"),
    "split"
] = "holdout_unlabeled"

df.loc[
    df["label_status"].eq("needs_review"),
    "split"
] = "holdout_conflict"

assert not (hard & xstest).any()
assert not (harmful & (hard | xstest)).any()

assert df.loc[hard, "split"].isin(
    ["test_orbench_hard", "holdout_conflict"]
).all()

assert df.loc[xstest, "split"].isin(
    ["test_xstest", "holdout_conflict"]
).all()

assert df.loc[harmful, "split"].eq(
    "test_harmful"
).all()

dev = df["split"].eq("development_candidate")
assert df.loc[dev, "label_status"].eq("provisional").all()
assert df.loc[dev, "is_safe"].isin([0, 1]).all()

print("Creating 80/10/10 development split...")

# Binary-label stratification with deterministic ordering.
# Note: this does not correct source-label confounding.
for label in (0, 1):
    indices = df.index[dev & df["is_safe"].eq(label)].tolist()
    indices.sort(key=lambda i: df.at[i, "prompt_hash"])

    rng = random.Random(SEED + label)
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * 0.80)
    n_cal = int(n * 0.10)

    train_end = n_train
    cal_end = n_train + n_cal

    df.loc[indices[:train_end], "split"] = "train"
    df.loc[indices[train_end:cal_end], "split"] = "calibration"
    df.loc[indices[cal_end:], "split"] = "test_id"

assert not df["split"].eq("development_candidate").any()

# Check label semantics in the reserved evaluation sets.
assert df.loc[
    df["split"].isin(["test_orbench_hard", "test_xstest"]),
    "is_safe"
].eq(1).all()

assert df.loc[
    df["split"].eq("test_harmful"),
    "is_safe"
].eq(0).all()

# Fixed expected partition sizes from prior audits.
expected = {
    "test_orbench_hard": 1319,
    "test_xstest": 448,
    "test_harmful": 630,
    "holdout_conflict": 2,
    "holdout_unlabeled": 99,
}

counts = df["split"].value_counts()
for name, n in expected.items():
    assert counts.get(name, 0) == n, (
        f"{name}: expected {n}, got {counts.get(name, 0)}"
    )

assert df["prompt_hash"].is_unique
assert len(df) == 107307
assert counts.sum() == len(df)

# Save the reproducible split table.
split_file = OUT / "calibguard_split_v1.csv"
df.to_csv(split_file, index=False, encoding="utf-8-sig")

summary = pd.crosstab(df["split"], df["is_safe"], dropna=False)
summary.columns = [
    "unlabeled" if pd.isna(v)
    else "safe" if v == 1
    else "unsafe" if v == 0
    else str(v)
    for v in summary.columns
]
summary["total"] = summary.sum(axis=1)

summary_file = OUT / "split_v1_summary.csv"
summary.to_csv(summary_file, encoding="utf-8-sig")

# Record configuration counts for each partition.
source_counter = Counter()
for split, config_list in zip(df["split"], configs):
    for config in set(config_list):
        source_counter[(split, config)] += 1

source_file = OUT / "split_v1_source_counts.csv"
pd.DataFrame(
    [
        {"split": split, "config": config, "count": count}
        for (split, config), count in sorted(source_counter.items())
    ]
).to_csv(source_file, index=False, encoding="utf-8-sig")

manifest = {
    "protocol": "CalibGuard split v1",
    "seed": SEED,
    "development_ratio": {
        "train": 0.80,
        "calibration": 0.10,
        "test_id": 0.10
    },
    "label_meaning": "is_safe=1 benign; 0 harmful (provisional)",
    "split_unit": "normalized prompt hash",
    "input_sha256": sha256_file(INPUT),
    "split_sha256": sha256_file(split_file),
    "counts": {k: int(v) for k, v in counts.items()},
    "limitations": [
        "Risk-category-derived labels are provisional",
        "Source-label confounding remains",
        "Near-duplicate leakage not yet audited",
        "Calibration/test exchangeability not established"
    ]
}

manifest_file = OUT / "split_v1_manifest.json"
manifest_file.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("CALIBGUARD SPLIT V1 - 80/10/10")
print("=" * 65)
print(summary.to_string())
print("\nTotal prompts:", len(df))
print("Random seed:", SEED)
print("\nSaved:")
print(split_file)
print(summary_file)
print(source_file)
print(manifest_file)
print("\nAll split integrity checks passed.")
