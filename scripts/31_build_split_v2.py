
import json
import random
import hashlib
from pathlib import Path

import pandas as pd

SEED = 42

INPUT = Path("data/processed/calibguard_canonical.csv")
OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)


def parse_bool(series):
    values = series.astype(str).str.strip().str.lower()
    assert values.isin(["true", "false"]).all()
    return values.eq("true")


def parse_configs(value):
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("Loading canonical dataset...")

df = pd.read_csv(INPUT, low_memory=False)

assert len(df) == 107307
assert df["prompt_hash"].is_unique

df["is_safe"] = pd.to_numeric(
    df["is_safe"], errors="coerce"
)

hard = parse_bool(df["reserved_hard_eval"])
xstest = parse_bool(df["reserved_xstest_eval"])

configs = df["configs"].apply(parse_configs)

harmful = configs.apply(
    lambda items: bool(
        {"HarmBench", "StrongREJECT"} & set(items)
    )
)

# Reserve external evaluation sets by unique prompt.
df["split"] = "development_candidate"

df.loc[hard, "split"] = "test_orbench_hard"
df.loc[xstest, "split"] = "test_xstest"
df.loc[harmful, "split"] = "test_harmful"

# Uncertain labels override evaluation reservations.
df.loc[
    df["label_status"] == "unlabeled",
    "split"
] = "holdout_unlabeled"

df.loc[
    df["label_status"] == "needs_review",
    "split"
] = "holdout_conflict"

assert not (hard & xstest).any()
assert not (harmful & (hard | xstest)).any()

dev = df["split"] == "development_candidate"

assert df.loc[dev, "label_status"].eq(
    "provisional"
).all()

assert df.loc[dev, "is_safe"].isin([0, 1]).all()

print("Building 70/30 split...")

# Split by binary label and normalized prompt hash.
for label in (0, 1):
    indices = df.index[
        dev & df["is_safe"].eq(label)
    ].tolist()

    indices.sort(
        key=lambda i: df.at[i, "prompt_hash"]
    )

    rng = random.Random(SEED + label)
    rng.shuffle(indices)

    n_train = int(len(indices) * 0.70)

    df.loc[indices[:n_train], "split"] = "train"
    df.loc[indices[n_train:], "split"] = "test_id"

# Integrity checks
assert not df["split"].eq(
    "development_candidate"
).any()

assert df.loc[hard, "split"].isin(
    ["test_orbench_hard", "holdout_conflict"]
).all()

assert df.loc[xstest, "split"].isin(
    ["test_xstest", "holdout_conflict"]
).all()

assert df.loc[harmful, "split"].eq(
    "test_harmful"
).all()

assert df["prompt_hash"].is_unique
assert len(df) == 107307

expected = {
    "test_orbench_hard": 1319,
    "test_xstest": 448,
    "test_harmful": 630,
    "holdout_conflict": 2,
    "holdout_unlabeled": 99,
}

counts = df["split"].value_counts()

for name, count in expected.items():
    assert counts.get(name, 0) == count, (
        f"{name} count mismatch"
    )

# Save V2; preserve V1 unchanged.
OUTPUT = OUT / "calibguard_split_v2.csv"

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

summary = pd.crosstab(
    df["split"],
    df["is_safe"],
    dropna=False
)

summary.columns = [
    "unlabeled" if pd.isna(c)
    else "safe" if c == 1
    else "unsafe" if c == 0
    else str(c)
    for c in summary.columns
]

summary["total"] = summary.sum(axis=1)

summary.to_csv(
    OUT / "split_v2_summary.csv",
    encoding="utf-8-sig"
)

manifest = {
    "protocol": "CalibGuard Train/Test V2",
    "seed": SEED,
    "development_ratio": {
        "train": 0.70,
        "test_id": 0.30
    },
    "input_sha256": file_hash(INPUT),
    "output_sha256": file_hash(OUTPUT),
    "split_counts": {
        k: int(v)
        for k, v in counts.items()
    },
    "notes": [
        "No standalone validation split",
        "External evaluation sets reserved",
        "Labels remain provisional",
        "Calibration must not use final test labels",
        "Near-duplicate leakage remains unaudited",
        "Source-label confounding remains"
    ]
}

(OUT / "split_v2_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("CALIBGUARD SPLIT V2 - TRAIN/TEST 70/30")
print("=" * 65)

print(summary.to_string())

print("\nTotal prompts:", len(df))
print("Random seed:", SEED)
print("Saved:", OUTPUT)
print("All V2 split checks passed.")
