
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

INPUT = "data/splits/calibguard_split_v2.csv"
MODEL_ID = "microsoft/deberta-v3-base"

BATCH_SIZE = 256

print("=" * 60)
print("CALIBGUARD - TOKEN LENGTH AUDIT")
print("=" * 60)

df = pd.read_csv(INPUT, low_memory=False)
train = df[df["split"] == "train"].copy()

assert len(train) == 73365
assert train["text"].notna().all()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

texts = train["text"].astype(str).tolist()
lengths = []

print("Counting token lengths...")

for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]

    result = tokenizer(
        batch,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    lengths.extend(
        len(ids) for ids in result["input_ids"]
    )

lengths = np.asarray(lengths, dtype=np.int64)

print("\nSamples:", len(lengths))
print("Mean:", round(float(lengths.mean()), 2))
print("Median:", float(np.median(lengths)))
print("P90:", float(np.percentile(lengths, 90)))
print("P95:", float(np.percentile(lengths, 95)))
print("P99:", float(np.percentile(lengths, 99)))
print("Max:", int(lengths.max()))

print("\nTruncation rates:")

for limit in (128, 256, 512):
    count = int((lengths > limit).sum())
    percent = 100 * count / len(lengths)

    print(
        f"max_length={limit}: "
        f"{count} truncated ({percent:.2f}%)"
    )

print("\nToken length audit complete.")
