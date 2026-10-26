
import torch
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "data/splits/calibguard_split_v2.csv"
MODEL_ID = "microsoft/deberta-v3-base"

MAX_LENGTH = 128
BATCH_SIZE = 4

df = pd.read_csv(DATA_PATH, low_memory=False)

train = df[df["split"] == "train"].copy()

assert len(train) == 73365

train["is_safe"] = pd.to_numeric(
    train["is_safe"], errors="raise"
)

assert train["is_safe"].isin([0, 1]).all()

train["unsafe_label"] = 1 - train["is_safe"]

assert (train["unsafe_label"] == 1).sum() == 17001
assert (train["unsafe_label"] == 0).sum() == 56364


class SafetyDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            k: v.squeeze(0)
            for k, v in encoding.items()
        }

        item["labels"] = torch.tensor(
            float(row["unsafe_label"]),
            dtype=torch.float32,
        )

        return item


print("=" * 60)
print("CALIBGUARD BINARY DATA LOADER TEST")
print("=" * 60)

print("Train samples:", len(train))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

dataset = SafetyDataset(train, tokenizer)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(loader))

print("\nBatch tensor shapes:")

for name, value in batch.items():
    print(f"{name}: {tuple(value.shape)}")

print("\nUnsafe labels:", batch["labels"].tolist())

assert batch["labels"].shape == (BATCH_SIZE,)
assert batch["input_ids"].shape == (
    BATCH_SIZE, MAX_LENGTH
)

assert torch.isin(
    batch["labels"],
    torch.tensor([0.0, 1.0])
).all().item()

print("\nBinary data loader test passed.")
