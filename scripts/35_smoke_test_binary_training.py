
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.optim import AdamW

MODEL_ID = "microsoft/deberta-v3-base"
DATA_PATH = "data/splits/calibguard_split_v2.csv"

DEVICE = "cpu"
MAX_LENGTH = 64
BATCH_SIZE = 2

torch.manual_seed(42)

print("=" * 60)
print("CALIBGUARD - BINARY TRAINING SMOKE TEST")
print("=" * 60)

print("\nLoading training samples...")

df = pd.read_csv(DATA_PATH, low_memory=False)
train = df[df["split"] == "train"].copy()

train["is_safe"] = pd.to_numeric(
    train["is_safe"], errors="raise"
)

assert len(train) == 73365
assert train["is_safe"].isin([0, 1]).all()

# Use one provisional safe and one provisional unsafe sample.
safe = train[train["is_safe"] == 1].iloc[0]
unsafe = train[train["is_safe"] == 0].iloc[0]

texts = [str(safe["text"]), str(unsafe["text"])]

# unsafe=1, safe=0
labels = torch.tensor(
    [0.0, 1.0],
    dtype=torch.float32,
    device=DEVICE,
)

print("Batch size:", len(texts))
print("Labels:", labels.tolist())

print("\nLoading tokenizer and encoder...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
encoder = AutoModel.from_pretrained(MODEL_ID)

# Freeze encoder for a lightweight CPU-only smoke test.
for parameter in encoder.parameters():
    parameter.requires_grad = False

encoder.eval()

class BinarySafetyModel(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone
        self.classifier = nn.Linear(
            backbone.config.hidden_size, 1
        )

    def forward(self, **inputs):
        with torch.no_grad():
            outputs = self.encoder(**inputs)

        cls_vector = outputs.last_hidden_state[:, 0, :]
        cls_vector = cls_vector.to(
        dtype=self.classifier.weight.dtype
        )
        logits = self.classifier(cls_vector).squeeze(-1)
        return logits


model = BinarySafetyModel(encoder).to(DEVICE)
model.train()

# Keep pretrained encoder in evaluation mode.
model.encoder.eval()
print(
    "Encoder parameter dtype:",
    next(model.encoder.parameters()).dtype
)
print(
    "Classifier dtype:",
    model.classifier.weight.dtype
)
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

encoded = {
    k: v.to(DEVICE)
    for k, v in encoded.items()
}

optimizer = AdamW(
    model.classifier.parameters(),
    lr=1e-3,
)

criterion = nn.BCEWithLogitsLoss()

print("\nRunning one optimization step...")

old_weights = model.classifier.weight.detach().clone()

optimizer.zero_grad()

logits = model(**encoded)
loss = criterion(logits, labels)

assert torch.isfinite(loss).item()

loss.backward()

assert model.classifier.weight.grad is not None

optimizer.step()

weight_change = (
    model.classifier.weight.detach() - old_weights
).abs().sum().item()

print("Logits shape:", tuple(logits.shape))
print("Loss:", round(loss.item(), 6))
print("Classifier weight change:", weight_change)

assert logits.shape == (BATCH_SIZE,)
assert weight_change > 0

print("\nBinary training smoke test passed.")
