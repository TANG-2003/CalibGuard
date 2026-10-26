
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "microsoft/deberta-v3-base"

print("=" * 60)
print("CalibGuard - Step 2.5: DeBERTa Encoder Test")
print("=" * 60)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

print("Loading pretrained encoder...")
model = AutoModel.from_pretrained(MODEL_ID)

model.eval()

print("\nModel class:", type(model).__name__)
print("Hidden size:", model.config.hidden_size)
print("Number of layers:", model.config.num_hidden_layers)

texts = [
    "How can I learn Python programming?",
    "Can you explain basic principles of network security?",
]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

print("\nRunning CPU forward pass...")

with torch.inference_mode():
    outputs = model(**encoded)

hidden = outputs.last_hidden_state

print("\nLast hidden state:", tuple(hidden.shape))

assert hidden.shape[0] == len(texts)
assert hidden.shape[-1] == model.config.hidden_size

print("\nEncoder forward test passed.")
