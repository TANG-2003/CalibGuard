
from transformers import AutoTokenizer

MODEL_ID = "microsoft/deberta-v3-base"

print("=" * 60)
print("CalibGuard - DeBERTa Tokenizer Test")
print("=" * 60)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

print("Tokenizer class:", type(tokenizer).__name__)
print("Vocabulary size:", len(tokenizer))

texts = [
    "How can I learn Python programming?",
    "Can you explain the basic principles of network security?",
]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

print("\nEncoded tensor shapes:")

for key, value in encoded.items():
    print(f"{key}: {tuple(value.shape)}")

decoded = tokenizer.decode(
    encoded["input_ids"][0],
    skip_special_tokens=True,
)

print("\nDecoded sample:")
print(decoded)

assert len(encoded["input_ids"]) == len(texts)
assert encoded["input_ids"].shape[1] <= 128

print("\nTokenizer test passed.")
