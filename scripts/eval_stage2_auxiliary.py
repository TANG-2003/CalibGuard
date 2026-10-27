import sys
import json
import hashlib
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, "scripts")
import formal_train_deberta_multitask as base

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
run = Path(f"results/stage2_full_seed{seed}_6ep_v1")
ckpt = run / "checkpoint_best"

config = json.loads((run / "run_config.json").read_text())
model_config = json.loads((ckpt / "model_config.json").read_text())

df = pd.read_csv("data/splits/calibguard_split_v2.csv", low_memory=False)
train = df.loc[df["split"] == "train"].copy()

safe = pd.to_numeric(train["is_safe"]).astype(int).to_numpy()
_, val_idx = train_test_split(
    range(len(train)),
    test_size=config["val_ratio"],
    random_state=config["split_seed"],
    stratify=safe,
)
val = train.iloc[val_idx].copy()

digest = hashlib.sha256(
    "\n".join(val["prompt_hash"].astype(str)).encode()
).hexdigest()

assert digest == config["validation_hash_sha256"], "Validation split mismatch"

val["risk_categories_parsed"] = val["risk_categories"].map(base.parse_label_list)
val["domains_parsed"] = val["domains"].map(base.parse_label_list)

risk_vocab = model_config["risk_vocab"]
domain_vocab = model_config["domain_vocab"]

dataset = base.MultiTaskDataset(val, risk_vocab, domain_vocab)
tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=False)

loader = DataLoader(
    dataset,
    batch_size=config["eval_batch_size"],
    shuffle=False,
    collate_fn=base.make_collator(tokenizer, config["max_length"]),
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = base.DebertaMultiTaskModel(
    config["model_id"],
    len(risk_vocab),
    len(domain_vocab),
    dropout=model_config["dropout"],
)
state = torch.load(ckpt / "model_state.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.to(device)

metrics, _, _, _ = base.evaluate(
    model, loader, device,
    device.type == "cuda",
    risk_vocab, domain_vocab,
)

assert metrics["evaluated_samples"] == 7337

output = run / "auxiliary_validation_metrics.json"
output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("Binary Unsafe F1:", metrics["unsafe_f1"])
print("Risk auxiliary:", json.dumps(metrics.get("risk_auxiliary"), indent=2))
print("Domain auxiliary:", json.dumps(metrics.get("domain_auxiliary"), indent=2))
print("Saved:", output)
