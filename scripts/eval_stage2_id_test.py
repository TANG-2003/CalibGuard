import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, "scripts")
import formal_train_deberta_multitask as base

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["binary", "risk", "domain", "full"], required=True)
parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
args = parser.parse_args()

run = Path(f"results/stage2_{args.variant}_seed{args.seed}_6ep_v1")
ckpt = run / "checkpoint_best"
out = run / "id_test_metrics.json"

if out.exists():
    raise RuntimeError(f"Existing test result; refusing overwrite: {out}")

config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
model_config = json.loads((ckpt / "model_config.json").read_text(encoding="utf-8"))
summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))

assert summary["epochs_completed"] == 6
assert summary["id_test_was_evaluated"] is False
assert config["seed"] == args.seed
assert config["variant"] == args.variant
assert config["split_seed"] == 42
assert model_config["epoch"] == summary["best_epoch"]

risk_vocab = model_config["risk_vocab"]
domain_vocab = model_config["domain_vocab"]

assert risk_vocab == config["risk_vocab"]
assert domain_vocab == config["domain_vocab"]
assert len(risk_vocab) == 22
assert len(domain_vocab) == 19

df = pd.read_csv(config["data_path"], low_memory=False)
test = df.loc[df["split"] == "test_id"].copy()

assert len(test) == 31444
assert test["prompt_hash"].is_unique

test["risk_categories_parsed"] = test["risk_categories"].map(base.parse_label_list)
test["domains_parsed"] = test["domains"].map(base.parse_label_list)

dataset = base.MultiTaskDataset(test, risk_vocab, domain_vocab)
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

state = torch.load(
    ckpt / "model_state.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state, strict=True)
model.to(device)

metrics, y, pred, prob = base.evaluate(
    model, loader, device,
    device.type == "cuda",
    risk_vocab, domain_vocab,
    max_batches=0,
)

assert metrics["evaluated_samples"] == 31444
assert metrics["safe_samples"] == 24157
assert metrics["unsafe_samples"] == 7287
assert metrics["threshold"] == 0.5

# Only report auxiliary heads that received task-specific supervision.
if args.variant == "binary":
    metrics.pop("risk_auxiliary", None)
    metrics.pop("domain_auxiliary", None)
elif args.variant == "risk":
    metrics.pop("domain_auxiliary", None)
elif args.variant == "domain":
    metrics.pop("risk_auxiliary", None)

result = {
    "variant": args.variant,
    "seed": args.seed,
    "training_epochs": 6,
    "selected_epoch": summary["best_epoch"],
    "evaluation_split": "test_id",
    "evaluation_data": "frozen_v2",
    "model_selection": "validation_unsafe_f1",
    "metrics": metrics,
}

out.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"SUCCESS: {args.variant}, seed={args.seed}")
print(f"Best Epoch: {summary['best_epoch']}")
print(f"Test Unsafe F1: {metrics['unsafe_f1']:.6f}")
print(f"Test Unsafe Recall: {metrics['unsafe_recall']:.6f}")
print(f"Test Safe FPR: {metrics['safe_false_positive_rate_over_refusal_proxy']:.6f}")
print(f"Saved: {out}")
