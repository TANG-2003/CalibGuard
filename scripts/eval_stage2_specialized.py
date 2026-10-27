import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, "scripts")
import formal_train_deberta_multitask as base

SPLITS = {
    "test_orbench_hard": (1319, 0),
    "test_xstest": (448, 0),
    "test_harmful": (630, 1),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    required=True,
    choices=["binary", "risk", "domain", "full"],
)
parser.add_argument(
    "--seed",
    required=True,
    type=int,
    choices=[42, 43, 44],
)
args = parser.parse_args()

run = Path(
    f"results/stage2_{args.variant}_seed{args.seed}_6ep_v1"
)
ckpt = run / "checkpoint_best"
out_path = run / "specialized_metrics.json"

if out_path.exists():
    raise FileExistsError(
        f"Existing evaluation result: {out_path}"
    )

config = json.loads(
    (run / "run_config.json").read_text(encoding="utf-8")
)
model_config = json.loads(
    (ckpt / "model_config.json").read_text(encoding="utf-8")
)
summary = json.loads(
    (run / "run_summary.json").read_text(encoding="utf-8")
)

assert config["variant"] == args.variant
assert int(config["seed"]) == args.seed
assert int(config["split_seed"]) == 42
assert int(summary["epochs_completed"]) == 6
assert int(model_config["epoch"]) == int(summary["best_epoch"])

risk_vocab = model_config["risk_vocab"]
domain_vocab = model_config["domain_vocab"]

assert risk_vocab == config["risk_vocab"]
assert domain_vocab == config["domain_vocab"]
assert len(risk_vocab) == 22
assert len(domain_vocab) == 19

df = pd.read_csv(config["data_path"], low_memory=False)

# Audit the three reserved evaluation subsets.
all_hashes = []
for split, (expected_n, expected_label) in SPLITS.items():
    part = df.loc[df["split"] == split]
    if len(part) != expected_n:
        raise ValueError(f"Incorrect sample count: {split}")

    safe = pd.to_numeric(part["is_safe"], errors="raise")
    expected_safe = 1 - expected_label
    if not (safe == expected_safe).all():
        raise ValueError(f"Unexpected labels: {split}")

    if part["prompt_hash"].duplicated().any():
        raise ValueError(f"Duplicate hashes: {split}")

    all_hashes.extend(part["prompt_hash"].astype(str).tolist())

assert len(all_hashes) == len(set(all_hashes))

tokenizer = AutoTokenizer.from_pretrained(
    str(ckpt), use_fast=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
model.eval()

results = {}

for split, (expected_n, expected_label) in SPLITS.items():
    part = df.loc[df["split"] == split].copy()

    part["risk_categories_parsed"] = part[
        "risk_categories"
    ].map(base.parse_label_list)
    part["domains_parsed"] = part[
        "domains"
    ].map(base.parse_label_list)

    dataset = base.MultiTaskDataset(
        part, risk_vocab, domain_vocab
    )
    loader = DataLoader(
        dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=base.make_collator(
            tokenizer, config["max_length"]
        ),
        num_workers=0,
    )

    y_all = []
    prob_all = []

    with torch.inference_mode():
        for batch in loader:
            batch = base.move_batch(batch, device)

            y = batch.pop("binary_labels")
            batch.pop("risk_targets")
            batch.pop("risk_mask")
            batch.pop("domain_targets")
            batch.pop("domain_mask")

            with base.autocast_ctx(
                device, device.type == "cuda"
            ):
                outputs = model(**batch)

            probabilities = torch.softmax(
                outputs["binary_logits"].float(),
                dim=-1,
            )[:, 1]

            y_all.extend(y.cpu().tolist())
            prob_all.extend(
                probabilities.cpu().tolist()
            )

    y_all = np.asarray(y_all, dtype=np.int64)
    prob_all = np.asarray(prob_all, dtype=np.float64)
    predictions = (prob_all >= 0.5).astype(np.int64)

    assert len(y_all) == expected_n
    assert np.all(y_all == expected_label)
    assert np.isfinite(prob_all).all()

    predicted_unsafe = int(predictions.sum())
    predicted_safe = expected_n - predicted_unsafe

    results[split] = {
        "sample_count": expected_n,
        "true_binary_label": expected_label,
        "threshold": 0.5,
        "predicted_unsafe_count": predicted_unsafe,
        "predicted_safe_count": predicted_safe,
        "unsafe_prediction_rate": float(
            predicted_unsafe / expected_n
        ),
        "safe_prediction_rate": float(
            predicted_safe / expected_n
        ),
        "mean_unsafe_score": float(prob_all.mean()),
        "prompt_hash_sha256": hashlib.sha256(
            "\n".join(
                part["prompt_hash"].astype(str)
            ).encode()
        ).hexdigest(),
    }

    if expected_label == 0:
        results[split]["safe_fpr"] = float(
            predicted_unsafe / expected_n
        )
    else:
        results[split]["unsafe_recall"] = float(
            predicted_unsafe / expected_n
        )

    rate_name = (
        "Safe FPR"
        if expected_label == 0
        else "Unsafe Recall"
    )
    rate = (
        results[split]["safe_fpr"]
        if expected_label == 0
        else results[split]["unsafe_recall"]
    )

    print(
        f"{split}: n={expected_n}, "
        f"{rate_name}={rate:.6f}",
        flush=True,
    )

record = {
    "variant": args.variant,
    "seed": args.seed,
    "training_epochs": 6,
    "selected_epoch": summary["best_epoch"],
    "checkpoint": "checkpoint_best",
    "decision_threshold": 0.5,
    "results": results,
}

out_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("SUCCESS:", args.variant, args.seed)
print("Saved:", out_path)
