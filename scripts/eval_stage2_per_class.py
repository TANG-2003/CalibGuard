import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, "scripts")
import formal_train_deberta_multitask as base

DATA = "data/splits/calibguard_split_v2.csv"
df = pd.read_csv(DATA, low_memory=False)
test = df.loc[df["split"] == "test_id"].copy()

assert len(test) == 31444
assert test["prompt_hash"].is_unique

test["risk_categories_parsed"] = test[
    "risk_categories"
].map(base.parse_label_list)

test["domains_parsed"] = test[
    "domains"
].map(base.parse_label_list)


def class_metrics(targets, probabilities, labels):
    predictions = (probabilities >= 0.5).astype(int)
    rows = []

    for j, name in enumerate(labels):
        yt = targets[:, j].astype(int)
        yp = predictions[:, j]

        p, r, f, _ = precision_recall_fscore_support(
            yt, yp,
            average="binary",
            zero_division=0,
        )

        support = int(yt.sum())
        negatives = int(len(yt) - support)

        ap = None
        if support > 0 and negatives > 0:
            ap = float(
                average_precision_score(
                    yt, probabilities[:, j]
                )
            )

        rows.append({
            "class_name": name,
            "labeled_samples": int(len(yt)),
            "support": support,
            "predicted_positive": int(yp.sum()),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "average_precision": ap,
        })

    return rows


for seed in [42, 43, 44]:
    run = Path(f"results/stage2_full_seed{seed}_6ep_v1")
    ckpt = run / "checkpoint_best"
    output = run / "per_class_id_test.json"

    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")

    cfg = json.loads(
        (run / "run_config.json").read_text(encoding="utf-8")
    )
    mc = json.loads(
        (ckpt / "model_config.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run / "run_summary.json").read_text(encoding="utf-8")
    )

    assert cfg["variant"] == "full"
    assert int(cfg["seed"]) == seed
    assert int(summary["epochs_completed"]) == 6
    assert int(mc["epoch"]) == int(summary["best_epoch"])

    risk_vocab = mc["risk_vocab"]
    domain_vocab = mc["domain_vocab"]

    assert len(risk_vocab) == 22
    assert len(domain_vocab) == 19
    assert risk_vocab == cfg["risk_vocab"]
    assert domain_vocab == cfg["domain_vocab"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt), use_fast=False
    )

    dataset = base.MultiTaskDataset(
        test, risk_vocab, domain_vocab
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=base.make_collator(
            tokenizer, cfg["max_length"]
        ),
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = base.DebertaMultiTaskModel(
        cfg["model_id"],
        len(risk_vocab),
        len(domain_vocab),
        dropout=mc["dropout"],
    )

    state = torch.load(
        ckpt / "model_state.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    collected = {
        "risk": {"targets": [], "prob": []},
        "domain": {"targets": [], "prob": []},
    }

    with torch.inference_mode():
        for batch in loader:
            batch = base.move_batch(batch, device)

            batch.pop("binary_labels")

            rt = batch.pop("risk_targets")
            rm = batch.pop("risk_mask")
            dt = batch.pop("domain_targets")
            dm = batch.pop("domain_mask")

            outputs = model(**batch)

            for key, target, mask in [
                ("risk", rt, rm),
                ("domain", dt, dm),
            ]:
                mask = mask.bool().reshape(-1)
                if not mask.any().item():
                    continue

                logits = outputs[f"{key}_logits"]
                prob = torch.sigmoid(logits[mask].float())

                collected[key]["targets"].append(
                    target[mask].cpu().numpy()
                )
                collected[key]["prob"].append(
                    prob.cpu().numpy()
                )

    result = {
        "seed": seed,
        "variant": "full",
        "evaluation_split": "test_id",
        "evaluation_samples": len(test),
        "training_epochs": 6,
        "selected_epoch": summary["best_epoch"],
        "auxiliary_threshold": 0.5,
    }

    for key, vocab in [
        ("risk", risk_vocab),
        ("domain", domain_vocab),
    ]:
        targets = np.concatenate(
            collected[key]["targets"], axis=0
        )
        probabilities = np.concatenate(
            collected[key]["prob"], axis=0
        )

        assert targets.shape == probabilities.shape
        assert targets.shape[1] == len(vocab)
        assert np.isfinite(probabilities).all()

        result[key] = {
            "labeled_samples": int(len(targets)),
            "per_class": class_metrics(
                targets, probabilities, vocab
            ),
        }

    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"SUCCESS seed={seed}: "
        f"risk={result['risk']['labeled_samples']}, "
        f"domain={result['domain']['labeled_samples']}",
        flush=True,
    )

    del model, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("All three seeds finished.")
