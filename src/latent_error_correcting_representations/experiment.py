from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import statistics
import torch

from .codes import METHODS, encode, decode, task_label
from .corruption import flip_exactly_k, adversarial_single_flip, nearest_opposite_valid_codeword
from .data import make_dataset
from .metrics import evaluate
from .model import MessageEncoder
from .seed import seed_everything
from .training import train_encoder, predict_messages


def run_seed(seed: int, out_dir: str | Path, epochs: int = 30) -> dict:
    seed_everything(seed)
    data = make_dataset(seed)
    model = MessageEncoder(input_dim=data.projection.shape[0])
    history = train_encoder(model, data.train, seed=seed, epochs=epochs)
    pred_message, true_message, true_y = predict_messages(model, data.test)

    encoder_bit_acc = float((pred_message == true_message).float().mean().item())
    encoder_task_acc = float((task_label(pred_message) == true_y).float().mean().item())

    result: dict = {
        "seed": seed,
        "encoder": {
            "final_train_loss": float(history[-1]),
            "message_bit_accuracy": encoder_bit_acc,
            "task_accuracy": encoder_task_acc,
        },
        "random_corruption": {},
        "adversarial_single_flip": {},
        "coherent_drift": {},
    }

    message = pred_message
    labels = task_label(message)

    for method in METHODS:
        codeword = encode(method, message)
        per_k = {}
        for k in (0, 1, 2):
            g = torch.Generator().manual_seed(seed * 1000 + k * 31 + len(method))
            corrupted = flip_exactly_k(codeword, k, g)
            decoded = decode(method, corrupted)
            metrics = evaluate(decoded.message, message, decoded.detected, decoded.corrected)
            per_k[str(k)] = asdict(metrics)
        result["random_corruption"][method] = per_k

        adv = adversarial_single_flip(method, codeword, labels)
        adv_dec = decode(method, adv)
        adv_metrics = evaluate(adv_dec.message, message, adv_dec.detected, adv_dec.corrected)
        result["adversarial_single_flip"][method] = asdict(adv_metrics)

        drifted = nearest_opposite_valid_codeword(method, message)
        drift_dec = decode(method, drifted)
        drift_metrics = evaluate(drift_dec.message, message, drift_dec.detected, drift_dec.corrected)
        result["coherent_drift"][method] = asdict(drift_metrics)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def summarize(results: list[dict]) -> dict:
    summary: dict = {"seeds": [r["seed"] for r in results], "encoder": {}, "random_corruption": {}, "adversarial_single_flip": {}, "coherent_drift": {}}
    for key in ("message_bit_accuracy", "task_accuracy"):
        summary["encoder"][key] = _mean_sd([r["encoder"][key] for r in results])

    for method in METHODS:
        summary["random_corruption"][method] = {}
        for k in ("0", "1", "2"):
            summary["random_corruption"][method][k] = {}
            for metric in ("message_bit_accuracy", "task_accuracy", "detection_rate", "correction_rate"):
                summary["random_corruption"][method][k][metric] = _mean_sd([
                    r["random_corruption"][method][k][metric] for r in results
                ])
        summary["adversarial_single_flip"][method] = {}
        summary["coherent_drift"][method] = {}
        for metric in ("message_bit_accuracy", "task_accuracy", "detection_rate", "correction_rate"):
            summary["adversarial_single_flip"][method][metric] = _mean_sd([
                r["adversarial_single_flip"][method][metric] for r in results
            ])
            summary["coherent_drift"][method][metric] = _mean_sd([
                r["coherent_drift"][method][metric] for r in results
            ])
    return summary
