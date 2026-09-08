from __future__ import annotations

from dataclasses import asdict
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import statistics
import numpy as np
import torch

from .codes import METHODS, encode, decode, task_label
from .corruption import flip_exactly_k, adversarial_single_flip, nearest_opposite_valid_codeword
from .data import make_dataset
from .metrics import evaluate
from .model import MessageEncoder
from .seed import seed_everything, validate_seed
from .training import train_encoder, predict_messages


def run_seed(seed: int, out_dir: str | Path, epochs: int = 30) -> dict:
    seed = validate_seed(seed)
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    epochs = int(epochs)
    data_options = dict(n_train=2500, n_test=800, input_dim=16, noise_std=0.35)
    model_options = dict(input_dim=data_options["input_dim"], hidden_dim=48)
    training_options = dict(epochs=epochs, batch_size=256, lr=2e-3, device="cpu")
    seed_everything(seed)
    data = make_dataset(seed, **data_options)
    model = MessageEncoder(**model_options)
    history = train_encoder(model, data.train, seed=seed, **training_options)
    pred_message, true_message, true_y = predict_messages(model, data.test)

    encoder_bit_acc = float((pred_message == true_message).float().mean().item())
    encoder_task_acc = float((task_label(pred_message) == true_y).float().mean().item())

    result: dict = {
        "seed": seed,
        "metadata": {
            "schema_version": 2,
            "metric_reference": "predicted_message_before_corruption",
            "end_to_end_reference": "ground_truth_message",
            "configuration": {
                "data": data_options,
                "model": model_options,
                "training": training_options,
                "exact_bit_flips": [0, 1, 2],
                "methods": list(METHODS),
            },
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        },
        "encoder": {
            "final_train_loss": float(history[-1]),
            "message_bit_accuracy": encoder_bit_acc,
            "task_accuracy": encoder_task_acc,
        },
        "random_corruption": {},
        "adversarial_single_flip": {},
        "coherent_drift": {},
        "end_to_end": {"random_corruption": {}, "adversarial_single_flip": {}, "coherent_drift": {}},
    }

    message = pred_message
    labels = task_label(message)

    for method in METHODS:
        codeword = encode(method, message)
        per_k = {}
        end_to_end_per_k = {}
        for k in (0, 1, 2):
            g = torch.Generator().manual_seed(seed * 1000 + k * 31 + len(method))
            corrupted = flip_exactly_k(codeword, k, g)
            decoded = decode(method, corrupted)
            metrics = evaluate(decoded.message, message, decoded.detected, decoded.corrected)
            per_k[str(k)] = asdict(metrics)
            end_to_end_per_k[str(k)] = _task_accuracy(decoded, true_message)
        result["random_corruption"][method] = per_k
        result["end_to_end"]["random_corruption"][method] = end_to_end_per_k

        adv = adversarial_single_flip(method, codeword, labels)
        adv_dec = decode(method, adv)
        adv_metrics = evaluate(adv_dec.message, message, adv_dec.detected, adv_dec.corrected)
        result["adversarial_single_flip"][method] = asdict(adv_metrics)
        result["end_to_end"]["adversarial_single_flip"][method] = _task_accuracy(adv_dec, true_message)

        drifted = nearest_opposite_valid_codeword(method, message)
        drift_dec = decode(method, drifted)
        drift_metrics = evaluate(drift_dec.message, message, drift_dec.detected, drift_dec.corrected)
        result["coherent_drift"][method] = asdict(drift_metrics)
        result["end_to_end"]["coherent_drift"][method] = _task_accuracy(drift_dec, true_message)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed_{seed}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _task_accuracy(decoded, true_message: torch.Tensor) -> dict[str, float]:
    metrics = evaluate(decoded.message, true_message, decoded.detected, decoded.corrected)
    return {"message_bit_accuracy": metrics.message_bit_accuracy, "task_accuracy": metrics.task_accuracy}


def _mean_sd(values: list[float]) -> dict:
    if not values or any(
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("accuracy and rate measurements must be finite numbers within [0, 1]")
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def summarize(results: list[dict]) -> dict:
    if not results:
        raise ValueError("at least one seed result is required")
    seeds = [validate_seed(result["seed"]) for result in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed results must be distinct; repeated seeds are not independent runs")
    metadata = results[0].get("metadata")
    if any(result.get("metadata") != metadata for result in results):
        raise ValueError("cannot aggregate runs with different configurations, environments or metric references")
    has_end_to_end = "end_to_end" in results[0]
    if any(("end_to_end" in result) != has_end_to_end for result in results):
        raise ValueError("cannot mix results with and without ground-truth metrics")
    summary: dict = {"seeds": [r["seed"] for r in results], "encoder": {}, "random_corruption": {}, "adversarial_single_flip": {}, "coherent_drift": {}}
    if metadata is not None:
        summary["metadata"] = metadata
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
    if has_end_to_end:
        summary["end_to_end"] = {"random_corruption": {}, "adversarial_single_flip": {}, "coherent_drift": {}}
        for method in METHODS:
            summary["end_to_end"]["random_corruption"][method] = {
                k: {
                    metric: _mean_sd([r["end_to_end"]["random_corruption"][method][k][metric] for r in results])
                    for metric in ("message_bit_accuracy", "task_accuracy")
                } for k in ("0", "1", "2")
            }
            for challenge in ("adversarial_single_flip", "coherent_drift"):
                summary["end_to_end"][challenge][method] = {
                    metric: _mean_sd([r["end_to_end"][challenge][method][metric] for r in results])
                    for metric in ("message_bit_accuracy", "task_accuracy")
                }
    return summary
