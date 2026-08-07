from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

METHOD_LABELS = {
    "uncoded": "Uncoded",
    "repetition3": "Repetition-3",
    "hamming74_detect": "Hamming detect",
    "hamming74_repair": "Hamming repair",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_random_corruption(summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    ks = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method, label in METHOD_LABELS.items():
        vals = [summary["random_corruption"][method][str(k)]["task_accuracy"]["mean"] for k in ks]
        ax.plot(ks, vals, marker="o", label=label)
    ax.set_xlabel("Exact latent bit flips per sample")
    ax.set_ylabel("Task accuracy")
    ax.set_xticks(ks)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, out / "random_corruption_task_accuracy.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method, label in METHOD_LABELS.items():
        vals = [summary["random_corruption"][method][str(k)]["detection_rate"]["mean"] for k in ks]
        ax.plot(ks, vals, marker="o", label=label)
    ax.set_xlabel("Exact latent bit flips per sample")
    ax.set_ylabel("Corruption detection rate")
    ax.set_xticks(ks)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, out / "random_corruption_detection.png")


def plot_adversarial(summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    methods = list(METHOD_LABELS)
    vals = [1.0 - summary["adversarial_single_flip"][m]["task_accuracy"]["mean"] for m in methods]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(methods))
    ax.bar(x, vals)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Adversarial single-flip failure rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out / "adversarial_single_flip.png")


def plot_coherent_drift(summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    methods = list(METHOD_LABELS)
    task_failure = [1.0 - summary["coherent_drift"][m]["task_accuracy"]["mean"] for m in methods]
    detection = [summary["coherent_drift"][m]["detection_rate"]["mean"] for m in methods]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.bar(x - width / 2, task_failure, width, label="Behavioral failure")
    ax.bar(x + width / 2, detection, width, label="Syndrome/detection signal")
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out / "coherent_drift_blind_spot.png")
