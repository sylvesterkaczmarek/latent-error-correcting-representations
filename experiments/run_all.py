from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_error_correcting_representations.experiment import run_seed, summarize
from latent_error_correcting_representations.plotting import plot_random_corruption, plot_adversarial, plot_coherent_drift


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29, 41, 53])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    runs_dir = out / "runs"
    results = [run_seed(seed, runs_dir, epochs=args.epochs) for seed in args.seeds]
    summary = summarize(results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_random_corruption(summary, out / "figures")
    plot_adversarial(summary, out / "figures")
    plot_coherent_drift(summary, out / "figures")
    print(json.dumps({
        "encoder_task_accuracy": summary["encoder"]["task_accuracy"],
        "hamming_single_flip_task_accuracy": summary["random_corruption"]["hamming74_repair"]["1"]["task_accuracy"],
        "hamming_coherent_drift_detection": summary["coherent_drift"]["hamming74_repair"]["detection_rate"],
        "hamming_coherent_drift_task_accuracy": summary["coherent_drift"]["hamming74_repair"]["task_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
