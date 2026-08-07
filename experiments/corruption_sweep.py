from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_error_correcting_representations.experiment import run_seed, summarize
from latent_error_correcting_representations.plotting import plot_random_corruption


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29, 41, 53])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out", default="results/corruption_sweep")
    args = p.parse_args()
    out = Path(args.out)
    results = [run_seed(s, out / "runs", epochs=args.epochs) for s in args.seeds]
    summary = summarize(results)
    (out / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary["random_corruption"], indent=2), encoding="utf-8")
    plot_random_corruption(summary, out / "figures")


if __name__ == "__main__":
    main()
