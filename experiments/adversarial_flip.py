from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

from latent_error_correcting_representations.experiment import run_seed, summarize
from latent_error_correcting_representations.plotting import plot_adversarial
from ._cli import challenge_report, parse_args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv, default_out="results/adversarial_flip")
    out = Path(args.out)
    results = [run_seed(s, out / "runs", epochs=args.epochs) for s in args.seeds]
    summary = summarize(results)
    (out / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    report = challenge_report(summary, "adversarial_single_flip")
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plot_adversarial(summary, out / "figures")


if __name__ == "__main__":
    main()
