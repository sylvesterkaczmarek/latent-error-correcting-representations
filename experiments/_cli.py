"""Shared argument validation for the installed experiment commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from latent_error_correcting_representations.seed import validate_seed


def _seed(value: str) -> int:
    try:
        return validate_seed(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seed must be an integer between 0 and 4294967295"
        ) from exc


def _epochs(value: str) -> int:
    try:
        epochs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("epochs must be a positive integer") from exc
    if epochs < 1:
        raise argparse.ArgumentTypeError("epochs must be a positive integer")
    return epochs


def parse_args(
    argv: Sequence[str] | None = None, *, default_out: str = "results"
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=_seed, default=[7, 17, 29, 41, 53])
    parser.add_argument("--epochs", type=_epochs, default=5)
    parser.add_argument("--out", default=default_out)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct seeds; repeats are not independent runs")
    return args


def challenge_report(summary: dict, challenge: str) -> dict:
    """Retain existing method paths alongside provenance and true task accuracy."""
    return {
        **summary[challenge],
        "seeds": summary["seeds"],
        "metadata": summary["metadata"],
        "end_to_end": summary["end_to_end"][challenge],
    }
