"""Execute or verify the separate selective-decoding benchmark."""

from __future__ import annotations
import argparse
import json
from latent_error_correcting_representations.study import execute, verify_study
from ._cli import _seed, _epochs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("run", "verify"))
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=_seed, default=[7, 17, 29, 41, 53])
    p.add_argument("--epochs", type=_epochs, default=5)
    args = p.parse_args(argv)
    if args.operation == "run":
        execute(args.seeds, args.epochs, args.out)
    print(json.dumps(verify_study(args.out), indent=2))


if __name__ == "__main__":
    main()
