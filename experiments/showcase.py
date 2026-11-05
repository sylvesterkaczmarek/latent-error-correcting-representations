"""Eight deterministic decoder checks, with no training or external service."""

from __future__ import annotations
import argparse
import json
import torch
from latent_error_correcting_representations.artifacts import (
    experiment_output,
    write_new_json,
)
from latent_error_correcting_representations.codes import encode, decode
from latent_error_correcting_representations.metrics import evaluate
from latent_error_correcting_representations.selective import (
    encode_hamming84,
    decode_hamming84,
    outcome_counts,
    nearest_valid_replacement,
)


def demonstrate(output):
    with experiment_output(
        output,
        {
            "study": "decoder-showcase",
            "scope": "synthetic four-bit tensors, no learned-model evaluation",
        },
    ) as out:
        truth = torch.tensor([[0, 0, 0, 0]])
        rows = []

        def save(name, expected, observed):
            rows.append(
                {
                    "case": name,
                    "expected": expected,
                    "observed": observed,
                    "passed": expected == observed,
                }
            )

        classic = encode("hamming74_repair", truth)
        one = classic.clone()
        one[:, 0] ^= 1
        save(
            "hamming_single_error_corrected",
            True,
            torch.equal(decode("hamming74_repair", one).message, truth),
        )
        two = classic.clone()
        two[:, [0, 1]] ^= 1
        save(
            "hamming_double_error_miscorrected",
            True,
            not torch.equal(decode("hamming74_repair", two).message, truth),
        )
        extended = encode_hamming84(truth)
        for count, expected in [
            (1, {"success": 1.0, "coverage": 1.0, "wrong": 0.0}),
            (2, {"success": 0.0, "coverage": 0.0, "wrong": 0.0}),
            (3, {"success": 0.0, "coverage": 1.0, "wrong": 1.0}),
        ]:
            altered = extended.clone()
            altered[:, list(range(count))] ^= 1
            metrics = outcome_counts(decode_hamming84(altered), truth)
            save(
                f"secded_{count}_bit_outcome",
                expected,
                {
                    "success": metrics["task_success_rate"],
                    "coverage": metrics["coverage"],
                    "wrong": metrics["wrong_acceptance_rate"],
                },
            )
        coherent = decode_hamming84(
            nearest_valid_replacement("hamming84_secded", truth)
        )
        metrics = outcome_counts(coherent, truth)
        save(
            "valid_codeword_has_wrong_semantics",
            {"wrong": 1.0, "detected": 0.0},
            {
                "wrong": metrics["wrong_acceptance_rate"],
                "detected": metrics["detection_rate"],
            },
        )
        wrong = truth.clone()
        wrong[:, 0] ^= 1
        decoded = decode_hamming84(encode_hamming84(wrong))
        save(
            "wrong_encoder_prediction_remains_wrong",
            {"prediction_preserved": 1.0, "true_task_success": 0.0},
            {
                "prediction_preserved": outcome_counts(decoded, wrong)[
                    "task_success_rate"
                ],
                "true_task_success": outcome_counts(decoded, truth)[
                    "task_success_rate"
                ],
            },
        )
        rejected = False
        try:
            evaluate(truth, truth, torch.tensor([False]), torch.tensor([True]))
        except ValueError:
            rejected = True
        save("unsupported_repair_evidence_rejected", True, rejected)
        summary = {
            "passed": sum(row["passed"] for row in rows),
            "total": len(rows),
            "cases": rows,
            "scope": "Exact synthetic decoder cases. SECDED is classical coding, not a learned safety guarantee.",
        }
        write_new_json(out / "summary.json", summary)
        for row in rows:
            print(f"{row['case']:45} {'PASS' if row['passed'] else 'FAIL'}")
        if summary["passed"] != summary["total"]:
            raise RuntimeError("showcase failed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(demonstrate(args.out), indent=2))


if __name__ == "__main__":
    main()
