"""Typed contracts for the declared selective-decoding study and its seed records."""

from __future__ import annotations
import math
import re
from .artifacts import canonical, digest
from .selective import SELECTIVE_METHODS

HASH_FIELDS = (
    "initial_model_sha256",
    "model_sha256",
    "checkpoint_sha256",
    "train_data_sha256",
    "test_data_sha256",
)
COUNT_FIELDS = (
    "n",
    "accepted",
    "abstained",
    "correct_accepted",
    "wrong_accepted",
    "detected",
    "repair_attempted",
    "repair_to_reference",
    "candidate_correct",
)
RATE_FIELDS = (
    "coverage",
    "task_success_rate",
    "wrong_acceptance_rate",
    "detection_rate",
    "conditional_accuracy",
)
CHALLENGES = (
    "random_0",
    "random_1",
    "random_2",
    "random_3",
    "random_4",
    "adversarial_1",
    "adversarial_2",
    "coherent_drift",
)
WIDTHS = {
    "uncoded": 4,
    "repetition3": 12,
    "hamming74_detect": 7,
    "hamming74_repair": 7,
    "hamming74_abstain": 7,
    "hamming84_secded": 8,
}


def fingerprint(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("evidence identity must be a lowercase SHA-256 digest")


def finite(value, *, probability=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except (OverflowError, TypeError):
        valid = False
    if not valid or (probability and value > 1):
        raise ValueError("invalid finite measurement")


def validate_plan(plan):
    """Accept the complete supported design, not arbitrary relabelled comparisons.

    The historical source/runtime metadata is validated but need not match the
    verifier's current installation. File identities do not authenticate origin.
    """
    try:
        keys = {
            "schema_version",
            "study",
            "seeds",
            "epochs",
            "methods",
            "challenges",
            "data",
            "model",
            "training",
            "bootstrap",
            "source_sha256",
            "environment",
            "randomization",
            "interpretation",
        }
        if type(plan) is not dict or plan.keys() != keys:
            raise ValueError("incomplete or unknown study plan fields")
        if (
            type(plan["schema_version"]) is not int
            or plan["schema_version"] != 1
            or plan["study"] != "selective-decoding-v1"
        ):
            raise ValueError("unsupported study identity or schema")
        seeds = plan["seeds"]
        if (
            type(seeds) is not list
            or not seeds
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("plan requires distinct integer seed identities")
        if type(plan["epochs"]) is not int or plan["epochs"] < 1:
            raise ValueError("epochs must be a positive integer")
        fixed = {
            "methods": list(SELECTIVE_METHODS),
            "challenges": list(CHALLENGES),
            "data": {
                "n_train": 2500,
                "n_test": 800,
                "input_dim": 16,
                "noise_std": 0.35,
            },
            "model": {"input_dim": 16, "hidden_dim": 48},
            "training": {"batch_size": 256, "lr": 0.002, "device": "cpu"},
            "bootstrap": {"seed": 2605, "resamples": 5000},
            "randomization": "SHA256(seed,width,k) initializes a CPU generator; equal-width policies share the same masks.",
        }
        for field, expected in fixed.items():
            if canonical(plan[field]) != canonical(expected):
                raise ValueError(
                    f"plan {field} differs from the supported predeclared design"
                )
        identity = plan["source_sha256"]
        if type(identity) is not dict or not identity:
            raise ValueError("missing source identity")
        for name, value in identity.items():
            if (
                type(name) is not str
                or re.fullmatch(r"[A-Za-z0-9_]+\.py", name) is None
            ):
                raise ValueError("invalid source module identity")
            fingerprint(value)
        environment = plan["environment"]
        if type(environment) is not dict or environment.keys() != {
            "python",
            "torch",
            "numpy",
            "threads",
            "deterministic_algorithms",
        }:
            raise ValueError("invalid runtime identity")
        if any(
            type(environment[k]) is not str or not environment[k].strip()
            for k in ("python", "torch", "numpy")
        ):
            raise ValueError("invalid runtime version")
        if (
            type(environment["threads"]) is not int
            or environment["threads"] < 1
            or environment["deterministic_algorithms"] is not True
        ):
            raise ValueError("invalid deterministic runtime settings")
        if (
            type(plan["interpretation"]) is not str
            or not plan["interpretation"].strip()
        ):
            raise ValueError("missing interpretation")
        canonical(plan)
    except (KeyError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid study plan") from exc


def validate_identity(record, plan):
    if (
        type(record) is not dict
        or type(record.get("seed")) is not int
        or record["seed"] not in plan["seeds"]
        or record.get("plan_sha256") != digest(plan)
    ):
        raise ValueError("record identity differs from its plan")
    for field in HASH_FIELDS:
        fingerprint(record.get(field))
    history = record.get("training_loss")
    if type(history) is not list or len(history) != plan["epochs"]:
        raise ValueError("training history does not match planned epochs")
    for value in history:
        finite(value)


def validate_bits(value, rows, width):
    if type(value) is not list or len(value) != rows:
        raise ValueError("recorded bit matrix has incorrect sample count")
    for row in value:
        if (
            type(row) is not list
            or len(row) != width
            or any(type(bit) is not int or bit not in (0, 1) for bit in row)
        ):
            raise ValueError(
                "recorded bits require exact integer zero/one values and dimensions"
            )


def validate_counts(row, n):
    if type(row) is not dict or row.keys() != set(COUNT_FIELDS + RATE_FIELDS):
        raise ValueError("invalid outcome count fields")
    if (
        any(type(row[k]) is not int or not 0 <= row[k] <= n for k in COUNT_FIELDS)
        or row["n"] != n
    ):
        raise ValueError(
            "outcome counts must be bounded integers with the planned denominator"
        )
    if (
        row["accepted"] + row["abstained"] != n
        or row["correct_accepted"] + row["wrong_accepted"] != row["accepted"]
        or not row["correct_accepted"]
        <= row["candidate_correct"]
        <= row["correct_accepted"] + row["abstained"]
        or row["repair_attempted"] > min(row["accepted"], row["detected"])
        or row["repair_to_reference"]
        > min(row["repair_attempted"], row["correct_accepted"])
    ):
        raise ValueError("contradictory outcome counts")
    values = {
        "coverage": row["accepted"] / n,
        "task_success_rate": row["correct_accepted"] / n,
        "wrong_acceptance_rate": row["wrong_accepted"] / n,
        "detection_rate": row["detected"] / n,
        "conditional_accuracy": row["correct_accepted"] / row["accepted"]
        if row["accepted"]
        else None,
    }
    for field, expected in values.items():
        actual = row[field]
        if expected is None:
            if actual is not None:
                raise ValueError("conditional accuracy must be null at zero coverage")
        else:
            finite(actual, probability=True)
            if not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("reported rate differs from integer outcome counts")


def validate_result(record, plan):
    try:
        validate_identity(record, plan)
        keys = {
            "seed",
            "plan_sha256",
            "encoder_task_accuracy",
            "encoder_bit_accuracy",
            "training_loss",
            "cases",
            *HASH_FIELDS,
        }
        if record.keys() != keys:
            raise ValueError("invalid seed result fields")
        for name in ("encoder_task_accuracy", "encoder_bit_accuracy"):
            finite(record[name], probability=True)
        task, bit = record["encoder_task_accuracy"], record["encoder_bit_accuracy"]
        if task > bit + 1e-12 or bit > 0.75 + 0.25 * task + 1e-12:
            raise ValueError("encoder task and bit accuracy are contradictory")
        expected = {(m, c) for m in plan["methods"] for c in plan["challenges"]}
        seen = set()
        if type(record["cases"]) is not list:
            raise ValueError("invalid case list")
        for row in record["cases"]:
            if type(row) is not dict or row.keys() != {
                "method",
                "challenge",
                "integrity",
                "end_to_end",
            }:
                raise ValueError("invalid seed cell fields")
            key = row["method"], row["challenge"]
            if key not in expected or key in seen:
                raise ValueError("duplicate or unknown seed cell")
            seen.add(key)
            for target in ("integrity", "end_to_end"):
                validate_counts(row[target], plan["data"]["n_test"])
            for field in ("n", "accepted", "abstained", "detected", "repair_attempted"):
                if row["integrity"][field] != row["end_to_end"][field]:
                    raise ValueError("reference-independent decision counts disagree")
        if seen != expected:
            raise ValueError("missing planned method/challenge cell")
    except (KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid seed result") from exc
