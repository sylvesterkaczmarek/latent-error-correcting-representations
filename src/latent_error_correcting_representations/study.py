"""Predeclared selective-decoding study with inspectable per-example decisions."""

from __future__ import annotations
import hashlib
import math
from pathlib import Path
import platform
import statistics
import numpy as np
import torch
from .artifacts import canonical, digest, read_json, write_new_json, experiment_output
from .codes import task_label
from .corruption import flip_exactly_k
from .data import make_dataset
from .model import MessageEncoder
from .training import train_encoder, predict_messages
from .seed import seed_everything, validate_seed
from .selective import (
    SELECTIVE_METHODS,
    selective_encode,
    selective_decode,
    outcome_counts,
    nearest_valid_replacement,
    adversarial_exact_flip,
)
from .exhaustive import enumerate_codes

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


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        h.update(canonical([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def source_identity():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }


def design(seeds, epochs):
    seeds = [validate_seed(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("distinct nonempty seed list required")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    return {
        "schema_version": 1,
        "study": "selective-decoding-v1",
        "seeds": seeds,
        "epochs": epochs,
        "methods": list(SELECTIVE_METHODS),
        "challenges": list(CHALLENGES),
        "data": {"n_train": 2500, "n_test": 800, "input_dim": 16, "noise_std": 0.35},
        "model": {"input_dim": 16, "hidden_dim": 48},
        "training": {"batch_size": 256, "lr": 0.002, "device": "cpu"},
        "bootstrap": {"seed": 2605, "resamples": 5000},
        "source_sha256": source_identity(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
        },
        "randomization": "SHA256(seed,width,k) initializes a CPU generator; equal-width policies share the same masks.",
        "interpretation": "Exact bit budgets do not equalize code rate. Per-example ground truth is separate from preservation of the encoder prediction. Three- and four-bit stress tests exceed SECDED guarantees. Replays are not new independent seeds.",
    }


def random_seed(seed, width, k):
    return int(digest([seed, width, k])[:15], 16)


def damage(method, codeword, predicted, seed, challenge):
    if challenge.startswith("random_"):
        k = int(challenge.split("_")[-1])
        generator = torch.Generator().manual_seed(
            random_seed(seed, codeword.shape[1], k)
        )
        return flip_exactly_k(codeword, k, generator)
    if challenge.startswith("adversarial_"):
        return adversarial_exact_flip(
            method, codeword, predicted, int(challenge.split("_")[-1])
        )
    if challenge == "coherent_drift":
        return nearest_valid_replacement(method, predicted)
    raise ValueError("unknown challenge")


def run_trial(plan, seed, out):
    seed_everything(seed)
    data = make_dataset(seed, **plan["data"])
    model = MessageEncoder(**plan["model"])
    initial_hash = tensor_digest(model.state_dict())
    history = train_encoder(
        model, data.train, seed=seed, epochs=plan["epochs"], **plan["training"]
    )
    predicted, truth, labels = predict_messages(model, data.test)
    if not torch.equal(labels, task_label(truth)):
        raise ValueError("task labels differ from semantic messages")
    evidence = {
        "seed": seed,
        "plan_sha256": digest(plan),
        "initial_model_sha256": initial_hash,
        "model_sha256": tensor_digest(model.state_dict()),
        "training_loss": history,
        "train_data_sha256": tensor_digest(
            {str(i): t for i, t in enumerate(data.train.tensors)}
        ),
        "test_data_sha256": tensor_digest(
            {str(i): t for i, t in enumerate(data.test.tensors)}
        ),
    }
    trace = {
        "schema_version": 1,
        **evidence,
        "predicted_messages": predicted.tolist(),
        "true_messages": truth.tolist(),
        "cases": [],
    }
    for method in plan["methods"]:
        codeword = selective_encode(method, predicted)
        for challenge in plan["challenges"]:
            damaged = damage(method, codeword, predicted, seed, challenge)
            decoded = selective_decode(method, damaged)
            trace["cases"].append(
                {
                    "method": method,
                    "challenge": challenge,
                    "damaged_words": damaged.tolist(),
                    "candidate_messages": decoded.candidate_message.tolist(),
                    "detected": decoded.detected.tolist(),
                    "repaired": decoded.repaired.tolist(),
                    "accepted": decoded.accepted.tolist(),
                    "task_labels": decoded.task_labels.tolist(),
                }
            )
    checkpoint = out / "checkpoints" / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as stream:
        torch.save(
            {"model": model.state_dict(), "seed": seed, "plan_sha256": digest(plan)},
            stream,
        )
    evidence["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trace["checkpoint_sha256"] = evidence["checkpoint_sha256"]
    result = summarize_trace(trace, plan)
    write_new_json(out / "traces" / f"seed_{seed}.json", trace, compact=True)
    write_new_json(out / "runs" / f"seed_{seed}.json", result)
    return result


def summarize_trace(trace, plan, *, verify_corruption=True):
    if (
        type(trace.get("seed")) is not int
        or trace["seed"] not in plan["seeds"]
        or trace.get("plan_sha256") != digest(plan)
    ):
        raise ValueError("trace identity does not match plan")
    if type(trace.get("schema_version")) is not int or trace["schema_version"] != 1:
        raise ValueError("unsupported trace schema")
    predicted = torch.tensor(trace["predicted_messages"])
    truth = torch.tensor(trace["true_messages"])
    if predicted.shape != (plan["data"]["n_test"], 4) or truth.shape != predicted.shape:
        raise ValueError("unexpected evaluation sample count")
    task_label(predicted)
    task_label(truth)
    cases = []
    seen = set()
    for row in trace["cases"]:
        key = row["method"], row["challenge"]
        if (
            key in seen
            or key[0] not in plan["methods"]
            or key[1] not in plan["challenges"]
        ):
            raise ValueError("duplicate or unknown method/challenge cell")
        seen.add(key)
        damaged = torch.tensor(row["damaged_words"])
        if verify_corruption:
            expected = damage(
                key[0],
                selective_encode(key[0], predicted),
                predicted,
                trace["seed"],
                key[1],
            )
            if not torch.equal(expected, damaged):
                raise ValueError(
                    "corruption evidence differs from predeclared challenge"
                )
        actual = selective_decode(key[0], damaged)
        for name in (
            "candidate_messages",
            "detected",
            "repaired",
            "accepted",
            "task_labels",
        ):
            values = getattr(
                actual, "candidate_message" if name == "candidate_messages" else name
            ).tolist()
            if canonical(row[name]) != canonical(values):
                raise ValueError("trace decisions differ from decoding recorded bits")
        cases.append(
            {
                "method": key[0],
                "challenge": key[1],
                "integrity": outcome_counts(actual, predicted),
                "end_to_end": outcome_counts(actual, truth),
            }
        )
    if seen != {(m, c) for m in plan["methods"] for c in plan["challenges"]}:
        raise ValueError("incomplete method/challenge evidence")
    history = trace["training_loss"]
    if len(history) != plan["epochs"] or any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in history
    ):
        raise ValueError("invalid training history")
    return {
        "seed": trace["seed"],
        "plan_sha256": digest(plan),
        "encoder_task_accuracy": int((task_label(predicted) == task_label(truth)).sum())
        / len(truth),
        "encoder_bit_accuracy": int((predicted == truth).sum()) / truth.numel(),
        "initial_model_sha256": trace["initial_model_sha256"],
        "model_sha256": trace["model_sha256"],
        "checkpoint_sha256": trace["checkpoint_sha256"],
        "train_data_sha256": trace["train_data_sha256"],
        "test_data_sha256": trace["test_data_sha256"],
        "training_loss": history,
        "cases": cases,
    }


def summary(results, plan):
    if any(type(row.get("seed")) is not int for row in results):
        raise ValueError("seed identities must be integers")
    if len(results) != len(plan["seeds"]) or sorted(
        r["seed"] for r in results
    ) != sorted(plan["seeds"]):
        raise ValueError(
            "summary requires exactly one complete result for every declared seed"
        )
    if any(row["plan_sha256"] != digest(plan) for row in results):
        raise ValueError("result is from a different plan")

    def mean_sd(values):
        usable = [x for x in values if x is not None]
        return {
            "mean": statistics.mean(usable) if usable else None,
            "sample_sd": statistics.stdev(usable) if len(usable) > 1 else None,
            "n_defined": len(usable),
            "n_seeds": len(values),
        }

    grouped = []
    indexed = [
        {(row["method"], row["challenge"]): row for row in result["cases"]}
        for result in results
    ]
    for method in plan["methods"]:
        for challenge in plan["challenges"]:
            item = {"method": method, "challenge": challenge}
            for target in ("integrity", "end_to_end"):
                records = [table[(method, challenge)][target] for table in indexed]
                item[target] = {
                    metric: mean_sd([r[metric] for r in records])
                    for metric in (
                        "coverage",
                        "task_success_rate",
                        "wrong_acceptance_rate",
                        "detection_rate",
                        "conditional_accuracy",
                    )
                }
            grouped.append(item)
    paired = []
    for control in ("hamming74_repair", "hamming74_abstain", "repetition3"):
        for challenge in ("random_2", "random_3", "random_4", "coherent_drift"):
            for target in ("integrity", "end_to_end"):
                for metric in (
                    "coverage",
                    "task_success_rate",
                    "wrong_acceptance_rate",
                ):
                    delta = [
                        table[("hamming84_secded", challenge)][target][metric]
                        - table[(control, challenge)][target][metric]
                        for table in indexed
                    ]
                    interval = None
                    if len(delta) > 1:
                        rng = np.random.default_rng(plan["bootstrap"]["seed"])
                        samples = np.asarray(delta)[
                            rng.integers(
                                len(delta),
                                size=(plan["bootstrap"]["resamples"], len(delta)),
                            )
                        ].mean(axis=1)
                        interval = [
                            float(x) for x in np.quantile(samples, [0.025, 0.975])
                        ]
                    paired.append(
                        {
                            "left": "hamming84_secded",
                            "right": control,
                            "challenge": challenge,
                            "reference": target,
                            "metric": metric,
                            "seed_differences": delta,
                            "mean_difference": statistics.mean(delta),
                            "percentile_95": interval,
                            "n_seed_pairs": len(delta),
                        }
                    )
    return {
        "schema_version": 1,
        "plan_sha256": digest(plan),
        "seeds": plan["seeds"],
        "encoder_task_accuracy": mean_sd([r["encoder_task_accuracy"] for r in results]),
        "comparisons": grouped,
        "paired": paired,
        "uncertainty": "Descriptive seed-paired bootstrap, 5000 resamples, no multiplicity-adjusted significance claim. Finite code-space enumeration is exact and needs no sampling interval.",
    }


def execute(seeds, epochs, output):
    plan = design(seeds, epochs)
    with experiment_output(output, plan) as out:
        records = []
        for seed in plan["seeds"]:
            records.append(run_trial(plan, seed, out))
            print(
                f"Seed {seed} completed all {len(SELECTIVE_METHODS) * len(CHALLENGES)} policy/challenge cells",
                flush=True,
            )
        write_new_json(out / "summary.json", summary(records, plan))
        write_new_json(out / "exhaustive.json", enumerate_codes())
    return plan


def verify_study(path):
    path = Path(path)
    plan = read_json(path / "plan.json")
    completion = read_json(path / "completion.json")
    if completion.get("status") != "COMPLETE":
        raise ValueError("experiment was not completed")
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("evidence tree must not contain symlinks")
    actual = {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file() and p != path / "completion.json"
    }
    if actual != completion["files"]:
        raise ValueError("experiment files differ from their completion manifest")
    expected_files = {"plan.json", "summary.json", "exhaustive.json"} | {
        f"{folder}/seed_{seed}.{suffix}"
        for seed in plan["seeds"]
        for folder, suffix in (
            ("runs", "json"),
            ("traces", "json"),
            ("checkpoints", "pt"),
        )
    }
    if set(actual) != expected_files:
        raise ValueError("unexpected or missing evidence files")
    records = []
    for seed in plan["seeds"]:
        trace = read_json(path / "traces" / f"seed_{seed}.json")
        result = summarize_trace(trace, plan)
        if result["seed"] != seed or canonical(result) != canonical(
            read_json(path / "runs" / f"seed_{seed}.json")
        ):
            raise ValueError("seed summary does not match trace")
        if result["checkpoint_sha256"] != actual[f"checkpoints/seed_{seed}.pt"]:
            raise ValueError("checkpoint digest does not match trace")
        records.append(result)
    if not equivalent(summary(records, plan), read_json(path / "summary.json")):
        raise ValueError("aggregate statistics do not match recorded seeds")
    if enumerate_codes() != read_json(path / "exhaustive.json"):
        raise ValueError("exhaustive code-space results do not reproduce")
    return {
        "status": "PASS",
        "seed_runs": len(records),
        "policy_challenge_cells": len(records)
        * len(SELECTIVE_METHODS)
        * len(CHALLENGES),
        "message_mask_pairs": read_json(path / "exhaustive.json")["message_mask_pairs"],
        "meaning": "Trace decoding, corruption budgets, aggregation and file consistency; not a new neural-training reproduction or authenticated provenance.",
    }


def equivalent(a, b):
    if type(a) is float and type(b) is float:
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return a == b
