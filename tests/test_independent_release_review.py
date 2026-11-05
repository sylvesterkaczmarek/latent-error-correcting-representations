"""Separately authored synthetic review cases, retained unchanged outside checkout."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import pytest
import torch
from latent_error_correcting_representations.artifacts import (
    read_json,
    canonical,
    digest,
)
from latent_error_correcting_representations.study import (
    execute,
    summarize_trace,
    summary,
    verify_study,
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    p = tmp_path_factory.mktemp("separate-review") / "study"
    execute([7], 1, p)
    return p


def save(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def reseal(path):
    files = {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file() and p.name != "completion.json"
    }
    save(path / "completion.json", {"status": "COMPLETE", "files": files})


@pytest.mark.parametrize(
    "kind",
    [
        "bool-schema",
        "float-schema",
        "wrong-study",
        "duplicate-seeds",
        "duplicate-method",
        "duplicate-challenge",
        "wrong-projection",
        "wrong-bootstrap",
    ],
)
def test_verifier_rejects_inconsistent_plan_even_when_checksums_rebuilt(
    evidence, tmp_path, kind
):
    target = tmp_path / "study"
    shutil.copytree(evidence, target)
    plan = read_json(target / "plan.json")
    if kind == "bool-schema":
        plan["schema_version"] = True
    if kind == "float-schema":
        plan["schema_version"] = 1.0
    if kind == "wrong-study":
        plan["study"] = "unknown-design"
    if kind == "duplicate-seeds":
        plan["seeds"] = [7, 7]
    if kind == "duplicate-method":
        plan["methods"].append(plan["methods"][0])
    if kind == "duplicate-challenge":
        plan["challenges"].append(plan["challenges"][0])
    if kind == "wrong-projection":
        plan["model"]["input_dim"] = 15
    if kind == "wrong-bootstrap":
        plan["bootstrap"]["resamples"] = 10
    save(target / "plan.json", plan)
    trace = read_json(target / "traces/seed_7.json")
    trace["plan_sha256"] = digest(plan)
    save(target / "traces/seed_7.json", trace)
    # Record derived values under the altered design where the existing API permits it.
    try:
        result = summarize_trace(trace, plan)
        save(target / "runs/seed_7.json", result)
        save(
            target / "summary.json",
            summary([deepcopy(result) for _ in plan["seeds"]], plan),
        )
    except ValueError:
        return
    reseal(target)
    with pytest.raises(ValueError):
        verify_study(target)


@pytest.mark.parametrize(
    "field",
    [
        "initial_model_sha256",
        "model_sha256",
        "train_data_sha256",
        "test_data_sha256",
        "checkpoint_sha256",
    ],
)
def test_trace_hashes_require_sha256_identity(evidence, field):
    plan = read_json(evidence / "plan.json")
    trace = read_json(evidence / "traces/seed_7.json")
    trace[field] = "not-a-digest"
    with pytest.raises(ValueError):
        summarize_trace(trace, plan)


@pytest.mark.parametrize(
    "field", ["predicted_messages", "true_messages", "damaged_words"]
)
@pytest.mark.parametrize("value", [True, 1.0])
def test_json_recorded_bits_have_canonical_integer_types(evidence, field, value):
    plan = read_json(evidence / "plan.json")
    trace = read_json(evidence / "traces/seed_7.json")
    matrix = trace["cases"][0][field] if field == "damaged_words" else trace[field]
    # Preserve bit values so this tests type sensitivity rather than a corruption difference.
    bit = matrix[0][0]
    matrix[0][0] = bool(bit) if type(value) is bool else float(bit)
    with pytest.raises(ValueError):
        summarize_trace(trace, plan)


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate-cell",
        "missing-cell",
        "unknown-cell",
        "nan-rate",
        "boolean-count",
        "bad-partition",
        "bad-rate",
        "wrong-encoder",
        "bad-checkpoint-hash",
    ],
)
def test_summary_rejects_invalid_seed_evidence(evidence, kind):
    plan = read_json(evidence / "plan.json")
    result = read_json(evidence / "runs/seed_7.json")
    counts = result["cases"][0]["integrity"]
    if kind == "duplicate-cell":
        result["cases"].append(deepcopy(result["cases"][0]))
    if kind == "missing-cell":
        result["cases"].pop()
    if kind == "unknown-cell":
        result["cases"].append({**deepcopy(result["cases"][0]), "method": "invented"})
    if kind == "nan-rate":
        counts["coverage"] = float("nan")
    if kind == "boolean-count":
        counts["n"] = True
    if kind == "bad-partition":
        counts["abstained"] = counts["n"]
    if kind == "bad-rate":
        counts["task_success_rate"] = 0.12345
    if kind == "wrong-encoder":
        result["encoder_task_accuracy"] = 1.1
    if kind == "bad-checkpoint-hash":
        result["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError):
        summary([result], plan)


def test_summary_does_not_alias_mutable_plan(evidence):
    plan = read_json(evidence / "plan.json")
    result = read_json(evidence / "runs/seed_7.json")
    output = summary([result], plan)
    plan["seeds"].append(17)
    assert output["seeds"] == [7]


@pytest.mark.parametrize(
    "kind", ["unreadable", "wrong-seed", "wrong-plan", "wrong-model", "wrong-dtype"]
)
def test_checkpoint_evidence_must_agree_with_recorded_model(evidence, tmp_path, kind):
    target = tmp_path / "study"
    shutil.copytree(evidence, target)
    p = target / "checkpoints/seed_7.pt"
    state = torch.load(p, weights_only=True, map_location="cpu")
    if kind == "unreadable":
        p.write_bytes(b"not a model checkpoint")
    else:
        if kind == "wrong-seed":
            state["seed"] = 8
        if kind == "wrong-plan":
            state["plan_sha256"] = "0" * 64
        if kind == "wrong-model":
            next(iter(state["model"].values())).fill_(0)
        if kind == "wrong-dtype":
            state["model"] = {k: v.double() for k, v in state["model"].items()}
        torch.save(state, p)
    trace = read_json(target / "traces/seed_7.json")
    trace["checkpoint_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    save(target / "traces/seed_7.json", trace)
    result = read_json(target / "runs/seed_7.json")
    result["checkpoint_sha256"] = trace["checkpoint_sha256"]
    save(target / "runs/seed_7.json", result)
    reseal(target)
    with pytest.raises(ValueError):
        verify_study(target)


def test_valid_record_and_legacy_digest_control(evidence):
    assert verify_study(evidence)["status"] == "PASS"
    plan = read_json(evidence / "plan.json")
    trace = read_json(evidence / "traces/seed_7.json")
    result = summarize_trace(trace, plan)
    assert canonical(summary([result], plan)) == canonical(
        read_json(evidence / "summary.json")
    )


@pytest.mark.parametrize(
    "field", ["plan_sha256", "model_sha256", "checkpoint_sha256", "train_data_sha256"]
)
def test_trace_identifiers_preserved_in_derived_summary(evidence, field):
    plan = read_json(evidence / "plan.json")
    trace = read_json(evidence / "traces/seed_7.json")
    assert summarize_trace(trace, plan)[field] == trace[field]


@pytest.mark.parametrize(
    "kind",
    [
        "negative-count",
        "fractional-count",
        "wrong-denominator",
        "accepted-partition",
        "candidate-partition",
        "repair-count",
        "conditional-zero",
        "cross-reference",
    ],
)
def test_outcome_count_relationships_cannot_be_aggregated(evidence, kind):
    plan = read_json(evidence / "plan.json")
    result = read_json(evidence / "runs/seed_7.json")
    counts = result["cases"][0]["integrity"]
    if kind == "negative-count":
        counts["detected"] = -1
    if kind == "fractional-count":
        counts["accepted"] = float(counts["accepted"])
    if kind == "wrong-denominator":
        counts["n"] -= 1
    if kind == "accepted-partition":
        counts["wrong_accepted"] += 1
    if kind == "candidate-partition":
        counts["candidate_correct"] = 0
    if kind == "repair-count":
        counts["repair_attempted"] = 1
    if kind == "conditional-zero":
        row = next(
            r
            for r in result["cases"]
            if r["method"] == "hamming84_secded" and r["challenge"] == "random_2"
        )
        row["integrity"]["conditional_accuracy"] = 1.0
    if kind == "cross-reference":
        counts["detected"] = 1
        counts["detection_rate"] = 1 / counts["n"]
    with pytest.raises(ValueError):
        summary([result], plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", True),
        ("seeds", [7, 7]),
        ("seeds", [True]),
        ("methods", []),
        ("schema_version", 1.0),
    ],
)
def test_plan_validation_precedes_checkpoint_deserialization(
    evidence, tmp_path, monkeypatch, field, value
):
    target = tmp_path / "study"
    shutil.copytree(evidence, target)
    plan = read_json(target / "plan.json")
    plan[field] = value
    save(target / "plan.json", plan)
    reseal(target)
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("loaded before validating plan")
    )
    with pytest.raises(ValueError):
        verify_study(target)


def test_checkpoint_validation_preserves_cpu_random_state(evidence):
    before = torch.get_rng_state().clone()
    assert verify_study(evidence)["status"] == "PASS"
    assert torch.equal(before, torch.get_rng_state())


def test_historical_complete_study_remains_verifiable():
    root = Path(__file__).resolve().parents[1]
    report = verify_study(root / "results/selective-v1")
    assert report["seed_runs"] == 5 and report["policy_challenge_cells"] == 240


@pytest.mark.parametrize(
    "kind", ["extra-field", "boolean-seed", "missing-weight", "nan-weight", "shape"]
)
def test_checkpoint_contract_rejects_incomplete_or_ambiguous_state(
    evidence, tmp_path, kind
):
    target = tmp_path / "study"
    shutil.copytree(evidence, target)
    p = target / "checkpoints/seed_7.pt"
    state = torch.load(p, weights_only=True, map_location="cpu")
    if kind == "extra-field":
        state["extra"] = True
    if kind == "boolean-seed":
        state["seed"] = True
    if kind == "missing-weight":
        state["model"].pop(next(iter(state["model"])))
    if kind == "nan-weight":
        next(iter(state["model"].values())).fill_(float("nan"))
    if kind == "shape":
        state["model"][next(iter(state["model"]))] = torch.ones(1)
    torch.save(state, p)
    trace = read_json(target / "traces/seed_7.json")
    trace["checkpoint_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    save(target / "traces/seed_7.json", trace)
    result = read_json(target / "runs/seed_7.json")
    result["checkpoint_sha256"] = trace["checkpoint_sha256"]
    save(target / "runs/seed_7.json", result)
    reseal(target)
    with pytest.raises(ValueError):
        verify_study(target)
