from copy import deepcopy
import hashlib
import json
import os
import pytest
import torch
from latent_error_correcting_representations.artifacts import (
    read_json,
    write_new_json,
    experiment_output,
)
from latent_error_correcting_representations.study import (
    design,
    summary,
    summarize_trace,
    verify_study,
    damage,
    equivalent,
)
from latent_error_correcting_representations.selective import messages, selective_encode


@pytest.mark.parametrize(
    "raw",
    ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":"\\ud800"}'],
)
def test_ambiguous_json_artifact_rejected(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises((ValueError, UnicodeError)):
        read_json(path)


def test_failed_json_validation_preserves_existing_output(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("original")
    with pytest.raises(ValueError):
        write_new_json(path, {"value": float("nan")})
    assert path.read_text() == "original"


def test_exclusive_json_never_replaces_existing_file_or_symlink(tmp_path):
    p = tmp_path / "record"
    write_new_json(p, {"value": 1})
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_json(p, {"value": 2})
    assert p.read_bytes() == before
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(p)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(FileExistsError):
        write_new_json(alias, {"value": 3})
    assert p.read_bytes() == before


def test_write_failure_removes_temporary_without_replacing_evidence(
    tmp_path, monkeypatch
):
    def fail(*args, **kwargs):
        raise OSError("synthetic filesystem failure")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError):
        write_new_json(tmp_path / "record", {"value": 1})
    assert not list(tmp_path.iterdir())


def test_interrupted_experiment_retains_failure_without_completion(tmp_path):
    out = tmp_path / "partial"
    with pytest.raises(RuntimeError):
        with experiment_output(out, {"design": "synthetic"}):
            write_new_json(out / "partial.json", {"partial": True})
            raise RuntimeError("controlled failure")
    assert read_json(out / "failure.json")["status"] == "FAILED"
    assert (out / "partial.json").exists() and not (out / "completion.json").exists()
    with pytest.raises(FileExistsError):
        with experiment_output(out, {}):
            pass


@pytest.mark.parametrize(
    "seeds,epochs", [([1, 1], 1), ([], 1), ([True], 1), ([1], True), ([1], 0)]
)
def test_study_plan_rejects_invalid_cells_or_training(seeds, epochs):
    with pytest.raises(ValueError):
        design(seeds, epochs)


def test_random_masks_are_paired_for_equal_width_policies():
    m = messages()
    mask = []
    for method in ("hamming74_detect", "hamming74_repair", "hamming74_abstain"):
        base = selective_encode(method, m)
        mask.append(damage(method, base, m, 7, "random_2") != base)
    assert torch.equal(mask[0], mask[1]) and torch.equal(mask[1], mask[2])


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    from latent_error_correcting_representations.study import execute

    path = tmp_path_factory.mktemp("root") / "study"
    execute([7], 1, path)
    return path


def test_evidence_verifier_recomputes_all_cases(executed):
    report = verify_study(executed)
    assert report["seed_runs"] == 1 and report["policy_challenge_cells"] == 48
    assert report["message_mask_pairs"] == 76032


@pytest.mark.parametrize(
    "change",
    ["duplicate", "missing", "decision", "corruption", "identity", "label-type"],
)
def test_trace_rejects_incomplete_or_altered_observations(executed, change):
    plan = read_json(executed / "plan.json")
    trace = read_json(executed / "traces/seed_7.json")
    if change == "duplicate":
        trace["cases"].append(deepcopy(trace["cases"][0]))
    elif change == "missing":
        trace["cases"].pop()
    elif change == "decision":
        trace["cases"][0]["accepted"][0] = False
    elif change == "corruption":
        trace["cases"][0]["damaged_words"][0][0] ^= 1
    elif change == "identity":
        trace["seed"] = 8
    elif change == "label-type":
        trace["cases"][0]["task_labels"][0] = float(trace["cases"][0]["task_labels"][0])
    with pytest.raises(ValueError):
        summarize_trace(trace, plan)


@pytest.mark.parametrize(
    "change", ["summary", "extra-file", "checkpoint", "incomplete"]
)
def test_manifest_rejects_changed_files(executed, tmp_path, change):
    import shutil

    target = tmp_path / "copy"
    shutil.copytree(executed, target)
    if change == "summary":
        (target / "summary.json").write_text("{}")
    elif change == "extra-file":
        (target / "unexpected.json").write_text("{}")
    elif change == "checkpoint":
        (target / "checkpoints/seed_7.pt").write_bytes(b"synthetic damaged checkpoint")
    else:
        (target / "completion.json").write_text('{"status":"FAILED"}')
    with pytest.raises(ValueError):
        verify_study(target)


def test_recomputed_manifest_does_not_hide_wrong_summary(executed, tmp_path):
    import shutil

    target = tmp_path / "copy"
    shutil.copytree(executed, target)
    path = target / "summary.json"
    value = read_json(path)
    value["encoder_task_accuracy"]["mean"] = 0.12345
    path.write_text(json.dumps(value))
    completion = read_json(target / "completion.json")
    completion["files"]["summary.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (target / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="aggregate"):
        verify_study(target)


def test_undertrained_encoder_metrics_do_not_hide_prediction_errors(executed):
    record = read_json(executed / "runs/seed_7.json")
    case = next(
        row
        for row in record["cases"]
        if row["method"] == "hamming84_secded" and row["challenge"] == "random_1"
    )
    assert case["integrity"]["task_success_rate"] == 1
    assert case["end_to_end"]["task_success_rate"] == record["encoder_task_accuracy"]
    # The contract holds even if a future training library reaches perfect accuracy in one epoch.
    from latent_error_correcting_representations.selective import (
        decode_hamming84,
        encode_hamming84,
        outcome_counts,
    )

    truth = messages()
    wrong = truth.clone()
    wrong[:, 0] ^= 1
    result = decode_hamming84(encode_hamming84(wrong))
    assert outcome_counts(result, wrong)["task_success_rate"] == 1
    assert outcome_counts(result, truth)["task_success_rate"] == 0


def test_bootstrap_does_not_expand_independent_sample_count(executed):
    plan = read_json(executed / "plan.json")
    row = read_json(executed / "runs/seed_7.json")
    report = summary([row], plan)
    assert all(
        p["n_seed_pairs"] == 1 and p["percentile_95"] is None for p in report["paired"]
    )
    with pytest.raises(ValueError):
        summary([row, deepcopy(row)], plan)


def test_rounding_tolerance_does_not_coerce_counts():
    assert equivalent({"rate": 0.5}, {"rate": 0.5 + 1e-15})
    assert not equivalent({"n": 1}, {"n": True})
    assert not equivalent({"n": 1}, {"n": 1.0})
    assert not equivalent({"rate": 0.5}, {"rate": 0.5001})


def test_showcase_is_executable_and_preserves_existing_output(tmp_path):
    from experiments.showcase import demonstrate

    out = tmp_path / "showcase"
    assert demonstrate(out)["passed"] == 8
    with pytest.raises(FileExistsError):
        demonstrate(out)


def test_overflowing_learning_rate_is_a_defined_input_error():
    from latent_error_correcting_representations.model import MessageEncoder
    from latent_error_correcting_representations.data import make_dataset
    from latent_error_correcting_representations.training import train_encoder

    model = MessageEncoder()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        train_encoder(model, make_dataset(7, n_train=8, n_test=2).train, 7, lr=10**400)
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )


def test_boolean_trace_schema_is_rejected(executed):
    trace = read_json(executed / "traces/seed_7.json")
    trace["schema_version"] = True
    with pytest.raises(ValueError):
        summarize_trace(trace, read_json(executed / "plan.json"))


def test_compact_trace_retains_identical_json_values(tmp_path):
    value = {"matrix": [[0, 1, 1, 0]] * 10, "flags": [True, False], "empty": None}
    pretty = write_new_json(tmp_path / "pretty.json", value)
    compact = write_new_json(tmp_path / "compact.json", value, compact=True)
    assert read_json(pretty) == read_json(compact) == value
    assert compact.stat().st_size < pretty.stat().st_size
    with pytest.raises(ValueError):
        write_new_json(tmp_path / "bad.json", value, compact="true")
