"""Independent audit probes for the original candidate; synthetic tensors only."""

from copy import deepcopy
import importlib
from types import SimpleNamespace
import pytest
import torch
from latent_error_correcting_representations import codes, experiment
from latent_error_correcting_representations.metrics import evaluate


@pytest.fixture
def run_record(tmp_path, monkeypatch):
    message = torch.zeros((4, 4), dtype=torch.int64)
    monkeypatch.setattr(
        experiment,
        "make_dataset",
        lambda *a, **k: SimpleNamespace(train=None, test=None),
    )
    monkeypatch.setattr(experiment, "train_encoder", lambda *a, **k: [0.25])
    monkeypatch.setattr(
        experiment,
        "predict_messages",
        lambda *a, **k: (message, message, torch.zeros(4, dtype=torch.int64)),
    )
    return experiment.run_seed(7, tmp_path / "runs", epochs=1)


@pytest.mark.parametrize(
    "method,width",
    [("decode_uncoded", 4), ("decode_repetition3", 12), ("decode_hamming74", 7)],
)
@pytest.mark.parametrize("repair", ["false", 1, None])
def test_repair_control_requires_real_boolean(method, width, repair):
    with pytest.raises((TypeError, ValueError)):
        getattr(codes, method)(torch.zeros((2, width)), repair=repair)


def test_correction_requires_detection_evidence():
    with pytest.raises(ValueError):
        evaluate(
            torch.zeros((2, 4)),
            torch.zeros((2, 4)),
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
        )


@pytest.mark.parametrize("mutation", ["bool-alias", "float-schema"])
def test_summary_metadata_cannot_hide_type_changes(run_record, mutation):
    other = deepcopy(run_record)
    other["seed"] = 8
    if mutation == "bool-alias":
        other["metadata"]["configuration"]["training"]["epochs"] = True
    else:
        other["metadata"]["schema_version"] = 2.0
    with pytest.raises(ValueError):
        experiment.summarize([run_record, other])


@pytest.mark.parametrize(
    "mutation",
    [
        "repair-without-detection",
        "bit-task-contradiction",
        "nonfinite-loss",
        "unknown-method",
    ],
)
def test_summary_rejects_internally_impossible_records(run_record, mutation):
    row = run_record["random_corruption"]["uncoded"]["0"]
    if mutation == "repair-without-detection":
        row["correction_rate"] = 1.0
    elif mutation == "bit-task-contradiction":
        row["message_bit_accuracy"] = 0.0
    elif mutation == "nonfinite-loss":
        run_record["encoder"]["final_train_loss"] = float("nan")
    else:
        run_record["coherent_drift"]["unreviewed"] = deepcopy(row)
    with pytest.raises(ValueError):
        experiment.summarize([run_record])


def test_summary_owns_metadata_snapshot(run_record):
    summary = experiment.summarize([run_record])
    before = deepcopy(summary)
    run_record["metadata"]["configuration"]["training"]["epochs"] = 999
    assert summary == before


@pytest.mark.parametrize(
    "module", ["run_all", "adversarial_flip", "coherent_drift", "corruption_sweep"]
)
def test_cli_refuses_to_overwrite_prior_experiment_before_training(
    tmp_path, monkeypatch, module
):
    command = importlib.import_module("experiments." + module)
    output = tmp_path / "published"
    output.mkdir()
    (output / "summary.json").write_text("existing evidence")
    monkeypatch.setattr(
        command,
        "run_seed",
        lambda *a, **k: pytest.fail(
            "training started before output collision was rejected"
        ),
    )
    with pytest.raises((ValueError, FileExistsError)):
        command.main(["--seeds", "7", "--epochs", "1", "--out", str(output)])
    assert (output / "summary.json").read_text() == "existing evidence"


def test_run_seed_refuses_existing_record_before_random_state_changes(
    tmp_path, monkeypatch
):
    destination = tmp_path / "seed_7.json"
    destination.write_text("preserve")
    monkeypatch.setattr(
        experiment,
        "seed_everything",
        lambda *a: pytest.fail("random state changed before collision was rejected"),
    )
    with pytest.raises((ValueError, FileExistsError)):
        experiment.run_seed(7, tmp_path, epochs=1)
    assert destination.read_text() == "preserve"
