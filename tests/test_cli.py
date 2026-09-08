from __future__ import annotations

import importlib
import json

import pytest

from experiments._cli import parse_args


@pytest.mark.parametrize(
    "module_name",
    ["run_all", "adversarial_flip", "coherent_drift", "corruption_sweep"],
)
@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "0"],
        ["--epochs", "-1"],
        ["--epochs", "1.5"],
        ["--seeds", "7", "-1"],
        ["--seeds", "7", "4294967296"],
        ["--seeds", "7", "7"],
        ["--seeds", "7", "1.5"],
    ],
)
def test_invalid_cli_does_not_run_or_write(module_name, arguments, monkeypatch, tmp_path):
    command = importlib.import_module(f"experiments.{module_name}")
    calls = []
    monkeypatch.setattr(command, "run_seed", lambda *args, **kwargs: calls.append(args))
    output = tmp_path / "unwritten"
    with pytest.raises(SystemExit) as error:
        command.main([*arguments, "--out", str(output)])
    assert error.value.code == 2
    assert calls == []
    assert not output.exists()


def test_reference_cli_defaults():
    args = parse_args([])
    assert args.seeds == [7, 17, 29, 41, 53]
    assert args.epochs == 5
    assert args.out == "results"


def test_cli_accepts_seed_endpoints_and_output_override(tmp_path):
    args = parse_args(
        ["--seeds", "0", "4294967295", "--epochs", "1", "--out", str(tmp_path)]
    )
    assert args.seeds == [0, 4294967295]
    assert args.epochs == 1
    assert args.out == str(tmp_path)


@pytest.mark.parametrize(
    "module_name, challenge, plot_name",
    [
        ("adversarial_flip", "adversarial_single_flip", "plot_adversarial"),
        ("coherent_drift", "coherent_drift", "plot_coherent_drift"),
        ("corruption_sweep", "random_corruption", "plot_random_corruption"),
    ],
)
def test_individual_reports_preserve_method_paths_and_ground_truth(
    module_name, challenge, plot_name, monkeypatch, tmp_path
):
    command = importlib.import_module(f"experiments.{module_name}")
    integrity = {"task_accuracy": {"mean": 1.0, "std": 0.0, "n": 1}}
    truth = {"task_accuracy": {"mean": 0.25, "std": 0.0, "n": 1}}
    if challenge == "random_corruption":
        integrity, truth = {"1": integrity}, {"1": truth}
    summary = {
        "seeds": [7],
        "metadata": {"schema_version": 2, "configuration": {"epochs": 1}},
        challenge: {"hamming74_repair": integrity},
        "end_to_end": {challenge: {"hamming74_repair": truth}},
    }
    calls = []

    def run_seed(seed, out_dir, *, epochs):
        calls.append((seed, out_dir, epochs))
        return {"seed": seed}

    monkeypatch.setattr(command, "run_seed", run_seed)
    monkeypatch.setattr(command, "summarize", lambda results: summary)
    plots = []
    monkeypatch.setattr(command, plot_name, lambda data, out: plots.append((data, out)))
    output = tmp_path / module_name
    command.main(["--seeds", "7", "--epochs", "1", "--out", str(output)])

    report = json.loads((output / "summary.json").read_text())
    assert report["hamming74_repair"] == integrity
    assert report["end_to_end"]["hamming74_repair"] == truth
    assert report["metadata"] == summary["metadata"]
    assert report["seeds"] == [7]
    assert challenge not in report
    assert challenge not in report["end_to_end"]
    assert calls == [(7, output / "runs", 1)]
    assert plots == [(summary, output / "figures")]
