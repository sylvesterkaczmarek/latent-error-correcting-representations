from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import latent_error_correcting_representations.experiment as experiment


@pytest.fixture
def wrong_encoder_run(monkeypatch, tmp_path):
    # An encoder mistakes true message 0001 for 0000 on every example.
    predicted = torch.zeros((2, 4), dtype=torch.int64)
    truth = predicted.clone()
    truth[:, 0] = 1
    monkeypatch.setattr(experiment, 'make_dataset', lambda *a, **kw: SimpleNamespace(train=None, test=None))
    monkeypatch.setattr(experiment, 'train_encoder', lambda *a, **kw: [0.25])
    monkeypatch.setattr(experiment, 'predict_messages', lambda *a, **kw: (predicted, truth, torch.ones(2, dtype=torch.int64)))
    return experiment.run_seed(7, tmp_path, epochs=1)


def test_integrity_is_separate_from_true_task_accuracy(wrong_encoder_run):
    result = wrong_encoder_run
    assert result['encoder']['task_accuracy'] == 0.0
    assert result['random_corruption']['hamming74_repair']['1']['task_accuracy'] == 1.0
    assert result['end_to_end']['random_corruption']['hamming74_repair']['1']['task_accuracy'] == 0.0
    assert result['coherent_drift']['uncoded']['task_accuracy'] == 0.0
    assert result['end_to_end']['coherent_drift']['uncoded']['task_accuracy'] == 1.0
    assert result['metadata']['metric_reference'] == 'predicted_message_before_corruption'
    assert result['metadata']['configuration']['training']['epochs'] == 1


def test_summary_retains_both_accuracy_references(wrong_encoder_run):
    summary = experiment.summarize([wrong_encoder_run])
    assert summary['end_to_end']['random_corruption']['hamming74_repair']['1']['task_accuracy'] == dict(mean=0.0, std=0.0, n=1)
    assert summary['random_corruption']['hamming74_repair']['1']['task_accuracy']['mean'] == 1.0


def test_repeated_seed_does_not_inflate_sample_count(wrong_encoder_run):
    with pytest.raises(ValueError, match='distinct'):
        experiment.summarize([wrong_encoder_run, deepcopy(wrong_encoder_run)])


@pytest.mark.parametrize('change', ['epochs', 'environment', 'metadata', 'end_to_end'])
def test_summary_rejects_incomparable_runs(wrong_encoder_run, change):
    other = deepcopy(wrong_encoder_run)
    other['seed'] = 8
    if change == 'epochs':
        other['metadata']['configuration']['training']['epochs'] = 10
    elif change == 'environment':
        other['metadata']['software']['torch'] = 'different version'
    else:
        del other[change]
    with pytest.raises(ValueError):
        experiment.summarize([wrong_encoder_run, other])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True])
def test_summary_rejects_invalid_measurements(wrong_encoder_run, value):
    wrong_encoder_run['random_corruption']['uncoded']['1']['task_accuracy'] = value
    with pytest.raises(ValueError, match='finite numbers'):
        experiment.summarize([wrong_encoder_run])


def test_empty_summary_is_explicit_error():
    with pytest.raises(ValueError, match='at least one'):
        experiment.summarize([])


@pytest.mark.parametrize('epochs', [0, -1, 1.5, True])
def test_invalid_run_settings_fail_before_seed_or_output(monkeypatch, tmp_path, epochs):
    monkeypatch.setattr(experiment, 'seed_everything', lambda _: pytest.fail('RNG state must not change'))
    output = tmp_path / 'new-output'
    with pytest.raises(ValueError, match='epochs'):
        experiment.run_seed(7, output, epochs=epochs)
    assert not output.exists()
