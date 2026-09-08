import pytest
import torch

from latent_error_correcting_representations.metrics import evaluate


def test_accuracy_uses_every_bit_and_task():
    truth = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    prediction = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]])
    result = evaluate(prediction, truth, torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert result.message_bit_accuracy == 7 / 8
    assert result.task_accuracy == 1 / 2
    assert result.detection_rate == result.correction_rate == 1 / 2


@pytest.mark.parametrize('truth', [torch.zeros(4), torch.zeros((1, 4)), torch.zeros((2, 5))])
def test_metrics_do_not_broadcast_incompatible_ground_truth(truth):
    with pytest.raises(ValueError):
        evaluate(torch.zeros((2, 4)), truth, torch.zeros(2), torch.zeros(2))


@pytest.mark.parametrize('flags', [torch.zeros((2, 1)), torch.zeros(1), torch.tensor([0.0, float('nan')]), torch.tensor([0, 2])])
@pytest.mark.parametrize('which', ['detected', 'corrected'])
def test_metrics_reject_incomplete_or_invalid_flags(flags, which):
    kwargs = dict(detected=torch.zeros(2), corrected=torch.zeros(2))
    kwargs[which] = flags
    with pytest.raises(ValueError):
        evaluate(torch.zeros((2, 4)), torch.zeros((2, 4)), **kwargs)


def test_empty_metrics_cannot_produce_nan_report():
    with pytest.raises(ValueError, match='non-empty'):
        evaluate(torch.empty((0, 4)), torch.empty((0, 4)), torch.empty(0), torch.empty(0))
