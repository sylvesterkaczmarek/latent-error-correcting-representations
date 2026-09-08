import numpy as np
import pytest
import torch
from latent_error_correcting_representations.data import make_dataset


def test_dataset_is_deterministic():
    a = make_dataset(7, n_train=32, n_test=16)
    b = make_dataset(7, n_train=32, n_test=16)
    for ta, tb in zip(a.train.tensors, b.train.tensors):
        assert torch.equal(ta, tb)
    for ta, tb in zip(a.test.tensors, b.test.tensors):
        assert torch.equal(ta, tb)


def test_noiseless_observations_and_labels_match_the_semantic_message():
    bundle = make_dataset(7, n_train=32, n_test=16, noise_std=0)
    for split in (bundle.train, bundle.test):
        observations, message, labels = split.tensors
        torch.testing.assert_close(observations, (2 * message - 1) @ bundle.projection.T)
        assert torch.equal(labels, (message.to(torch.int64) * torch.tensor([1, 2, 4, 8])).sum(dim=1))


@pytest.mark.parametrize("name", ["n_train", "n_test", "input_dim"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_dataset_sizes_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        make_dataset(7, **{name: value})


@pytest.mark.parametrize("noise_std", [float("nan"), float("inf"), -0.1, True])
def test_invalid_noise_does_not_generate_misleading_observations(noise_std):
    with pytest.raises(ValueError, match="noise_std"):
        make_dataset(7, noise_std=noise_std)


def test_float32_noise_overflow_is_rejected():
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="finite in float32"):
            make_dataset(7, n_train=8, n_test=4, noise_std=1e100)
