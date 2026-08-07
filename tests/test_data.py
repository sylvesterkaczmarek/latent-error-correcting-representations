import torch
from latent_error_correcting_representations.data import make_dataset


def test_dataset_is_deterministic():
    a = make_dataset(7, n_train=32, n_test=16)
    b = make_dataset(7, n_train=32, n_test=16)
    for ta, tb in zip(a.train.tensors, b.train.tensors):
        assert torch.equal(ta, tb)
    for ta, tb in zip(a.test.tensors, b.test.tensors):
        assert torch.equal(ta, tb)
