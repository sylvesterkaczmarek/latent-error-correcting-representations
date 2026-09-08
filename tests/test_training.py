import random

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from latent_error_correcting_representations.data import make_dataset
from latent_error_correcting_representations.model import MessageEncoder, hard_message
from latent_error_correcting_representations.seed import seed_everything, validate_seed
from latent_error_correcting_representations.training import predict_messages, train_encoder


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_hard_message_uses_logit_sign_even_when_sigmoid_rounds_to_half(dtype):
    small = torch.finfo(dtype).eps / 4
    logits = torch.tensor([[-small, 0, small, -1]], dtype=dtype)
    assert torch.equal(hard_message(logits), torch.tensor([[0, 1, 1, 0]]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_hard_message_rejects_non_finite_logits(value):
    with pytest.raises(ValueError, match="finite real"):
        hard_message(torch.tensor([[0, 0, 0, value]]))


@pytest.mark.parametrize("name", ["input_dim", "hidden_dim"])
def test_encoder_cannot_have_an_empty_layer(name):
    with pytest.raises(ValueError, match=name):
        MessageEncoder(**{name: 0})


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed_does_not_partially_reseed_generators(seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed)
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_numpy_integer_seed_is_normalised():
    assert type(validate_seed(np.int64(7))) is int
    assert validate_seed(np.int64(7)) == 7


def test_deterministic_algorithm_failure_is_reported(monkeypatch):
    def fail(_):
        raise RuntimeError("deterministic execution unavailable")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", fail)
    with pytest.raises(RuntimeError, match="deterministic execution unavailable"):
        seed_everything(7)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"epochs": -1}, {"epochs": True}, {"batch_size": 0}, {"lr": 0}, {"lr": float("nan")}, {"lr": float("inf")}])
def test_invalid_training_options_fail_before_changing_model(kwargs):
    model = MessageEncoder()
    model.eval()
    before = {name: parameter.clone() for name, parameter in model.state_dict().items()}
    data = make_dataset(7, n_train=8, n_test=4)
    with pytest.raises(ValueError):
        train_encoder(model, data.train, seed=7, **kwargs)
    assert model.training is False
    assert all(torch.equal(before[name], parameter) for name, parameter in model.state_dict().items())


@pytest.mark.parametrize("operation", [train_encoder, predict_messages])
def test_empty_dataset_fails_with_a_clear_error(operation):
    model = MessageEncoder()
    empty = TensorDataset(torch.empty(0, 16), torch.empty(0, 4), torch.empty(0, dtype=torch.int64))
    kwargs = {"seed": 7, "epochs": 1} if operation is train_encoder else {}
    with pytest.raises(ValueError, match="dataset must not be empty"):
        operation(model, empty, **kwargs)


@pytest.mark.parametrize("operation", [train_encoder, predict_messages])
@pytest.mark.parametrize("corrupt", ["observations", "messages", "labels"])
def test_invalid_semantic_batches_are_rejected(operation, corrupt):
    model = MessageEncoder()
    tensors = [tensor.clone() for tensor in make_dataset(7, n_train=8, n_test=4).train.tensors]
    if corrupt == "observations":
        tensors[0][0, 0] = float("nan")
    elif corrupt == "messages":
        tensors[1][0, 0] = 0.5
    else:
        tensors[2][0] = (tensors[2][0] + 1) % 16
    kwargs = {"seed": 7, "epochs": 1} if operation is train_encoder else {}
    with pytest.raises(ValueError, match={"observations": "observations", "messages": "binary", "labels": "task identities"}[corrupt]):
        operation(model, TensorDataset(*tensors), **kwargs)


def test_prediction_restores_mixed_modes_and_does_not_consume_training_rng():
    model = MessageEncoder()
    model.train()
    model.net[0].eval()
    modes = [module.training for module in model.modules()]
    data = make_dataset(7, n_train=8, n_test=4)
    rng_state = torch.get_rng_state().clone()
    predict_messages(model, data.test, batch_size=2)
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_non_finite_prediction_fails_and_restores_training_mode():
    model = MessageEncoder()
    with torch.no_grad():
        model.net[-1].weight[0, 0] = float("nan")
    data = make_dataset(7, n_train=8, n_test=4)
    with pytest.raises(ValueError, match="finite real logits"):
        predict_messages(model, data.test)
    assert model.training is True


def test_non_finite_gradient_fails_before_optimizer_update():
    model = MessageEncoder()
    first = next(model.parameters())
    first.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    before = {name: parameter.clone() for name, parameter in model.state_dict().items()}
    data = make_dataset(7, n_train=8, n_test=4)
    with pytest.raises(FloatingPointError, match="gradients are not finite"):
        train_encoder(model, data.train, seed=7, epochs=1)
    assert all(torch.equal(before[name], parameter) for name, parameter in model.state_dict().items())


def test_finite_logits_with_overflowing_loss_fail_before_optimizer_update():
    model = MessageEncoder()
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.fill_(torch.finfo(torch.float32).max / 2)
    before = {name: parameter.clone() for name, parameter in model.state_dict().items()}
    data = make_dataset(7, n_train=8, n_test=4)
    with pytest.raises(FloatingPointError, match="loss is not finite"):
        train_encoder(model, data.train, seed=7, epochs=1)
    assert all(torch.equal(before[name], parameter) for name, parameter in model.state_dict().items())
