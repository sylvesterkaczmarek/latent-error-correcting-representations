import torch

from latent_error_correcting_representations.codes import hamming74_syndrome, task_label
from latent_error_correcting_representations.corruption import nearest_opposite_valid_codeword


def test_coherent_drift_is_valid_but_changes_behavior():
    messages = torch.tensor([[0, 0, 0, 0], [1, 0, 1, 1]], dtype=torch.int64)
    drifted = nearest_opposite_valid_codeword("hamming74_repair", messages)
    assert torch.equal(hamming74_syndrome(drifted), torch.zeros(messages.shape[0], dtype=torch.int64))
    decoded_messages = drifted[:, [2, 4, 5, 6]]
    assert torch.all(task_label(decoded_messages) != task_label(messages))
