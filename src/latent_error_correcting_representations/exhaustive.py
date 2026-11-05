"""Complete finite code-space enumeration with integer outcome denominators."""

from itertools import combinations
import torch
from .selective import (
    SELECTIVE_METHODS,
    messages,
    selective_encode,
    selective_decode,
    outcome_counts,
    nearest_valid_replacement,
)


def enumerate_codes():
    truth = messages()
    output = {}
    for method in SELECTIVE_METHODS:
        codebook = selective_encode(method, truth)
        width = codebook.shape[1]
        distance = (codebook[:, None] != codebook[None]).sum(dim=2)
        minimum = int(distance[~torch.eye(16, dtype=torch.bool)].min().item())
        budgets = {}
        for k in range(width + 1):
            patterns = list(combinations(range(width), k))
            altered = codebook[:, None].expand(-1, len(patterns), -1).clone()
            for index, positions in enumerate(patterns):
                if positions:
                    altered[:, index, list(positions)] ^= 1
            reference = truth[:, None].expand(-1, len(patterns), -1).reshape(-1, 4)
            budgets[str(k)] = outcome_counts(
                selective_decode(method, altered.reshape(-1, width)), reference
            )
        output[method] = {
            "message_bits": 4,
            "encoded_bits": width,
            "redundancy_bits": width - 4,
            "code_rate": 4 / width,
            "minimum_distance": minimum,
            "exact_bit_flips": budgets,
            "nearest_valid_replacement": outcome_counts(
                selective_decode(method, nearest_valid_replacement(method, truth)),
                truth,
            ),
        }
    return {
        "schema_version": 1,
        "enumeration": "Every one of 16 messages times every binary corruption mask for each policy.",
        "message_mask_pairs": sum(
            sum(row["n"] for row in value["exact_bit_flips"].values())
            for value in output.values()
        ),
        "interpretation": "Exact finite counts, not statistical estimates or independent trained-model trials.",
        "methods": output,
    }
