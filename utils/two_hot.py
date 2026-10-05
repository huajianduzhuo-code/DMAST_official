import numpy as np

import torch
import torch.nn.functional as F


def symlog(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)
    else:
        return np.sign(x) * np.log(np.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.FloatTensor, x_min: int = -1, x_max: int = 1, num_buckets: int = 255) -> torch.FloatTensor:
    x = torch.clip(x, x_min + 1e-4, x_max - 1e-4)
    buckets = torch.linspace(x_min, x_max, num_buckets, device=x.device, dtype=x.dtype)
    k = torch.searchsorted(buckets, x) - 1
    k = torch.clip(k, 0, num_buckets - 2)
    values = torch.stack((buckets[k + 1] - x, x - buckets[k]), dim=-1) / (buckets[k + 1] - buckets[k]).unsqueeze(-1)
    two_hots = torch.scatter(x.new_zeros(*x.size(), num_buckets), dim=-1, index=torch.stack((k, k + 1), dim=-1), src=values)

    return two_hots


def compute_softmax_over_buckets(logits: torch.FloatTensor, x_min: int = -1, x_max: int = 1) -> torch.FloatTensor:
    num_buckets = logits.size(-1)
    buckets = torch.linspace(x_min, x_max, num_buckets, device=logits.device, dtype=logits.dtype)
    probs = F.softmax(logits, dim=-1)

    return probs @ buckets
