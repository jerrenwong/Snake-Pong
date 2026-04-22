"""Action-selection helpers for single-env / per-env Q-lookups.

Batched rollout paths in vec_rollout.py / vec_rollout_gpu.py have their own
helpers. These are for code paths that work on single observations at a
time (play server, eval, play.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models import N_ACTIONS


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(obs).unsqueeze(0).to(device)


@torch.no_grad()
def greedy_action(q_net: nn.Module, obs: np.ndarray, device: torch.device) -> int:
    q = q_net(obs_to_tensor(obs, device))
    # Bootstrapped output: (1, K, A) → mean over heads.
    if q.dim() == 3:
        q = q.mean(dim=1)
    return int(q.argmax(dim=1).item())


@torch.no_grad()
def epsilon_greedy_action(
    q_net: nn.Module,
    obs: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return greedy_action(q_net, obs, device)
