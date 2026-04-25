"""TD losses for Snake-Pong DQN variants.

Two variants:
  - compute_loss: single-head (mlp or dueling). Standard Double-DQN.
  - compute_loss_bootstrapped: multi-head with per-head bootstrap masks.

Both accept batches with either numpy arrays or torch tensors (the
_to_dev helper handles the conversion automatically).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_dev(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if x.device == device else x.to(device)
    return torch.from_numpy(x).to(device)


def compute_loss(
    q_net: nn.Module,
    target_net: nn.Module,
    batch: dict,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Double-DQN loss. Expects q_net output shape (B, n_actions)."""
    obs = _to_dev(batch["obs"], device)
    action = _to_dev(batch["action"], device)
    reward = _to_dev(batch["reward"], device)
    next_obs = _to_dev(batch["next_obs"], device)
    done = _to_dev(batch["done"], device)

    q = q_net(obs).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_a = q_net(next_obs).argmax(dim=1, keepdim=True)
        next_q = target_net(next_obs).gather(1, next_a).squeeze(1)
        target = reward + gamma * (1.0 - done) * next_q
    return F.smooth_l1_loss(q, target)


def compute_loss_bootstrapped(
    q_net: nn.Module,
    target_net: nn.Module,
    batch: dict,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-head masked Double-DQN loss for BootstrappedQNetwork variants.

    Expects batch to include `mask` of shape (B, K). Each head k only updates
    on transitions where mask[:, k] == 1. Without masks heads would converge.
    """
    obs = _to_dev(batch["obs"], device)
    action = _to_dev(batch["action"], device)
    reward = _to_dev(batch["reward"], device)
    next_obs = _to_dev(batch["next_obs"], device)
    done = _to_dev(batch["done"], device)
    mask = _to_dev(batch["mask"], device).to(torch.float32)  # (B, K)

    q_all = q_net(obs)                                         # (B, K, A)
    q = q_all.gather(
        2, action.view(-1, 1, 1).expand(-1, q_all.size(1), 1),
    ).squeeze(-1)                                               # (B, K)

    with torch.no_grad():
        next_q_online = q_net(next_obs)                         # (B, K, A)
        next_a = next_q_online.argmax(dim=-1, keepdim=True)     # (B, K, 1)
        next_q_target = target_net(next_obs).gather(2, next_a).squeeze(-1)
        target = reward.unsqueeze(-1) + gamma * (1.0 - done.unsqueeze(-1)) * next_q_target

    per_example = F.smooth_l1_loss(q, target, reduction="none")  # (B, K)
    denom = mask.sum(dim=0).clamp(min=1.0)
    loss_per_head = (per_example * mask).sum(dim=0) / denom      # (K,)
    # Store per-head detail so train loop can log them individually.
    compute_loss_bootstrapped.last_per_head = loss_per_head.detach()
    return loss_per_head.mean()
