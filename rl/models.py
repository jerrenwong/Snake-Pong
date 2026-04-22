"""Q-network architectures for Snake-Pong DQN training.

Four variants, selected via build_q_net(arch, ...):
  - mlp: plain 4-layer MLP → (B, n_actions)
  - dueling: shared trunk + (V, A) heads → (B, n_actions) via V+A−mean(A)
  - bootstrapped: K parallel Q-heads sharing a trunk → (B, K, n_actions)
  - bootstrapped_dueling: K parallel dueling (V, A) pairs → (B, K, n_actions)

Bootstrapped variants are the main multi-head approach. The K heads share a
trunk (compute/parameter efficient) but have independent decoder MLPs.
During rollout each env picks one active head; during training, bootstrap
masks keep heads distinct.

Output shape contract:
  - mlp, dueling:                 (B, n_actions)
  - bootstrapped, bootstrapped_dueling: (B, K, n_actions)

Downstream code (loss, action selection) branches on `q.dim()`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


N_ACTIONS = 4


class QNetwork(nn.Module):
    """Plain MLP (4 layers)."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DuelingQNetwork(nn.Module):
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) − mean_a(A(s,·)).

    Shared trunk; separate value (1-D) and advantage (n_actions-D) heads.
    Output shape (B, n_actions) — drop-in for QNetwork.
    """

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=1, keepdim=True)


class BootstrappedQNetwork(nn.Module):
    """Bootstrapped DQN: K parallel Q-heads sharing a trunk.

    Forward returns (B, K, n_actions). See rl/models.py docstring.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = N_ACTIONS,
        hidden: int = 256,
        n_heads: int = 5,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )
            for _ in range(n_heads)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        return torch.stack([head(h) for head in self.heads], dim=1)

    def mean_q(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).mean(dim=1)

    def head_q(self, obs: torch.Tensor, head_idx: int) -> torch.Tensor:
        h = self.trunk(obs)
        return self.heads[head_idx](h)


class BootstrappedDuelingQNetwork(nn.Module):
    """Bootstrapped DQN where each of K heads is a dueling (V, A) pair."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = N_ACTIONS,
        hidden: int = 256,
        n_heads: int = 5,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_heads)
        ])
        self.adv_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )
            for _ in range(n_heads)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        qs = []
        for v_head, a_head in zip(self.value_heads, self.adv_heads):
            v = v_head(h)
            a = a_head(h)
            qs.append(v + a - a.mean(dim=1, keepdim=True))
        return torch.stack(qs, dim=1)


def build_q_net(
    arch: str,
    obs_dim: int,
    n_actions: int = N_ACTIONS,
    n_heads: int = 5,
    hidden: int = 256,
) -> nn.Module:
    """Central factory — all call sites go through this."""
    if arch == "mlp":
        return QNetwork(obs_dim, n_actions, hidden=hidden)
    if arch == "dueling":
        return DuelingQNetwork(obs_dim, n_actions, hidden=hidden)
    if arch == "bootstrapped":
        return BootstrappedQNetwork(obs_dim, n_actions, n_heads=n_heads, hidden=hidden)
    if arch == "bootstrapped_dueling":
        return BootstrappedDuelingQNetwork(obs_dim, n_actions, n_heads=n_heads, hidden=hidden)
    raise ValueError(f"Unknown model arch: {arch}")
