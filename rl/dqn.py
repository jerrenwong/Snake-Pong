"""DQN model + replay buffer for Snake-Pong self-play.

Uses a compact MLP over a flat feature vector: positions of both snake bodies,
ball position, and ball velocity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 4


class QNetwork(nn.Module):
    """Plain MLP over the flat observation vector."""

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
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,·)).

    Shared trunk; separate value (1-D) and advantage (n_actions-D) heads.
    Output matches `QNetwork`'s (B, n_actions) shape, so it's a drop-in
    replacement elsewhere.
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

    Forward returns (B, K, n_actions): per-head Q-values. During rollouts
    pick a random head (or ensemble mean); during training each head only
    updates on a random bootstrap-sampled subset of transitions.
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
        """Returns (B, K, n_actions)."""
        h = self.trunk(obs)
        outs = torch.stack([head(h) for head in self.heads], dim=1)  # (B, K, A)
        return outs

    def mean_q(self, obs: torch.Tensor) -> torch.Tensor:
        """Ensemble mean over heads → (B, n_actions)."""
        return self.forward(obs).mean(dim=1)

    def head_q(self, obs: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Single-head Q-values → (B, n_actions)."""
        h = self.trunk(obs)
        return self.heads[head_idx](h)


def build_q_net(arch: str, obs_dim: int, n_actions: int = N_ACTIONS, n_heads: int = 5) -> nn.Module:
    """Construct a Q-network by name. Use this everywhere to keep arch central."""
    if arch == "mlp":
        return QNetwork(obs_dim, n_actions)
    if arch == "dueling":
        return DuelingQNetwork(obs_dim, n_actions)
    if arch == "bootstrapped":
        return BootstrappedQNetwork(obs_dim, n_actions, n_heads=n_heads)
    raise ValueError(f"Unknown model arch: {arch}")


@dataclass
class Transition:
    obs: np.ndarray       # (D,) float32
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool            # terminal (not truncation)
    mask: np.ndarray | None = None  # (K,) bool — Bootstrapped-DQN per-head mask. None for non-bootstrap.


class ReplayBuffer:
    """CPU-backed numpy replay buffer (paired with the numpy VecRollout).

    If `n_heads > 0`, also stores a (K,) bootstrap mask per transition
    for Bootstrapped DQN training.
    """

    def __init__(self, capacity: int, obs_dim: int, n_heads: int = 0):
        self.capacity = capacity
        self.size = 0
        self.idx = 0
        self.n_heads = n_heads
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        if n_heads > 0:
            self.mask = np.zeros((capacity, n_heads), dtype=np.float32)
        else:
            self.mask = None

    def push(self, t: Transition) -> None:
        i = self.idx
        self.obs[i] = t.obs
        self.action[i] = t.action
        self.reward[i] = t.reward
        self.next_obs[i] = t.next_obs
        self.done[i] = 1.0 if t.done else 0.0
        if self.mask is not None and t.mask is not None:
            self.mask[i] = t.mask
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        out = {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }
        if self.mask is not None:
            out["mask"] = self.mask[idx]
        return out


class GpuReplayBuffer:
    """GPU-resident replay buffer. Tensors live on `device`; push/sample
    never round-trip to CPU. Designed to pair with VecRolloutGPU.

    If `n_heads > 0`, also stores a (K,) bootstrap mask per transition.
    """

    def __init__(self, capacity: int, obs_dim: int, device: torch.device, n_heads: int = 0):
        self.capacity = capacity
        self.size = 0
        self.idx = 0
        self.device = device
        self.n_heads = n_heads
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((capacity,), dtype=torch.int64, device=device)
        self.reward = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((capacity,), dtype=torch.float32, device=device)
        if n_heads > 0:
            self.mask = torch.zeros((capacity, n_heads), dtype=torch.float32, device=device)
        else:
            self.mask = None

    def push_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Batched insert of B transitions. All tensors must be on self.device."""
        b = obs.shape[0]
        positions = (torch.arange(b, device=self.device) + self.idx) % self.capacity
        self.obs[positions] = obs
        self.action[positions] = action.to(torch.int64)
        self.reward[positions] = reward.to(torch.float32)
        self.next_obs[positions] = next_obs
        self.done[positions] = done.to(torch.float32)
        if self.mask is not None and mask is not None:
            self.mask[positions] = mask.to(torch.float32)
        self.idx = int((self.idx + b) % self.capacity)
        self.size = min(self.size + b, self.capacity)

    def sample(self, batch_size: int, generator: torch.Generator):
        idx = torch.randint(0, self.size, (batch_size,), generator=generator, device=self.device)
        out = {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }
        if self.mask is not None:
            out["mask"] = self.mask[idx]
        return out


def _to_dev(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if x.device == device else x.to(device)
    return torch.from_numpy(x).to(device)


def compute_loss_bootstrapped(
    q_net: "BootstrappedQNetwork",
    target_net: "BootstrappedQNetwork",
    batch: dict,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-head masked Double-DQN loss.

    batch must contain `mask` of shape (B, K): bool/float, True where a
    given head should update on that transition. Without masks the heads
    would converge — masks enforce bootstrap diversity.
    """
    obs = _to_dev(batch["obs"], device)
    action = _to_dev(batch["action"], device)
    reward = _to_dev(batch["reward"], device)
    next_obs = _to_dev(batch["next_obs"], device)
    done = _to_dev(batch["done"], device)
    mask = _to_dev(batch["mask"], device).to(torch.float32)  # (B, K)

    # All-heads Q for current obs: (B, K, A); gather on action dim
    q_all = q_net(obs)  # (B, K, A)
    q = q_all.gather(2, action.view(-1, 1, 1).expand(-1, q_all.size(1), 1)).squeeze(-1)  # (B, K)

    with torch.no_grad():
        # Double-DQN per head: online picks action, target evaluates
        next_q_online = q_net(next_obs)  # (B, K, A)
        next_a = next_q_online.argmax(dim=-1, keepdim=True)  # (B, K, 1)
        next_q_target = target_net(next_obs).gather(2, next_a).squeeze(-1)  # (B, K)
        target = reward.unsqueeze(-1) + gamma * (1.0 - done.unsqueeze(-1)) * next_q_target  # (B, K)

    per_example = F.smooth_l1_loss(q, target, reduction="none")  # (B, K)
    # Normalize per-head so each head gets equal weight regardless of mask mean
    denom = mask.sum(dim=0).clamp(min=1.0)
    loss = (per_example * mask).sum(dim=0) / denom  # (K,)
    return loss.mean()


def compute_loss(
    q_net: nn.Module,
    target_net: nn.Module,
    batch: dict,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    obs = _to_dev(batch["obs"], device)
    action = _to_dev(batch["action"], device)
    reward = _to_dev(batch["reward"], device)
    next_obs = _to_dev(batch["next_obs"], device)
    done = _to_dev(batch["done"], device)

    q = q_net(obs).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: action selection by online net, evaluation by target.
        next_a = q_net(next_obs).argmax(dim=1, keepdim=True)
        next_q = target_net(next_obs).gather(1, next_a).squeeze(1)
        target = reward + gamma * (1.0 - done) * next_q
    return F.smooth_l1_loss(q, target)


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(obs).unsqueeze(0).to(device)


@torch.no_grad()
def greedy_action(q_net: nn.Module, obs: np.ndarray, device: torch.device) -> int:
    q = q_net(obs_to_tensor(obs, device))
    # Bootstrapped: (1, K, A) → mean over heads
    if q.dim() == 3:
        q = q.mean(dim=1)
    return int(q.argmax(dim=1).item())


@torch.no_grad()
def epsilon_greedy_action(
    q_net: QNetwork, obs: np.ndarray, epsilon: float, device: torch.device, rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return greedy_action(q_net, obs, device)
