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


def build_q_net(arch: str, obs_dim: int, n_actions: int = N_ACTIONS) -> nn.Module:
    """Construct a Q-network by name. Use this everywhere to keep arch central."""
    if arch == "mlp":
        return QNetwork(obs_dim, n_actions)
    if arch == "dueling":
        return DuelingQNetwork(obs_dim, n_actions)
    raise ValueError(f"Unknown model arch: {arch}")


@dataclass
class Transition:
    obs: np.ndarray       # (D,) float32
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool            # terminal (not truncation)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.size = 0
        self.idx = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def push(self, t: Transition) -> None:
        i = self.idx
        self.obs[i] = t.obs
        self.action[i] = t.action
        self.reward[i] = t.reward
        self.next_obs[i] = t.next_obs
        self.done[i] = 1.0 if t.done else 0.0
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }


def compute_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    batch: dict,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    obs = torch.from_numpy(batch["obs"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    reward = torch.from_numpy(batch["reward"]).to(device)
    next_obs = torch.from_numpy(batch["next_obs"]).to(device)
    done = torch.from_numpy(batch["done"]).to(device)

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
def greedy_action(q_net: QNetwork, obs: np.ndarray, device: torch.device) -> int:
    q = q_net(obs_to_tensor(obs, device))
    return int(q.argmax(dim=1).item())


@torch.no_grad()
def epsilon_greedy_action(
    q_net: QNetwork, obs: np.ndarray, epsilon: float, device: torch.device, rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return greedy_action(q_net, obs, device)
