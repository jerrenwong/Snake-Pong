"""DQN model + replay buffer for Snake-Pong self-play."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gym_env import NUM_CHANNELS, SCALAR_DIM
from .env import COLS, ROWS

N_ACTIONS = 4


class QNetwork(nn.Module):
    """Small CNN over the board + MLP head mixing in scalar features."""

    def __init__(self, n_actions: int = N_ACTIONS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        # After stride-2 conv: (64, ceil(ROWS/2), ceil(COLS/2))
        conv_h = (ROWS + 1) // 2
        conv_w = (COLS + 1) // 2
        self.flatten_dim = 64 * conv_h * conv_w
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim + SCALAR_DIM, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = self.conv(grid)
        h = h.reshape(h.size(0), -1)
        h = torch.cat([h, scalars], dim=1)
        return self.head(h)


@dataclass
class Transition:
    grid: np.ndarray      # (C, H, W) float32
    scalars: np.ndarray   # (D,) float32
    action: int
    reward: float
    next_grid: np.ndarray
    next_scalars: np.ndarray
    done: bool            # terminal (not truncation) — bootstrap stops here


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.idx = 0
        self.grid = np.zeros((capacity, NUM_CHANNELS, ROWS, COLS), dtype=np.float32)
        self.scalars = np.zeros((capacity, SCALAR_DIM), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_grid = np.zeros_like(self.grid)
        self.next_scalars = np.zeros_like(self.scalars)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def push(self, t: Transition) -> None:
        i = self.idx
        self.grid[i] = t.grid
        self.scalars[i] = t.scalars
        self.action[i] = t.action
        self.reward[i] = t.reward
        self.next_grid[i] = t.next_grid
        self.next_scalars[i] = t.next_scalars
        self.done[i] = 1.0 if t.done else 0.0
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "grid": self.grid[idx],
            "scalars": self.scalars[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_grid": self.next_grid[idx],
            "next_scalars": self.next_scalars[idx],
            "done": self.done[idx],
        }


def compute_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    batch: dict,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    grid = torch.from_numpy(batch["grid"]).to(device)
    scalars = torch.from_numpy(batch["scalars"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    reward = torch.from_numpy(batch["reward"]).to(device)
    next_grid = torch.from_numpy(batch["next_grid"]).to(device)
    next_scalars = torch.from_numpy(batch["next_scalars"]).to(device)
    done = torch.from_numpy(batch["done"]).to(device)

    q = q_net(grid, scalars).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: action selection by online net, evaluation by target.
        next_a = q_net(next_grid, next_scalars).argmax(dim=1, keepdim=True)
        next_q = target_net(next_grid, next_scalars).gather(1, next_a).squeeze(1)
        target = reward + gamma * (1.0 - done) * next_q
    return F.smooth_l1_loss(q, target)


def obs_to_tensors(obs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.from_numpy(obs["grid"]).unsqueeze(0).to(device)
    s = torch.from_numpy(obs["scalars"]).unsqueeze(0).to(device)
    return g, s


@torch.no_grad()
def greedy_action(q_net: QNetwork, obs: dict, device: torch.device) -> int:
    g, s = obs_to_tensors(obs, device)
    q = q_net(g, s)
    return int(q.argmax(dim=1).item())


@torch.no_grad()
def epsilon_greedy_action(
    q_net: QNetwork, obs: dict, epsilon: float, device: torch.device, rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return greedy_action(q_net, obs, device)
