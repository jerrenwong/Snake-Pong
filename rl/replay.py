"""Replay buffers for Snake-Pong DQN.

Two implementations with the same interface:
  - ReplayBuffer: CPU-backed numpy arrays (paired with VecRollout)
  - GpuReplayBuffer: device-backed torch tensors (paired with VecRolloutGPU)

When `n_heads > 0`, both also store a (K,) bootstrap mask per transition
for Bootstrapped DQN training. The mask is a Bernoulli(p) vector; head k
only trains on transitions where mask[k] = 1, which prevents all heads
from converging to the same function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray       # (D,) float32
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool            # terminal (not truncation)
    mask: np.ndarray | None = None  # (K,) bool — Bootstrapped-DQN per-head mask


class ReplayBuffer:
    """CPU-backed numpy ring buffer."""

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
        self.mask = np.zeros((capacity, n_heads), dtype=np.float32) if n_heads > 0 else None

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
    """GPU-resident ring buffer. push_batch / sample stay on device.

    Paired with VecRolloutGPU so transitions never round-trip to CPU.
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
        self.mask = (
            torch.zeros((capacity, n_heads), dtype=torch.float32, device=device)
            if n_heads > 0 else None
        )

    def push_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
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
