"""Self-play opponent pool: freeze snapshots and sample among them."""
from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch

from .dqn import QNetwork, greedy_action, epsilon_greedy_action
from .gym_env import OpponentPolicy


def make_policy(q_net: QNetwork, device: torch.device, epsilon: float = 0.05,
                rng: Optional[np.random.Generator] = None) -> OpponentPolicy:
    """Return an OpponentPolicy (obs-dict -> action) backed by a frozen Q-net."""
    q_net = q_net.eval()
    r = rng if rng is not None else np.random.default_rng()

    def policy(obs: dict) -> int:
        return epsilon_greedy_action(q_net, obs, epsilon, device, r)

    return policy


def random_policy(rng: Optional[np.random.Generator] = None) -> OpponentPolicy:
    r = rng if rng is not None else np.random.default_rng()

    def policy(obs: dict) -> int:
        return int(r.integers(0, 4))

    return policy


class OpponentPool:
    """Stores frozen Q-network snapshots and samples opponent policies from them.

    Also keeps a weighted chance of sampling a random-action opponent to prevent
    self-play collapse (important while the agent is still bad).
    """

    def __init__(
        self,
        device: torch.device,
        max_snapshots: int = 10,
        random_prob: float = 0.25,
        opponent_epsilon: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        self.device = device
        self.max_snapshots = max_snapshots
        self.random_prob = random_prob
        self.opponent_epsilon = opponent_epsilon
        self._snapshots: list[QNetwork] = []
        self._rng = rng if rng is not None else np.random.default_rng()

    def add_snapshot(self, q_net: QNetwork) -> None:
        snap = copy.deepcopy(q_net).to(self.device)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self._snapshots.append(snap)
        if len(self._snapshots) > self.max_snapshots:
            # Drop the oldest (but keep the first in case we want a reference).
            # Simple FIFO after the first.
            self._snapshots.pop(1 if len(self._snapshots) > 1 else 0)

    def sample(self) -> OpponentPolicy:
        if not self._snapshots or self._rng.random() < self.random_prob:
            return random_policy(self._rng)
        # Bias towards recent snapshots (but not exclusively): linear weighting.
        n = len(self._snapshots)
        weights = np.arange(1, n + 1, dtype=np.float64)
        weights /= weights.sum()
        idx = int(self._rng.choice(n, p=weights))
        return make_policy(self._snapshots[idx], self.device, self.opponent_epsilon, self._rng)

    def __len__(self) -> int:
        return len(self._snapshots)
