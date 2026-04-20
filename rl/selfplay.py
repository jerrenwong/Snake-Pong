"""Self-play opponent pool + benchmark set.

Anti-catastrophic-forgetting strategy
-------------------------------------
The training pool uses **reservoir sampling** so every historical snapshot
has an equal probability of staying in the pool. One slot is reserved for
the most recent snapshot so the learner always sees a current-skill
opponent. Sampling during matches is uniform over the pool.

A separate **BenchmarkSet** holds fixed snapshots (taken at 0%, 25%, 50%,
75% milestones plus rolling-latest) used only for evaluation. Win rate
against each benchmark is logged to reveal regression.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .dqn import QNetwork, epsilon_greedy_action
from .gym_env import OpponentPolicy


def make_policy(
    q_net: QNetwork, device: torch.device, epsilon: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> OpponentPolicy:
    """Return an OpponentPolicy backed by a frozen Q-net."""
    q_net = q_net.eval()
    r = rng if rng is not None else np.random.default_rng()

    def policy(obs):
        return epsilon_greedy_action(q_net, obs, epsilon, device, r)

    return policy


def random_policy(rng: Optional[np.random.Generator] = None) -> OpponentPolicy:
    r = rng if rng is not None else np.random.default_rng()

    def policy(obs):
        return int(r.integers(0, 4))

    return policy


def _freeze_snapshot(q_net: QNetwork, device: torch.device) -> QNetwork:
    snap = copy.deepcopy(q_net).to(device)
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


class OpponentPool:
    """Reservoir-sampled pool of historical snapshots, with a reserved slot
    for the most recent snapshot and a configurable fraction of random-opponent
    samples.
    """

    def __init__(
        self,
        device: torch.device,
        max_snapshots: int = 30,
        random_prob: float = 0.25,
        opponent_epsilon: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        self.device = device
        self.max_snapshots = max_snapshots
        self.random_prob = random_prob
        self.opponent_epsilon = opponent_epsilon
        self._rng = rng if rng is not None else np.random.default_rng()

        # Reservoir over non-latest slots.
        self._reservoir: list[QNetwork] = []
        self._latest: Optional[QNetwork] = None
        self._total_added: int = 0  # total snapshots ever added

    def add_snapshot(self, q_net: QNetwork) -> None:
        """Add a new snapshot. Latest always kept; reservoir-sample the rest."""
        new_snap = _freeze_snapshot(q_net, self.device)
        self._total_added += 1

        prev_latest = self._latest
        self._latest = new_snap

        # The previous latest is now a candidate for reservoir placement.
        if prev_latest is None:
            return

        reservoir_cap = max(0, self.max_snapshots - 1)
        if reservoir_cap == 0:
            return

        # Reservoir sampling over the stream of "previous latest" snapshots.
        # self._total_added counts *all* snapshots; the reservoir stream has
        # (total_added - 1) items (the previous latest of each add). The
        # current candidate's 1-based index is total_added - 1.
        stream_index = self._total_added - 1  # 1, 2, 3, ...
        if stream_index <= reservoir_cap:
            self._reservoir.append(prev_latest)
        else:
            j = int(self._rng.integers(0, stream_index))
            if j < reservoir_cap:
                self._reservoir[j] = prev_latest
        # (If not accepted, prev_latest is discarded.)

    def _all_snapshots(self) -> list[QNetwork]:
        out = list(self._reservoir)
        if self._latest is not None:
            out.append(self._latest)
        return out

    def sample(self) -> OpponentPolicy:
        pool = self._all_snapshots()
        if not pool or self._rng.random() < self.random_prob:
            return random_policy(self._rng)
        idx = int(self._rng.integers(0, len(pool)))
        return make_policy(pool[idx], self.device, self.opponent_epsilon, self._rng)

    def __len__(self) -> int:
        return len(self._all_snapshots())


@dataclass
class Benchmark:
    name: str
    policy: OpponentPolicy


class BenchmarkSet:
    """Fixed set of opponents for evaluation. Captures snapshots at
    user-specified iteration milestones plus a rolling `snap_latest`.
    """

    def __init__(
        self,
        device: torch.device,
        total_iters: int,
        rng: Optional[np.random.Generator] = None,
        opponent_epsilon: float = 0.0,  # greedy for fair benchmarking
    ):
        self.device = device
        self.total_iters = total_iters
        self._rng = rng if rng is not None else np.random.default_rng()
        self.opponent_epsilon = opponent_epsilon
        # Milestone iter -> name
        q1 = max(1, total_iters // 4)
        q2 = max(2, total_iters // 2)
        q3 = max(3, (3 * total_iters) // 4)
        self._milestones: dict[int, str] = {q1: "snap_25", q2: "snap_50", q3: "snap_75"}

        self._snapshots: dict[str, QNetwork] = {}  # name -> frozen net
        self._first_taken = False
        self._latest: Optional[QNetwork] = None

    def on_iter_end(self, iter_1based: int, q_net: QNetwork) -> None:
        """Call at end of each iteration. Captures `snap_first` on first
        call, milestone snaps, and updates rolling latest.
        """
        if not self._first_taken:
            self._snapshots["snap_first"] = _freeze_snapshot(q_net, self.device)
            self._first_taken = True
        if iter_1based in self._milestones:
            self._snapshots[self._milestones[iter_1based]] = _freeze_snapshot(q_net, self.device)
        self._latest = _freeze_snapshot(q_net, self.device)

    def names(self) -> list[str]:
        names = ["random"]
        for key in ("snap_first", "snap_25", "snap_50", "snap_75"):
            if key in self._snapshots:
                names.append(key)
        if self._latest is not None:
            names.append("snap_latest")
        return names

    def policy_for(self, name: str) -> Optional[OpponentPolicy]:
        if name == "random":
            return random_policy(self._rng)
        if name == "snap_latest":
            if self._latest is None:
                return None
            return make_policy(self._latest, self.device, self.opponent_epsilon, self._rng)
        snap = self._snapshots.get(name)
        if snap is None:
            return None
        return make_policy(snap, self.device, self.opponent_epsilon, self._rng)
