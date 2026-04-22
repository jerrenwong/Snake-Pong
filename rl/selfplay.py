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
import torch.nn as nn

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


# ── Scripted / heuristic opponents ───────────────────────────────────────────
# These play a fixed, *stable* policy that doesn't change with training. They
# are used in the benchmark set so "best-by-eval" picks can't be inflated by
# training-time weakening of the opponent pool.
#
# Obs layout (flat vector, egocentric — learner always on "left"):
#   [0..2L)  own body (head-first, x/y pairs normalized 0..1)
#   [2L..4L) opp body (same format)
#   [4L..4L+4)  ball (x_norm, y_norm, vx_scaled, vy_scaled)
#
# Action encoding (egocentric after mirror): 0=up, 1=down, 2=left, 3=right.
def _obs_components(obs, snake_length):
    L = snake_length
    own_head = obs[0:2]
    opp_head = obs[2*L:2*L+2]
    ball = obs[4*L:4*L+4]
    return own_head, opp_head, ball


def chase_ball_policy(snake_length: int = 4) -> OpponentPolicy:
    """Always move toward the ball. No lookahead — sometimes runs into walls,
    but gives the learner a reasonable moving target to beat.
    """
    def policy(obs):
        own_head, _, ball = _obs_components(obs, snake_length)
        dx = ball[0] - own_head[0]
        dy = ball[1] - own_head[1]
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2   # right or left
        return 1 if dy > 0 else 0        # down or up
    return policy


def defender_policy(snake_length: int = 4) -> OpponentPolicy:
    """Stay near own goal line (x ≈ 0 in egocentric view) and match the ball's
    y-position. Conservative, hard to beat by naive attackers.
    """
    # "Home x" is low-left quadrant of the board; y tracks ball.y.
    def policy(obs):
        own_head, _, ball = _obs_components(obs, snake_length)
        # Match y first; only move horizontally if already aligned.
        dy = ball[1] - own_head[1]
        if abs(dy) > 0.04:  # ~1 cell in normalized units
            return 1 if dy > 0 else 0
        # y aligned — drift toward home x (keep head near own goal, but not 0).
        home_x = 0.2
        dx = home_x - own_head[0]
        if abs(dx) > 0.04:
            return 3 if dx > 0 else 2
        # Default: stay still-ish (pick up to avoid self-collision)
        return 0
    return policy


def vertical_oscillator_policy(snake_length: int = 4) -> OpponentPolicy:
    """Bounce up and down on the board's far side. Naive but forces the
    learner to time shots, not just aim at center.
    """
    state = {"dir_down": True}
    def policy(obs):
        own_head, _, _ = _obs_components(obs, snake_length)
        if own_head[1] < 0.08:
            state["dir_down"] = True
        elif own_head[1] > 0.92:
            state["dir_down"] = False
        return 1 if state["dir_down"] else 0
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

    def sample_snapshot(self) -> tuple[Optional[QNetwork], float]:
        """Variant used by vectorized rollout: returns (q_net_or_None, epsilon).
        q_net=None means uniform-random opponent.
        """
        pool = self._all_snapshots()
        if not pool or self._rng.random() < self.random_prob:
            return None, 0.0
        idx = int(self._rng.integers(0, len(pool)))
        return pool[idx], self.opponent_epsilon

    def __len__(self) -> int:
        return len(self._all_snapshots())


@dataclass
class Benchmark:
    name: str
    policy: OpponentPolicy


class BenchmarkSet:
    """Fixed-opponent eval set.

    Two categories:
      - STABLE benchmarks: external trained checkpoints (e.g. v5, v8) loaded
        once, frozen. Their difficulty NEVER changes across training, so
        best-by-eval selection on these is monotone-comparable.
        Optional scripted bots can also be registered.
      - SNAPSHOT benchmarks (snap_first / snap_25 / snap_50 / snap_75 /
        snap_latest): self-play history. Useful for measuring forgetting,
        NOT for "best" picks since they evolve with training.

    `stable_names()` returns only stable (external + scripted) benchmarks.
    """

    def __init__(
        self,
        device: torch.device,
        total_iters: int,
        snake_length: int = 4,
        rng: Optional[np.random.Generator] = None,
        opponent_epsilon: float = 0.0,  # greedy for fair benchmarking
        include_random: bool = False,
        scripted_names: Optional[list[str]] = None,
        external_checkpoints: Optional[list[tuple[str, str]]] = None,
    ):
        """external_checkpoints: list of (name, path) tuples. Each checkpoint
        is loaded, frozen, and used as a stable benchmark opponent.
        """
        self.device = device
        self.total_iters = total_iters
        self.snake_length = snake_length
        self._rng = rng if rng is not None else np.random.default_rng()
        self.opponent_epsilon = opponent_epsilon
        self.include_random = include_random

        self._scripted_names = list(scripted_names) if scripted_names else []
        # External checkpoints: load and freeze now. Each entry is
        # (name, path) or (name, path, head_idx).
        self._external: dict[str, nn.Module] = {}
        for entry in (external_checkpoints or []):
            if len(entry) == 2:
                name, path = entry
                head = None
            elif len(entry) == 3:
                name, path, head = entry
            else:
                raise ValueError(f"Invalid external_checkpoints entry: {entry!r}")
            q = _load_external_checkpoint(path, device, head=head)
            self._external[name] = q

        q1 = max(1, total_iters // 4)
        q2 = max(2, total_iters // 2)
        q3 = max(3, (3 * total_iters) // 4)
        self._milestones: dict[int, str] = {q1: "snap_25", q2: "snap_50", q3: "snap_75"}
        self._snapshots: dict[str, QNetwork] = {}
        self._first_taken = False
        self._latest: Optional[QNetwork] = None

    def on_iter_end(self, iter_1based: int, q_net: QNetwork) -> None:
        if not self._first_taken:
            self._snapshots["snap_first"] = _freeze_snapshot(q_net, self.device)
            self._first_taken = True
        if iter_1based in self._milestones:
            self._snapshots[self._milestones[iter_1based]] = _freeze_snapshot(q_net, self.device)
        self._latest = _freeze_snapshot(q_net, self.device)

    def stable_names(self) -> list[str]:
        return list(self._external.keys()) + list(self._scripted_names)

    def names(self) -> list[str]:
        names = self.stable_names()
        if self.include_random:
            names.append("random")
        for key in ("snap_first", "snap_25", "snap_50", "snap_75"):
            if key in self._snapshots:
                names.append(key)
        if self._latest is not None:
            names.append("snap_latest")
        return names

    def policy_for(self, name: str) -> Optional[OpponentPolicy]:
        if name == "random":
            return random_policy(self._rng)
        if name == "scripted_chase":
            return chase_ball_policy(self.snake_length)
        if name == "scripted_defender":
            return defender_policy(self.snake_length)
        if name == "scripted_oscillator":
            return vertical_oscillator_policy(self.snake_length)
        if name in self._external:
            return make_policy(self._external[name], self.device, self.opponent_epsilon, self._rng)
        if name == "snap_latest":
            if self._latest is None:
                return None
            return make_policy(self._latest, self.device, self.opponent_epsilon, self._rng)
        snap = self._snapshots.get(name)
        if snap is None:
            return None
        return make_policy(snap, self.device, self.opponent_epsilon, self._rng)


class _HeadWrapper(nn.Module):
    """Freezes a bootstrapped net to a single head index during forward."""
    def __init__(self, base: nn.Module, head_idx: int):
        super().__init__()
        self.base = base
        self.head_idx = head_idx
    @torch.no_grad()
    def forward(self, obs):
        q = self.base(obs)
        # (B, K, A) -> (B, A)
        return q[:, self.head_idx, :]


def _load_external_checkpoint(path: str, device: torch.device, head: Optional[int] = None) -> nn.Module:
    """Load a saved Q-net as a frozen benchmark. `head` selects a specific
    head (0..K-1) for bootstrapped nets; None = use net's default output
    (mean-reduced by the downstream action-selection helpers).
    """
    from .dqn import build_q_net
    from .gym_env import obs_dim
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "q_net" in ckpt:
        state = ckpt["q_net"]
        cfg = ckpt.get("config", {}) or {}
    else:
        state = ckpt
        cfg = {}
    L = cfg.get("snake_length", 4)
    arch = cfg.get("model_arch", "mlp")
    n_heads = cfg.get("n_heads", 5)
    hidden = cfg.get("hidden_size", 256)
    q = build_q_net(arch, obs_dim(L), n_heads=n_heads, hidden=hidden).to(device).eval()
    q.load_state_dict(state)
    for p in q.parameters():
        p.requires_grad_(False)
    if head is not None:
        if "bootstrap" not in arch:
            raise ValueError(f"head={head} specified but ckpt {path} is non-bootstrap arch {arch}")
        q = _HeadWrapper(q, head).to(device).eval()
    return q


def parse_benchmark_spec(spec: str) -> tuple[str, str, Optional[int]]:
    """Parse 'name:path[:head=N]' into (name, path, head_or_None).

    Examples:
      v5:rl/runs/v5/best.pt             → ('v5', 'rl/runs/v5/best.pt', None)
      v8:rl/runs/v8/best.pt             → ('v8', 'rl/runs/v8/best.pt', None)
      v8_h4:rl/runs/v8/best.pt:head=4   → ('v8_h4', 'rl/runs/v8/best.pt', 4)
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid benchmark spec {spec!r}. Expected 'name:path[:head=N]'.")
    name, path = parts[0], parts[1]
    head: Optional[int] = None
    for extra in parts[2:]:
        if extra.startswith("head="):
            head = int(extra.split("=", 1)[1])
    return name, path, head
