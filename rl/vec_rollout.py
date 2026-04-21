"""Vectorized rollout: N parallel games stepped in a single numpy op, with
batched Q-net forwards for both learner and opponent.

Backed by `VectorSnakePongGame`. All per-step work (env stepping + obs
building + Q forwards) is batched. This is typically 5-10x faster than the
earlier Python-loop version.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .env import COLS, ROWS, VectorSnakePongGame
from .dqn import N_ACTIONS, ReplayBuffer, Transition


# ── Action mirror array (0=up,1=down,2=left,3=right): for side-2 players the
# x-axis is mirrored (left<->right, up/down unchanged).
_MIRROR_ACTION = np.array([0, 1, 3, 2], dtype=np.int64)


def _build_obs_batch(
    vec: VectorSnakePongGame,
    learner_sides: np.ndarray,  # (N,) int in {1,2}
    interp_ball: bool,
) -> np.ndarray:
    """Egocentric flat obs vector for each of N games. Shape (N, 4*L+4)."""
    n = vec.n
    L = vec.snake_length
    mult = vec.snake_multiplier
    nx, ny = COLS - 1, ROWS - 1

    mirror = (learner_sides == 2)  # (N,) bool
    # Own / opp bodies: pick by learner_side. Build (N, 2, L, 2) indexed.
    # Simpler: we'll make a (N, L, 2) for own, (N, L, 2) for opp.
    bodies = vec.bodies  # (N, 2, L, 2)
    own_bodies = np.where(
        learner_sides[:, None, None] == 1,
        bodies[:, 0, :, :],
        bodies[:, 1, :, :],
    )  # (N, L, 2)
    opp_bodies = np.where(
        learner_sides[:, None, None] == 1,
        bodies[:, 1, :, :],
        bodies[:, 0, :, :],
    )  # (N, L, 2)

    own_body_float = own_bodies.astype(np.float32)  # (N, L, 2)
    opp_body_float = opp_bodies.astype(np.float32)

    # Mirror x for side-2 players
    if mirror.any():
        own_body_float[mirror, :, 0] = nx - own_body_float[mirror, :, 0]
        opp_body_float[mirror, :, 0] = nx - opp_body_float[mirror, :, 0]

    own_body_float[..., 0] /= nx
    own_body_float[..., 1] /= ny
    opp_body_float[..., 0] /= nx
    opp_body_float[..., 1] /= ny

    own_flat = own_body_float.reshape(n, L * 2)
    opp_flat = opp_body_float.reshape(n, L * 2)

    # Ball: optionally interpolate by phase.
    phase = vec.phase  # (N,) int
    bx = vec.balls[:, 0].astype(np.float32)
    by = vec.balls[:, 1].astype(np.float32)
    vx = vec.balls[:, 2].astype(np.float32)
    vy = vec.balls[:, 3].astype(np.float32)

    if interp_ball and mult > 1:
        frac = phase.astype(np.float32) / mult
        bx = bx + frac * vx
        by = by + frac * vy
        vx = vx / mult
        vy = vy / mult

    # Mirror x for side-2 players (both position and vx).
    if mirror.any():
        bx = np.where(mirror, nx - bx, bx)
        vx = np.where(mirror, -vx, vx)

    ball = np.stack([bx / nx, by / ny, vx, vy], axis=-1).astype(np.float32)  # (N, 4)
    return np.concatenate([own_flat, opp_flat, ball], axis=1)


@torch.no_grad()
def _batched_q_actions(
    q_net: nn.Module,
    obs_batch: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    obs_t = torch.from_numpy(obs_batch).to(device)
    q = q_net(obs_t)
    actions = q.argmax(dim=1).cpu().numpy()
    if epsilon > 0.0:
        mask = rng.random(len(obs_batch)) < epsilon
        if mask.any():
            actions[mask] = rng.integers(0, N_ACTIONS, size=int(mask.sum()))
    return actions.astype(np.int64)


class VecRollout:
    """Owns N parallel games (via VectorSnakePongGame). `collect()` advances
    all of them and auto-resets on terminal/truncation.
    """

    def __init__(
        self,
        n_envs: int,
        q_net: nn.Module,
        device: torch.device,
        snake_length: int = 4,
        snake_multiplier: int = 1,
        max_steps: int = 500,
        interp_ball: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_envs = n_envs
        self.q_net = q_net
        self.device = device
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.max_steps = max_steps
        self.interp_ball = interp_ball
        self._rng = rng if rng is not None else np.random.default_rng()

        self.vec = VectorSnakePongGame(
            n_games=n_envs, snake_length=snake_length,
            snake_multiplier=snake_multiplier,
            seed=int(self._rng.integers(1 << 31)),
        )
        self.learner_sides = np.where(
            self._rng.random(n_envs) < 0.5, 1, 2,
        ).astype(np.int64)
        self.ep_lengths = np.zeros(n_envs, dtype=np.int64)

        self._opp_q: Optional[nn.Module] = None
        self._opp_is_random: bool = True
        self._opp_epsilon: float = 0.0

    def set_opponent(self, opp_q_net: Optional[nn.Module], opp_epsilon: float = 0.05) -> None:
        self._opp_q = opp_q_net
        self._opp_is_random = opp_q_net is None
        self._opp_epsilon = opp_epsilon

    def _reset_mask(self, mask: np.ndarray) -> None:
        if not mask.any():
            return
        self.vec.reset(mask)
        # Re-pick sides for reset games
        k = int(mask.sum())
        self.learner_sides[mask] = np.where(self._rng.random(k) < 0.5, 1, 2)
        self.ep_lengths[mask] = 0

    def collect(
        self,
        replay: ReplayBuffer,
        n_transitions: int,
        epsilon: float,
    ) -> list[dict]:
        """Run until `n_transitions` learner-perspective transitions are pushed
        to `replay`. Returns list of stats dicts (one per completed episode).
        """
        completed: list[dict] = []
        collected = 0
        n = self.n_envs

        while collected < n_transitions:
            # Build learner + opponent obs batches.
            learner_obs = _build_obs_batch(self.vec, self.learner_sides, self.interp_ball)
            opp_sides = 3 - self.learner_sides
            if self._opp_is_random:
                opp_actions_ego = self._rng.integers(0, N_ACTIONS, size=n).astype(np.int64)
            else:
                opp_obs = _build_obs_batch(self.vec, opp_sides, self.interp_ball)
                opp_actions_ego = _batched_q_actions(
                    self._opp_q, opp_obs, self._opp_epsilon, self.device, self._rng,
                )

            learner_actions_ego = _batched_q_actions(
                self.q_net, learner_obs, epsilon, self.device, self._rng,
            )

            # Map egocentric actions → real-board actions (mirror if side 2).
            learner_real = np.where(
                self.learner_sides == 1, learner_actions_ego,
                _MIRROR_ACTION[learner_actions_ego],
            )
            opp_real = np.where(
                opp_sides == 1, opp_actions_ego,
                _MIRROR_ACTION[opp_actions_ego],
            )

            # Pack (N, 2) actions: column 0 = s1 action, column 1 = s2 action.
            actions = np.zeros((n, 2), dtype=np.int64)
            a1 = np.where(self.learner_sides == 1, learner_real, opp_real)
            a2 = np.where(self.learner_sides == 1, opp_real, learner_real)
            actions[:, 0] = a1
            actions[:, 1] = a2

            result = self.vec.step(actions)
            self.ep_lengths += 1

            # Rewards (learner POV)
            scorer = result["scorer"]  # int in {0,1,2,3} (3 = draw)
            won = (scorer == self.learner_sides)
            lost = (scorer != 0) & (scorer != 3) & ~won
            rewards = np.where(won, 1.0, np.where(lost, -1.0, 0.0)).astype(np.float32)

            terminated = result["terminated"] | (scorer == 3)
            truncated = (~terminated) & (self.ep_lengths >= self.max_steps)

            # Build next-obs (post-step). Truncated games get their current
            # obs as "next" and a non-terminal done flag (False).
            next_obs = _build_obs_batch(self.vec, self.learner_sides, self.interp_ball)

            # Push transitions
            for i in range(n):
                replay.push(Transition(
                    obs=learner_obs[i],
                    action=int(learner_actions_ego[i]),
                    reward=float(rewards[i]),
                    next_obs=next_obs[i],
                    done=bool(terminated[i]),
                ))
            collected += n

            # Record completed episodes + reset.
            reset_mask = terminated | truncated
            if reset_mask.any():
                idxs = np.nonzero(reset_mask)[0]
                for i in idxs:
                    si = int(scorer[i])
                    is_term = bool(terminated[i])
                    is_trunc = bool(truncated[i]) and not is_term
                    w = bool((si == int(self.learner_sides[i])))
                    term_label = (
                        "truncated" if is_trunc
                        else ("draw" if si == 3 else "scored")
                    )
                    completed.append({
                        "length": int(self.ep_lengths[i]),
                        "won": w,
                        "scorer": si,
                        "terminal": term_label,
                    })
                # Clear `done` before reset so we can step again.
                self.vec.done[reset_mask] = False
                self._reset_mask(reset_mask)

            if collected >= n_transitions:
                break

        return completed
