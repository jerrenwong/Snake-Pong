"""Vectorized rollout: run N parallel SnakePongGame instances and batch the
Q-network forward passes for both learner and opponent actions.

Each env step still runs as pure Python (no numpy-vectorized game sim) but
the big cost — the Q-net forward — is amortized over N envs in a single
GPU call. With N=32–64 this is typically ~5–10× faster than one-at-a-time
rollouts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .env import SnakePongGame
from .gym_env import _build_obs, _mirror_action
from .dqn import QNetwork, N_ACTIONS, ReplayBuffer, Transition


@torch.no_grad()
def _batched_q_actions(
    q_net: QNetwork,
    obs_batch: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """One batched forward pass + per-env ε-greedy. Returns (N,) int64."""
    obs_t = torch.from_numpy(obs_batch).to(device)
    q = q_net(obs_t)
    actions = q.argmax(dim=1).cpu().numpy()
    if epsilon > 0.0:
        mask = rng.random(len(obs_batch)) < epsilon
        if mask.any():
            actions[mask] = rng.integers(0, N_ACTIONS, size=int(mask.sum()))
    return actions.astype(np.int64)


class VecRollout:
    """Owns N parallel games. `collect(replay, n_steps)` advances all N envs
    for `n_steps` ticks (auto-resetting terminated games) and returns a list
    of completed-episode stat dicts.
    """

    def __init__(
        self,
        n_envs: int,
        q_net: QNetwork,
        device: torch.device,
        snake_length: int = 4,
        snake_multiplier: int = 1,
        max_steps: int = 500,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_envs = n_envs
        self.q_net = q_net
        self.device = device
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.max_steps = max_steps
        self._rng = rng if rng is not None else np.random.default_rng()

        self.games: list[SnakePongGame] = []
        self.learner_sides: np.ndarray = np.zeros(n_envs, dtype=np.int64)
        self.ep_lengths: np.ndarray = np.zeros(n_envs, dtype=np.int64)
        # Opponent config — set via set_opponent()
        self._opp_q_net: Optional[QNetwork] = None
        self._opp_is_random: bool = True
        self._opp_epsilon: float = 0.0

        self._init_envs()

    def _init_envs(self) -> None:
        self.games = [
            SnakePongGame(self.snake_length, snake_multiplier=self.snake_multiplier, seed=int(self._rng.integers(1 << 31)))
            for _ in range(self.n_envs)
        ]
        for i in range(self.n_envs):
            self.learner_sides[i] = 1 if self._rng.random() < 0.5 else 2
            self.ep_lengths[i] = 0

    def set_opponent(
        self,
        opp_q_net: Optional[QNetwork],
        opp_epsilon: float = 0.05,
    ) -> None:
        """Configure opponent. opp_q_net=None → uniform random opponent."""
        self._opp_q_net = opp_q_net
        self._opp_is_random = opp_q_net is None
        self._opp_epsilon = opp_epsilon

    def _reset_env(self, i: int) -> None:
        self.games[i] = SnakePongGame(
            self.snake_length,
            snake_multiplier=self.snake_multiplier,
            seed=int(self._rng.integers(1 << 31)),
        )
        self.learner_sides[i] = 1 if self._rng.random() < 0.5 else 2
        self.ep_lengths[i] = 0

    def _build_learner_obs_batch(self) -> np.ndarray:
        return np.stack([
            _build_obs(self.games[i], int(self.learner_sides[i]))
            for i in range(self.n_envs)
        ], axis=0)

    def _build_opp_obs_batch(self) -> np.ndarray:
        return np.stack([
            _build_obs(self.games[i], 3 - int(self.learner_sides[i]))
            for i in range(self.n_envs)
        ], axis=0)

    def collect(
        self,
        replay: ReplayBuffer,
        n_transitions: int,
        epsilon: float,
    ) -> list[dict]:
        """Run until `n_transitions` learner-perspective transitions are
        pushed to `replay`. Returns list of stats dicts, one per episode
        that completed during this call.
        """
        completed: list[dict] = []
        collected = 0

        while collected < n_transitions:
            # Build learner + opponent obs batches.
            learner_obs = self._build_learner_obs_batch()
            if self._opp_is_random:
                opp_actions = self._rng.integers(0, N_ACTIONS, size=self.n_envs).astype(np.int64)
            else:
                opp_obs = self._build_opp_obs_batch()
                opp_actions = _batched_q_actions(
                    self._opp_q_net, opp_obs, self._opp_epsilon, self.device, self._rng,
                )

            learner_actions = _batched_q_actions(
                self.q_net, learner_obs, epsilon, self.device, self._rng,
            )

            # Step each env (serial Python).
            for i in range(self.n_envs):
                side = int(self.learner_sides[i])
                la = int(learner_actions[i])
                oa = int(opp_actions[i])
                real_a_learner = la if side == 1 else _mirror_action(la)
                real_a_opp = oa if (3 - side) == 1 else _mirror_action(oa)
                if side == 1:
                    a1, a2 = real_a_learner, real_a_opp
                else:
                    a1, a2 = real_a_opp, real_a_learner

                result = self.games[i].step(a1, a2)
                self.ep_lengths[i] += 1

                if result.scorer is None or result.scorer == 0:
                    reward = 0.0
                elif result.scorer == side:
                    reward = 1.0
                else:
                    reward = -1.0

                terminated = self.games[i].done
                truncated = (not terminated) and self.ep_lengths[i] >= self.max_steps

                next_obs = _build_obs(self.games[i], side)
                replay.push(Transition(
                    obs=learner_obs[i],
                    action=la,
                    reward=reward,
                    next_obs=next_obs,
                    done=bool(terminated),
                ))
                collected += 1

                if terminated or truncated:
                    completed.append({
                        "length": int(self.ep_lengths[i]),
                        "won": bool(result.scorer == side),
                        "scorer": result.scorer,
                        "terminal": "truncated" if truncated and not terminated else
                                    ("scored" if result.scorer in (1, 2) else "draw"),
                    })
                    self._reset_env(i)

                if collected >= n_transitions:
                    break

        return completed
