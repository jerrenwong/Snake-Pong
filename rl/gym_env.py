"""Single-agent gymnasium wrapper over SnakePongGame with self-play opponent.

Design
------
- The environment exposes one snake (the "learner") and wraps the other with a
  caller-supplied policy.
- Observations are *egocentric*: always presented as if the learner is on the
  left side. When the learner plays as s2 (right), we mirror the x-axis and
  flip actions / x-velocity accordingly.
- Reward is zero-sum: +1 on scoring, -1 on being scored / dying.

Observation is a flat vector:
    [own body (head-first, padded to snake_length) as (x,y) pairs normalized,
     opp body same,
     ball (x, y, vx, vy) — position normalized, velocity in {-1,0,1}]

Snake bodies have a fixed length (no power-ups), so no padding mask needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import COLS, ROWS, SnakePongGame, Snake


# Action mirror for right-side player (mirror x axis):
#   up<->up, down<->down, left<->right
_MIRROR_ACTION = {0: 0, 1: 1, 2: 3, 3: 2}


OpponentPolicy = Callable[[np.ndarray], int]
"""A callable that takes an egocentric observation array and returns an action 0-3."""


def obs_dim(snake_length: int) -> int:
    # own body: snake_length * 2 + opp body: snake_length * 2 + ball: 4
    return 4 * snake_length + 4


def random_opponent(obs: np.ndarray, rng: np.random.Generator | None = None) -> int:
    r = rng if rng is not None else np.random.default_rng()
    return int(r.integers(0, 4))


def _encode_body(snake: Snake, mirror: bool) -> np.ndarray:
    out = np.empty(len(snake.body) * 2, dtype=np.float32)
    nx = COLS - 1
    ny = ROWS - 1
    for i, (x, y) in enumerate(snake.body):
        xm = (nx - x) if mirror else x
        out[2 * i] = xm / nx
        out[2 * i + 1] = y / ny
    return out


def _build_obs(game: SnakePongGame, as_player: int) -> np.ndarray:
    """Flat egocentric observation. `as_player` is 1 or 2."""
    mirror = as_player == 2
    own = game.s1 if as_player == 1 else game.s2
    opp = game.s2 if as_player == 1 else game.s1

    own_body = _encode_body(own, mirror)
    opp_body = _encode_body(opp, mirror)

    bx = (COLS - 1 - game.ball.x) if mirror else game.ball.x
    ball_vx = -game.ball.vx if mirror else game.ball.vx
    ball = np.array([
        bx / (COLS - 1),
        game.ball.y / (ROWS - 1),
        float(ball_vx),
        float(game.ball.vy),
    ], dtype=np.float32)

    return np.concatenate([own_body, opp_body, ball], axis=0)


def _mirror_action(action: int) -> int:
    return _MIRROR_ACTION[int(action)]


@dataclass
class EpisodeStats:
    length: int = 0
    scorer: Optional[int] = None
    terminal: str = ""


class SnakePongSelfPlayEnv(gym.Env):
    """Single-agent env where the learner plays one side; opponent is a fixed policy."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_policy: Optional[OpponentPolicy] = None,
        learner_side: Literal[1, 2, "random"] = "random",
        snake_length: int = 4,
        snake_multiplier: int = 1,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.opponent_policy = opponent_policy if opponent_policy is not None else random_opponent
        self.learner_side_arg = learner_side
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._game = SnakePongGame(
            snake_length=snake_length, snake_multiplier=snake_multiplier,
            seed=int(self._rng.integers(1 << 31)),
        )
        self._learner: int = 1

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim(snake_length),), dtype=np.float32,
        )
        self._last_stats = EpisodeStats()

    def set_opponent(self, policy: OpponentPolicy) -> None:
        self.opponent_policy = policy

    def _pick_side(self) -> int:
        if self.learner_side_arg == "random":
            return 1 if self._rng.random() < 0.5 else 2
        return int(self.learner_side_arg)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._learner = self._pick_side()
        self._game = SnakePongGame(
            snake_length=self.snake_length,
            snake_multiplier=self.snake_multiplier,
            seed=int(self._rng.integers(1 << 31)),
        )
        self._last_stats = EpisodeStats()
        obs = _build_obs(self._game, as_player=self._learner)
        return obs, {"learner_side": self._learner}

    def step(self, action: int):
        real_a_learner = action if self._learner == 1 else _mirror_action(action)

        opp_side = 2 if self._learner == 1 else 1
        opp_obs = _build_obs(self._game, as_player=opp_side)
        opp_action = int(self.opponent_policy(opp_obs))
        real_a_opp = opp_action if opp_side == 1 else _mirror_action(opp_action)

        if self._learner == 1:
            a1, a2 = real_a_learner, real_a_opp
        else:
            a1, a2 = real_a_opp, real_a_learner

        result = self._game.step(a1, a2)
        self._last_stats.length += 1

        if result.scorer is None:
            reward = 0.0
        elif result.scorer == 0:
            reward = 0.0
        elif result.scorer == self._learner:
            reward = 1.0
        else:
            reward = -1.0

        terminated = self._game.done
        truncated = (not terminated) and self._last_stats.length >= self.max_steps
        if terminated:
            self._last_stats.scorer = result.scorer
            if result.s1_died and result.s2_died:
                self._last_stats.terminal = "both_died"
            elif result.s1_died:
                self._last_stats.terminal = "s1_died"
            elif result.s2_died:
                self._last_stats.terminal = "s2_died"
            else:
                self._last_stats.terminal = "scored"

        obs = _build_obs(self._game, as_player=self._learner)
        info: dict[str, Any] = {"learner_side": self._learner}
        if terminated or truncated:
            info["episode_stats"] = {
                "length": self._last_stats.length,
                "scorer": self._last_stats.scorer,
                "terminal": self._last_stats.terminal,
                "learner_won": result.scorer == self._learner,
            }
        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        rows = []
        for y in range(ROWS):
            row = []
            for x in range(COLS):
                ch = "."
                if (x, y) in self._game.s1.body[1:]:
                    ch = "o"
                if self._game.s1.head == (x, y):
                    ch = "O"
                if (x, y) in self._game.s2.body[1:]:
                    ch = "x"
                if self._game.s2.head == (x, y):
                    ch = "X"
                if (self._game.ball.x, self._game.ball.y) == (x, y):
                    ch = "*"
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows)
