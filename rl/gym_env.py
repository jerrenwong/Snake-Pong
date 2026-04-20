"""Single-agent gymnasium wrapper over SnakePongGame with self-play opponent.

Design
------
- The environment exposes one snake (the "learner") and wraps the other with a
  caller-supplied policy.
- Observations are *egocentric*: always presented as if the learner is on the
  left side. When the learner plays as s2 (right), we mirror the board
  horizontally and flip actions/velocity accordingly.
- Reward is zero-sum: +1 on scoring, -1 on being scored / dying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import COLS, ROWS, SnakePongGame, Snake, Ball

# 6 channels: own_head, own_body, opp_head, opp_body, ball, ball_vel_dir_onehot_flat
NUM_CHANNELS = 5
SCALAR_DIM = 6  # ball_vx, ball_vy, own_dx, own_dy, opp_dx, opp_dy

# Action mirror for right-side player (mirror x axis):
#   up<->up, down<->down, left<->right
_MIRROR_ACTION = {0: 0, 1: 1, 2: 3, 3: 2}


OpponentPolicy = Callable[[dict], int]
"""A callable that takes an egocentric observation dict and returns an action 0-3."""


def random_opponent(obs: dict, rng: np.random.Generator | None = None) -> int:
    r = rng if rng is not None else np.random.default_rng()
    return int(r.integers(0, 4))


def _draw_snake(grid: np.ndarray, snake: Snake, head_ch: int, body_ch: int, mirror: bool) -> None:
    for i, (x, y) in enumerate(snake.body):
        xm = (COLS - 1 - x) if mirror else x
        if 0 <= xm < COLS and 0 <= y < ROWS:
            if i == 0:
                grid[head_ch, y, xm] = 1.0
            else:
                grid[body_ch, y, xm] = 1.0


def _build_obs(game: SnakePongGame, as_player: int) -> dict:
    """Build egocentric observation. `as_player` is 1 or 2."""
    mirror = as_player == 2
    grid = np.zeros((NUM_CHANNELS, ROWS, COLS), dtype=np.float32)

    own = game.s1 if as_player == 1 else game.s2
    opp = game.s2 if as_player == 1 else game.s1

    _draw_snake(grid, own, head_ch=0, body_ch=1, mirror=mirror)
    _draw_snake(grid, opp, head_ch=2, body_ch=3, mirror=mirror)
    bx = (COLS - 1 - game.ball.x) if mirror else game.ball.x
    if 0 <= bx < COLS and 0 <= game.ball.y < ROWS:
        grid[4, game.ball.y, bx] = 1.0

    ball_vx = -game.ball.vx if mirror else game.ball.vx
    own_dx = -own.dx if mirror else own.dx
    opp_dx = -opp.dx if mirror else opp.dx

    scalars = np.array([
        ball_vx, game.ball.vy,
        own_dx, own.dy,
        opp_dx, opp.dy,
    ], dtype=np.float32)

    return {"grid": grid, "scalars": scalars}


def _mirror_action(action: int) -> int:
    return _MIRROR_ACTION[int(action)]


@dataclass
class EpisodeStats:
    length: int = 0
    scorer: Optional[int] = None  # 1, 2, 0 (draw), or None (timeout)
    terminal: str = ""


class SnakePongSelfPlayEnv(gym.Env):
    """Single-agent env where the learner plays one side; opponent is a fixed policy."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_policy: Optional[OpponentPolicy] = None,
        learner_side: Literal[1, 2, "random"] = "random",
        snake_length: int = 4,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.opponent_policy = opponent_policy if opponent_policy is not None else random_opponent
        self.learner_side_arg = learner_side
        self.snake_length = snake_length
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._game = SnakePongGame(snake_length=snake_length, seed=int(self._rng.integers(1 << 31)))
        self._learner: int = 1

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, shape=(NUM_CHANNELS, ROWS, COLS), dtype=np.float32),
            "scalars": spaces.Box(-1.0, 1.0, shape=(SCALAR_DIM,), dtype=np.float32),
        })
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
            seed=int(self._rng.integers(1 << 31)),
        )
        self._last_stats = EpisodeStats()
        obs = _build_obs(self._game, as_player=self._learner)
        return obs, {"learner_side": self._learner}

    def step(self, action: int):
        # Learner action is egocentric. Map back to real board for the player's side.
        real_a_learner = action if self._learner == 1 else _mirror_action(action)

        # Opponent observes from its own egocentric view.
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

        # Reward for learner
        if result.scorer is None:
            reward = 0.0
        elif result.scorer == 0:  # both died — draw, no reward
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

