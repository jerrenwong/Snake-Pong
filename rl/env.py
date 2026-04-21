"""Headless Python port of Snake-Pong game logic.

Mirrors src/logic.js. Grid is COLS x ROWS (width x height), origin top-left.
Two snakes: s1 on the left, s2 on the right. One ball.

Simplifications vs. the JS original:
- No power-ups, no map walls, no snake speed multipliers per snake.
- `snake_multiplier` controls ball tick rate: ball moves every `mult`-th env step.

This file provides two game implementations:
- `SnakePongGame`: scalar, one-game-at-a-time (used by eval, render, play).
- `VectorSnakePongGame`: numpy-vectorized across N games (used by training
  rollouts — no Python loop in the hot path). ~5-10x faster for large N.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

COLS = 36
ROWS = 26
WALL_L = COLS // 2 - 1
WALL_R = COLS // 2

# Action encoding (egocentric after observation mirroring):
#   0 = up, 1 = down, 2 = left, 3 = right
ACTION_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_ACTION_DELTAS_ARR = np.array(ACTION_DELTAS, dtype=np.int32)  # (4, 2)


@dataclass
class Snake:
    body: list[tuple[int, int]]
    dx: int
    dy: int

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]


@dataclass
class Ball:
    x: int
    y: int
    vx: int
    vy: int


@dataclass
class StepResult:
    scorer: Optional[Literal[1, 2, 0]]  # 1=s1, 2=s2, 0=draw (both die), None=nothing
    s1_died: bool = False
    s2_died: bool = False
    ball_bounced: bool = False


def _make_snakes(length: int) -> tuple[Snake, Snake]:
    hy = ROWS // 2
    h1x = WALL_L // 2
    h2x = WALL_R + (COLS - WALL_R) // 2
    s1 = Snake(
        body=[(h1x - i, hy) for i in range(length)],
        dx=1, dy=0,
    )
    s2 = Snake(
        body=[(h2x + i, hy) for i in range(length)],
        dx=-1, dy=0,
    )
    return s1, s2


def _make_ball(rng: np.random.Generator) -> Ball:
    # Use rng.choice (same as VectorSnakePongGame) for parity.
    side = int(rng.choice([-1, 1], size=1).item())
    vy = int(rng.choice([-1, 1], size=1).item())
    bx = WALL_L // 2 if side < 0 else WALL_R + (COLS - WALL_R) // 2
    return Ball(x=bx, y=ROWS // 2, vx=side, vy=vy)


# ──────────────────────────────────────────────────────────────────────────────
# Scalar sim (unchanged from before — for eval, render, play).
# ──────────────────────────────────────────────────────────────────────────────
class SnakePongGame:
    """Two-player Snake-Pong sim. Both players step simultaneously each frame.

    `snake_multiplier`: snake moves every env step, ball moves every
    `snake_multiplier`-th step. Matches the JS `snakeMultiplier` setting.
    """

    def __init__(
        self,
        snake_length: int = 4,
        snake_multiplier: int = 1,
        seed: Optional[int] = None,
    ):
        assert snake_multiplier >= 1
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.rng = np.random.default_rng(seed)
        self.s1: Snake
        self.s2: Snake
        self.ball: Ball
        self.steps = 0
        self.done = False
        self.reset()

    @property
    def phase(self) -> int:
        """Substep index within current ball-tick cycle, ∈ [0, snake_multiplier).

        - `0` means "ball just moved" (or reset).
        - `snake_multiplier - 1` means "ball about to move on the next step".
        """
        return self.steps % self.snake_multiplier

    def reset(self) -> None:
        self.s1, self.s2 = _make_snakes(self.snake_length)
        self.ball = _make_ball(self.rng)
        self.steps = 0
        self.done = False

    def _apply_action(self, snake: Snake, action: int) -> None:
        dx, dy = ACTION_DELTAS[action]
        if dx != 0 and snake.dx == -dx:
            return
        if dy != 0 and snake.dy == -dy:
            return
        snake.dx, snake.dy = dx, dy

    @staticmethod
    def _step_snake(snake: Snake) -> None:
        hx, hy = snake.head
        snake.body.insert(0, (hx + snake.dx, hy + snake.dy))
        snake.body.pop()

    @staticmethod
    def _snake_out_or_self(snake: Snake) -> bool:
        hx, hy = snake.head
        if hx < 0 or hx >= COLS or hy < 0 or hy >= ROWS:
            return True
        return any((cx, cy) == (hx, hy) for (cx, cy) in snake.body[1:])

    @staticmethod
    def _snakes_collide(s1: Snake, s2: Snake) -> Optional[Literal["both", "s1", "s2"]]:
        h1, h2 = s1.head, s2.head
        if h1 == h2:
            return "both"
        if h1 in s2.body:
            return "s1"
        if h2 in s1.body:
            return "s2"
        return None

    def _step_ball(self) -> tuple[Optional[Literal[1, 2]], bool]:
        ball = self.ball
        nx = ball.x + ball.vx
        ny = ball.y + ball.vy
        bounced = False
        if ny < 0 or ny >= ROWS:
            ball.vy = -ball.vy
            ny = ball.y + ball.vy
            bounced = True
        if nx < 0:
            return 2, bounced
        if nx >= COLS:
            return 1, bounced
        all_segs = set(self.s1.body) | set(self.s2.body)
        h_hit = (nx, ball.y) in all_segs
        v_hit = (ball.x, ny) in all_segs
        d_hit = (not h_hit) and (not v_hit) and ((nx, ny) in all_segs)
        if h_hit or v_hit or d_hit:
            if h_hit:
                ball.vx = -ball.vx
            if v_hit:
                ball.vy = -ball.vy
            if d_hit:
                ball.vx = -ball.vx
                ball.vy = -ball.vy
            nx = ball.x + ball.vx
            ny = max(0, min(ROWS - 1, ball.y + ball.vy))
            bounced = True
        ball.x, ball.y = nx, ny
        return None, bounced

    def step(self, action1: int, action2: int) -> StepResult:
        if self.done:
            raise RuntimeError("step() called after episode end. Call reset().")
        self.steps += 1
        self._apply_action(self.s1, action1)
        self._apply_action(self.s2, action2)
        self._step_snake(self.s1)
        self._step_snake(self.s2)

        d1 = self._snake_out_or_self(self.s1)
        d2 = self._snake_out_or_self(self.s2)
        if d1 and d2:
            self.done = True
            return StepResult(scorer=0, s1_died=True, s2_died=True)
        if d1:
            self.done = True
            return StepResult(scorer=2, s1_died=True)
        if d2:
            self.done = True
            return StepResult(scorer=1, s2_died=True)

        col = self._snakes_collide(self.s1, self.s2)
        if col == "both":
            self.done = True
            return StepResult(scorer=0, s1_died=True, s2_died=True)
        if col == "s1":
            self.done = True
            return StepResult(scorer=2, s1_died=True)
        if col == "s2":
            self.done = True
            return StepResult(scorer=1, s2_died=True)

        scorer: Optional[Literal[1, 2]] = None
        bounced = False
        if self.steps % self.snake_multiplier == 0:
            scorer, bounced = self._step_ball()
        if scorer is not None:
            self.done = True
        return StepResult(scorer=scorer, ball_bounced=bounced)


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized sim: N parallel games stepped in batched numpy ops.
# ──────────────────────────────────────────────────────────────────────────────
class VectorSnakePongGame:
    """N parallel Snake-Pong games stepped jointly in a single batched op.

    State tensors (all int32 unless noted):
        bodies: (N, 2, L, 2)   — [game, player, segment, xy]
        dirs:   (N, 2, 2)      — [game, player, xy]
        balls:  (N, 4)         — [x, y, vx, vy]
        steps:  (N,)
        done:   (N,) bool
    `phase` is `steps % snake_multiplier` (elementwise).
    """

    def __init__(
        self,
        n_games: int,
        snake_length: int = 4,
        snake_multiplier: int = 1,
        seed: Optional[int] = None,
    ):
        assert n_games > 0 and snake_length > 1 and snake_multiplier >= 1
        self.n = n_games
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.rng = np.random.default_rng(seed)

        self.bodies = np.zeros((n_games, 2, snake_length, 2), dtype=np.int32)
        self.dirs = np.zeros((n_games, 2, 2), dtype=np.int32)
        self.balls = np.zeros((n_games, 4), dtype=np.int32)
        self.steps = np.zeros(n_games, dtype=np.int32)
        self.done = np.zeros(n_games, dtype=bool)
        self._init_all(np.arange(n_games))

    @property
    def phase(self) -> np.ndarray:
        return self.steps % self.snake_multiplier

    # ── Initialization ──────────────────────────────────────────────────────
    def _init_all(self, idx: np.ndarray) -> None:
        """Initialize/reset the games at `idx` (int array)."""
        if idx.size == 0:
            return
        hy = ROWS // 2
        h1x = WALL_L // 2
        h2x = WALL_R + (COLS - WALL_R) // 2
        L = self.snake_length

        # Snake 1 body: head-first. Segment i at (h1x - i, hy).
        for i in range(L):
            self.bodies[idx, 0, i, 0] = h1x - i
            self.bodies[idx, 0, i, 1] = hy
            self.bodies[idx, 1, i, 0] = h2x + i
            self.bodies[idx, 1, i, 1] = hy
        self.dirs[idx, 0, 0] = 1
        self.dirs[idx, 0, 1] = 0
        self.dirs[idx, 1, 0] = -1
        self.dirs[idx, 1, 1] = 0

        # Random ball: side ± chooses which half, then vy ±1
        k = idx.size
        side = self.rng.choice([-1, 1], size=k).astype(np.int32)
        bx = np.where(side < 0, h1x, h2x).astype(np.int32)
        vy = self.rng.choice([-1, 1], size=k).astype(np.int32)
        self.balls[idx, 0] = bx
        self.balls[idx, 1] = hy
        self.balls[idx, 2] = side
        self.balls[idx, 3] = vy

        self.steps[idx] = 0
        self.done[idx] = False

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Reset games where `mask` is True (or all games if mask is None)."""
        if mask is None:
            idx = np.arange(self.n)
        else:
            idx = np.nonzero(mask)[0]
        self._init_all(idx)

    # ── Step ────────────────────────────────────────────────────────────────
    def _apply_actions(self, actions: np.ndarray) -> None:
        """actions: (n, 2) int in [0, 4). Updates self.dirs, respecting
        the 'cannot reverse' rule.
        """
        new_dirs = _ACTION_DELTAS_ARR[actions]  # (n, 2, 2)
        # Reversal detection: new_dir is opposite of current (and non-zero).
        opposite = (new_dirs[..., 0] == -self.dirs[..., 0]) & \
                   (new_dirs[..., 1] == -self.dirs[..., 1])
        # Note: original logic only rejects if the axis being changed is
        # opposite; but since ACTION_DELTAS entries all have exactly one
        # non-zero component, opposite detection on the full vector is the
        # same condition.
        mask = opposite[..., None]
        self.dirs = np.where(mask, self.dirs, new_dirs)

    def _step_snakes(self) -> None:
        """Shift bodies: new head = old head + dir; segments shift by 1; tail dropped."""
        new_head = self.bodies[:, :, 0, :] + self.dirs  # (n, 2, 2)
        # Shift: body[:, :, 1:] = body[:, :, :-1] then body[:, :, 0] = new_head
        self.bodies[:, :, 1:, :] = self.bodies[:, :, :-1, :]
        self.bodies[:, :, 0, :] = new_head

    def _snake_out_or_self(self) -> np.ndarray:
        """Returns (n, 2) bool: True where snake's head is out-of-bounds or on own body[1:]."""
        heads = self.bodies[:, :, 0, :]  # (n, 2, 2)
        out = (
            (heads[..., 0] < 0) | (heads[..., 0] >= COLS) |
            (heads[..., 1] < 0) | (heads[..., 1] >= ROWS)
        )
        tail = self.bodies[:, :, 1:, :]  # (n, 2, L-1, 2)
        match = (tail == heads[:, :, None, :]).all(axis=-1)  # (n, 2, L-1)
        self_hit = match.any(axis=-1)  # (n, 2)
        return out | self_hit

    def _snakes_collide(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (both_heads_same, s1_in_s2, s2_in_s1): each (n,) bool."""
        h1 = self.bodies[:, 0, 0, :]  # (n, 2)
        h2 = self.bodies[:, 1, 0, :]
        both = (h1 == h2).all(axis=-1)
        s2_body = self.bodies[:, 1, :, :]  # (n, L, 2)
        s1_body = self.bodies[:, 0, :, :]
        s1_in_s2 = (s2_body == h1[:, None, :]).all(axis=-1).any(axis=-1)
        s2_in_s1 = (s1_body == h2[:, None, :]).all(axis=-1).any(axis=-1)
        # If heads coincide, don't also count crossing-body.
        s1_in_s2 = s1_in_s2 & ~both
        s2_in_s1 = s2_in_s1 & ~both
        return both, s1_in_s2, s2_in_s1

    def _step_ball(self, ball_tick_mask: np.ndarray) -> np.ndarray:
        """Step the ball for games in `ball_tick_mask`. Returns scorer array
        shape (n,) int: 0 = no score, 1 = s1 scored, 2 = s2 scored.
        Mutates self.balls.
        """
        scorer = np.zeros(self.n, dtype=np.int8)
        if not ball_tick_mask.any():
            return scorer

        x = self.balls[:, 0].copy()
        y = self.balls[:, 1].copy()
        vx = self.balls[:, 2].copy()
        vy = self.balls[:, 3].copy()

        # Candidate next position
        ny_raw = y + vy
        # Top/bottom bounce: flip vy, recompute ny from original y
        top_bounce = (ny_raw < 0) | (ny_raw >= ROWS)
        vy = np.where(top_bounce, -vy, vy)
        ny = np.where(top_bounce, y + vy, ny_raw)

        nx = x + vx
        # Scoring: only for games where ball actually ticks
        scored_left = (nx < 0) & ball_tick_mask
        scored_right = (nx >= COLS) & ball_tick_mask
        scorer = np.where(scored_right, 1, scorer)
        scorer = np.where(scored_left, 2, scorer)

        # Games that ticked and didn't score → check body bounces
        still_active = ball_tick_mask & ~scored_left & ~scored_right

        # Flat list of all body cells per game: (n, 2*L, 2)
        all_bodies = self.bodies.reshape(self.n, 2 * self.snake_length, 2)
        h_target = np.stack([nx, y], axis=-1)  # (n, 2)
        v_target = np.stack([x, ny], axis=-1)
        d_target = np.stack([nx, ny], axis=-1)
        h_hit = (all_bodies == h_target[:, None, :]).all(axis=-1).any(axis=-1)
        v_hit = (all_bodies == v_target[:, None, :]).all(axis=-1).any(axis=-1)
        d_raw = (all_bodies == d_target[:, None, :]).all(axis=-1).any(axis=-1)
        d_hit = d_raw & ~h_hit & ~v_hit

        # Only active (non-scored) games body-bounce
        h_hit = h_hit & still_active
        v_hit = v_hit & still_active
        d_hit = d_hit & still_active
        any_hit = h_hit | v_hit | d_hit

        # Apply bounces to vx/vy
        vx = np.where(h_hit | d_hit, -vx, vx)
        vy = np.where(v_hit | d_hit, -vy, vy)

        # Recompute final position: use body-bounced velocity from original x/y
        nx_final = np.where(any_hit, x + vx, nx)
        ny_final_raw = np.where(any_hit, y + vy, ny)
        ny_final = np.clip(ny_final_raw, 0, ROWS - 1)

        # Write back only for still_active games (others: unchanged)
        self.balls[still_active, 0] = nx_final[still_active]
        self.balls[still_active, 1] = ny_final[still_active]
        self.balls[still_active, 2] = vx[still_active]
        self.balls[still_active, 3] = vy[still_active]
        return scorer

    def step(self, actions: np.ndarray) -> dict:
        """actions: (n, 2) int in [0, 4).

        Returns a dict of numpy arrays (all shape (n,) unless noted):
        - scorer: int8 in {0, 1, 2, 3}; 0 = none, 1 = s1, 2 = s2, 3 = draw (both died)
        - s1_died, s2_died: bool
        - terminated: bool (any terminal event this step)
        """
        if np.any(self.done):
            raise RuntimeError("step() called with some games done. Reset first.")
        self.steps += 1
        self._apply_actions(actions)
        self._step_snakes()

        # Deaths / collisions
        out_self = self._snake_out_or_self()  # (n, 2)
        s1_died = out_self[:, 0]
        s2_died = out_self[:, 1]
        both_heads, s1_in_s2, s2_in_s1 = self._snakes_collide()
        s1_died = s1_died | both_heads | s1_in_s2
        s2_died = s2_died | both_heads | s2_in_s1

        # Resolve death → scorer
        draw = s1_died & s2_died
        scorer = np.zeros(self.n, dtype=np.int8)
        scorer = np.where(draw, 3, scorer)
        scorer = np.where(s1_died & ~draw, 2, scorer)  # s1 died → s2 scores
        scorer = np.where(s2_died & ~draw, 1, scorer)

        died_any = s1_died | s2_died

        # Ball tick for games where no death yet, and phase says ball moves.
        ball_tick = (~died_any) & (self.steps % self.snake_multiplier == 0)
        ball_scorer = self._step_ball(ball_tick)  # 0/1/2, only filled on ticks

        # Ball score overrides for non-died games
        scorer = np.where(died_any, scorer, ball_scorer)

        terminated = scorer != 0
        self.done = self.done | terminated

        return {
            "scorer": scorer,
            "s1_died": s1_died,
            "s2_died": s2_died,
            "terminated": terminated,
        }
