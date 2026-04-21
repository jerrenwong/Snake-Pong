"""Headless Python port of Snake-Pong game logic.

Mirrors src/logic.js. Grid is COLS x ROWS (width x height), origin top-left.
Two snakes: s1 on the left, s2 on the right. One ball.

Simplifications vs. the JS original:
- No power-ups, no map walls, no snake speed multipliers.
- Snake and ball move exactly once per env step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

COLS = 36
ROWS = 26
WALL_L = COLS // 2 - 1
WALL_R = COLS // 2

# Action encoding (egocentric after observation mirroring):
#   0 = up, 1 = down, 2 = left, 3 = right
ACTION_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


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
    side = -1 if rng.random() < 0.5 else 1
    bx = WALL_L // 2 if side < 0 else WALL_R + (COLS - WALL_R) // 2
    return Ball(
        x=bx,
        y=ROWS // 2,
        vx=side,
        vy=1 if rng.random() < 0.5 else -1,
    )


class SnakePongGame:
    """Two-player Snake-Pong sim. Both players step simultaneously each frame.

    `snake_multiplier`: snake moves every env step, but the ball only moves
    every `snake_multiplier`-th step. Matches the JS game's `snakeMultiplier`
    setting (snake tick rate relative to ball tick rate).
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

    def reset(self) -> None:
        self.s1, self.s2 = _make_snakes(self.snake_length)
        self.ball = _make_ball(self.rng)
        self.steps = 0
        self.done = False

    # ── Action application ──────────────────────────────────────────────────
    def _apply_action(self, snake: Snake, action: int) -> None:
        dx, dy = ACTION_DELTAS[action]
        # Cannot reverse direction
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
        """Returns (scorer, bounced). Mutates self.ball. Matches logic.js `stepBall`."""
        ball = self.ball
        nx = ball.x + ball.vx
        ny = ball.y + ball.vy

        # Top/bottom bounce
        bounced = False
        if ny < 0 or ny >= ROWS:
            ball.vy = -ball.vy
            ny = ball.y + ball.vy
            bounced = True

        # Ball exits left → P2 scores. Exits right → P1 scores.
        if nx < 0:
            return 2, bounced
        if nx >= COLS:
            return 1, bounced

        # Snake body obstacles
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

    # ── Public step ─────────────────────────────────────────────────────────
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

        # Ball moves every snake_multiplier-th env step (snakes move every step).
        scorer: Optional[Literal[1, 2]] = None
        bounced = False
        if self.steps % self.snake_multiplier == 0:
            scorer, bounced = self._step_ball()
        if scorer is not None:
            self.done = True
        return StepResult(scorer=scorer, ball_bounced=bounced)
