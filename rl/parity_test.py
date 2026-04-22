"""Exhaustive parity test between the JS game (src/logic.js, src/main.js)
and our Python training env (rl/env.py).

Method
------
We transcribe src/logic.js into Python (`JSLogicPort`), following the JS
control-flow line-for-line. The tick loop from main.js is also reproduced
faithfully:
  - Every snake tick: apply both players' actions; step both snakes;
    check deaths + snake-snake collisions.
  - Every ball tick (snake_multiplier-th snake tick): step ball.

Then we run many random trajectories through BOTH:
  A) JSLogicPort (should match the real JS game behavior)
  B) rl/env.py::SnakePongGame (our training env)

…and diff the state at every step. Any divergence → bug in our env.

The JS port deliberately uses the same data structures as logic.js
(dicts with {x,y,dir}, list-based bodies, no numpy) so the translation
is as literal as possible and easy to audit.
"""
from __future__ import annotations

import json
import numpy as np

from .env import SnakePongGame, COLS, ROWS, WALL_L, WALL_R, ACTION_DELTAS


# ─── JS-faithful port of src/logic.js ─────────────────────────────────────────
# Transcription notes:
# - JS `Math.floor` → int division // for non-negative ints in our range
# - JS `Array.from({length:len}, (_, i) => ...)` → list comprehension
# - JS `snake.body.unshift(head)` → body.insert(0, head)
# - JS `Set.has("x,y")` → check via `(x,y) in set_of_tuples`

def js_create_snakes(length: int):
    hy = ROWS // 2
    h1x = WALL_L // 2
    h2x = WALL_R + (COLS - WALL_R) // 2
    s1 = {
        "body":    [{"x": h1x - i, "y": hy} for i in range(length)],
        "dir":     {"x":  1, "y": 0},
        "nextDir": {"x":  1, "y": 0},
    }
    s2 = {
        "body":    [{"x": h2x + i, "y": hy} for i in range(length)],
        "dir":     {"x": -1, "y": 0},
        "nextDir": {"x": -1, "y": 0},
    }
    return s1, s2


def js_create_ball(rng: np.random.Generator):
    """Ported from src/logic.js::createBall (but seeded for determinism)."""
    side = -1 if rng.random() < 0.5 else 1
    bx = WALL_L // 2 if side < 0 else WALL_R + (COLS - WALL_R) // 2
    return {
        "x":  bx,
        "y":  ROWS // 2,
        "vx": side,
        "vy": 1 if rng.random() < 0.5 else -1,
    }


def js_step_snake(snake):
    """Ported from src/logic.js::stepSnake."""
    snake["dir"] = snake["nextDir"]
    head = snake["body"][0]
    snake["body"].insert(0, {
        "x": head["x"] + snake["dir"]["x"],
        "y": head["y"] + snake["dir"]["y"],
    })
    snake["body"].pop()


def js_snake_hits_death(snake, wallSet=None):
    """Ported from src/logic.js::snakeHitsDeath (wallSet always None here)."""
    h = snake["body"][0]
    if h["x"] < 0 or h["x"] >= COLS or h["y"] < 0 or h["y"] >= ROWS:
        return True
    for c in snake["body"][1:]:
        if c["x"] == h["x"] and c["y"] == h["y"]:
            return True
    return False


def js_snakes_collide(s1, s2):
    """Ported from src/logic.js::snakesCollide."""
    h1, h2 = s1["body"][0], s2["body"][0]
    if h1["x"] == h2["x"] and h1["y"] == h2["y"]:
        return "both"
    if any(c["x"] == h1["x"] and c["y"] == h1["y"] for c in s2["body"]):
        return "s1"
    if any(c["x"] == h2["x"] and c["y"] == h2["y"] for c in s1["body"]):
        return "s2"
    return None


def js_step_ball(ball, s1, s2):
    """Ported from src/logic.js::stepBall. Returns scorer (1|2) or None."""
    nx = ball["x"] + ball["vx"]
    ny = ball["y"] + ball["vy"]

    # Top/bottom bounce
    if ny < 0 or ny >= ROWS:
        ball["vy"] = -ball["vy"]
        ny = ball["y"] + ball["vy"]

    if nx < 0:
        return 2
    if nx >= COLS:
        return 1

    # Obstacle check: snake bodies (no walls in this game)
    allSegs = s1["body"] + s2["body"]

    def cell_in_segs(x, y):
        for c in allSegs:
            if c["x"] == x and c["y"] == y:
                return True
        return False

    hHit = cell_in_segs(nx, ball["y"])
    vHit = cell_in_segs(ball["x"], ny)
    dHit = (not hHit) and (not vHit) and cell_in_segs(nx, ny)

    if hHit or vHit or dHit:
        if hHit:
            ball["vx"] = -ball["vx"]
        if vHit:
            ball["vy"] = -ball["vy"]
        if dHit:
            ball["vx"] = -ball["vx"]
            ball["vy"] = -ball["vy"]
        nx = ball["x"] + ball["vx"]
        ny = max(0, min(ROWS - 1, ball["y"] + ball["vy"]))

    ball["x"] = nx
    ball["y"] = ny
    return None


# ─── Faithful reproduction of main.js's tick loop ─────────────────────────────
def js_set_next_dir(snake, dx, dy):
    """src/main.js::onDirection{P1,P2} — reversal check vs current dir."""
    if dx != 0 and snake["dir"]["x"] == -dx:
        return
    if dy != 0 and snake["dir"]["y"] == -dy:
        return
    snake["nextDir"] = {"x": dx, "y": dy}


class JSLogicPort:
    """Drives a game using the JS-ported logic functions above, following
    main.js's tick-loop ordering (snake ticks first, then ball ticks at
    snake_multiplier intervals).
    """

    def __init__(self, snake_length=4, snake_multiplier=1, seed=None):
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.s1, self.s2 = js_create_snakes(self.snake_length)
        self.ball = js_create_ball(self.rng)
        self.steps = 0
        self.done = False
        self.scorer = None
        self.s1_died = False
        self.s2_died = False

    def step(self, a1, a2):
        """One snake tick: apply actions, move both snakes, check deaths,
        and if it's a ball-tick step (steps % mult == 0), step the ball.
        """
        if self.done:
            raise RuntimeError("step() called after game over")
        self.steps += 1
        # Apply actions — matches main.js's input handlers (set nextDir if not
        # a reversal of current dir)
        dx1, dy1 = ACTION_DELTAS[a1]
        dx2, dy2 = ACTION_DELTAS[a2]
        js_set_next_dir(self.s1, dx1, dy1)
        js_set_next_dir(self.s2, dx2, dy2)
        # Move
        js_step_snake(self.s1)
        js_step_snake(self.s2)
        # Death checks (JS does them in this order)
        d1 = js_snake_hits_death(self.s1)
        d2 = js_snake_hits_death(self.s2)
        if d1 and d2:
            self.done = True; self.s1_died = True; self.s2_died = True
            self.scorer = 0
            return
        if d1:
            self.done = True; self.s1_died = True
            self.scorer = 2
            return
        if d2:
            self.done = True; self.s2_died = True
            self.scorer = 1
            return
        col = js_snakes_collide(self.s1, self.s2)
        if col == "both":
            self.done = True; self.s1_died = True; self.s2_died = True
            self.scorer = 0
            return
        if col == "s1":
            self.done = True; self.s1_died = True
            self.scorer = 2
            return
        if col == "s2":
            self.done = True; self.s2_died = True
            self.scorer = 1
            return
        # Ball tick on multiplier-th step
        if self.steps % self.snake_multiplier == 0:
            scorer = js_step_ball(self.ball, self.s1, self.s2)
            if scorer is not None:
                self.done = True
                self.scorer = scorer


# ─── Parity comparison ────────────────────────────────────────────────────────
def py_state_as_dict(g: SnakePongGame):
    """Normalise SnakePongGame state to the same dict shape JSLogicPort uses."""
    return {
        "s1_body": [(x, y) for (x, y) in g.s1.body],
        "s2_body": [(x, y) for (x, y) in g.s2.body],
        "s1_dir": (g.s1.dx, g.s1.dy),
        "s2_dir": (g.s2.dx, g.s2.dy),
        "ball": (g.ball.x, g.ball.y, g.ball.vx, g.ball.vy),
        "done": g.done,
    }


def js_state_as_dict(js: JSLogicPort):
    return {
        "s1_body": [(c["x"], c["y"]) for c in js.s1["body"]],
        "s2_body": [(c["x"], c["y"]) for c in js.s2["body"]],
        "s1_dir": (js.s1["dir"]["x"], js.s1["dir"]["y"]),
        "s2_dir": (js.s2["dir"]["x"], js.s2["dir"]["y"]),
        "ball": (js.ball["x"], js.ball["y"], js.ball["vx"], js.ball["vy"]),
        "done": js.done,
    }


def compare_run(snake_length, snake_multiplier, seed, max_steps=200, verbose=False):
    """Single random trajectory: run both, report first divergence (if any).

    We FORCE identical initial states (same ball spawn, same bodies) before
    stepping, so any divergence is purely step-logic related — not RNG.
    """
    py = SnakePongGame(snake_length, snake_multiplier, seed=seed)
    js = JSLogicPort(snake_length, snake_multiplier, seed=seed)
    # Force JS port to match py's initial ball state (both have the same
    # 50/50 distribution; RNG consumption just differs).
    js.ball = {"x": py.ball.x, "y": py.ball.y, "vx": py.ball.vx, "vy": py.ball.vy}
    s0_py = py_state_as_dict(py)
    s0_js = js_state_as_dict(js)
    if s0_py != s0_js:
        print(f"INITIAL DIVERGE (seed={seed}): py={s0_py}  js={s0_js}")
        return "init_diff"
    rng = np.random.default_rng(seed * 9973 + 1)
    for t in range(max_steps):
        if py.done or js.done:
            if py.done != js.done:
                print(f"DONE DIVERGE t={t} seed={seed}: py.done={py.done} js.done={js.done}")
                return "done_diff"
            break
        a1 = int(rng.integers(0, 4))
        a2 = int(rng.integers(0, 4))
        py.step(a1, a2)
        js.step(a1, a2)
        s_py = py_state_as_dict(py)
        s_js = js_state_as_dict(js)
        if s_py != s_js:
            print(f"STATE DIVERGE seed={seed} t={t} actions=({a1},{a2})")
            print(f"  py: {s_py}")
            print(f"  js: {s_js}")
            return "state_diff"
        if verbose and t < 3:
            print(f"  t={t} {s_py}")
    return "ok"


def main():
    total = divergent = 0
    first_failures = []
    for snake_length in [3, 4, 5, 6]:
        for snake_multiplier in [1, 2, 3]:
            for seed in range(25):
                total += 1
                res = compare_run(snake_length, snake_multiplier, seed, max_steps=300)
                if res != "ok":
                    divergent += 1
                    if len(first_failures) < 5:
                        first_failures.append((snake_length, snake_multiplier, seed, res))
    print(f"\nRan {total} trajectories across (L, mult) ∈ {{3,4,5,6}} × {{1,2,3}}, seeds 0-24.")
    print(f"  divergent: {divergent}")
    if divergent == 0:
        print("  → EXACT PARITY: rl/env.py == JS-port.")
    else:
        print(f"  first failures: {first_failures}")


if __name__ == "__main__":
    main()
