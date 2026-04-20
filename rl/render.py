"""PIL-based pixel-art rendering of SnakePongGame, and per-episode recorder.

Produces an (T, 3, H, W) uint8 tensor suitable for wandb.Video.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw

from .env import COLS, ROWS, SnakePongGame
from .gym_env import SnakePongSelfPlayEnv, _build_obs


CELL = 12
W_PX = COLS * CELL
H_PX = ROWS * CELL

# Colours match src/renderer.js / JS game:
BG         = (10, 10, 18)
GRID_LINE  = (28, 28, 40)
S1_HEAD    = (90, 200, 255)   # brighter blue
S1_BODY    = (34, 170, 255)   # #2af
S2_HEAD    = (255, 140, 90)   # brighter orange
S2_BODY    = (255, 102, 34)   # #f62
BALL_COLOR = (245, 245, 245)


def _cell_rect(x: int, y: int) -> tuple[int, int, int, int]:
    x0, y0 = x * CELL, y * CELL
    # 1-px inset for visible grid-ish look
    return (x0 + 1, y0 + 1, x0 + CELL - 1, y0 + CELL - 1)


def render_frame(game: SnakePongGame) -> Image.Image:
    """Render current game state as a PIL RGB image."""
    img = Image.new("RGB", (W_PX, H_PX), BG)
    d = ImageDraw.Draw(img)

    # Centre divider (visual — no gameplay effect in simplified sim)
    mid_x = COLS // 2
    d.line([(mid_x * CELL, 0), (mid_x * CELL, H_PX)], fill=GRID_LINE, width=1)

    # s1
    for i, (x, y) in enumerate(game.s1.body):
        color = S1_HEAD if i == 0 else S1_BODY
        d.rectangle(_cell_rect(x, y), fill=color)
    # s2
    for i, (x, y) in enumerate(game.s2.body):
        color = S2_HEAD if i == 0 else S2_BODY
        d.rectangle(_cell_rect(x, y), fill=color)
    # ball
    d.rectangle(_cell_rect(game.ball.x, game.ball.y), fill=BALL_COLOR)

    return img


def _image_to_chw(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    return arr.transpose(2, 0, 1)           # (3, H, W)


def record_episode(
    learner_action_fn: Callable[[np.ndarray], int],
    opponent_policy: Callable[[np.ndarray], int],
    snake_length: int = 4,
    max_steps: int = 500,
    seed: Optional[int] = None,
    learner_side: int | str = "random",
) -> tuple[np.ndarray, dict]:
    """Play one episode and return (frames, info).

    frames: (T, 3, H, W) uint8 ndarray — T = episode length + 1 (includes initial).
    info: {won, length, terminal, scorer, learner_side}
    """
    env = SnakePongSelfPlayEnv(
        opponent_policy=opponent_policy,
        learner_side=learner_side,
        snake_length=snake_length,
        max_steps=max_steps,
        seed=seed,
    )
    obs, reset_info = env.reset()
    frames = [_image_to_chw(render_frame(env._game))]

    episode_info = {
        "won": False, "length": 0, "terminal": "", "scorer": None,
        "learner_side": reset_info["learner_side"],
    }
    while True:
        action = learner_action_fn(obs)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(_image_to_chw(render_frame(env._game)))
        if term or trunc:
            stats = info.get("episode_stats", {})
            episode_info.update({
                "won": stats.get("learner_won", False),
                "length": stats.get("length", 0),
                "terminal": stats.get("terminal", "truncated"),
                "scorer": stats.get("scorer"),
            })
            break

    return np.stack(frames, axis=0), episode_info


def save_gif(frames: np.ndarray, path: str, fps: int = 10) -> None:
    """Save a (T, 3, H, W) uint8 array as a GIF."""
    import imageio.v2 as imageio
    # imageio wants (T, H, W, 3)
    imgs = frames.transpose(0, 2, 3, 1)
    imageio.mimsave(path, list(imgs), format="GIF", duration=1.0 / fps)
