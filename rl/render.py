"""PIL-based pixel-art rendering of SnakePongGame, and per-episode recorder.

Produces an (T, 3, H, W) uint8 tensor suitable for wandb.Video.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw

from .env import ACTION_DELTAS, COLS, ROWS, Snake, SnakePongGame
from .gym_env import SnakePongSelfPlayEnv, _MIRROR_ACTION, _build_obs


def _legal_ego_mask(own: Snake, opp: Snake, side: int) -> list[bool]:
    """For the four ego actions {0=up,1=down,2=left,3=right}, return which
    ones would NOT immediately kill `own`. Same check applied in the
    browser's safety filter and during GPU-batched evaluation."""
    head_x, head_y = own.body[0]
    cur_dx, cur_dy = own.dx, own.dy
    own_block = set(own.body[:-1])  # tail vacates next step
    opp_block = set(opp.body)
    legal = [False, False, False, False]
    for ego in range(4):
        real = ego if side == 1 else _MIRROR_ACTION[ego]
        rdx, rdy = ACTION_DELTAS[real]
        if (rdx, rdy) == (-cur_dx, -cur_dy):
            rdx, rdy = cur_dx, cur_dy  # engine ignores reversal
        nh = (head_x + rdx, head_y + rdy)
        if nh[0] < 0 or nh[0] >= COLS or nh[1] < 0 or nh[1] >= ROWS:
            continue
        if nh in own_block or nh in opp_block:
            continue
        legal[ego] = True
    return legal


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
    snake_multiplier: int = 1,
    max_steps: int = 500,
    interp_ball: bool = True,
    seed: Optional[int] = None,
    learner_side: int | str = "random",
    safety_filter: bool = False,
    learner_q_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    opponent_q_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[np.ndarray, dict]:
    """Play one episode and return (frames, info).

    frames: (T, 3, H, W) uint8 ndarray — T = episode length + 1 (includes initial).
    info: {won, length, terminal, scorer, learner_side}

    If `safety_filter=True` AND both `learner_q_fn` and `opponent_q_fn`
    are provided (returning length-4 Q-value / logit arrays for the 4 ego
    actions), illegal actions are masked out before argmax on each side —
    matching the in-browser never-crash filter. When `safety_filter=False`
    the original (int-returning) callables are used unchanged.
    """
    # When safety_filter is on, we substitute the opponent_policy fed into
    # the env with a safety-wrapped version that masks its Q-values using
    # the env's LIVE game state. Because the wrapper closes over `env`,
    # we need to build env WITHOUT opponent first, then patch.
    env = SnakePongSelfPlayEnv(
        opponent_policy=opponent_policy,  # will be patched below if filtering
        learner_side=learner_side,
        snake_length=snake_length,
        snake_multiplier=snake_multiplier,
        max_steps=max_steps,
        interp_ball=interp_ball,
        seed=seed,
    )

    if safety_filter:
        if learner_q_fn is None or opponent_q_fn is None:
            raise ValueError("safety_filter=True requires learner_q_fn AND opponent_q_fn")
        def _safe_opp_policy(obs: np.ndarray) -> int:
            q = np.asarray(opponent_q_fn(obs), dtype=np.float32)
            # Opp is whichever snake isn't the learner.
            opp_side = 2 if env._learner == 1 else 1
            own = env._game.s2 if opp_side == 2 else env._game.s1
            other = env._game.s1 if opp_side == 2 else env._game.s2
            legal = _legal_ego_mask(own, other, opp_side)
            masked = np.where(legal, q, -np.inf)
            if not any(legal):
                return int(q.argmax())
            return int(masked.argmax())
        env.set_opponent(_safe_opp_policy)

        def _safe_learner_action(obs: np.ndarray) -> int:
            q = np.asarray(learner_q_fn(obs), dtype=np.float32)
            own = env._game.s1 if env._learner == 1 else env._game.s2
            other = env._game.s2 if env._learner == 1 else env._game.s1
            legal = _legal_ego_mask(own, other, env._learner)
            masked = np.where(legal, q, -np.inf)
            if not any(legal):
                return int(q.argmax())
            return int(masked.argmax())
        learner_action_fn = _safe_learner_action

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
