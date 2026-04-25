"""Deterministic 4-configuration evaluator.

When two safety-filtered argmax policies meet, the trajectory is fully
determined by the env's init RNG. The init has only four meaningful axes
once the ego obs is taken into account:

    ball_side ∈ {-1, +1}     # which half the ball spawns in (= initial vx)
    ball_vy   ∈ {-1, +1}     # initial vertical velocity
    (learner_side is irrelevant: obs is egocentric and mirrored, so swapping
     learner_side just relabels.)

So with two players we run exactly 4 games and the win rate is one of
{0, 25, 50, 75, 100} %. Drastically faster than rolling 100 stochastic
episodes per pair, exact for deterministic agents.

Usage:
    from rl.det_eval import det_winrate_grid
    out = det_winrate_grid(
        models={"v16_u500": ac1, "v16_u3000": ac2},
        device="cuda", snake_length=4, snake_multiplier=3,
        max_steps=800, interp_ball=True,
    )
    # out['win_rate'] is a (M, M) numpy array in {0, .25, .5, .75, 1}.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .env import COLS, ROWS, WALL_L, WALL_R
from .env_torch import TorchVectorSnakePongGame
from .vec_rollout_gpu import (
    _build_obs_batch_gpu,
    _mirror_action_tensor,
)
from .vec_tournament import _safety_mask_ego_subset


# ── Deterministic init ───────────────────────────────────────────────────────
def _force_init(vec: TorchVectorSnakePongGame, configs: list[tuple[int, int]]) -> None:
    """Force vec[i] to initial state config[i] = (ball_side ±1, ball_vy ±1).
    Snake bodies / dirs / positions remain at their default startup state.
    Learner side is fixed at 1 across all configs (egocentric obs makes
    side-2 a redundant reflection).
    """
    n = vec.n
    assert len(configs) == n, "one config per env slot"
    device = vec.device
    L = vec.snake_length
    hy = ROWS // 2
    h1x = WALL_L // 2
    h2x = WALL_R + (COLS - WALL_R) // 2

    offsets = torch.arange(L, dtype=torch.int32, device=device)
    s1_x = h1x - offsets
    s2_x = h2x + offsets
    y_const = torch.full((L,), hy, dtype=torch.int32, device=device)

    idx = torch.arange(n, device=device)
    vec.bodies[idx, 0, :, 0] = s1_x.unsqueeze(0).expand(n, L)
    vec.bodies[idx, 0, :, 1] = y_const.unsqueeze(0).expand(n, L)
    vec.bodies[idx, 1, :, 0] = s2_x.unsqueeze(0).expand(n, L)
    vec.bodies[idx, 1, :, 1] = y_const.unsqueeze(0).expand(n, L)
    vec.dirs[idx, 0, 0] = 1
    vec.dirs[idx, 0, 1] = 0
    vec.dirs[idx, 1, 0] = -1
    vec.dirs[idx, 1, 1] = 0

    sides = torch.tensor([c[0] for c in configs], dtype=torch.int32, device=device)
    vys = torch.tensor([c[1] for c in configs], dtype=torch.int32, device=device)
    bx = torch.where(sides < 0, torch.tensor(h1x, dtype=torch.int32, device=device),
                     torch.tensor(h2x, dtype=torch.int32, device=device))
    vec.balls[idx, 0] = bx
    vec.balls[idx, 1] = hy
    vec.balls[idx, 2] = sides
    vec.balls[idx, 3] = vys
    vec.steps[idx] = 0
    vec.done[idx] = False


# ── Pair eval ────────────────────────────────────────────────────────────────
@torch.no_grad()
def det_winrate_pair(
    model_a: nn.Module, model_b: nn.Module,
    device: torch.device,
    snake_length: int = 4,
    snake_multiplier: int = 3,
    max_steps: int = 800,
    interp_ball: bool = True,
) -> tuple[float, float, float]:
    """Returns (a_wins, b_wins, draws) as floats in {0,.25,.5,.75,1}, summing
    to 1.0. `a` plays as side-1, `b` plays as side-2.
    Both sides use safety filter + argmax (deterministic).
    """
    configs = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    vec = TorchVectorSnakePongGame(
        n_games=4, snake_length=snake_length,
        snake_multiplier=snake_multiplier, device=device,
        seed=0,
    )
    _force_init(vec, configs)
    learner_sides = torch.full((4,), 1, dtype=torch.int64, device=device)
    opp_sides = torch.full((4,), 2, dtype=torch.int64, device=device)
    mirror_t = _mirror_action_tensor(device)

    a_wins = 0
    b_wins = 0
    draws = 0
    finished = torch.zeros(4, dtype=torch.bool, device=device)
    ep_lengths = torch.zeros(4, dtype=torch.int64, device=device)

    for _ in range(max_steps):
        if bool(finished.all().item()):
            break
        learner_obs = _build_obs_batch_gpu(vec, learner_sides, interp_ball)
        opp_obs = _build_obs_batch_gpu(vec, opp_sides, interp_ball)

        idx_all = torch.arange(4, device=device, dtype=torch.int64)
        q_a = model_a(learner_obs)
        if q_a.dim() == 3:
            q_a = q_a.mean(dim=1)
        a_ego = _safety_mask_ego_subset(q_a, vec, learner_sides, idx_all, mirror_t)

        q_b = model_b(opp_obs)
        if q_b.dim() == 3:
            q_b = q_b.mean(dim=1)
        b_ego = _safety_mask_ego_subset(q_b, vec, opp_sides, idx_all, mirror_t)

        a_real = torch.where(learner_sides == 1, a_ego, mirror_t[a_ego])
        b_real = torch.where(opp_sides == 1, b_ego, mirror_t[b_ego])
        actions = torch.zeros((4, 2), dtype=torch.int64, device=device)
        actions[:, 0] = a_real
        actions[:, 1] = b_real

        # Keep already-finished slots from breaking step()'s "no done envs"
        # check. We ignore any results from those slots anyway.
        vec.done = vec.done & ~finished
        result = vec.step(actions)
        ep_lengths += 1

        scorer = result["scorer"]
        terminated = result["terminated"] | (scorer == 3)
        truncated = (~terminated) & (ep_lengths >= max_steps)
        new_done = (terminated | truncated) & ~finished

        if new_done.any():
            sc = scorer[new_done].cpu().numpy()
            tr = truncated[new_done].cpu().numpy()
            for s, t in zip(sc, tr):
                if t and s == 0:
                    draws += 1
                elif s == 1:
                    a_wins += 1
                elif s == 2:
                    b_wins += 1
                else:
                    draws += 1
            finished |= new_done

    return a_wins / 4.0, b_wins / 4.0, draws / 4.0


# ── Grid eval (M × M) ────────────────────────────────────────────────────────
@torch.no_grad()
def det_winrate_grid(
    models: Dict[str, nn.Module],
    device: torch.device | str = "cuda",
    snake_length: int = 4,
    snake_multiplier: int = 3,
    max_steps: int = 800,
    interp_ball: bool = True,
) -> dict:
    """Run det_winrate_pair for every ordered pair (a, b). Returns dict with
    win_rate / loss_rate / draw_rate matrices (M × M, float in {0,.25,.5,.75,1}).
    """
    device = torch.device(device)
    names = list(models.keys())
    M = len(names)
    win = np.zeros((M, M), dtype=np.float32)
    loss = np.zeros((M, M), dtype=np.float32)
    draw = np.zeros((M, M), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            w, l, d = det_winrate_pair(
                models[a], models[b], device=device,
                snake_length=snake_length, snake_multiplier=snake_multiplier,
                max_steps=max_steps, interp_ball=interp_ball,
            )
            win[i, j] = w; loss[i, j] = l; draw[i, j] = d
    return {"names": names, "win_rate": win, "loss_rate": loss, "draw_rate": draw}


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from .big_tournament import _load_generic

    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help="name:path[:head=N], repeat for each model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--snake-multiplier", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    mods: Dict[str, nn.Module] = {}
    for spec in args.model:
        parts = spec.split(":")
        name, path = parts[0], parts[1]
        head = None
        for ex in parts[2:]:
            if ex.startswith("head="):
                head = int(ex.split("=", 1)[1])
        mods[name] = _load_generic(path, device, head=head)
    res = det_winrate_grid(mods, device=device, snake_multiplier=args.snake_multiplier)
    M = len(res["names"])
    print("Deterministic win-rate grid (row=A, col=B), each cell ∈ {0,.25,.5,.75,1}:")
    hdr = "         " + "".join(f"{n:>9s}" for n in res["names"])
    print(hdr)
    for i, n in enumerate(res["names"]):
        row = f"{n:>8s} "
        for j in range(M):
            row += f"{res['win_rate'][i,j]:8.0%} "
        print(row)
