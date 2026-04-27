"""GPU-resident rollout helpers: obs building, action mirroring, and
batched Q-net forwards. Used by the PPO rollout (`ppo_rollout.py`),
the deterministic evaluator (`det_eval.py`), and the tournament runner
(`vec_tournament.py`).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .env import COLS, ROWS
from .env_torch import TorchVectorSnakePongGame
from .models import N_ACTIONS


_MIRROR_ACTION_T: Optional[torch.Tensor] = None  # initialized lazily per device


def _mirror_action_tensor(device: torch.device) -> torch.Tensor:
    global _MIRROR_ACTION_T
    if _MIRROR_ACTION_T is None or _MIRROR_ACTION_T.device != device:
        _MIRROR_ACTION_T = torch.tensor([0, 1, 3, 2], dtype=torch.int64, device=device)
    return _MIRROR_ACTION_T


@torch.no_grad()
def _build_obs_batch_gpu(
    vec: TorchVectorSnakePongGame,
    learner_sides: torch.Tensor,  # (N,) int64 in {1, 2}
    interp_ball: bool,
) -> torch.Tensor:
    """Returns (N, 4*L+4) float32 tensor on device."""
    n = vec.n
    L = vec.snake_length
    mult = vec.snake_multiplier
    nx, ny = COLS - 1, ROWS - 1
    device = vec.device

    mirror = (learner_sides == 2)  # (N,)

    bodies = vec.bodies  # (N, 2, L, 2) int32
    # Pick own/opp bodies by side
    pick_own = (learner_sides == 1).view(-1, 1, 1).expand(n, L, 2)
    own_bodies = torch.where(pick_own, bodies[:, 0, :, :], bodies[:, 1, :, :])  # (N, L, 2)
    opp_bodies = torch.where(pick_own, bodies[:, 1, :, :], bodies[:, 0, :, :])

    own_body_f = own_bodies.to(torch.float32)
    opp_body_f = opp_bodies.to(torch.float32)

    # Mirror x
    mirror_x = mirror.view(-1, 1)
    own_body_f_x = torch.where(mirror_x, nx - own_body_f[..., 0], own_body_f[..., 0])
    opp_body_f_x = torch.where(mirror_x, nx - opp_body_f[..., 0], opp_body_f[..., 0])
    own_body_norm = torch.stack([own_body_f_x / nx, own_body_f[..., 1] / ny], dim=-1)
    opp_body_norm = torch.stack([opp_body_f_x / nx, opp_body_f[..., 1] / ny], dim=-1)

    own_flat = own_body_norm.reshape(n, L * 2)
    opp_flat = opp_body_norm.reshape(n, L * 2)

    # Ball interpolation
    bx = vec.balls[:, 0].to(torch.float32)
    by = vec.balls[:, 1].to(torch.float32)
    vx = vec.balls[:, 2].to(torch.float32)
    vy = vec.balls[:, 3].to(torch.float32)
    if interp_ball and mult > 1:
        frac = vec.phase.to(torch.float32) / mult
        bx = bx + frac * vx
        by = by + frac * vy
        vx = vx / mult
        vy = vy / mult

    bx = torch.where(mirror, nx - bx, bx)
    vx = torch.where(mirror, -vx, vx)

    ball = torch.stack([bx / nx, by / ny, vx, vy], dim=-1)  # (N, 4)
    return torch.cat([own_flat, opp_flat, ball], dim=1)


@torch.no_grad()
def _batched_q_actions_gpu(
    q_net: nn.Module,
    obs_batch: torch.Tensor,  # (N, D) on device
    epsilon: float,
    gen: torch.Generator,
    active_heads: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns (N,) int64 action indices on device.

    If q_net outputs (N, K, A) (Bootstrapped DQN) and `active_heads` is
    provided, gather per-env head; otherwise mean over heads.
    """
    q = q_net(obs_batch)
    if q.dim() == 3:
        if active_heads is not None:
            q = q.gather(1, active_heads.view(-1, 1, 1).expand(-1, 1, q.size(-1))).squeeze(1)
        else:
            q = q.mean(dim=1)
    actions = q.argmax(dim=1)
    if epsilon > 0.0:
        rand = torch.rand(obs_batch.shape[0], generator=gen, device=obs_batch.device)
        mask = rand < epsilon
        if mask.any():
            rand_a = torch.randint(0, N_ACTIONS, (int(mask.sum().item()),),
                                   generator=gen, device=obs_batch.device, dtype=torch.int64)
            actions = actions.masked_scatter(mask, rand_a)
    return actions

