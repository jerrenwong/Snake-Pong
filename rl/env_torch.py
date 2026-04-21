"""Torch-backed vectorized Snake-Pong sim.

Mirrors `VectorSnakePongGame` (numpy) but all state lives on GPU. Hot-path
obs building and Q-net forwards run end-to-end on device with no per-step
CPU↔GPU transfers.

API matches VectorSnakePongGame: step(actions) returns a dict of torch
tensors on `device`.
"""
from __future__ import annotations

from typing import Optional

import torch

from .env import COLS, ROWS, WALL_L, WALL_R, ACTION_DELTAS


_ACTION_DELTAS_T = torch.tensor(ACTION_DELTAS, dtype=torch.int32)  # (4, 2); will .to(device) in ctor


class TorchVectorSnakePongGame:
    """N parallel games whose state tensors live on `device`.

    State (all on device):
        bodies: (N, 2, L, 2) int32
        dirs:   (N, 2, 2)    int32
        balls:  (N, 4)       int32
        steps:  (N,)          int32
        done:   (N,)          bool
    """

    def __init__(
        self,
        n_games: int,
        snake_length: int = 4,
        snake_multiplier: int = 1,
        device: str | torch.device = "cuda",
        seed: Optional[int] = None,
    ):
        assert n_games > 0 and snake_length > 1 and snake_multiplier >= 1
        self.n = n_games
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.device = torch.device(device)
        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        self._action_deltas = _ACTION_DELTAS_T.to(self.device)

        self.bodies = torch.zeros((n_games, 2, snake_length, 2), dtype=torch.int32, device=self.device)
        self.dirs = torch.zeros((n_games, 2, 2), dtype=torch.int32, device=self.device)
        self.balls = torch.zeros((n_games, 4), dtype=torch.int32, device=self.device)
        self.steps = torch.zeros(n_games, dtype=torch.int32, device=self.device)
        self.done = torch.zeros(n_games, dtype=torch.bool, device=self.device)

        self._init_mask(torch.ones(n_games, dtype=torch.bool, device=self.device))

    @property
    def phase(self) -> torch.Tensor:
        return self.steps % self.snake_multiplier

    def _init_mask(self, mask: torch.Tensor) -> None:
        k = int(mask.sum().item())
        if k == 0:
            return
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        hy = ROWS // 2
        h1x = WALL_L // 2
        h2x = WALL_R + (COLS - WALL_R) // 2
        L = self.snake_length

        # Snake bodies (broadcast across k games in idx)
        offsets = torch.arange(L, dtype=torch.int32, device=self.device)  # (L,)
        s1_x = h1x - offsets  # (L,)
        s2_x = h2x + offsets
        y_const = torch.full((L,), hy, dtype=torch.int32, device=self.device)

        # bodies[idx, 0, :, 0] = s1_x (broadcast over idx)
        self.bodies[idx, 0, :, 0] = s1_x.unsqueeze(0).expand(k, -1)
        self.bodies[idx, 0, :, 1] = y_const.unsqueeze(0).expand(k, -1)
        self.bodies[idx, 1, :, 0] = s2_x.unsqueeze(0).expand(k, -1)
        self.bodies[idx, 1, :, 1] = y_const.unsqueeze(0).expand(k, -1)

        self.dirs[idx, 0, 0] = 1
        self.dirs[idx, 0, 1] = 0
        self.dirs[idx, 1, 0] = -1
        self.dirs[idx, 1, 1] = 0

        # Random ball: side ±1, vy ±1
        side_sign = (torch.randint(0, 2, (k,), generator=self._gen, device=self.device) * 2 - 1).to(torch.int32)
        vy_sign = (torch.randint(0, 2, (k,), generator=self._gen, device=self.device) * 2 - 1).to(torch.int32)
        bx = torch.where(side_sign < 0,
                         torch.tensor(h1x, dtype=torch.int32, device=self.device),
                         torch.tensor(h2x, dtype=torch.int32, device=self.device))
        self.balls[idx, 0] = bx
        self.balls[idx, 1] = hy
        self.balls[idx, 2] = side_sign
        self.balls[idx, 3] = vy_sign

        self.steps[idx] = 0
        self.done[idx] = False

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._init_mask(mask)

    def _apply_actions(self, actions: torch.Tensor) -> None:
        """actions: (n, 2) int. Updates self.dirs, respecting 'no reverse'."""
        new_dirs = self._action_deltas[actions]  # (n, 2, 2)
        opposite = ((new_dirs[..., 0] == -self.dirs[..., 0]) &
                    (new_dirs[..., 1] == -self.dirs[..., 1]))
        mask = opposite.unsqueeze(-1)
        self.dirs = torch.where(mask, self.dirs, new_dirs)

    def _step_snakes(self) -> None:
        new_head = self.bodies[:, :, 0, :] + self.dirs  # (n, 2, 2)
        self.bodies[:, :, 1:, :] = self.bodies[:, :, :-1, :].clone()
        self.bodies[:, :, 0, :] = new_head

    def _snake_out_or_self(self) -> torch.Tensor:
        heads = self.bodies[:, :, 0, :]  # (n, 2, 2)
        out = ((heads[..., 0] < 0) | (heads[..., 0] >= COLS) |
               (heads[..., 1] < 0) | (heads[..., 1] >= ROWS))
        tail = self.bodies[:, :, 1:, :]
        match = (tail == heads[:, :, None, :]).all(dim=-1)  # (n, 2, L-1)
        self_hit = match.any(dim=-1)
        return out | self_hit

    def _snakes_collide(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.bodies[:, 0, 0, :]  # (n, 2)
        h2 = self.bodies[:, 1, 0, :]
        both = (h1 == h2).all(dim=-1)
        s2_body = self.bodies[:, 1, :, :]
        s1_body = self.bodies[:, 0, :, :]
        s1_in_s2 = (s2_body == h1[:, None, :]).all(dim=-1).any(dim=-1)
        s2_in_s1 = (s1_body == h2[:, None, :]).all(dim=-1).any(dim=-1)
        s1_in_s2 = s1_in_s2 & ~both
        s2_in_s1 = s2_in_s1 & ~both
        return both, s1_in_s2, s2_in_s1

    def _step_ball(self, ball_tick_mask: torch.Tensor) -> torch.Tensor:
        """Returns scorer (N,) int8: 0 none, 1 s1, 2 s2."""
        scorer = torch.zeros(self.n, dtype=torch.int8, device=self.device)
        if not ball_tick_mask.any():
            return scorer

        x = self.balls[:, 0].clone()
        y = self.balls[:, 1].clone()
        vx = self.balls[:, 2].clone()
        vy = self.balls[:, 3].clone()

        ny_raw = y + vy
        top_bounce = (ny_raw < 0) | (ny_raw >= ROWS)
        vy = torch.where(top_bounce, -vy, vy)
        ny = torch.where(top_bounce, y + vy, ny_raw)

        nx = x + vx
        scored_left = (nx < 0) & ball_tick_mask
        scored_right = (nx >= COLS) & ball_tick_mask
        scorer = torch.where(scored_right, torch.tensor(1, dtype=torch.int8, device=self.device), scorer)
        scorer = torch.where(scored_left, torch.tensor(2, dtype=torch.int8, device=self.device), scorer)

        still_active = ball_tick_mask & ~scored_left & ~scored_right

        all_bodies = self.bodies.reshape(self.n, 2 * self.snake_length, 2)
        h_target = torch.stack([nx, y], dim=-1)
        v_target = torch.stack([x, ny], dim=-1)
        d_target = torch.stack([nx, ny], dim=-1)
        h_hit = (all_bodies == h_target[:, None, :]).all(dim=-1).any(dim=-1)
        v_hit = (all_bodies == v_target[:, None, :]).all(dim=-1).any(dim=-1)
        d_raw = (all_bodies == d_target[:, None, :]).all(dim=-1).any(dim=-1)
        d_hit = d_raw & ~h_hit & ~v_hit

        h_hit = h_hit & still_active
        v_hit = v_hit & still_active
        d_hit = d_hit & still_active
        any_hit = h_hit | v_hit | d_hit

        vx = torch.where(h_hit | d_hit, -vx, vx)
        vy = torch.where(v_hit | d_hit, -vy, vy)

        nx_final = torch.where(any_hit, x + vx, nx)
        ny_final_raw = torch.where(any_hit, y + vy, ny)
        ny_final = ny_final_raw.clamp(0, ROWS - 1)

        # Write back only for still_active games
        self.balls[still_active, 0] = nx_final[still_active]
        self.balls[still_active, 1] = ny_final[still_active]
        self.balls[still_active, 2] = vx[still_active]
        self.balls[still_active, 3] = vy[still_active]
        return scorer

    def step(self, actions: torch.Tensor) -> dict:
        """actions: (n, 2) int64 on device.

        Returns dict with torch tensors on device:
          scorer (int8): 0 none, 1 s1, 2 s2, 3 draw
          s1_died, s2_died: bool
          terminated: bool
        """
        if self.done.any():
            raise RuntimeError("step() called with some games done. Reset first.")
        self.steps += 1
        self._apply_actions(actions)
        self._step_snakes()

        out_self = self._snake_out_or_self()
        s1_died = out_self[:, 0]
        s2_died = out_self[:, 1]
        both_heads, s1_in_s2, s2_in_s1 = self._snakes_collide()
        s1_died = s1_died | both_heads | s1_in_s2
        s2_died = s2_died | both_heads | s2_in_s1

        draw = s1_died & s2_died
        scorer = torch.zeros(self.n, dtype=torch.int8, device=self.device)
        scorer = torch.where(draw, torch.tensor(3, dtype=torch.int8, device=self.device), scorer)
        scorer = torch.where(s1_died & ~draw, torch.tensor(2, dtype=torch.int8, device=self.device), scorer)
        scorer = torch.where(s2_died & ~draw, torch.tensor(1, dtype=torch.int8, device=self.device), scorer)

        died_any = s1_died | s2_died

        ball_tick = (~died_any) & ((self.steps % self.snake_multiplier) == 0)
        ball_scorer = self._step_ball(ball_tick)

        scorer = torch.where(died_any, scorer, ball_scorer)
        terminated = scorer != 0
        self.done = self.done | terminated

        return {
            "scorer": scorer,
            "s1_died": s1_died,
            "s2_died": s2_died,
            "terminated": terminated,
        }
