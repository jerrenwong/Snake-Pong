"""GPU-resident rollout: env state + obs building + Q-net forwards all on
device. Transitions are copied to CPU once per batched step for the replay
buffer (which lives on CPU).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .dqn import N_ACTIONS, GpuReplayBuffer, ReplayBuffer, Transition
from .env import COLS, ROWS
from .env_torch import TorchVectorSnakePongGame


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


class VecRolloutGPU:
    """Owns N parallel games with state on GPU. Obs, Q-forwards, and env
    steps all run on device. Replay buffer is still on CPU; we batch-copy
    transitions at the end of each iter step.
    """

    def __init__(
        self,
        n_envs: int,
        q_net: nn.Module,
        device: torch.device,
        snake_length: int = 4,
        snake_multiplier: int = 1,
        max_steps: int = 500,
        interp_ball: bool = True,
        n_heads: int = 0,
        bootstrap_mask_prob: float = 0.5,
        death_penalty: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_envs = n_envs
        self.q_net = q_net
        self.device = device
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.max_steps = max_steps
        self.interp_ball = interp_ball
        self.n_heads = n_heads
        self.bootstrap_mask_prob = bootstrap_mask_prob
        self.death_penalty = float(death_penalty)
        self._np_rng = rng if rng is not None else np.random.default_rng()

        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(self._np_rng.integers(1 << 31)))

        self.vec = TorchVectorSnakePongGame(
            n_games=n_envs, snake_length=snake_length,
            snake_multiplier=snake_multiplier, device=device,
            seed=int(self._np_rng.integers(1 << 31)),
        )
        # Initialize learner_sides randomly on device
        half_mask = torch.rand(n_envs, generator=self._gen, device=device) < 0.5
        self.learner_sides = torch.where(half_mask, 1, 2).to(torch.int64)
        self.ep_lengths = torch.zeros(n_envs, dtype=torch.int64, device=device)
        # Bootstrapped DQN: per-env active head
        if n_heads > 0:
            self.active_heads = torch.randint(
                0, n_heads, (n_envs,), generator=self._gen, device=device, dtype=torch.int64,
            )
        else:
            self.active_heads = None  # type: ignore[assignment]

        self._opp_q: Optional[nn.Module] = None
        self._opp_is_random: bool = True
        self._opp_epsilon: float = 0.0

    def set_opponent(self, opp_q_net: Optional[nn.Module], opp_epsilon: float = 0.05) -> None:
        self._opp_q = opp_q_net
        self._opp_is_random = opp_q_net is None
        self._opp_epsilon = opp_epsilon

    def _reset_mask(self, mask: torch.Tensor) -> None:
        if not mask.any():
            return
        self.vec.reset(mask)
        k = int(mask.sum().item())
        new_sides = torch.where(
            torch.rand(k, generator=self._gen, device=self.device) < 0.5,
            1, 2,
        ).to(torch.int64)
        self.learner_sides[mask] = new_sides
        self.ep_lengths[mask] = 0
        if self.active_heads is not None and self.n_heads > 0:
            self.active_heads[mask] = torch.randint(
                0, self.n_heads, (k,), generator=self._gen, device=self.device, dtype=torch.int64,
            )

    def collect(
        self,
        replay,  # ReplayBuffer (numpy) or GpuReplayBuffer (torch)
        n_transitions: int,
        epsilon: float,
    ) -> list[dict]:
        completed: list[dict] = []
        collected = 0
        n = self.n_envs
        mirror_t = _mirror_action_tensor(self.device)
        gpu_buffer = isinstance(replay, GpuReplayBuffer)

        while collected < n_transitions:
            learner_obs = _build_obs_batch_gpu(self.vec, self.learner_sides, self.interp_ball)
            opp_sides = 3 - self.learner_sides

            if self._opp_is_random:
                opp_actions_ego = torch.randint(0, N_ACTIONS, (n,), generator=self._gen,
                                                device=self.device, dtype=torch.int64)
            else:
                opp_obs = _build_obs_batch_gpu(self.vec, opp_sides, self.interp_ball)
                # Head-vs-head: if opponent is bootstrapped (K>1 heads),
                # sample a per-env random head so each env meets a different
                # opponent style. Pool snapshots are frozen so this is safe.
                opp_K = getattr(self._opp_q, "n_heads", 1)
                if opp_K and opp_K > 1:
                    opp_heads = torch.randint(
                        0, opp_K, (n,), generator=self._gen,
                        device=self.device, dtype=torch.int64,
                    )
                else:
                    opp_heads = None
                opp_actions_ego = _batched_q_actions_gpu(
                    self._opp_q, opp_obs, self._opp_epsilon, self._gen,
                    active_heads=opp_heads,
                )

            learner_actions_ego = _batched_q_actions_gpu(
                self.q_net, learner_obs, epsilon, self._gen,
                active_heads=self.active_heads,
            )

            learner_real = torch.where(
                self.learner_sides == 1, learner_actions_ego,
                mirror_t[learner_actions_ego],
            )
            opp_real = torch.where(
                opp_sides == 1, opp_actions_ego,
                mirror_t[opp_actions_ego],
            )

            actions = torch.zeros((n, 2), dtype=torch.int64, device=self.device)
            is_side1 = self.learner_sides == 1
            actions[:, 0] = torch.where(is_side1, learner_real, opp_real)
            actions[:, 1] = torch.where(is_side1, opp_real, learner_real)

            result = self.vec.step(actions)
            self.ep_lengths += 1

            scorer = result["scorer"]  # int8
            won = (scorer.to(torch.int64) == self.learner_sides)
            lost = (scorer != 0) & (scorer != 3) & ~won
            rewards = torch.where(won, 1.0, torch.where(lost, -1.0, 0.0)).to(torch.float32)
            # Optional shaping: extra penalty if the learner-side snake died
            # from wall / self / inter-snake collision (encourages risk-aversion).
            if self.death_penalty != 0.0:
                s1_died = result["s1_died"]
                s2_died = result["s2_died"]
                learner_died = torch.where(self.learner_sides == 1, s1_died, s2_died)
                rewards = rewards + torch.where(
                    learner_died,
                    torch.tensor(self.death_penalty, dtype=torch.float32, device=self.device),
                    torch.tensor(0.0, dtype=torch.float32, device=self.device),
                )

            terminated = result["terminated"] | (scorer == 3)
            truncated = (~terminated) & (self.ep_lengths >= self.max_steps)

            next_obs = _build_obs_batch_gpu(self.vec, self.learner_sides, self.interp_ball)

            # Bootstrapped: generate (N, K) mask per step for buffer.
            mask_gpu: Optional[torch.Tensor] = None
            if self.n_heads > 0:
                rand = torch.rand(n, self.n_heads, generator=self._gen, device=self.device)
                mask_gpu = (rand < self.bootstrap_mask_prob).to(torch.float32)

            # Push to replay. GPU buffer: stay on device. CPU buffer: copy once.
            if gpu_buffer:
                replay.push_batch(
                    obs=learner_obs, action=learner_actions_ego,
                    reward=rewards, next_obs=next_obs, done=terminated,
                    mask=mask_gpu,
                )
            else:
                lo_cpu = learner_obs.cpu().numpy()
                la_cpu = learner_actions_ego.cpu().numpy()
                rw_cpu = rewards.cpu().numpy()
                no_cpu = next_obs.cpu().numpy()
                done_cpu = terminated.cpu().numpy()
                mk_cpu = mask_gpu.cpu().numpy() if mask_gpu is not None else None
                for i in range(n):
                    replay.push(Transition(
                        obs=lo_cpu[i], action=int(la_cpu[i]), reward=float(rw_cpu[i]),
                        next_obs=no_cpu[i], done=bool(done_cpu[i]),
                        mask=(mk_cpu[i] if mk_cpu is not None else None),
                    ))
            collected += n

            # Episode bookkeeping — needs CPU-side info for logging only.
            reset_mask = terminated | truncated
            if reset_mask.any():
                idxs = reset_mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
                sc_cpu = scorer.cpu().numpy()
                sides_cpu = self.learner_sides.cpu().numpy()
                trunc_cpu = truncated.cpu().numpy()
                term_cpu = terminated.cpu().numpy()
                lens_cpu = self.ep_lengths.cpu().numpy()
                for i in idxs:
                    si = int(sc_cpu[i])
                    is_trunc = bool(trunc_cpu[i]) and not bool(term_cpu[i])
                    w = si == int(sides_cpu[i])
                    term_label = (
                        "truncated" if is_trunc
                        else ("draw" if si == 3 else "scored")
                    )
                    completed.append({
                        "length": int(lens_cpu[i]),
                        "won": bool(w),
                        "scorer": si,
                        "terminal": term_label,
                    })
                self.vec.done[reset_mask] = False
                self._reset_mask(reset_mask)

            if collected >= n_transitions:
                break

        return completed
