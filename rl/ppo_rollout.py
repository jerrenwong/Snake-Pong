"""GPU on-policy rollout collector for PPO.

Mirrors rl/vec_rollout_gpu.VecRolloutGPU's env plumbing (vec env, obs
building, opponent pool, action mirroring) but writes to per-step tensor
buffers instead of a replay buffer. One rollout is `T` env steps across
`N` parallel games → (T, N, ...) tensors suitable for GAE.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import COLS
from .env_torch import TorchVectorSnakePongGame
from .vec_rollout_gpu import (
    N_ACTIONS,
    _batched_q_actions_gpu,
    _build_obs_batch_gpu,
    _mirror_action_tensor,
)
from .ppo_model import ActorCritic


class PPORollout:
    """Owns N parallel games + preallocated (T, N) storage tensors.

    Usage per PPO update:
        ro.collect(actor_critic, rollout_steps=T)
        last_value = ro.bootstrap_value(actor_critic)   # V(next_obs_{T-1})
        advs, rets = compute_gae(ro.rewards, ro.values, ro.dones,
                                 last_value, ro.next_done,
                                 gamma, lam)
        # Flatten (T, N, ...) → (T*N, ...) and feed PPO epochs.

    State persists across updates (next_obs / next_done carry over) — that's
    the "no reset at update boundaries" rule from the PPO blog. Done envs
    auto-reset during collect().
    """

    def __init__(
        self,
        n_envs: int,
        rollout_steps: int,
        obs_dim: int,
        device: torch.device,
        snake_length: int = 4,
        snake_multiplier: int = 2,
        max_steps: int = 800,
        interp_ball: bool = True,
        shape_coef: float = 0.0,
        shape_gamma: float = 0.99,
        bf16: bool = False,
        learner_safety_filter: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.device = device
        self.snake_length = snake_length
        self.snake_multiplier = snake_multiplier
        self.max_steps = max_steps
        self.interp_ball = interp_ball
        # Potential-based reward shaping (Ng et al. 1999): preserves the optimal
        # policy while giving a dense per-step signal. Φ ∝ -distance(ball, opp_goal)
        # normalized to [-1, 0]; shaped reward per step is
        #   α · (γ·Φ(s') - Φ(s)),
        # with Φ(terminal) = 0. Set shape_coef=0 to disable.
        self.shape_coef = shape_coef
        self.shape_gamma = shape_gamma
        # bf16 autocast for actor/critic/opponent forwards during rollout.
        # Doesn't change stored logits/values (we cast back to fp32 in the
        # rollout loop) — just speeds up the big matmuls on Ampere+.
        self.bf16 = bf16
        # Mask the learner's action distribution to legal moves before sampling.
        # Makes the learned policy consistent with the browser's never-crash
        # filter. The per-step legal mask is stored in `self.legal` so the
        # update phase can re-apply it to the recomputed logits.
        self.learner_safety_filter = learner_safety_filter

        self._np_rng = rng if rng is not None else np.random.default_rng()
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(int(self._np_rng.integers(1 << 31)))

        self.vec = TorchVectorSnakePongGame(
            n_games=n_envs, snake_length=snake_length,
            snake_multiplier=snake_multiplier, device=device,
            seed=int(self._np_rng.integers(1 << 31)),
        )
        half = torch.rand(n_envs, generator=self._gen, device=device) < 0.5
        self.learner_sides = torch.where(half, 1, 2).to(torch.int64)
        self.ep_lengths = torch.zeros(n_envs, dtype=torch.int64, device=device)
        self.ep_returns = torch.zeros(n_envs, dtype=torch.float32, device=device)

        # Preallocated rollout buffers (T, N, ...) on device.
        T, N, D = rollout_steps, n_envs, obs_dim
        self.obs = torch.zeros((T, N, D), dtype=torch.float32, device=device)
        self.actions = torch.zeros((T, N), dtype=torch.int64, device=device)
        self.logprobs = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.dones = torch.zeros((T, N), dtype=torch.bool, device=device)

        # GPU-resident episode-event log — populated per step without touching
        # CPU. Bulk-transferred once at end of rollout for Python-side logging.
        self._ev_done = torch.zeros((T, N), dtype=torch.bool, device=device)
        self._ev_truncated = torch.zeros((T, N), dtype=torch.bool, device=device)
        self._ev_length = torch.zeros((T, N), dtype=torch.int32, device=device)
        self._ev_return = torch.zeros((T, N), dtype=torch.float32, device=device)
        self._ev_won = torch.zeros((T, N), dtype=torch.bool, device=device)
        self._ev_scorer = torch.zeros((T, N), dtype=torch.int8, device=device)

        # Per-step legal-action mask for the learner (only allocated when
        # safety filter is on). Shape (T, N, 4) bool — True = legal.
        if learner_safety_filter:
            self.legal = torch.zeros((T, N, 4), dtype=torch.bool, device=device)
        else:
            self.legal = None

        # Carry-over across update boundaries.
        self.next_obs = _build_obs_batch_gpu(self.vec, self.learner_sides, interp_ball)
        self.next_done = torch.zeros(n_envs, dtype=torch.bool, device=device)

        self._opp_net: Optional[nn.Module] = None
        self._opp_is_random: bool = True
        self._opp_epsilon: float = 0.0
        self._opp_safety_filter: bool = False

    # ── Opponent ──────────────────────────────────────────────────────────
    def set_opponent(
        self, opp_net: Optional[nn.Module], opp_epsilon: float = 0.05,
        opp_safety_filter: bool = False,
    ) -> None:
        """None opponent → uniform-random actions. Otherwise opp_net is any
        module returning (B, A) or (B, K, A); the existing DQN plumbing
        handles both via _batched_q_actions_gpu.

        opp_safety_filter: if True, mask out opponent actions that would
        immediately kill the opponent snake before argmax (matches the
        in-browser v12 filter). Forces the learner to beat the opponent at
        real play rather than exploiting crash behavior.
        """
        self._opp_net = opp_net
        self._opp_is_random = opp_net is None
        self._opp_epsilon = opp_epsilon
        self._opp_safety_filter = opp_safety_filter

    # ── Internal: reset done envs (sync-free) ────────────────────────────
    def _reset(self, mask: torch.Tensor) -> None:
        """Apply reset where mask[i]=True. Always O(N); no GPU→CPU sync.
        The per-env torch.where() is cheap compared to eliminating the stall."""
        self.vec.reset(mask)
        new_sides_all = torch.where(
            torch.rand(self.n_envs, generator=self._gen, device=self.device) < 0.5,
            1, 2,
        ).to(torch.int64)
        self.learner_sides = torch.where(mask, new_sides_all, self.learner_sides)
        zeros_len = torch.zeros_like(self.ep_lengths)
        zeros_ret = torch.zeros_like(self.ep_returns)
        self.ep_lengths = torch.where(mask, zeros_len, self.ep_lengths)
        self.ep_returns = torch.where(mask, zeros_ret, self.ep_returns)

    # ── Main collect loop ────────────────────────────────────────────────
    @torch.no_grad()
    def collect(self, actor_critic: ActorCritic) -> list[dict]:
        """Collect `rollout_steps` transitions and write into self.* buffers.

        Returns a list of completed-episode dicts (for logging: won/length).
        """
        completed: list[dict] = []
        mirror_t = _mirror_action_tensor(self.device)
        T, N = self.rollout_steps, self.n_envs

        for t in range(T):
            # Persist the pre-step observation and the done flag that applies
            # to this timestep (i.e. the mask saying "env started fresh for
            # this step"). PPO/GAE uses this as the (t+1) slot for t-1.
            obs_t = self.next_obs
            done_t = self.next_done
            self.obs[t] = obs_t
            self.dones[t] = done_t

            # Learner acts stochastically. bf16 autocast wraps the matmuls;
            # we cast results back to fp32 before storing so the rollout
            # buffers match the (fp32) update-phase expectations.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                logits = actor_critic.actor(obs_t)
                value = actor_critic.critic(obs_t)
            logits = logits.float()
            value = value.float()

            # Safety-mask logits before sampling. If every action would kill
            # the snake (trapped), leave logits untouched so Categorical stays
            # valid. Store the mask so the update phase can apply the same
            # filter on recomputed logits.
            if self.learner_safety_filter:
                from .vec_tournament import _legal_mask_ego_subset
                all_idx = torch.arange(N, device=self.device, dtype=torch.int64)
                legal = _legal_mask_ego_subset(
                    self.vec, self.learner_sides, all_idx, mirror_t,
                )                                         # (N, 4) bool
                any_legal = legal.any(dim=1, keepdim=True)
                # Where any action is legal, mask illegals to -inf.
                masked = logits.masked_fill(~legal, float("-inf"))
                logits = torch.where(any_legal, masked, logits)
                self.legal[t] = legal

            dist = Categorical(logits=logits)
            action_ego = dist.sample()
            logp = dist.log_prob(action_ego)

            self.actions[t] = action_ego
            self.logprobs[t] = logp
            self.values[t] = value

            # Opponent action.
            opp_sides = 3 - self.learner_sides
            if self._opp_is_random:
                opp_action_ego = torch.randint(
                    0, N_ACTIONS, (N,), generator=self._gen,
                    device=self.device, dtype=torch.int64,
                )
            else:
                opp_obs = _build_obs_batch_gpu(self.vec, opp_sides, self.interp_ball)
                opp_K = getattr(self._opp_net, "n_heads", 1) or 1
                opp_heads = None
                if opp_K > 1:
                    opp_heads = torch.randint(
                        0, opp_K, (N,), generator=self._gen,
                        device=self.device, dtype=torch.int64,
                    )
                opp_autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16)
                if self._opp_safety_filter:
                    from .vec_tournament import _safety_mask_ego_subset
                    with opp_autocast_ctx:
                        q_opp = self._opp_net(opp_obs)
                    q_opp = q_opp.float()
                    if q_opp.dim() == 3:
                        if opp_heads is not None:
                            q_opp = q_opp.gather(
                                1, opp_heads.view(-1, 1, 1).expand(-1, 1, q_opp.size(-1)),
                            ).squeeze(1)
                        else:
                            q_opp = q_opp.mean(dim=1)
                    all_idx = torch.arange(N, device=self.device, dtype=torch.int64)
                    opp_action_ego = _safety_mask_ego_subset(
                        q_opp, self.vec, opp_sides, all_idx, mirror_t,
                    )
                    # Optional ε-random exploration on top of the filter.
                    if self._opp_epsilon > 0.0:
                        rand = torch.rand(N, generator=self._gen, device=self.device)
                        mask = rand < self._opp_epsilon
                        if mask.any():
                            n_r = int(mask.sum().item())
                            rand_a = torch.randint(
                                0, N_ACTIONS, (n_r,), generator=self._gen,
                                device=self.device, dtype=torch.int64,
                            )
                            opp_action_ego = opp_action_ego.masked_scatter(mask, rand_a)
                else:
                    with opp_autocast_ctx:
                        opp_action_ego = _batched_q_actions_gpu(
                            self._opp_net, opp_obs, self._opp_epsilon, self._gen,
                            active_heads=opp_heads,
                        )

            # Mirror egocentric → real-board actions.
            learner_real = torch.where(
                self.learner_sides == 1, action_ego, mirror_t[action_ego],
            )
            opp_real = torch.where(
                opp_sides == 1, opp_action_ego, mirror_t[opp_action_ego],
            )
            real_actions = torch.zeros((N, 2), dtype=torch.int64, device=self.device)
            is_side1 = self.learner_sides == 1
            real_actions[:, 0] = torch.where(is_side1, learner_real, opp_real)
            real_actions[:, 1] = torch.where(is_side1, opp_real, learner_real)

            # Potential Φ(s) BEFORE stepping — normalised ball distance to
            # learner's opponent-goal, ∈ [-1, 0]. Learner on side 1 (left)
            # wants ball → COLS-1; learner on side 2 mirrors.
            if self.shape_coef != 0.0:
                bx_pre = self.vec.balls[:, 0].to(torch.float32)
                phi_pre = torch.where(
                    self.learner_sides == 1,
                    (bx_pre - (COLS - 1)) / (COLS - 1),   # -1 at left, 0 at right
                    -bx_pre / (COLS - 1),                  # 0 at left, -1 at right
                )
            else:
                phi_pre = None  # unused

            result = self.vec.step(real_actions)
            self.ep_lengths += 1

            scorer = result["scorer"]
            won = (scorer.to(torch.int64) == self.learner_sides)
            lost = (scorer != 0) & (scorer != 3) & ~won
            reward = torch.where(won, 1.0, torch.where(lost, -1.0, 0.0)).to(torch.float32)

            # Potential-based shaping. Φ(s') evaluated at the new ball position,
            # BEFORE auto-reset zeroes it. On terminal steps we use Φ(terminal)=0
            # to satisfy the PBRS correctness condition.
            if self.shape_coef != 0.0:
                bx_post = self.vec.balls[:, 0].to(torch.float32)
                phi_post_raw = torch.where(
                    self.learner_sides == 1,
                    (bx_post - (COLS - 1)) / (COLS - 1),
                    -bx_post / (COLS - 1),
                )
                term_now = result["terminated"] | (scorer == 3)
                phi_post = torch.where(term_now, torch.zeros_like(phi_post_raw), phi_post_raw)
                shape = self.shape_coef * (self.shape_gamma * phi_post - phi_pre)
                reward = reward + shape

            self.rewards[t] = reward
            self.ep_returns += reward

            terminated = result["terminated"] | (scorer == 3)
            truncated = (~terminated) & (self.ep_lengths >= self.max_steps)
            self.next_done = terminated

            reset_mask = terminated | truncated
            # Record per-step episode event on GPU (sync-free). Unreset slots
            # just leave their _ev_* entries at zero/default which we'll filter
            # out later via _ev_done.
            self._ev_done[t] = reset_mask
            self._ev_truncated[t] = truncated & ~terminated
            self._ev_length[t] = self.ep_lengths.to(torch.int32)
            self._ev_return[t] = self.ep_returns
            self._ev_won[t] = won
            self._ev_scorer[t] = scorer

            # Auto-reset without sync.
            self.vec.done = self.vec.done & ~reset_mask
            self._reset(reset_mask)

            self.next_obs = _build_obs_batch_gpu(
                self.vec, self.learner_sides, self.interp_ball,
            )

        # ONE bulk GPU→CPU transfer for all completed episodes. This is the
        # only .cpu() call in the rollout.
        ev_mask = self._ev_done
        if bool(ev_mask.any().item()):
            idxs = ev_mask.nonzero(as_tuple=False)  # (K, 2) — (t, env)
            lengths = self._ev_length[ev_mask].cpu().numpy()
            returns = self._ev_return[ev_mask].cpu().numpy()
            wons = self._ev_won[ev_mask].cpu().numpy()
            scorers = self._ev_scorer[ev_mask].cpu().numpy()
            truncs = self._ev_truncated[ev_mask].cpu().numpy()
            for i in range(lengths.shape[0]):
                si = int(scorers[i])
                is_trunc = bool(truncs[i])
                completed.append({
                    "length": int(lengths[i]),
                    "return": float(returns[i]),
                    "won": bool(wons[i]),
                    "scorer": si,
                    "terminal": ("truncated" if is_trunc
                                 else ("draw" if si == 3 else "scored")),
                })

        return completed

    @torch.no_grad()
    def bootstrap_value(self, actor_critic: ActorCritic) -> torch.Tensor:
        """V(next_obs_{T-1}) for bootstrapping GAE on the last timestep."""
        return actor_critic.critic(self.next_obs)

    # ── Flatten (T, N, ...) buffers for minibatch SGD ────────────────────
    def flat_views(self) -> dict[str, torch.Tensor]:
        """Return flattened (T*N, ...) views — same memory, no copy."""
        T, N = self.rollout_steps, self.n_envs
        out = {
            "obs": self.obs.reshape(T * N, self.obs_dim),
            "actions": self.actions.reshape(T * N),
            "logprobs": self.logprobs.reshape(T * N),
            "values": self.values.reshape(T * N),
        }
        if self.legal is not None:
            out["legal"] = self.legal.reshape(T * N, 4)
        return out
