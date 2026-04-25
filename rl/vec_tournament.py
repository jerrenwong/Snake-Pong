"""Batched GPU tournament: run all-vs-all matchups in parallel.

Usage:
    from rl.vec_tournament import run_tournament

    results = run_tournament(
        models={"v5": q_v5, "v8": q_v8, "v9_h0": q_v9_h0, ...},
        games_per_pair=50,
        snake_length=4, snake_multiplier=2, max_steps=800, interp_ball=True,
        device="cuda",
    )
    # results["win_rate"] is a (M, M) numpy array where row=learner, col=opponent.

Key idea: build ONE vectorized env with N_slots = M × M × G game instances.
At every env step, group slots by their learner model, do one batched forward
per unique model, then do the same for opponents. GPU does a handful of
reasonably-sized batched matmuls instead of N_slots batch-1 forwards.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .env import COLS, ROWS
from .env_torch import TorchVectorSnakePongGame
from .vec_rollout_gpu import _build_obs_batch_gpu, _mirror_action_tensor


@torch.no_grad()
def _legal_mask_ego_subset(
    vec: TorchVectorSnakePongGame,
    sides: torch.Tensor,      # (K,) int64 in {1, 2}
    idx: torch.Tensor,        # (K,) int64 global env indices of these slots
    mirror_t: torch.Tensor,   # (4,) int64 mirror table
) -> torch.Tensor:
    """Returns (K, 4) bool — True where the ego action is non-fatal.

    Same geometric check used by the argmax-masking helper, but returns the
    raw legality mask so a stochastic sampler (Categorical) can mask its
    logits before sampling, keeping PPO's log-prob math consistent with the
    effective policy.
    """
    K = sides.shape[0]
    device = sides.device
    L = vec.snake_length
    action_deltas = torch.tensor(
        [[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=torch.int32, device=device,
    )

    own_idx = (sides - 1).to(torch.int64)
    is_side2 = sides == 2
    bodies = vec.bodies[idx]
    dirs = vec.dirs[idx]

    own_body = bodies.gather(
        1, own_idx.view(-1, 1, 1, 1).expand(-1, 1, L, 2),
    ).squeeze(1)
    opp_body = bodies.gather(
        1, (1 - own_idx).view(-1, 1, 1, 1).expand(-1, 1, L, 2),
    ).squeeze(1)
    own_head = own_body[:, 0, :]
    own_dir = dirs.gather(
        1, own_idx.view(-1, 1, 1).expand(-1, 1, 2),
    ).squeeze(1)

    ego_range = torch.arange(4, device=device)
    real_action = torch.where(
        is_side2.view(-1, 1), mirror_t[ego_range].view(1, -1).expand(K, 4),
        ego_range.view(1, -1).expand(K, 4),
    )
    real_delta = action_deltas[real_action]
    is_reverse = ((real_delta[..., 0] == -own_dir[:, 0:1]) &
                  (real_delta[..., 1] == -own_dir[:, 1:2]))
    effective = torch.where(
        is_reverse.unsqueeze(-1),
        own_dir.unsqueeze(1).expand(-1, 4, -1),
        real_delta,
    )
    new_head = own_head.unsqueeze(1) + effective

    oob = ((new_head[..., 0] < 0) | (new_head[..., 0] >= COLS) |
           (new_head[..., 1] < 0) | (new_head[..., 1] >= ROWS))
    own_block = own_body[:, :-1, :]
    self_hit = ((new_head.unsqueeze(2) == own_block.unsqueeze(1))
                .all(dim=-1).any(dim=-1))
    opp_hit = ((new_head.unsqueeze(2) == opp_body.unsqueeze(1))
               .all(dim=-1).any(dim=-1))
    return ~(oob | self_hit | opp_hit)                             # legal = True


@torch.no_grad()
def _safety_mask_ego_subset(
    q: torch.Tensor,          # (K, 4) — raw Q/logits for ego actions of K slots
    vec: TorchVectorSnakePongGame,
    sides: torch.Tensor,      # (K,) int64 in {1, 2}
    idx: torch.Tensor,        # (K,) int64 global env indices of these slots
    mirror_t: torch.Tensor,   # (4,) int64 mirror table
) -> torch.Tensor:
    """Mask out ego actions that would immediately kill the acting snake.

    Mirrors src/ai_local.js `_legalActionsP2` but GPU-batched and side-aware.
    `idx` picks the K slots out of vec's full N-env state so this works on
    any subset (e.g. one model's share of the tournament slots).

    Returns (K,) — argmax(q) restricted to legal actions. If every action for
    a slot is fatal, falls back to raw argmax (doomed regardless).
    """
    K = q.shape[0]
    device = q.device
    L = vec.snake_length
    action_deltas = torch.tensor(
        [[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=torch.int32, device=device,
    )  # (4, 2)

    own_idx = (sides - 1).to(torch.int64)      # (K,) 0 or 1
    is_side2 = sides == 2
    bodies = vec.bodies[idx]                    # (K, 2, L, 2)
    dirs = vec.dirs[idx]                        # (K, 2, 2)

    own_body = bodies.gather(
        1, own_idx.view(-1, 1, 1, 1).expand(-1, 1, L, 2),
    ).squeeze(1)                                 # (K, L, 2)
    opp_body = bodies.gather(
        1, (1 - own_idx).view(-1, 1, 1, 1).expand(-1, 1, L, 2),
    ).squeeze(1)                                 # (K, L, 2)
    own_head = own_body[:, 0, :]                 # (K, 2)
    own_dir = dirs.gather(
        1, own_idx.view(-1, 1, 1).expand(-1, 1, 2),
    ).squeeze(1)                                 # (K, 2)

    ego_range = torch.arange(4, device=device)
    real_action = torch.where(
        is_side2.view(-1, 1), mirror_t[ego_range].view(1, -1).expand(K, 4),
        ego_range.view(1, -1).expand(K, 4),
    )                                             # (K, 4)
    real_delta = action_deltas[real_action]       # (K, 4, 2)

    is_reverse = ((real_delta[..., 0] == -own_dir[:, 0:1]) &
                  (real_delta[..., 1] == -own_dir[:, 1:2]))
    effective = torch.where(
        is_reverse.unsqueeze(-1),
        own_dir.unsqueeze(1).expand(-1, 4, -1),
        real_delta,
    )                                              # (K, 4, 2)
    new_head = own_head.unsqueeze(1) + effective   # (K, 4, 2)

    oob = ((new_head[..., 0] < 0) | (new_head[..., 0] >= COLS) |
           (new_head[..., 1] < 0) | (new_head[..., 1] >= ROWS))

    own_block = own_body[:, :-1, :]                # (K, L-1, 2)
    self_hit = ((new_head.unsqueeze(2) == own_block.unsqueeze(1))
                .all(dim=-1).any(dim=-1))
    opp_hit = ((new_head.unsqueeze(2) == opp_body.unsqueeze(1))
               .all(dim=-1).any(dim=-1))
    illegal = oob | self_hit | opp_hit
    q_masked = q.masked_fill(illegal, float("-inf"))
    all_illegal = illegal.all(dim=1)
    return torch.where(all_illegal, q.argmax(dim=1), q_masked.argmax(dim=1))


@torch.no_grad()
def run_tournament(
    models: Dict[str, nn.Module],
    games_per_pair: int = 50,
    snake_length: int = 4,
    snake_multiplier: int = 2,
    max_steps: int = 800,
    interp_ball: bool = True,
    device: str | torch.device = "cuda",
    seed: int = 12345,
    show_progress: bool = True,
    safety_filter: bool = False,
) -> dict:
    """Run a full M × M tournament. Returns dict with keys:
      'names', 'wins', 'losses', 'draws', 'win_rate' (all M×M arrays).
    """
    device = torch.device(device)
    names = list(models.keys())
    M = len(names)
    G = games_per_pair
    N = M * M * G

    gen = torch.Generator(device=device).manual_seed(seed)

    # Per-slot: learner idx, opponent idx (shape (N,))
    learner_idx = torch.arange(N, device=device) // (M * G) % M
    opp_idx = (torch.arange(N, device=device) // G) % M
    # pair_idx in flattened M*M layout
    pair_idx = learner_idx * M + opp_idx

    # Env (N parallel games)
    vec = TorchVectorSnakePongGame(
        n_games=N, snake_length=snake_length,
        snake_multiplier=snake_multiplier, device=device,
        seed=seed,
    )
    # Per-slot state
    learner_sides = torch.where(
        torch.rand(N, generator=gen, device=device) < 0.5, 1, 2,
    ).to(torch.int64)
    ep_lengths = torch.zeros(N, dtype=torch.int64, device=device)

    # Accumulators: (M*M,) counters — wins, losses, draws per pair
    wins = torch.zeros(M * M, dtype=torch.int64, device=device)
    losses = torch.zeros(M * M, dtype=torch.int64, device=device)
    draws = torch.zeros(M * M, dtype=torch.int64, device=device)
    remaining = torch.full((M * M,), G, dtype=torch.int64, device=device)

    # Build a list of tensors for per-model slot masks so we don't recompute
    learner_masks = [(learner_idx == m) for m in range(M)]
    opp_masks = [(opp_idx == m) for m in range(M)]

    model_list = [models[n] for n in names]
    mirror_t = _mirror_action_tensor(device)

    def _act_all(obs_all: torch.Tensor, which: str, sides: torch.Tensor) -> torch.Tensor:
        """For each unique model, do a batched forward on its subset of slots,
        scatter actions back. `which` = 'learner' or 'opp'. `sides` is the
        (N,) tensor of which snake side this actor is playing — needed for
        the optional safety filter (reuses vec.bodies / vec.dirs).
        """
        masks = learner_masks if which == "learner" else opp_masks
        actions = torch.zeros(N, dtype=torch.int64, device=device)
        for m, model in enumerate(model_list):
            mask = masks[m]
            if not mask.any():
                continue
            sub_obs = obs_all[mask]
            q = model(sub_obs)
            if q.dim() == 3:
                q = q.mean(dim=1)
            if safety_filter:
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                a = _safety_mask_ego_subset(q, vec, sides[mask], idx, mirror_t)
            else:
                a = q.argmax(dim=1)
            actions[mask] = a
        return actions

    step = 0
    while True:
        active = (remaining[pair_idx] > 0)  # slot active iff its pair has games left
        if not active.any():
            break

        learner_obs = _build_obs_batch_gpu(vec, learner_sides, interp_ball)
        opp_sides = 3 - learner_sides
        opp_obs = _build_obs_batch_gpu(vec, opp_sides, interp_ball)

        learner_a_ego = _act_all(learner_obs, "learner", learner_sides)
        opp_a_ego = _act_all(opp_obs, "opp", opp_sides)

        # Egocentric → real-board
        learner_real = torch.where(
            learner_sides == 1, learner_a_ego,
            mirror_t[learner_a_ego],
        )
        opp_real = torch.where(
            opp_sides == 1, opp_a_ego,
            mirror_t[opp_a_ego],
        )

        actions = torch.zeros((N, 2), dtype=torch.int64, device=device)
        is_s1 = learner_sides == 1
        actions[:, 0] = torch.where(is_s1, learner_real, opp_real)
        actions[:, 1] = torch.where(is_s1, opp_real, learner_real)

        result = vec.step(actions)
        ep_lengths += 1

        scorer = result["scorer"]  # int8: 0 none, 1 s1, 2 s2, 3 draw
        won = (scorer.to(torch.int64) == learner_sides)
        lost = (scorer != 0) & (scorer != 3) & ~won
        drew = scorer == 3

        terminated = result["terminated"] | drew
        truncated = (~terminated) & (ep_lengths >= max_steps)
        # Reset ALL terminated/truncated slots (regardless of pair quota) so
        # vec.done stays clean for the next step. Only COUNT for active slots.
        reset_mask_all = terminated | truncated

        # Count only for slots whose pair still has quota
        count_mask = reset_mask_all & active
        if count_mask.any():
            done_idx = count_mask.nonzero(as_tuple=False).squeeze(1)
            pa_done = pair_idx[done_idx]
            w_mask = won[done_idx]
            l_mask = lost[done_idx]
            d_mask = drew[done_idx] | (truncated[done_idx] & ~terminated[done_idx])
            wins.scatter_add_(0, pa_done, w_mask.to(torch.int64))
            losses.scatter_add_(0, pa_done, l_mask.to(torch.int64))
            draws.scatter_add_(0, pa_done, d_mask.to(torch.int64))
            finished_counts = torch.zeros(M * M, dtype=torch.int64, device=device)
            finished_counts.scatter_add_(
                0, pa_done, torch.ones_like(pa_done, dtype=torch.int64),
            )
            remaining = torch.clamp(remaining - finished_counts, min=0)

        if reset_mask_all.any():
            vec.done[reset_mask_all] = False
            vec.reset(reset_mask_all)
            k = int(reset_mask_all.sum().item())
            learner_sides[reset_mask_all] = torch.where(
                torch.rand(k, generator=gen, device=device) < 0.5,
                1, 2,
            ).to(torch.int64)
            ep_lengths[reset_mask_all] = 0

        step += 1
        if show_progress and step % 50 == 0:
            done_games = (G - remaining).sum().item()
            total_games = M * M * G
            print(f"  step {step}: {done_games}/{total_games} games completed "
                  f"({done_games/total_games:.1%})")

    # Clamp counts to G in case we overshot (possible: multiple slots finish
    # the same step and push beyond quota).
    wins_mat = wins.view(M, M).clamp(max=G).cpu().numpy()
    losses_mat = losses.view(M, M).clamp(max=G).cpu().numpy()
    draws_mat = draws.view(M, M).clamp(max=G).cpu().numpy()

    # Win rate = wins / (wins + losses + draws)
    total_mat = np.maximum(wins_mat + losses_mat + draws_mat, 1)
    win_rate = wins_mat.astype(np.float32) / total_mat

    return {
        "names": names,
        "wins": wins_mat,
        "losses": losses_mat,
        "draws": draws_mat,
        "win_rate": win_rate,
    }


def print_tournament(res: dict) -> None:
    names = res["names"]
    M = len(names)
    wr = res["win_rate"]
    # Matrix
    print("Win-rate matrix (row=learner, col=opponent):")
    hdr = "".join(f"{n:>9s}" for n in names)
    print("         " + hdr)
    for i, n in enumerate(names):
        row = f"{n:>8s} "
        for j in range(M):
            row += f"{wr[i,j]:8.1%} "
        print(row)
    # Ranking (exclude diagonal)
    rows = []
    for i, n in enumerate(names):
        idx = np.arange(M) != i
        p = wr[i, idx].mean()
        q = wr[idx, i].mean()
        rows.append((n, p, q, p - q))
    rows.sort(key=lambda r: -r[3])
    print("\nRanking (excl. diagonal):")
    for n, p, q, net in rows:
        print(f"  {n:>10s}: wins {p:5.1%}  losses {q:5.1%}  net {net:+.1%}")
