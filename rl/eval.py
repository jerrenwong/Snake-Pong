"""Evaluation utilities for Snake-Pong self-play.

Runs a learner (greedy Q-policy) against a given opponent for a fixed number
of episodes and returns summary stats. Used for benchmark-set evaluation to
monitor catastrophic forgetting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .actions import greedy_action
from .gym_env import SnakePongSelfPlayEnv, OpponentPolicy
from .models import QNetwork


def evaluate(
    q_net: QNetwork,
    opponent_policy: OpponentPolicy,
    n_episodes: int,
    device: torch.device,
    snake_length: int = 4,
    snake_multiplier: int = 1,
    max_steps: int = 500,
    interp_ball: bool = True,
    seed: Optional[int] = None,
) -> dict[str, float]:
    env = SnakePongSelfPlayEnv(
        opponent_policy=opponent_policy,
        snake_length=snake_length,
        snake_multiplier=snake_multiplier,
        max_steps=max_steps,
        interp_ball=interp_ball,
        seed=seed,
    )

    wins = losses = draws = truncs = 0
    total_len = 0
    own_deaths = opp_deaths = ball_misses = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        while True:
            a = greedy_action(q_net, obs, device)
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc:
                stats = info["episode_stats"]
                total_len += stats["length"]
                if trunc and not term:
                    truncs += 1
                    draws += 1
                elif stats["learner_won"]:
                    wins += 1
                    # Why did we win? Opp died, or we scored.
                    term_kind = stats["terminal"]
                    if "died" in term_kind and "s1" not in term_kind and "s2" not in term_kind:
                        pass
                    opp_deaths += 1 if "died" in term_kind and term_kind != "scored" else 0
                elif stats["scorer"] == 0:
                    draws += 1
                else:
                    losses += 1
                break

    n = max(1, n_episodes)
    return {
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
        "trunc_rate": truncs / n,
        "avg_len": total_len / n,
        "n_episodes": n_episodes,
    }
