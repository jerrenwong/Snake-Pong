"""Visualize / evaluate a trained Snake-Pong DQN.

Usage:
    python -m rl.play --checkpoint rl/runs/default/latest.pt --episodes 5
    python -m rl.play --checkpoint rl/runs/default/latest.pt --vs-random --episodes 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .dqn import QNetwork, greedy_action
from .gym_env import SnakePongSelfPlayEnv
from .selfplay import make_policy, random_policy


def load_q(checkpoint: str, device: torch.device) -> QNetwork:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["q_net"] if isinstance(ckpt, dict) and "q_net" in ckpt else ckpt
    q = QNetwork().to(device)
    q.load_state_dict(state)
    q.eval()
    return q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", action="store_true")
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--vs-random", action="store_true",
                   help="Opponent is random. Default: opponent is the same checkpoint (self-play eval).")
    p.add_argument("--vs-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    q = load_q(args.checkpoint, device)

    if args.vs_random:
        opp = random_policy(rng)
    elif args.vs_checkpoint:
        opp_q = load_q(args.vs_checkpoint, device)
        opp = make_policy(opp_q, device, epsilon=0.0, rng=rng)
    else:
        opp = make_policy(q, device, epsilon=0.0, rng=rng)

    env = SnakePongSelfPlayEnv(opponent_policy=opp, seed=args.seed)

    wins = 0
    losses = 0
    draws = 0
    total_len = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        ep_len = 0
        while True:
            a = greedy_action(q, obs, device)
            obs, r, term, trunc, info = env.step(a)
            ep_len += 1
            if args.render:
                print("\033[2J\033[H" + env.render())
                print(f"ep {ep+1} step {ep_len} side={info['learner_side']} r={r}")
                time.sleep(args.sleep)
            if term or trunc:
                stats = info["episode_stats"]
                won = stats["learner_won"]
                if won:
                    wins += 1
                elif stats["scorer"] == 0 or stats["terminal"] == "truncated":
                    draws += 1
                else:
                    losses += 1
                total_len += ep_len
                break
    total = wins + losses + draws
    print(f"Episodes: {total}  wins={wins} ({wins/total:.1%})  "
          f"losses={losses} ({losses/total:.1%})  draws={draws} ({draws/total:.1%})  "
          f"avg_len={total_len/total:.1f}")


if __name__ == "__main__":
    main()
