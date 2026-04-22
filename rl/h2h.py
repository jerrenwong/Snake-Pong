"""Head-to-head match: two checkpoints play each other for N games.

Usage:
    python -m rl.h2h --a rl/runs/exp_dueling/best.pt --b rl/runs/exp_plain/best.pt -n 100

Prints win rates in both directions + an overall summary.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .dqn import build_q_net
from .gym_env import SnakePongSelfPlayEnv, obs_dim
from .selfplay import make_policy


def load_q(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["q_net"] if isinstance(ckpt, dict) and "q_net" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    arch = cfg.get("model_arch", "mlp")
    L = cfg.get("snake_length", 4)
    M = cfg.get("snake_multiplier", 1)
    interp = cfg.get("interp_ball_obs", False)
    hidden = cfg.get("hidden_size", 256)
    n_heads = cfg.get("n_heads", 5)
    q = build_q_net(arch, obs_dim(L), n_heads=n_heads, hidden=hidden).to(device).eval()
    q.load_state_dict(state)
    return q, L, M, interp, arch


@torch.no_grad()
def _greedy_action(q_net, obs_np, device):
    t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    q = q_net(t)
    if q.dim() == 3:
        q = q.mean(dim=1)
    return int(q.argmax(dim=1).item())


def run_match(learner, opp_q, n, device, snake_length, snake_multiplier, interp, seed):
    rng = np.random.default_rng(seed)
    opp_pol = make_policy(opp_q, device, epsilon=0.0, rng=rng)
    env = SnakePongSelfPlayEnv(
        opponent_policy=opp_pol,
        snake_length=snake_length, snake_multiplier=snake_multiplier,
        max_steps=800, interp_ball=interp, seed=seed,
    )
    wins = losses = draws = 0
    total_len = 0
    for _ in range(n):
        obs, _ = env.reset()
        while True:
            a = _greedy_action(learner, obs, device)
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc:
                s = info["episode_stats"]
                total_len += s["length"]
                if s["learner_won"]:
                    wins += 1
                elif s.get("scorer") == 0 or s["terminal"] == "truncated":
                    draws += 1
                else:
                    losses += 1
                break
    return wins, losses, draws, total_len / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="Checkpoint A")
    p.add_argument("--b", required=True, help="Checkpoint B")
    p.add_argument("-n", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    a_q, aL, aM, aI, aA = load_q(args.a, device)
    b_q, bL, bM, bI, bA = load_q(args.b, device)

    print(f"A: {args.a}  arch={aA} L={aL} M={aM} interp={aI}")
    print(f"B: {args.b}  arch={bA} L={bL} M={bM} interp={bI}")
    if (aL, aM, aI) != (bL, bM, bI):
        print(f"WARNING: configs differ — picking A's for env. (A={aL,aM,aI}  B={bL,bM,bI})")

    wa, la, da, avg_a = run_match(
        a_q, b_q, args.n, device, aL, aM, aI, args.seed,
    )
    wb, lb, db, avg_b = run_match(
        b_q, a_q, args.n, device, bL, bM, bI, args.seed + 1,
    )
    print()
    print(f"A (learner) vs B (opponent), {args.n} games: W={wa} L={la} D={da}  (avg_len={avg_a:.0f})")
    print(f"B (learner) vs A (opponent), {args.n} games: W={wb} L={lb} D={db}  (avg_len={avg_b:.0f})")
    print()
    a_score = wa + lb  # A wins + A wins again (as opponent when B loses)
    b_score = wb + la
    total = a_score + b_score
    print(f"Overall: A={a_score}/{total} ({a_score/total:.0%})  B={b_score}/{total} ({b_score/total:.0%})")


if __name__ == "__main__":
    main()
