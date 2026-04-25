"""v12 PPO vs v11 / v9_h4 / v5 tournament.

Loads PPO ActorCritic + DQN checkpoints into a single vec_tournament bracket.
PPO enters as its actor module (returns (B, 4) logits → argmax = policy
action), which plugs into vec_tournament's argmax-per-head code path
unchanged.

Usage:
    python -m rl.run_v12_tournament \
        --games-per-pair 80 \
        --v12 rl/runs/v12/latest.pt \
        --v12-best rl/runs/v12/best.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .gym_env import obs_dim
from .models import build_q_net
from .ppo_model import build_actor_critic
from .run_v11_tournament import FixedHead, _load_full
from .vec_tournament import print_tournament, run_tournament


def _load_ppo_actor(path: str, device: torch.device) -> nn.Module:
    """Load a PPO checkpoint and return the actor module (logits head)."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    ac = build_actor_critic(
        obs_dim=obs_dim(cfg["snake_length"]),
        hidden=cfg.get("hidden_size", 64),
    ).to(device).eval()
    ac.load_state_dict(ckpt.get("ac") or ckpt["q_net"])
    for p in ac.parameters():
        p.requires_grad_(False)
    return ac.actor.eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games-per-pair", type=int, default=80)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--v12", default="rl/runs/v12/latest.pt",
                   help="Path to v12 (PPO) final checkpoint.")
    p.add_argument("--v12-best", default="rl/runs/v12/best.pt",
                   help="Path to v12 best.pt (by stable-benchmark eval).")
    args = p.parse_args()
    device = torch.device(args.device)

    print("Loading models...")
    v12_final = _load_ppo_actor(args.v12, device)
    v12_best = _load_ppo_actor(args.v12_best, device)

    v11_final_raw, _ = _load_full("rl/runs/v11/latest.pt", device)  # multi-head; tournament will mean-reduce
    v11_best_raw, _ = _load_full("rl/runs/v11/best.pt", device)

    v9_full, _ = _load_full("rl/runs/v9/play_h4_final.pt", device)
    v9_h4 = FixedHead(v9_full, 4).eval()

    v5, _ = _load_full("rl/runs/v5/best.pt", device)

    models = {
        "v12_final":  v12_final,
        "v12_best":   v12_best,
        "v11_final":  v11_final_raw,
        "v11_best":   v11_best_raw,
        "v9_h4":      v9_h4,
        "v5":         v5,
    }
    res = run_tournament(
        models, games_per_pair=args.games_per_pair, device=device, seed=7,
    )
    print_tournament(res)


if __name__ == "__main__":
    main()
