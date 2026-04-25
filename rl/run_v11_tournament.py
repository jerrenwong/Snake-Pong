"""v11 vs v9 vs stable-benchmarks tournament.

Two passes:
  1. Mean-reduction matchup: v11_best (mean), v11_final (mean), v9_h4, v5, plain.
  2. If v11 is competitive, per-head matchup: v11_best/h{0..4} vs v9_h4.
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn

from .ai_guest import _load_q
from .models import build_q_net
from .gym_env import obs_dim
from .vec_tournament import run_tournament, print_tournament


def _load_full(ckpt_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    q = build_q_net(
        cfg["model_arch"], obs_dim(cfg["snake_length"]),
        n_heads=cfg.get("n_heads", 5), hidden=cfg.get("hidden_size", 256),
    ).to(device).eval()
    q.load_state_dict(ckpt["q_net"])
    for p in q.parameters():
        p.requires_grad_(False)
    return q, cfg


class FixedHead(nn.Module):
    """Wrap a bootstrapped/independent-ensemble net as (B, A) for a fixed head."""

    def __init__(self, base: nn.Module, head_idx: int):
        super().__init__()
        self.base = base
        self.h = head_idx

    def forward(self, obs):
        q = self.base(obs)
        if q.dim() == 3:
            return q[:, self.h, :]
        return q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games-per-pair", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per-head", action="store_true",
                   help="Also run per-head matchup of v11_best heads vs v9_h4.")
    args = p.parse_args()
    device = torch.device(args.device)

    print("Loading models...")
    v11_best, _ = _load_full("rl/runs/v11/best.pt", device)
    v11_final, _ = _load_full("rl/runs/v11/latest.pt", device)
    v9_full, _ = _load_full("rl/runs/v9/play_h4_final.pt", device)
    v9_h4 = FixedHead(v9_full, 4).eval()
    v5, _ = _load_full("rl/runs/v5/best.pt", device)
    plain, _ = _load_full("rl/runs/exp_plain/best.pt", device)

    # Pass 1: mean reduction (the default in vec_tournament for (B,K,A) models).
    print("\n=== Pass 1: mean-reduction ===")
    models_p1 = {
        "v11_best": v11_best,
        "v11_final": v11_final,
        "v9_h4": v9_h4,
        "v5": v5,
        "plain": plain,
    }
    res_p1 = run_tournament(
        models_p1, games_per_pair=args.games_per_pair, device=device, seed=1,
    )
    print_tournament(res_p1)

    if args.per_head:
        print("\n=== Pass 2: v11_best per-head vs v9_h4 / v5 ===")
        models_p2 = {f"v11b_h{k}": FixedHead(v11_best, k).eval() for k in range(5)}
        models_p2.update({"v9_h4": v9_h4, "v5": v5})
        res_p2 = run_tournament(
            models_p2, games_per_pair=args.games_per_pair, device=device, seed=2,
        )
        print_tournament(res_p2)


if __name__ == "__main__":
    main()
