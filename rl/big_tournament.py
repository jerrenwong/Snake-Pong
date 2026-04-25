"""Big round-robin tournament across every checkpoint we have.

Loads DQN era models (plain, dueling, bootstrapped, independent-ensemble) via
the existing `selfplay._load_external_checkpoint` + head helper, and PPO
actor-critics via `ppo_model.build_actor_critic`. All enter as (B, A)-output
modules (for (B,K,A) ensembles, vec_tournament mean-reduces; for PPO actors
the logits argmax equals the greedy policy action).

Usage:
    python -m rl.big_tournament --games-per-pair 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .gym_env import obs_dim
from .models import build_q_net
from .ppo_model import build_actor_critic
from .selfplay import _HeadWrapper, _load_external_checkpoint
from .vec_tournament import print_tournament, run_tournament


def _load_generic(path: str, device: torch.device, head: int | None = None) -> nn.Module:
    """Works for both DQN checkpoints and PPO actor-critic checkpoints.

    Detects PPO via `config.model_arch == 'ppo_ac'` and returns the actor;
    otherwise defers to the DQN loader (which handles head extraction and
    ensemble archs).
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {}) or {}
    arch = cfg.get("model_arch", "mlp")
    if arch == "ppo_ac":
        assert head is None, f"PPO checkpoint {path} is single-output; drop head=…"
        ac = build_actor_critic(
            obs_dim=obs_dim(cfg["snake_length"]),
            hidden=cfg.get("hidden_size", 64),
        ).to(device).eval()
        ac.load_state_dict(ckpt.get("ac") or ckpt["q_net"])
        for p in ac.parameters():
            p.requires_grad_(False)
        return ac.actor.eval()
    return _load_external_checkpoint(path, device, head=head)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games-per-pair", type=int, default=80)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top-k", type=int, default=4,
                   help="Print the top-K ranking (by net-vs-field).")
    p.add_argument("--safety-filter", action="store_true", default=False,
                   help="Apply the never-crash safety mask to every model's "
                        "action selection. Matches the in-browser filter.")
    args = p.parse_args()
    device = torch.device(args.device)

    entries: list[tuple[str, str, int | None]] = [
        # (name, path, optional head for bootstrap)
        # v3/v4 excluded: different obs dim (L=6) from v5+.
        ("v5",        "rl/runs/v5/best.pt",                        None),
        ("v8",        "rl/runs/v8/best.pt",                        None),
        ("v9_h4",     "rl/runs/v9/play_h4_final.pt",               4),
        ("v11_best",  "rl/runs/v11/best.pt",                       None),
        ("v11_final", "rl/runs/v11/latest.pt",                     None),
        ("v12_sm",    "rl/runs/v12_small/best.pt",                 None),
        ("v12@1400",  "rl/runs/v12/snap_u1400.pt",                 None),
        ("v12@2000",  "rl/runs/v12/snap_latest.pt",                None),
        ("v12@4500",  "rl/runs/v12/snap_u4500.pt",                 None),
        ("v12@5000",  "rl/runs/v12/snap_final.pt",                 None),
        ("v13_r1",    "rl/runs/v13/run_1/latest.pt",               None),
        ("v13_r5",    "rl/runs/v13/run_5/latest.pt",               None),
        ("v14_best",  "rl/runs/v14/best.pt",                       None),
        ("v14_final", "rl/runs/v14/snap_final.pt",                 None),
        ("v15_s0",    "rl/runs/v15_hill/snake_0/failed.pt",        None),
        ("v15_s1",    "rl/runs/v15_hill/snake_1/failed.pt",        None),
    ]

    print("Loading models...")
    models: dict[str, nn.Module] = {}
    for name, path, head in entries:
        if not Path(path).exists():
            print(f"  skip {name}: {path} not found")
            continue
        try:
            models[name] = _load_generic(path, device, head=head)
            print(f"  loaded {name}  ({path})")
        except Exception as e:
            print(f"  skip {name}: load failed: {e}")

    if len(models) < 2:
        raise SystemExit("Need at least 2 models for tournament.")

    res = run_tournament(
        models, games_per_pair=args.games_per_pair, device=device, seed=17,
        safety_filter=args.safety_filter,
    )
    print_tournament(res)

    # Top-K by net (+wins% - losses%)
    import numpy as np
    names = res["names"]
    wr = res["win_rate"]
    M = len(names)
    rows = []
    for i, n in enumerate(names):
        idx = np.arange(M) != i
        p_wins = wr[i, idx].mean()
        p_losses = wr[idx, i].mean()
        rows.append((n, p_wins, p_losses, p_wins - p_losses))
    rows.sort(key=lambda r: -r[3])
    print(f"\n=== Top {args.top_k} opponents to train against ===")
    for n, p, q, net in rows[: args.top_k]:
        print(f"  {n:>12s}: wins {p:5.1%}  losses {q:5.1%}  net {net:+.1%}")


if __name__ == "__main__":
    main()
