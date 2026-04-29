"""Final tournament among all 114 v18d pool snapshots.

Loads every checkpoint in `rl/runs/v18d_league/pool/`, runs a round-robin
tournament with safety filter on both sides, prints rankings, and dumps
the full win/loss/draw matrices to JSON for later analysis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from rl.gym_env import obs_dim
from rl.ppo_league import _ActorQWrapper
from rl.ppo_model import build_actor_critic
from rl.vec_tournament import run_tournament


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_dir = Path("rl/runs/v18d_league/pool")
    ckpts = sorted(pool_dir.glob("*.pt"))
    print(f"Loading {len(ckpts)} pool snapshots from {pool_dir}")

    models: dict[str, torch.nn.Module] = {}
    for p in ckpts:
        ck = torch.load(p, map_location=device, weights_only=False)
        cfg = ck["config"]
        ac = build_actor_critic(
            obs_dim=obs_dim(cfg["snake_length"]),
            hidden=cfg["hidden_size"],
        ).to(device).eval()
        ac.load_state_dict(ck["ac"])
        for prm in ac.parameters():
            prm.requires_grad_(False)
        models[p.stem] = _ActorQWrapper(ac).eval()

    games_per_pair = 16
    M = len(models)
    n_games = M * M * games_per_pair
    print(f"Tournament: {M} models × {M} × {games_per_pair} = {n_games:,} parallel games")
    print(f"Memory estimate: ~{n_games * 130 // (1024 * 1024):.1f} GB env state")

    t0 = time.time()
    res = run_tournament(
        models, games_per_pair=games_per_pair, device=device, seed=42,
        snake_length=4, snake_multiplier=2, max_steps=800, interp_ball=True,
        safety_filter=True, show_progress=True,
    )
    print(f"\nTournament wall time: {time.time() - t0:.1f}s")

    out_path = Path("rl/runs/v18d_league/final_tournament.json")
    json.dump(
        {
            "names": res["names"],
            "wins": res["wins"].tolist(),
            "losses": res["losses"].tolist(),
            "draws": res["draws"].tolist(),
            "win_rate": res["win_rate"].tolist(),
            "games_per_pair": games_per_pair,
        },
        out_path.open("w"),
    )
    print(f"Wrote full results to {out_path}")

    # Rankings — exclude self-match diagonal.
    names = res["names"]
    wr = np.asarray(res["win_rate"], dtype=np.float64)
    losses = np.asarray(res["losses"], dtype=np.float64)
    draws = np.asarray(res["draws"], dtype=np.float64)
    wins = np.asarray(res["wins"], dtype=np.float64)
    np.fill_diagonal(wr, np.nan)

    total_per_pair = wins + losses + draws
    np.fill_diagonal(total_per_pair, 1)  # avoid div-by-zero on diag
    loss_rate = losses / np.maximum(total_per_pair, 1)
    draw_rate = draws / np.maximum(total_per_pair, 1)
    np.fill_diagonal(loss_rate, np.nan)
    np.fill_diagonal(draw_rate, np.nan)

    mean_wr = np.nanmean(wr, axis=1)
    mean_lr = np.nanmean(loss_rate, axis=1)
    mean_dr = np.nanmean(draw_rate, axis=1)
    ranking = np.argsort(-mean_wr)

    print(f"\n{'rank':>4} {'name':<30} {'W%':>6} {'L%':>6} {'D%':>6}")
    print("-" * 60)
    for r, idx in enumerate(ranking[:25]):
        print(f"{r+1:>4} {names[idx]:<30} "
              f"{mean_wr[idx]*100:>5.1f}% {mean_lr[idx]*100:>5.1f}% "
              f"{mean_dr[idx]*100:>5.1f}%")

    # Per-lineage best
    print("\nBest per lineage:")
    for prefix in ("lineage_0_", "lineage_1_", "lineage_2_", "lineage_3_"):
        best_idx = -1
        best_wr = -1.0
        for i, n in enumerate(names):
            if n.startswith(prefix) and mean_wr[i] > best_wr:
                best_wr = float(mean_wr[i])
                best_idx = i
        if best_idx >= 0:
            print(f"  {names[best_idx]}: mean_wr={best_wr*100:.1f}%  "
                  f"loss={mean_lr[best_idx]*100:.1f}%  "
                  f"draw={mean_dr[best_idx]*100:.1f}%")


if __name__ == "__main__":
    main()
