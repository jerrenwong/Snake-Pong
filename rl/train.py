"""Self-play DQN training entrypoint for Snake-Pong.

Usage:
    python -m rl.train --iters 200 --episodes-per-iter 20 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .dqn import (
    N_ACTIONS, QNetwork, ReplayBuffer, Transition,
    compute_loss, epsilon_greedy_action,
)
from .gym_env import SnakePongSelfPlayEnv
from .selfplay import OpponentPool


def run_episode(
    env: SnakePongSelfPlayEnv,
    q_net: QNetwork,
    replay: ReplayBuffer,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    obs, _ = env.reset()
    ep_reward = 0.0
    ep_len = 0
    while True:
        action = epsilon_greedy_action(q_net, obs, epsilon, device, rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Only treat terminal (not truncation) as absorbing — Sutton & Barto.
        replay.push(Transition(
            grid=obs["grid"], scalars=obs["scalars"],
            action=action, reward=float(reward),
            next_grid=next_obs["grid"], next_scalars=next_obs["scalars"],
            done=bool(terminated),
        ))
        obs = next_obs
        ep_reward += reward
        ep_len += 1
        if terminated or truncated:
            stats = info.get("episode_stats", {})
            return {
                "reward": ep_reward,
                "length": ep_len,
                "won": stats.get("learner_won", False),
                "terminal": stats.get("terminal", "truncated"),
            }


def train(cfg: argparse.Namespace) -> None:
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    pool = OpponentPool(device, max_snapshots=cfg.pool_size,
                        random_prob=cfg.opponent_random_prob,
                        opponent_epsilon=cfg.opponent_epsilon, rng=rng)

    env = SnakePongSelfPlayEnv(
        opponent_policy=pool.sample(),
        snake_length=cfg.snake_length,
        max_steps=cfg.max_steps,
        seed=int(rng.integers(1 << 31)),
    )

    replay = ReplayBuffer(cfg.replay_capacity)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    win_hist = deque(maxlen=100)
    len_hist = deque(maxlen=100)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")

    total_env_steps = 0
    total_grad_steps = 0

    # Linear epsilon schedule by env step
    def epsilon_at(step: int) -> float:
        frac = min(1.0, step / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    start_time = time.time()

    for it in range(cfg.iters):
        env.set_opponent(pool.sample())

        # 1) Rollout
        for ep in range(cfg.episodes_per_iter):
            ep_stats = run_episode(env, q_net, replay, epsilon_at(total_env_steps), device, rng)
            total_env_steps += ep_stats["length"]
            win_hist.append(1 if ep_stats["won"] else 0)
            len_hist.append(ep_stats["length"])

        # 2) Train
        if replay.size >= cfg.min_replay:
            for _ in range(cfg.grad_steps_per_iter):
                batch = replay.sample(cfg.batch_size, rng)
                loss = compute_loss(q_net, target_net, batch, cfg.gamma, device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                total_grad_steps += 1

                if total_grad_steps % cfg.target_update_every == 0:
                    target_net.load_state_dict(q_net.state_dict())

        # 3) Snapshot to pool
        if (it + 1) % cfg.snapshot_every == 0:
            pool.add_snapshot(q_net)

        # 4) Log + checkpoint
        elapsed = time.time() - start_time
        win_rate = float(np.mean(win_hist)) if win_hist else 0.0
        avg_len = float(np.mean(len_hist)) if len_hist else 0.0
        row = {
            "iter": it + 1,
            "env_steps": total_env_steps,
            "grad_steps": total_grad_steps,
            "epsilon": epsilon_at(total_env_steps),
            "win_rate_100": win_rate,
            "avg_len_100": avg_len,
            "pool_size": len(pool),
            "elapsed_s": round(elapsed, 1),
        }
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        if (it + 1) % cfg.print_every == 0:
            print(f"[{it+1:4d}/{cfg.iters}] steps={total_env_steps:>7d} "
                  f"eps={row['epsilon']:.2f} win100={win_rate:.2%} "
                  f"len100={avg_len:.1f} pool={len(pool)} elapsed={elapsed:.0f}s")

        if (it + 1) % cfg.save_every == 0:
            torch.save({
                "q_net": q_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(cfg),
                "iter": it + 1,
            }, out_dir / f"checkpoint_iter{it+1:05d}.pt")
            torch.save(q_net.state_dict(), out_dir / "latest.pt")

    torch.save({
        "q_net": q_net.state_dict(),
        "config": vars(cfg),
        "iter": cfg.iters,
    }, out_dir / "final.pt")
    torch.save(q_net.state_dict(), out_dir / "latest.pt")
    log_f.close()
    print(f"Done. Checkpoints in {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Outer loop
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=20)
    p.add_argument("--grad-steps-per-iter", type=int, default=100)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--target-update-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--print-every", type=int, default=1)
    # DQN
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--replay-capacity", type=int, default=100_000)
    p.add_argument("--min-replay", type=int, default=5_000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    # Env
    p.add_argument("--snake-length", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=500)
    # Self-play
    p.add_argument("--pool-size", type=int, default=10)
    p.add_argument("--opponent-random-prob", type=float, default=0.25)
    p.add_argument("--opponent-epsilon", type=float, default=0.05)
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="rl/runs/default")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
