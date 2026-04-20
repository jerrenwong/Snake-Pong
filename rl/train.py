"""Self-play DQN training entrypoint for Snake-Pong.

Logs to Weights & Biases: per-iter metrics, periodic benchmark evaluations
(monitors catastrophic forgetting), sample episode videos, and checkpoint
artifacts.

Usage:
    export WANDB_API_KEY=<your_key>
    python -m rl.train --iters 200 --out-dir rl/runs/v1

For offline / disabled wandb:
    WANDB_MODE=offline python -m rl.train ...
    python -m rl.train --wandb-mode disabled ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .dqn import (
    N_ACTIONS, QNetwork, ReplayBuffer, Transition,
    compute_loss, epsilon_greedy_action, greedy_action,
)
from .gym_env import SnakePongSelfPlayEnv, obs_dim
from .eval import evaluate
from .render import record_episode
from .selfplay import BenchmarkSet, OpponentPool


def _greedy_fn_factory(q_net: QNetwork, device: torch.device):
    def fn(obs):
        return greedy_action(q_net, obs, device)
    return fn


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
        replay.push(Transition(
            obs=obs, action=action, reward=float(reward),
            next_obs=next_obs, done=bool(terminated),
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


def _init_wandb(cfg: argparse.Namespace):
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; running without wandb.")
        return None
    if cfg.wandb_mode == "disabled":
        print("[wandb] disabled via --wandb-mode.")
        return None
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config=vars(cfg),
            dir=cfg.out_dir,
        )
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}); proceeding without wandb.")
        return None


def _log_checkpoint(run, ckpt_path: Path, iter_: int, aliases: list[str]) -> None:
    if run is None:
        return
    try:
        import wandb
        artifact = wandb.Artifact(
            name=f"{run.project or 'snake-pong'}-model",
            type="model",
            metadata={"iter": iter_},
        )
        artifact.add_file(str(ckpt_path))
        run.log_artifact(artifact, aliases=aliases)
    except Exception as e:
        print(f"[wandb] artifact upload failed: {e}")


def _log_video(run, frames: np.ndarray, key: str, step: int, fps: int = 10) -> None:
    """frames: (T, 3, H, W) uint8 ndarray."""
    if run is None:
        return
    try:
        import wandb
        run.log({key: wandb.Video(frames, fps=fps, format="gif")}, step=step)
    except Exception as e:
        print(f"[wandb] video upload failed: {e}")


def train(cfg: argparse.Namespace) -> None:
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = _init_wandb(cfg)

    d_obs = obs_dim(cfg.snake_length)
    q_net = QNetwork(d_obs).to(device)
    target_net = QNetwork(d_obs).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    pool = OpponentPool(
        device, max_snapshots=cfg.pool_size,
        random_prob=cfg.opponent_random_prob,
        opponent_epsilon=cfg.opponent_epsilon, rng=rng,
    )
    benchmarks = BenchmarkSet(device, total_iters=cfg.iters, rng=rng)

    env = SnakePongSelfPlayEnv(
        opponent_policy=pool.sample(),
        snake_length=cfg.snake_length,
        max_steps=cfg.max_steps,
        seed=int(rng.integers(1 << 31)),
    )

    replay = ReplayBuffer(cfg.replay_capacity, d_obs)

    win_hist = deque(maxlen=100)
    len_hist = deque(maxlen=100)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")

    total_env_steps = 0
    total_grad_steps = 0

    def epsilon_at(step: int) -> float:
        frac = min(1.0, step / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    video_opponents = [s.strip() for s in cfg.video_opponents.split(",") if s.strip()]

    start_time = time.time()

    for it in range(cfg.iters):
        iter_1based = it + 1
        env.set_opponent(pool.sample())

        # 1) Rollout
        for ep in range(cfg.episodes_per_iter):
            ep_stats = run_episode(env, q_net, replay, epsilon_at(total_env_steps), device, rng)
            total_env_steps += ep_stats["length"]
            win_hist.append(1 if ep_stats["won"] else 0)
            len_hist.append(ep_stats["length"])

        # 2) Train
        iter_losses: list[float] = []
        iter_q_mean: list[float] = []
        iter_q_max: list[float] = []
        if replay.size >= cfg.min_replay:
            for _ in range(cfg.grad_steps_per_iter):
                batch = replay.sample(cfg.batch_size, rng)
                loss = compute_loss(q_net, target_net, batch, cfg.gamma, device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                total_grad_steps += 1
                iter_losses.append(float(loss.item()))
                # Cheap Q diagnostics on last forward.
                with torch.no_grad():
                    obs_t = torch.from_numpy(batch["obs"]).to(device)
                    q_all = q_net(obs_t)
                    iter_q_mean.append(float(q_all.mean().item()))
                    iter_q_max.append(float(q_all.max().item()))
                if total_grad_steps % cfg.target_update_every == 0:
                    target_net.load_state_dict(q_net.state_dict())

        # 3) Snapshot to training pool
        if iter_1based % cfg.snapshot_every == 0:
            pool.add_snapshot(q_net)

        # 4) Update benchmark set (captures snap_first on first call and
        #    milestone snaps at 25/50/75%; rolling latest every iter).
        benchmarks.on_iter_end(iter_1based, q_net)

        # 5) Per-iter metrics + logging
        elapsed = time.time() - start_time
        win_rate = float(np.mean(win_hist)) if win_hist else 0.0
        avg_len = float(np.mean(len_hist)) if len_hist else 0.0
        metrics = {
            "train/iter": iter_1based,
            "train/env_steps": total_env_steps,
            "train/grad_steps": total_grad_steps,
            "train/epsilon": epsilon_at(total_env_steps),
            "train/win_rate_100": win_rate,
            "train/avg_len_100": avg_len,
            "train/pool_size": len(pool),
            "train/elapsed_s": round(elapsed, 1),
        }
        if iter_losses:
            metrics["train/loss"] = float(np.mean(iter_losses))
            metrics["train/q_mean"] = float(np.mean(iter_q_mean))
            metrics["train/q_max"] = float(np.mean(iter_q_max))

        # 6) Periodic benchmark evaluation
        do_eval = (iter_1based % cfg.eval_every == 0) or (iter_1based == cfg.iters)
        if do_eval:
            for name in benchmarks.names():
                opp = benchmarks.policy_for(name)
                if opp is None:
                    continue
                stats = evaluate(
                    q_net, opp, cfg.eval_episodes, device,
                    snake_length=cfg.snake_length, max_steps=cfg.max_steps,
                    seed=int(rng.integers(1 << 31)),
                )
                for k, v in stats.items():
                    metrics[f"eval/{name}/{k}"] = v

        # 7) Periodic video upload
        do_video = (iter_1based % cfg.video_every == 0) or (iter_1based == cfg.iters)
        if do_video and run is not None:
            action_fn = _greedy_fn_factory(q_net, device)
            for name in video_opponents:
                opp = benchmarks.policy_for(name)
                if opp is None:
                    continue
                frames, vinfo = record_episode(
                    action_fn, opp,
                    snake_length=cfg.snake_length,
                    max_steps=cfg.max_steps,
                    seed=int(rng.integers(1 << 31)),
                )
                _log_video(run, frames, key=f"video/vs_{name}", step=iter_1based)
                metrics[f"video/vs_{name}/won"] = 1.0 if vinfo["won"] else 0.0
                metrics[f"video/vs_{name}/length"] = vinfo["length"]

        # 8) Flush row to jsonl + wandb
        log_f.write(json.dumps(metrics) + "\n")
        log_f.flush()
        if run is not None:
            run.log(metrics, step=iter_1based)

        if iter_1based % cfg.print_every == 0:
            print(f"[{iter_1based:4d}/{cfg.iters}] steps={total_env_steps:>7d} "
                  f"eps={metrics['train/epsilon']:.2f} win100={win_rate:.2%} "
                  f"len100={avg_len:.1f} pool={len(pool)} elapsed={elapsed:.0f}s")

        # 9) Checkpoint save + artifact upload
        if iter_1based % cfg.save_every == 0 or iter_1based == cfg.iters:
            ckpt_path = out_dir / f"checkpoint_iter{iter_1based:05d}.pt"
            torch.save({
                "q_net": q_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(cfg),
                "iter": iter_1based,
            }, ckpt_path)
            torch.save(q_net.state_dict(), out_dir / "latest.pt")
            aliases = ["latest"]
            if iter_1based == cfg.iters:
                aliases.append("final")
            _log_checkpoint(run, ckpt_path, iter_1based, aliases)

    log_f.close()
    if run is not None:
        run.finish()
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
    p.add_argument("--pool-size", type=int, default=30)
    p.add_argument("--opponent-random-prob", type=float, default=0.25)
    p.add_argument("--opponent-epsilon", type=float, default=0.05)
    # Evaluation
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--video-every", type=int, default=25)
    p.add_argument("--video-opponents", type=str, default="random,snap_latest")
    # wandb
    p.add_argument("--wandb-project", type=str, default="snake-pong-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", "online"),
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-run-name", type=str, default=None)
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="rl/runs/default")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
