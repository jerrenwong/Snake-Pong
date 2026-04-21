"""Self-play DQN training entrypoint for Snake-Pong.

Vectorized rollouts: N parallel envs with batched Q-net forwards.

Usage:
    export WANDB_API_KEY=<your_key>
    python -m rl.train --iters 5000 --n-envs 32 --out-dir rl/runs/v2

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

from .dqn import build_q_net, ReplayBuffer, compute_loss, greedy_action
from .gym_env import obs_dim
from .eval import evaluate
from .render import record_episode
from .selfplay import BenchmarkSet, OpponentPool
from .vec_rollout import VecRollout


def _greedy_fn_factory(q_net: QNetwork, device: torch.device):
    def fn(obs):
        return greedy_action(q_net, obs, device)
    return fn


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
    q_net = build_q_net(cfg.model_arch, d_obs).to(device)
    target_net = build_q_net(cfg.model_arch, d_obs).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    # Cosine LR decay to `lr * lr_decay_min_ratio` over the full run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.iters, eta_min=cfg.lr * cfg.lr_decay_min_ratio,
    )

    pool = OpponentPool(
        device, max_snapshots=cfg.pool_size,
        random_prob=cfg.opponent_random_prob,
        opponent_epsilon=cfg.opponent_epsilon, rng=rng,
    )
    benchmarks = BenchmarkSet(device, total_iters=cfg.iters, rng=rng)

    vec = VecRollout(
        n_envs=cfg.n_envs, q_net=q_net, device=device,
        snake_length=cfg.snake_length, snake_multiplier=cfg.snake_multiplier,
        max_steps=cfg.max_steps, rng=rng,
    )

    replay = ReplayBuffer(cfg.replay_capacity, d_obs)

    win_hist = deque(maxlen=100)
    len_hist = deque(maxlen=100)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")

    total_env_steps = 0
    total_grad_steps = 0
    best_eval_avg = -1.0
    best_eval_iter = -1

    # Denser eval in the last `late_eval_frac` of training for reliable peak-picking.
    late_eval_start_iter = int(cfg.iters * (1.0 - cfg.late_eval_frac))

    def current_eval_every(it1: int) -> int:
        return max(1, cfg.eval_every // 2) if it1 > late_eval_start_iter else cfg.eval_every

    def epsilon_at(step: int) -> float:
        frac = min(1.0, step / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    video_opponents = [s.strip() for s in cfg.video_opponents.split(",") if s.strip()]

    start_time = time.time()

    for it in range(cfg.iters):
        iter_1based = it + 1

        # 1) Rollout — vectorized. Set a fresh opponent (shared across all N envs
        #    during this iter; re-sampled each iter for diversity).
        opp_q, opp_eps = pool.sample_snapshot()
        vec.set_opponent(opp_q, opp_eps)
        ep_stats = vec.collect(
            replay,
            n_transitions=cfg.rollout_steps_per_iter,
            epsilon=epsilon_at(total_env_steps),
        )
        total_env_steps += cfg.rollout_steps_per_iter
        for s in ep_stats:
            win_hist.append(1 if s["won"] else 0)
            len_hist.append(s["length"])

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
                with torch.no_grad():
                    obs_t = torch.from_numpy(batch["obs"]).to(device)
                    q_all = q_net(obs_t)
                    iter_q_mean.append(float(q_all.mean().item()))
                    iter_q_max.append(float(q_all.max().item()))
                if total_grad_steps % cfg.target_update_every == 0:
                    target_net.load_state_dict(q_net.state_dict())
            scheduler.step()

        # 3) Snapshot to training pool
        if iter_1based % cfg.snapshot_every == 0:
            pool.add_snapshot(q_net)

        # 4) Update benchmark set
        benchmarks.on_iter_end(iter_1based, q_net)

        # 5) Per-iter metrics
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

        # 6) Periodic benchmark evaluation (denser in final stretch)
        eval_every_now = current_eval_every(iter_1based)
        do_eval = (iter_1based % eval_every_now == 0) or (iter_1based == cfg.iters)
        metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        if do_eval:
            eval_win_rates: list[float] = []
            for name in benchmarks.names():
                opp = benchmarks.policy_for(name)
                if opp is None:
                    continue
                stats = evaluate(
                    q_net, opp, cfg.eval_episodes, device,
                    snake_length=cfg.snake_length, snake_multiplier=cfg.snake_multiplier,
                    max_steps=cfg.max_steps, seed=int(rng.integers(1 << 31)),
                )
                for k, v in stats.items():
                    metrics[f"eval/{name}/{k}"] = v
                eval_win_rates.append(stats["win_rate"])
            if eval_win_rates:
                eval_avg = float(np.mean(eval_win_rates))
                metrics["eval/avg_win_rate"] = eval_avg
                # Best-by-eval checkpoint — protects against late-stage drift.
                if eval_avg > best_eval_avg:
                    best_eval_avg = eval_avg
                    best_eval_iter = iter_1based
                    torch.save({
                        "q_net": q_net.state_dict(),
                        "config": vars(cfg),
                        "iter": iter_1based,
                        "eval_avg": eval_avg,
                    }, out_dir / "best.pt")
                metrics["eval/best_avg_so_far"] = best_eval_avg
                metrics["eval/best_iter_so_far"] = best_eval_iter

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
                    snake_length=cfg.snake_length, snake_multiplier=cfg.snake_multiplier,
                    max_steps=cfg.max_steps, seed=int(rng.integers(1 << 31)),
                )
                _log_video(run, frames, key=f"video/vs_{name}", step=iter_1based)
                metrics[f"video/vs_{name}/won"] = 1.0 if vinfo["won"] else 0.0
                metrics[f"video/vs_{name}/length"] = vinfo["length"]

        # 8) Flush
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
            torch.save({
                "q_net": q_net.state_dict(),
                "config": vars(cfg),
                "iter": iter_1based,
            }, out_dir / "latest.pt")
            aliases = ["latest"]
            if iter_1based == cfg.iters:
                aliases.append("final")
            _log_checkpoint(run, ckpt_path, iter_1based, aliases)

    log_f.close()
    if run is not None:
        # Log the best checkpoint as an artifact with alias "best" so it's
        # easy to pull from wandb after training.
        best_path = out_dir / "best.pt"
        if best_path.exists():
            _log_checkpoint(run, best_path, best_eval_iter, ["best"])
        run.finish()
    print(f"Done. Checkpoints in {out_dir}  (best.pt from iter {best_eval_iter} "
          f"with avg eval win rate {best_eval_avg:.1%})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Outer loop
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=32,
                   help="Parallel envs for vectorized rollouts.")
    p.add_argument("--rollout-steps-per-iter", type=int, default=600,
                   help="Transitions to collect per iter (across all envs).")
    p.add_argument("--grad-steps-per-iter", type=int, default=100)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--target-update-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--print-every", type=int, default=1)
    # DQN
    p.add_argument("--model-arch", type=str, default="dueling", choices=["mlp", "dueling"],
                   help="Q-network architecture. 'dueling' = Dueling DQN.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-decay-min-ratio", type=float, default=0.1,
                   help="CosineAnnealingLR decays LR to lr * this ratio over training.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--min-replay", type=int, default=5_000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    # Env
    p.add_argument("--snake-length", type=int, default=4)
    p.add_argument("--snake-multiplier", type=int, default=1,
                   help="Snake ticks per ball tick; matches JS game's snakeMultiplier.")
    p.add_argument("--max-steps", type=int, default=500)
    # Self-play
    p.add_argument("--pool-size", type=int, default=30)
    p.add_argument("--opponent-random-prob", type=float, default=0.25)
    p.add_argument("--opponent-epsilon", type=float, default=0.05)
    # Evaluation
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--late-eval-frac", type=float, default=0.33,
                   help="Final fraction of training where --eval-every is halved for denser eval.")
    p.add_argument("--video-every", type=int, default=2000)
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
