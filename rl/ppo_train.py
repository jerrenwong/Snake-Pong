"""PPO self-play training entrypoint for Snake-Pong.

Follows the "37 implementation details of PPO" checklist for state-vector
envs. Reuses the existing GPU vec-env / obs builder / opponent pool / stable
benchmark eval from the DQN pipeline.

Usage:
    export WANDB_API_KEY=<key>
    python -m rl.ppo_train \
        --total-updates 1800 \
        --n-envs 4096 --rollout-steps 64 \
        --n-epochs 3 --n-minibatches 8 \
        --lr 2.5e-4 --clip-coef 0.2 \
        --benchmark v5:rl/runs/v5/best.pt \
        --benchmark plain:rl/runs/exp_plain/best.pt \
        --out-dir rl/runs/v12

Anything you pass at training time is also baked into the checkpoint `config`
so downstream tournament / eval / export tooling can reconstruct the net.
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
import torch.nn as nn
from torch.distributions import Categorical

from .actions import greedy_action
from .eval import evaluate
from .gym_env import obs_dim
from .ppo_model import ActorCritic, ActorOpponentAdapter, build_actor_critic
from .ppo_rollout import PPORollout
from .ppo_utils import approx_kl_k3, compute_gae, explained_variance, linear_lr
from .render import record_episode
from .selfplay import BenchmarkSet, OpponentPool, parse_benchmark_spec


# ── wandb helpers ─────────────────────────────────────────────────────────────
def _init_wandb(cfg: argparse.Namespace):
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping.")
        return None
    if cfg.wandb_mode == "disabled":
        print("[wandb] disabled via --wandb-mode.")
        return None
    try:
        return wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity,
            name=cfg.wandb_run_name, mode=cfg.wandb_mode,
            config=vars(cfg), dir=cfg.out_dir,
        )
    except Exception as e:
        print(f"[wandb] init failed ({e}); proceeding without wandb.")
        return None


def _log_video(run, frames: np.ndarray, key: str, step: int, fps: int = 10) -> None:
    if run is None:
        return
    try:
        import wandb
        run.log({key: wandb.Video(frames, fps=fps, format="gif")}, step=step)
    except Exception as e:
        print(f"[wandb] video upload failed: {e}")


def _log_checkpoint(run, ckpt_path: Path, update: int, aliases: list[str]) -> None:
    if run is None:
        return
    try:
        import wandb
        artifact = wandb.Artifact(
            name=f"{run.project or 'snake-pong'}-ppo-model",
            type="model", metadata={"update": update},
        )
        artifact.add_file(str(ckpt_path))
        run.log_artifact(artifact, aliases=aliases)
    except Exception as e:
        print(f"[wandb] artifact upload failed: {e}")


# ── Save / config ────────────────────────────────────────────────────────────
def _save_ckpt(
    ac: ActorCritic, opt: torch.optim.Optimizer, cfg: argparse.Namespace,
    update: int, path: Path,
) -> None:
    torch.save({
        "q_net": ac.state_dict(),      # keep key name for tooling compat
        "ac":    ac.state_dict(),      # canonical name for PPO consumers
        "optimizer": opt.state_dict(),
        "iter": update,
        "config": {**vars(cfg), "model_arch": "ppo_ac"},
    }, path)


# ── Greedy eval wrapper so eval.evaluate() can drive an Actor ─────────────────
class _ActorGreedyWrapper(nn.Module):
    """Makes an Actor look like a (B,A) Q-net for evaluate()/greedy_action."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac

    @torch.no_grad()
    def forward(self, obs):
        return self.ac.actor(obs)


# ── Main training loop ───────────────────────────────────────────────────────
def train(cfg: argparse.Namespace) -> None:
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dims
    D = obs_dim(cfg.snake_length)
    batch_size = cfg.n_envs * cfg.rollout_steps
    mb_size = batch_size // cfg.n_minibatches
    assert mb_size * cfg.n_minibatches == batch_size, "batch_size must divide n_minibatches"

    # Network + optimizer
    ac = build_actor_critic(obs_dim=D, hidden=cfg.hidden_size).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    # Optional torch.compile — fuses the Linear/Tanh chain in actor & critic,
    # slashes kernel-launch overhead during rollout (the biggest remaining
    # bottleneck for this tiny MLP). `reduce-overhead` mode uses CUDA graphs
    # internally so per-step launches are amortized.
    if cfg.compile:
        print("[compile] wrapping actor + critic with torch.compile(reduce-overhead)")
        ac.actor = torch.compile(ac.actor, mode="reduce-overhead")
        ac.critic = torch.compile(ac.critic, mode="reduce-overhead")

    # Rollout collector (persistent env state across updates)
    rollout = PPORollout(
        n_envs=cfg.n_envs, rollout_steps=cfg.rollout_steps, obs_dim=D,
        device=device, snake_length=cfg.snake_length,
        snake_multiplier=cfg.snake_multiplier, max_steps=cfg.max_steps,
        interp_ball=cfg.interp_ball_obs,
        shape_coef=cfg.shape_coef, shape_gamma=cfg.gamma,
        bf16=cfg.bf16, rng=rng,
    )

    # Self-play pool + benchmark set (reuse DQN scaffolding)
    pool = OpponentPool(
        device=device, max_snapshots=cfg.pool_size,
        random_prob=cfg.opponent_random_prob,
        opponent_epsilon=cfg.opponent_epsilon, rng=rng,
    )
    external_benchmarks = [parse_benchmark_spec(s) for s in cfg.benchmark]
    scripted_list = (cfg.scripted_benchmarks.split(",")
                     if cfg.scripted_benchmarks else None)
    if scripted_list:
        scripted_list = [s.strip() for s in scripted_list if s.strip()]
    benchmarks = BenchmarkSet(
        device=device, total_iters=cfg.total_updates,
        snake_length=cfg.snake_length, rng=rng,
        opponent_epsilon=0.0, include_random=cfg.include_random_benchmark,
        scripted_names=scripted_list,
        external_checkpoints=external_benchmarks,
    )

    # wandb
    run = _init_wandb(cfg)

    # Rolling episodic stats (for tracking return / terminal breakdown / length).
    # Breakdown sums to ~1 modulo small lag: win + loss + draw + trunc.
    rolling_ret = deque(maxlen=200)
    rolling_len = deque(maxlen=200)
    rolling_win = deque(maxlen=200)
    rolling_loss = deque(maxlen=200)
    rolling_draw = deque(maxlen=200)
    rolling_trunc = deque(maxlen=200)

    # Best checkpoint tracking — by avg win-rate on stable benchmarks only.
    best_eval = -1.0
    total_env_steps = 0
    t_start = time.time()

    greedy_wrapper = _ActorGreedyWrapper(ac)

    def current_lr(u: int) -> float:
        return linear_lr(u, cfg.total_updates, cfg.lr)

    for update in range(cfg.total_updates):
        # Anneal learning rate.
        lr_now = current_lr(update)
        for g in opt.param_groups:
            g["lr"] = lr_now

        # Sample opponent once per update (fixed across all N_envs for this rollout).
        opp_net, opp_eps = pool.sample_snapshot()
        rollout.set_opponent(opp_net, opp_eps, opp_safety_filter=cfg.opp_safety_filter)

        # --- Rollout ---------------------------------------------------------
        t_rollout = time.time()
        completed = rollout.collect(ac)
        t_rollout = time.time() - t_rollout
        total_env_steps += cfg.n_envs * cfg.rollout_steps

        for e in completed:
            rolling_ret.append(e["return"])
            rolling_len.append(e["length"])
            term = e["terminal"]
            won = bool(e["won"])
            is_trunc = term == "truncated"
            is_draw = term == "draw"
            rolling_win.append(1.0 if won else 0.0)
            rolling_trunc.append(1.0 if is_trunc else 0.0)
            rolling_draw.append(1.0 if is_draw else 0.0)
            rolling_loss.append(1.0 if (not won and not is_trunc and not is_draw) else 0.0)

        # --- Bootstrap + GAE -------------------------------------------------
        with torch.no_grad():
            last_value = rollout.bootstrap_value(ac)  # (N,)
        advs, rets = compute_gae(
            rewards=rollout.rewards, values=rollout.values, dones=rollout.dones,
            last_value=last_value, last_done=rollout.next_done,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )

        # Flatten (T, N, ...) → (B, ...) for SGD
        flat = rollout.flat_views()
        b_obs = flat["obs"]
        b_actions = flat["actions"]
        b_logprobs = flat["logprobs"]
        b_values = flat["values"]
        b_advs = advs.reshape(-1)
        b_rets = rets.reshape(-1)

        # --- PPO epochs over minibatches ------------------------------------
        t_update = time.time()
        clip_fracs: list[float] = []
        approx_kls: list[float] = []
        pg_losses: list[float] = []
        v_losses: list[float] = []
        ent_losses: list[float] = []
        ratio_first_mb: float = float("nan")
        stopped_early_at: int = -1

        indices = torch.arange(batch_size, device=device)
        for epoch in range(cfg.n_epochs):
            perm = indices[torch.randperm(batch_size, device=device)]
            for mb_start in range(0, batch_size, mb_size):
                mb_idx = perm[mb_start: mb_start + mb_size]
                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logp = b_logprobs[mb_idx]
                mb_old_value = b_values[mb_idx]
                mb_advs = b_advs[mb_idx]
                mb_rets = b_rets[mb_idx]

                # Per-minibatch advantage normalization.
                if cfg.norm_adv:
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                # Mixed precision: run the forward under autocast(bf16), keep
                # the loss accumulation in fp32 (default). bf16 has the fp32
                # exponent range so no loss scaler is needed (unlike fp16).
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    new_logp, entropy, new_value = ac.evaluate_actions(mb_obs, mb_actions)
                # Cast back to fp32 for loss math — PPO's ratio/clip logic is
                # sensitive to numerics; the small bf16 mantissa is fine for
                # the matmuls but not for the exp(logratio) → clip chain.
                new_logp = new_logp.float()
                entropy = entropy.float()
                new_value = new_value.float()
                logratio = new_logp - mb_old_logp
                ratio = logratio.exp()

                if epoch == 0 and mb_start == 0:
                    ratio_first_mb = float(ratio.mean().item())

                # Clipped surrogate policy loss.
                unclipped = -mb_advs * ratio
                clipped = -mb_advs * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(unclipped, clipped).mean()

                # Clipped value loss (see blog core detail #9).
                if cfg.clip_vloss:
                    v_clipped = mb_old_value + torch.clamp(
                        new_value - mb_old_value, -cfg.clip_coef, cfg.clip_coef,
                    )
                    v_loss_unclipped = (new_value - mb_rets) ** 2
                    v_loss_clipped = (v_clipped - mb_rets) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb_rets) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    approx_kl = float(approx_kl_k3(logratio).item())
                    clipfrac = float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                approx_kls.append(approx_kl)
                clip_fracs.append(clipfrac)
                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                ent_losses.append(float(ent_loss.item()))

            # KL early-stop per epoch (averaged across all minibatches this epoch).
            if cfg.target_kl is not None and approx_kls:
                recent_kl = float(np.mean(approx_kls[-cfg.n_minibatches:]))
                if recent_kl > cfg.target_kl:
                    stopped_early_at = epoch + 1
                    break
        t_update = time.time() - t_update
        ev = explained_variance(b_values, b_rets)

        # --- Logging ---------------------------------------------------------
        elapsed = time.time() - t_start
        log_metrics = {
            "train/update": update + 1,
            "train/env_steps": total_env_steps,
            "train/lr": lr_now,
            "train/pool_size": len(pool),
            "train/rollout_sec": t_rollout,
            "train/update_sec": t_update,
            "train/ratio_first_mb": ratio_first_mb,
            "train/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "train/clipfrac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "train/pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "train/v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "train/entropy": float(np.mean(ent_losses)) if ent_losses else 0.0,
            "train/explained_variance": ev,
            "train/stopped_early_at": stopped_early_at,
            "train/rolling_return_200": (float(np.mean(rolling_ret))
                                          if rolling_ret else 0.0),
            "train/rolling_len_200": (float(np.mean(rolling_len))
                                       if rolling_len else 0.0),
            "train/rolling_win_200": (float(np.mean(rolling_win))
                                       if rolling_win else 0.0),
            "train/rolling_loss_200": (float(np.mean(rolling_loss))
                                        if rolling_loss else 0.0),
            "train/rolling_draw_200": (float(np.mean(rolling_draw))
                                        if rolling_draw else 0.0),
            "train/rolling_trunc_200": (float(np.mean(rolling_trunc))
                                         if rolling_trunc else 0.0),
            "train/episodes_this_update": len(completed),
            "train/elapsed_sec": elapsed,
        }

        # Snapshot opponent pool periodically.
        if (update + 1) % cfg.snapshot_every == 0:
            # Snapshot just the actor (opponent uses it as a Q-net surrogate).
            pool.add_snapshot(_ActorGreedyWrapper(ac))
            benchmarks.on_iter_end(update + 1, _ActorGreedyWrapper(ac))

        # Stable-benchmark eval (evaluate() expects a (B,A)-returning net;
        # the wrapper gives it one).
        if (update + 1) % cfg.eval_every == 0 or update == cfg.total_updates - 1:
            stable_wrs: list[float] = []
            for name in benchmarks.stable_names():
                policy = benchmarks.policy_for(name)
                if policy is None:
                    continue
                res = evaluate(
                    greedy_wrapper, policy, n_episodes=cfg.eval_episodes,
                    device=device, snake_length=cfg.snake_length,
                    snake_multiplier=cfg.snake_multiplier, max_steps=cfg.max_steps,
                    interp_ball=cfg.interp_ball_obs,
                    seed=cfg.seed + update,
                )
                log_metrics[f"eval/{name}/win_rate"] = res["win_rate"]
                log_metrics[f"eval/{name}/avg_len"] = res["avg_len"]
                stable_wrs.append(res["win_rate"])
            if stable_wrs:
                avg_wr = float(np.mean(stable_wrs))
                log_metrics["eval/stable_avg_win_rate"] = avg_wr
                if avg_wr > best_eval:
                    best_eval = avg_wr
                    best_path = out_dir / "best.pt"
                    _save_ckpt(ac, opt, cfg, update + 1, best_path)
                    _log_checkpoint(run, best_path, update + 1, ["best"])
                    print(f"[update {update + 1}] new best eval {avg_wr:.3f}")

        # Video every k updates
        if cfg.video_every and (update + 1) % cfg.video_every == 0:
            def learner_fn(obs):
                return greedy_action(greedy_wrapper, obs, device)
            for opp_name in (cfg.video_opponents or "").split(","):
                opp_name = opp_name.strip()
                if not opp_name:
                    continue
                policy = benchmarks.policy_for(opp_name)
                if policy is None:
                    continue
                frames, _vinfo = record_episode(
                    learner_fn, policy,
                    snake_length=cfg.snake_length,
                    snake_multiplier=cfg.snake_multiplier,
                    max_steps=cfg.max_steps,
                    interp_ball=cfg.interp_ball_obs,
                    seed=cfg.seed + update,
                )
                _log_video(run, frames, f"video/vs_{opp_name}", update + 1)

        # Checkpoint + wandb log
        if (update + 1) % cfg.save_every == 0 or update == cfg.total_updates - 1:
            latest = out_dir / "latest.pt"
            _save_ckpt(ac, opt, cfg, update + 1, latest)
            aliases = ["latest"]
            if update == cfg.total_updates - 1:
                aliases.append("final")
            _log_checkpoint(run, latest, update + 1, aliases)

        if run is not None:
            try:
                run.log(log_metrics, step=update + 1)
            except Exception as e:
                print(f"[wandb] log failed: {e}")

        # Console progress line.
        if (update + 1) % cfg.print_every == 0 or update == cfg.total_updates - 1:
            print(
                f"[{update + 1}/{cfg.total_updates}] "
                f"steps={total_env_steps} lr={lr_now:.2e} "
                f"kl={log_metrics['train/approx_kl']:.4f} "
                f"clipf={log_metrics['train/clipfrac']:.3f} "
                f"pg={log_metrics['train/pg_loss']:+.4f} "
                f"v={log_metrics['train/v_loss']:.4f} "
                f"ent={log_metrics['train/entropy']:.3f} "
                f"evar={ev:.3f} "
                f"w/l/d/t={log_metrics['train/rolling_win_200']:.1%}/"
                f"{log_metrics['train/rolling_loss_200']:.1%}/"
                f"{log_metrics['train/rolling_draw_200']:.1%}/"
                f"{log_metrics['train/rolling_trunc_200']:.1%} "
                f"len200={log_metrics['train/rolling_len_200']:.1f} "
                f"pool={len(pool)} "
                f"rollout={t_rollout:.2f}s update={t_update:.2f}s "
                f"elapsed={elapsed:.0f}s"
            )

    # Done.
    final = out_dir / "latest.pt"
    _save_ckpt(ac, opt, cfg, cfg.total_updates, final)
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
    print(f"Done. Checkpoints in {out_dir}  (best.pt avg eval win rate {best_eval:.3f})")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Training cadence
    p.add_argument("--total-updates", type=int, default=1800)
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-minibatches", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=800)

    # PPO hyperparams
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--adam-eps", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--shape-coef", type=float, default=0.0,
                   help="Potential-based distance shaping α. 0 disables. 0.1 "
                        "is a reasonable first try — gives each ball step ~3%% "
                        "of the terminal reward magnitude.")
    p.add_argument("--opp-safety-filter", action="store_true", default=False,
                   help="Mask opponent illegal actions during rollouts. Forces "
                        "the learner to beat opponents at play rather than "
                        "exploiting crash behavior — matches the in-browser "
                        "filter applied to all AIs.")
    p.add_argument("--compile", action="store_true", default=False,
                   help="Wrap actor and critic with torch.compile(reduce-overhead). "
                        "Fuses Linear/Tanh kernels; usually ~1.5-2× rollout "
                        "throughput for small MLPs.")
    p.add_argument("--bf16", action="store_true", default=False,
                   help="Enable bf16 autocast for actor/critic forwards. "
                        "On Ampere+ (3090/4090/A100/H100) bf16 matmul is "
                        "~4× faster than fp32 — big update-phase win for "
                        "this tiny MLP. Losses stay in fp32 for PPO stability.")
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")

    # Model
    p.add_argument("--hidden-size", type=int, default=64)

    # Env
    p.add_argument("--snake-length", type=int, default=4)
    p.add_argument("--snake-multiplier", type=int, default=2)
    p.add_argument("--interp-ball-obs", action="store_true", default=True)
    p.add_argument("--no-interp-ball-obs", dest="interp_ball_obs", action="store_false")

    # Self-play pool
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--opponent-random-prob", type=float, default=0.1)
    p.add_argument("--opponent-epsilon", type=float, default=0.0)
    p.add_argument("--snapshot-every", type=int, default=20)

    # Benchmarks / eval
    p.add_argument("--benchmark", action="append", default=[],
                   help="Stable benchmark spec, 'name:path[:head=N]'. Pass multiple times.")
    p.add_argument("--scripted-benchmarks", type=str, default="",
                   help="Comma-separated scripted benchmark names.")
    p.add_argument("--include-random-benchmark", action="store_true", default=False)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=40)

    # Video / checkpoint
    p.add_argument("--video-every", type=int, default=200)
    p.add_argument("--video-opponents", type=str, default="random,snap_latest")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--print-every", type=int, default=5)

    # Infra
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="rl/runs/v12")

    # wandb
    p.add_argument("--wandb-project", default="snake-pong-rl")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=("online", "offline", "disabled"))
    p.add_argument("--wandb-run-name", default=None)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
