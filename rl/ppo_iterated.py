"""Iterated experts: PPO trained against a FIXED set of strong opponents,
auto-restarted each time the agent beats every opponent at a given threshold.

Produces a collection of independently-trained expert snakes in
`rl/runs/v13/run_{0,1,2,...}/`. Runs until wall-clock budget runs out.

Design (per user request):
  1. Fresh random-init ActorCritic per "snake".
  2. Opponent set loaded from --opponents paths (strongest from the big
     tournament). Each training update samples ONE opponent uniformly from the
     set — diverse curriculum across the run.
  3. Potential-based distance-to-goal shaping (Ng et al. 1999) for denser
     signal; same form as in ppo_train.py.
  4. Periodic per-opponent eval: if min-opponent win rate ≥ threshold, save
     this snake as "converged" and start a fresh one.
  5. If max_updates_per_snake is hit first, save anyway and move on — don't
     get stuck on a slow learner.

Usage:
    python -m rl.ppo_iterated \
        --opponents v12@1400:rl/runs/v12/snap_u1400.pt \
        --opponents v12@4500:rl/runs/v12/snap_u4500.pt \
        --opponents v11:rl/runs/v11/latest.pt \
        --opponents v9_h4:rl/runs/v9/play_h4_final.pt:head=4 \
        --hours-budget 5.0 --shape-coef 0.1 \
        --n-envs 65536 --rollout-steps 128 \
        --out-dir rl/runs/v13
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

from .actions import greedy_action
from .big_tournament import _load_generic
from .eval import evaluate
from .gym_env import obs_dim
from .ppo_model import ActorCritic, build_actor_critic
from .ppo_rollout import PPORollout
from .ppo_utils import approx_kl_k3, compute_gae, explained_variance, linear_lr
from .selfplay import make_policy


# ── Wandb (optional) ──────────────────────────────────────────────────────────
def _init_wandb(cfg: argparse.Namespace):
    try:
        import wandb
    except ImportError:
        return None
    if cfg.wandb_mode == "disabled":
        return None
    try:
        return wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity,
            name=cfg.wandb_run_name, mode=cfg.wandb_mode,
            config=vars(cfg), dir=cfg.out_dir,
        )
    except Exception as e:
        print(f"[wandb] init failed ({e}); proceeding without.")
        return None


# ── Opponent spec parsing ────────────────────────────────────────────────────
def _parse_opp(spec: str) -> tuple[str, str, int | None]:
    """Parse 'name:path[:head=N]' → (name, path, head_or_None)."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"bad opponent spec {spec!r}")
    name, path = parts[0], parts[1]
    head = None
    for extra in parts[2:]:
        if extra.startswith("head="):
            head = int(extra.split("=", 1)[1])
    return name, path, head


# ── Per-snake training loop ──────────────────────────────────────────────────
class _ActorQWrapper(nn.Module):
    """Wrap Actor so evaluate()/greedy_action() treats it as a Q-net."""
    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac
    @torch.no_grad()
    def forward(self, obs):
        return self.ac.actor(obs)


def _save_snake(
    ac: ActorCritic, opt: torch.optim.Optimizer, cfg: argparse.Namespace,
    update: int, path: Path, converged: bool, eval_snapshot: dict,
) -> None:
    torch.save({
        "q_net": ac.state_dict(),
        "ac":    ac.state_dict(),
        "optimizer": opt.state_dict(),
        "iter": update,
        "config": {**vars(cfg), "model_arch": "ppo_ac"},
        "converged": converged,
        "eval_snapshot": eval_snapshot,
    }, path)


def train_one_snake(
    run_idx: int, cfg: argparse.Namespace, device: torch.device,
    opponents: list[tuple[str, nn.Module]],  # (name, module)
    seed: int, run_dir: Path, wandb_run,
) -> dict:
    """Train a single snake until convergence or max_updates_per_snake.

    Returns a summary dict {run_idx, updates, env_steps, converged,
    per_opp_win_rates, wall_sec, time_budget_used_sec}.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    D = obs_dim(cfg.snake_length)
    batch_size = cfg.n_envs * cfg.rollout_steps
    mb_size = batch_size // cfg.n_minibatches
    assert mb_size * cfg.n_minibatches == batch_size

    ac = build_actor_critic(obs_dim=D, hidden=cfg.hidden_size).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    rollout = PPORollout(
        n_envs=cfg.n_envs, rollout_steps=cfg.rollout_steps, obs_dim=D,
        device=device, snake_length=cfg.snake_length,
        snake_multiplier=cfg.snake_multiplier, max_steps=cfg.max_steps,
        interp_ball=cfg.interp_ball_obs,
        shape_coef=cfg.shape_coef, shape_gamma=cfg.gamma, rng=rng,
    )

    rolling_win = deque(maxlen=200)
    rolling_len = deque(maxlen=200)

    t_snake_start = time.time()
    t_global_start = wandb_run["global_t0"] if isinstance(wandb_run, dict) else time.time()

    greedy_q = _ActorQWrapper(ac)
    last_eval: dict[str, float] = {name: 0.0 for name, _ in opponents}
    converged = False
    best_min_eval = 0.0

    for update in range(cfg.max_updates_per_snake):
        # Linear anneal (full budget for this snake)
        lr_now = linear_lr(update, cfg.max_updates_per_snake, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now

        # Sample one opponent for this update (fixed across all envs).
        opp_idx = int(rng.integers(0, len(opponents)))
        opp_name, opp_net = opponents[opp_idx]
        rollout.set_opponent(
            opp_net, cfg.opponent_epsilon,
            opp_safety_filter=cfg.opp_safety_filter,
        )

        completed = rollout.collect(ac)
        for e in completed:
            rolling_win.append(1.0 if e["won"] else 0.0)
            rolling_len.append(e["length"])

        with torch.no_grad():
            last_value = rollout.bootstrap_value(ac)
        advs, rets = compute_gae(
            rollout.rewards, rollout.values, rollout.dones,
            last_value, rollout.next_done, cfg.gamma, cfg.gae_lambda,
        )

        flat = rollout.flat_views()
        b_obs, b_actions = flat["obs"], flat["actions"]
        b_logp, b_values = flat["logprobs"], flat["values"]
        b_advs, b_rets = advs.reshape(-1), rets.reshape(-1)

        pg_losses, v_losses, ent_losses, kls, clips = [], [], [], [], []
        stopped_early = -1
        indices = torch.arange(batch_size, device=device)
        for epoch in range(cfg.n_epochs):
            perm = indices[torch.randperm(batch_size, device=device)]
            for s in range(0, batch_size, mb_size):
                idx = perm[s: s + mb_size]
                mb_obs = b_obs[idx]; mb_act = b_actions[idx]
                mb_old_logp = b_logp[idx]; mb_old_v = b_values[idx]
                mb_adv = b_advs[idx]; mb_ret = b_rets[idx]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_logp, ent, new_v = ac.evaluate_actions(mb_obs, mb_act)
                logratio = new_logp - mb_old_logp
                ratio = logratio.exp()
                unclipped = -mb_adv * ratio
                clipped = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg = torch.max(unclipped, clipped).mean()
                if cfg.clip_vloss:
                    v_clip = mb_old_v + torch.clamp(new_v - mb_old_v, -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max((new_v - mb_ret) ** 2, (v_clip - mb_ret) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - mb_ret) ** 2).mean()
                ent_loss = ent.mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    kls.append(float(approx_kl_k3(logratio).item()))
                    clips.append(float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()))
                pg_losses.append(float(pg.item()))
                v_losses.append(float(v_loss.item()))
                ent_losses.append(float(ent_loss.item()))

            if cfg.target_kl is not None and kls and float(np.mean(kls[-cfg.n_minibatches:])) > cfg.target_kl:
                stopped_early = epoch + 1
                break

        ev = explained_variance(b_values, b_rets)
        t_elapsed = time.time() - t_snake_start
        t_budget_used = time.time() - t_global_start
        log = {
            "snake": run_idx,
            "snake_update": update + 1,
            "global_update": run_idx * cfg.max_updates_per_snake + update + 1,
            "budget_used_sec": t_budget_used,
            "snake_elapsed_sec": t_elapsed,
            "lr": lr_now,
            "pool/opp_sampled": opp_name,
            "pool/opp_idx": opp_idx,
            "train/approx_kl": float(np.mean(kls)) if kls else 0.0,
            "train/clipfrac": float(np.mean(clips)) if clips else 0.0,
            "train/pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "train/v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "train/entropy": float(np.mean(ent_losses)) if ent_losses else 0.0,
            "train/explained_variance": ev,
            "train/stopped_early": stopped_early,
            "train/rolling_win_200": float(np.mean(rolling_win)) if rolling_win else 0.0,
            "train/rolling_len_200": float(np.mean(rolling_len)) if rolling_len else 0.0,
        }

        # Convergence check — per-opponent eval.
        if (update + 1) % cfg.convergence_check_every == 0:
            wins = {}
            for name, opp_mod in opponents:
                policy = make_policy(opp_mod, device, epsilon=0.0,
                                     rng=np.random.default_rng(seed + update))
                res = evaluate(
                    greedy_q, policy, n_episodes=cfg.convergence_episodes,
                    device=device, snake_length=cfg.snake_length,
                    snake_multiplier=cfg.snake_multiplier,
                    max_steps=cfg.max_steps, interp_ball=cfg.interp_ball_obs,
                    seed=seed + update,
                )
                wins[name] = res["win_rate"]
                log[f"eval/{name}/win_rate"] = res["win_rate"]
                log[f"eval/{name}/avg_len"] = res["avg_len"]
                log[f"eval/{name}/draw_rate"] = res["draw_rate"]
            min_win = min(wins.values()) if wins else 0.0
            avg_win = float(np.mean(list(wins.values()))) if wins else 0.0
            log["eval/min_win"] = min_win
            log["eval/avg_win"] = avg_win
            last_eval = wins
            if min_win > best_min_eval:
                best_min_eval = min_win
                _save_snake(ac, opt, cfg, update + 1, run_dir / "best.pt",
                            converged=False, eval_snapshot=wins)
            if min_win >= cfg.convergence_threshold:
                converged = True

        if cfg.wandb_mode != "disabled" and wandb_run is not None and not isinstance(wandb_run, dict):
            try:
                wandb_run.log(log, step=log["global_update"])
            except Exception:
                pass

        if (update + 1) % cfg.print_every == 0 or converged:
            win_summary = " ".join(f"{n}={w*100:.0f}%" for n, w in last_eval.items())
            print(f"  [run{run_idx}/u{update+1}] opp={opp_name} "
                  f"win200={log['train/rolling_win_200']*100:.0f}% "
                  f"ent={log['train/entropy']:.3f} "
                  f"kl={log['train/approx_kl']:.4f} "
                  f"ev={log['train/explained_variance']:.2f} "
                  f"[{win_summary}]  "
                  f"budget={t_budget_used/3600:.2f}h")

        # Global budget check: stop inside the snake if we're out of time.
        if t_budget_used > cfg.hours_budget * 3600:
            break
        if converged:
            break

    # Final save for this snake.
    _save_snake(ac, opt, cfg, update + 1, run_dir / "latest.pt",
                converged=converged, eval_snapshot=last_eval)
    summary = {
        "run_idx": run_idx,
        "updates": update + 1,
        "env_steps": (update + 1) * cfg.n_envs * cfg.rollout_steps,
        "converged": converged,
        "per_opp_win_rates": last_eval,
        "best_min_eval": best_min_eval,
        "wall_sec": time.time() - t_snake_start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ── Outer loop ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()

    # Opponent set (required)
    p.add_argument("--opponents", action="append", required=True,
                   help="Opponent spec 'name:path[:head=N]'. Repeat for more.")
    p.add_argument("--opponent-epsilon", type=float, default=0.0)
    p.add_argument("--opp-safety-filter", action="store_true", default=False,
                   help="Mask opponent illegal actions during rollouts. Forces "
                        "the learner to beat the opponent without relying on "
                        "crash-exploitation — matches browser play conditions.")

    # Budget & convergence
    p.add_argument("--hours-budget", type=float, default=5.0)
    p.add_argument("--max-updates-per-snake", type=int, default=2000)
    p.add_argument("--convergence-threshold", type=float, default=0.95,
                   help="Snake is considered converged when min win-rate across "
                        "all opponents ≥ this. 1.0 = perfect.")
    p.add_argument("--convergence-check-every", type=int, default=50)
    p.add_argument("--convergence-episodes", type=int, default=60)

    # PPO + env — same defaults as ppo_train
    p.add_argument("--n-envs", type=int, default=65536)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-minibatches", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--adam-eps", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--shape-coef", type=float, default=0.1)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--snake-length", type=int, default=4)
    p.add_argument("--snake-multiplier", type=int, default=2)
    p.add_argument("--interp-ball-obs", action="store_true", default=True)
    p.add_argument("--no-interp-ball-obs", dest="interp_ball_obs", action="store_false")
    p.add_argument("--print-every", type=int, default=10)

    # Infra
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="rl/runs/v13")

    # wandb
    p.add_argument("--wandb-project", default="snake-pong-rl")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=("online", "offline", "disabled"))
    p.add_argument("--wandb-run-name", default=None)

    cfg = p.parse_args()

    device = torch.device(cfg.device)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load opponent set once — reused across all snakes.
    print(f"Loading {len(cfg.opponents)} opponent(s)...")
    opponents: list[tuple[str, nn.Module]] = []
    for spec in cfg.opponents:
        name, path, head = _parse_opp(spec)
        m = _load_generic(path, device, head=head)
        opponents.append((name, m))
        print(f"  loaded {name:>10s} ← {path}")

    # wandb (one run spans all snakes).
    run = _init_wandb(cfg)
    # Package a "global_t0" into wandb_run placeholder for train_one_snake.
    global_t0 = time.time()
    pack = run if run is not None else {"global_t0": global_t0}
    if run is not None:
        # attach budget t0 as a run attr
        try:
            run.config.update({"global_t0": global_t0}, allow_val_change=True)
        except Exception:
            pass

    # Outer loop — one snake at a time.
    summaries = []
    run_idx = 0
    budget_sec = cfg.hours_budget * 3600
    while (time.time() - global_t0) < budget_sec:
        run_dir = out_root / f"run_{run_idx}"
        print(f"\n=== SNAKE {run_idx} (run_dir={run_dir}) ===  "
              f"budget used {(time.time()-global_t0)/3600:.2f}h / {cfg.hours_budget}h")
        # Pass global_t0 via a shared dict; actual wandb run still usable for logging.
        wandb_ref = run if run is not None else pack
        if run is not None:
            # Ensure train_one_snake sees the t0. Monkey-patch via cfg attr.
            pass
        # Patch in-memory: set global_t0 as a cfg attribute so train_one_snake can read it.
        cfg._global_t0 = global_t0  # type: ignore[attr-defined]
        summary = train_one_snake(
            run_idx=run_idx, cfg=cfg, device=device, opponents=opponents,
            seed=cfg.seed + 1000 * run_idx, run_dir=run_dir, wandb_run=wandb_ref,
        )
        summaries.append(summary)
        print(f"  snake {run_idx} → converged={summary['converged']}  "
              f"updates={summary['updates']}  "
              f"min_win={summary['best_min_eval']:.2f}  "
              f"wall={summary['wall_sec']:.0f}s")
        # Persist the running summary
        (out_root / "summaries.json").write_text(json.dumps(summaries, indent=2))
        run_idx += 1

    print(f"\nDone. Trained {run_idx} snakes in {(time.time()-global_t0)/3600:.2f}h.")
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
