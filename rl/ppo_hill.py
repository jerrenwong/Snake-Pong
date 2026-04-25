"""King-of-the-hill PPO training.

Monotone-only self-play: a new snake is trained against the current pool. It
is admitted to the pool ONLY if it wins 100% of `--games-per-check` games
against EVERY pool member simultaneously. Otherwise the snake is discarded
(saved but not pooled). The pool therefore only ever grows with strictly
dominant agents, yielding a provably-monotone skill ladder.

Key differences vs `ppo_iterated`:
  - Initial pool is loaded from external checkpoints (all known eras: v5, v8,
    v9_h4, v11_*, v12_*, v13_*, v14_*).
  - Promotion criterion is strict (1.0 win-rate vs every pool member) by
    default; `--no-losses` relaxes to "0 losses, draws allowed".
  - Eval is GPU-batched via `vec_tournament.run_tournament` (safety filter
    on both sides) — avoids the CPU-bound evaluate() path.
  - Pool grows in-process — each promotion appends a frozen snapshot of the
    just-converged ActorCritic.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .big_tournament import _load_generic
from .gym_env import obs_dim
from .ppo_model import ActorCritic, build_actor_critic
from .ppo_rollout import PPORollout
from .ppo_utils import approx_kl_k3, compute_gae, explained_variance, linear_lr
from .vec_tournament import run_tournament


class _ActorQWrapper(nn.Module):
    """Plug an ActorCritic into modules-expecting-Q-net tooling."""
    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac
    @torch.no_grad()
    def forward(self, obs):
        return self.ac.actor(obs)


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


def _parse_spec(spec: str) -> tuple[str, str, int | None]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"bad spec {spec!r}")
    name, path = parts[0], parts[1]
    head = None
    for extra in parts[2:]:
        if extra.startswith("head="):
            head = int(extra.split("=", 1)[1])
    return name, path, head


def _save_snake(
    ac: ActorCritic, opt: torch.optim.Optimizer, cfg: argparse.Namespace,
    update: int, path: Path, promoted: bool, eval_snapshot: dict,
) -> None:
    torch.save({
        "q_net": ac.state_dict(),
        "ac":    ac.state_dict(),
        "optimizer": opt.state_dict(),
        "iter": update,
        "config": {**vars(cfg), "model_arch": "ppo_ac"},
        "promoted": promoted,
        "eval_snapshot": eval_snapshot,
    }, path)


def _eval_vs_pool(
    ac: ActorCritic,
    pool: dict[str, nn.Module],
    games_per_check: int,
    cfg: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Return {pool_name: win_rate_of_ac_vs_pool_name} using GPU tournament."""
    learner = _ActorQWrapper(ac).to(device).eval()
    models = {"__learner__": learner, **pool}
    res = run_tournament(
        models, games_per_pair=games_per_check, device=device, seed=seed,
        snake_length=cfg.snake_length, snake_multiplier=cfg.snake_multiplier,
        max_steps=cfg.max_steps, interp_ball=cfg.interp_ball_obs,
        safety_filter=True, show_progress=False,
    )
    names = res["names"]
    wr = res["win_rate"]
    losses_mat = res["losses"]
    i0 = names.index("__learner__")
    out: dict[str, dict[str, float]] = {}
    for j, n in enumerate(names):
        if j == i0:
            continue
        out[n] = {
            "win_rate":  float(wr[i0, j]),
            "loss_rate": float(losses_mat[i0, j] / max(1, games_per_check)),
        }
    return out


def train_one_snake(
    snake_idx: int, cfg: argparse.Namespace, device: torch.device,
    pool: dict[str, nn.Module], seed: int, run_dir: Path, wandb_run,
    global_t0: float,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    D = obs_dim(cfg.snake_length)
    batch_size = cfg.n_envs * cfg.rollout_steps
    mb_size = batch_size // cfg.n_minibatches

    ac = build_actor_critic(obs_dim=D, hidden=cfg.hidden_size).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    rollout = PPORollout(
        n_envs=cfg.n_envs, rollout_steps=cfg.rollout_steps, obs_dim=D,
        device=device, snake_length=cfg.snake_length,
        snake_multiplier=cfg.snake_multiplier, max_steps=cfg.max_steps,
        interp_ball=cfg.interp_ball_obs,
        shape_coef=0.0, shape_gamma=cfg.gamma,
        bf16=cfg.bf16, learner_safety_filter=True, rng=rng,
    )

    pool_names = list(pool.keys())
    t_snake_start = time.time()
    promoted = False
    last_eval: dict[str, dict[str, float]] = {}
    best_min_win = 0.0

    # Rolling per-opponent win / loss trackers — fed by training rollouts.
    # maxlen per opp chosen so ~each opp gets ~300 recent episodes at steady
    # state (pool_size * 200 eps/update gives plenty).
    ROLL = max(300, 10 * cfg.check_every // len(pool_names))
    roll_wins = {n: deque(maxlen=ROLL) for n in pool_names}
    roll_losses = {n: deque(maxlen=ROLL) for n in pool_names}
    # Opponents appear in pool dynamically once we promote; helper to extend.
    def _ensure_roll(name: str) -> None:
        if name not in roll_wins:
            roll_wins[name] = deque(maxlen=ROLL)
            roll_losses[name] = deque(maxlen=ROLL)

    for update in range(cfg.max_updates_per_snake):
        lr_now = linear_lr(update, cfg.max_updates_per_snake, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now

        # Uniform-random opponent this update.
        pool_names = list(pool.keys())
        opp_name = pool_names[int(rng.integers(0, len(pool_names)))]
        opp_net = pool[opp_name]
        rollout.set_opponent(opp_net, cfg.opponent_epsilon, opp_safety_filter=True)

        completed = rollout.collect(ac)
        # Tag completed episodes with the active opponent and fold into rolling trackers.
        _ensure_roll(opp_name)
        for e in completed:
            roll_wins[opp_name].append(1.0 if e["won"] else 0.0)
            roll_losses[opp_name].append(
                1.0 if (not e["won"] and e["terminal"] != "truncated" and e["scorer"] != 3) else 0.0
            )

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
        b_legal = flat.get("legal")  # (T*N, 4) bool or None

        # PPO epochs
        pg_losses, v_losses, ent_losses, kls, clips = [], [], [], [], []
        indices = torch.arange(batch_size, device=device)
        for epoch in range(cfg.n_epochs):
            perm = indices[torch.randperm(batch_size, device=device)]
            for s in range(0, batch_size, mb_size):
                idx = perm[s: s + mb_size]
                mb_obs = b_obs[idx]; mb_act = b_actions[idx]
                mb_old_logp = b_logp[idx]; mb_old_v = b_values[idx]
                mb_adv = b_advs[idx]; mb_ret = b_rets[idx]
                mb_legal = b_legal[idx] if b_legal is not None else None
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    new_logp, ent, new_v = ac.evaluate_actions(mb_obs, mb_act, legal=mb_legal)
                new_logp = new_logp.float(); ent = ent.float(); new_v = new_v.float()
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
            if cfg.target_kl is not None and float(np.mean(kls[-cfg.n_minibatches:])) > cfg.target_kl:
                break

        ev = explained_variance(b_values, b_rets)

        log_metrics = {
            "snake_idx": snake_idx,
            "snake_update": update + 1,
            "pool_size": len(pool),
            "budget_used_h": (time.time() - global_t0) / 3600,
            "snake_elapsed_h": (time.time() - t_snake_start) / 3600,
            "lr": lr_now,
            "opp_sampled": opp_name,
            "train/approx_kl": float(np.mean(kls)),
            "train/clipfrac": float(np.mean(clips)),
            "train/pg_loss": float(np.mean(pg_losses)),
            "train/v_loss": float(np.mean(v_losses)),
            "train/entropy": float(np.mean(ent_losses)),
            "train/explained_variance": ev,
        }

        # Log rolling per-opponent stats on every update (cheap — just deque means).
        roll_win_rates: dict[str, float] = {}
        roll_loss_rates: dict[str, float] = {}
        for n in pool_names:
            w = roll_wins.get(n) or deque()
            l = roll_losses.get(n) or deque()
            roll_win_rates[n] = float(np.mean(w)) if w else 0.0
            roll_loss_rates[n] = float(np.mean(l)) if l else 0.0
            log_metrics[f"roll/{n}/win"] = roll_win_rates[n]
            log_metrics[f"roll/{n}/loss"] = roll_loss_rates[n]

        # Promotion check uses rolling training stats as a CHEAP trigger, then
        # a quick greedy tournament as the final confirmation (because the
        # rollout policy samples stochastically; greedy is what actually runs).
        if (update + 1) % cfg.check_every == 0:
            # Require every opponent to have enough recent episodes logged.
            enough_data = all(len(roll_wins[n]) >= max(30, cfg.games_per_check // 3) for n in pool_names)
            if enough_data:
                min_win = min(roll_win_rates[n] for n in pool_names)
                max_loss = max(roll_loss_rates[n] for n in pool_names)
                best_min_win = max(best_min_win, min_win)
                log_metrics["roll/min_win"] = min_win
                log_metrics["roll/max_loss"] = max_loss

                candidate = (cfg.strict and min_win >= 1.0 - 1e-9) or \
                            ((not cfg.strict) and max_loss <= 0.0)
                if candidate:
                    # Confirm with greedy eval before promotion.
                    t_eval0 = time.time()
                    eval_out = _eval_vs_pool(
                        ac, pool, games_per_check=cfg.games_per_check,
                        cfg=cfg, device=device, seed=seed + update,
                    )
                    t_eval = time.time() - t_eval0
                    last_eval = eval_out
                    g_min_win = min(v["win_rate"] for v in eval_out.values())
                    g_max_loss = max(v["loss_rate"] for v in eval_out.values())
                    log_metrics["eval/min_win"] = g_min_win
                    log_metrics["eval/max_loss"] = g_max_loss
                    log_metrics["eval/sec"] = t_eval
                    for n, r in eval_out.items():
                        log_metrics[f"eval/{n}/win_rate"] = r["win_rate"]
                        log_metrics[f"eval/{n}/loss_rate"] = r["loss_rate"]

                    strict_ok = g_min_win >= 1.0 - 1e-9
                    no_loss_ok = g_max_loss <= 0.0
                    if cfg.strict and strict_ok:
                        promoted = True
                    elif (not cfg.strict) and no_loss_ok:
                        promoted = True

        if wandb_run is not None:
            try:
                wandb_run.log(log_metrics, step=snake_idx * cfg.max_updates_per_snake + update + 1)
            except Exception:
                pass

        if (update + 1) % cfg.print_every == 0 or promoted:
            tail = ""
            if last_eval:
                worst = min(last_eval.items(), key=lambda kv: kv[1]["win_rate"])
                tail = f"  worst={worst[0]}({worst[1]['win_rate']*100:.0f}%)"
            print(f"  [snake{snake_idx}/u{update+1}/pool{len(pool)}] "
                  f"opp={opp_name}  ent={log_metrics['train/entropy']:.3f}  "
                  f"kl={log_metrics['train/approx_kl']:.4f}  "
                  f"ev={log_metrics['train/explained_variance']:.2f}"
                  f"{tail}  budget={log_metrics['budget_used_h']:.2f}h")

        if promoted:
            break
        if (time.time() - global_t0) > cfg.hours_budget * 3600:
            break

    # Save (whether promoted or not).
    path = run_dir / ("promoted.pt" if promoted else "failed.pt")
    _save_snake(ac, opt, cfg, update + 1, path, promoted=promoted,
                eval_snapshot=last_eval)

    return {
        "snake_idx": snake_idx,
        "updates": update + 1,
        "promoted": promoted,
        "best_min_win": best_min_win,
        "last_eval": last_eval,
        "wall_sec": time.time() - t_snake_start,
    }


def main() -> None:
    p = argparse.ArgumentParser()

    # Initial pool (required) — every external snapshot, name:path[:head=N]
    p.add_argument("--initial", action="append", required=True,
                   help="Initial pool entry. Repeat for each.")

    # Budget / promotion
    p.add_argument("--hours-budget", type=float, default=5.0)
    p.add_argument("--max-updates-per-snake", type=int, default=3000)
    p.add_argument("--check-every", type=int, default=200)
    p.add_argument("--games-per-check", type=int, default=100)
    p.add_argument("--strict", action="store_true", default=True,
                   help="Require 100%% wins vs EVERY pool member (your rule).")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="Relax to 0%% losses (draws allowed).")

    # Opponent sampling
    p.add_argument("--opponent-epsilon", type=float, default=0.0)

    # PPO / env
    p.add_argument("--n-envs", type=int, default=131072)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-minibatches", type=int, default=32)
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
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--snake-length", type=int, default=4)
    p.add_argument("--snake-multiplier", type=int, default=2)
    p.add_argument("--interp-ball-obs", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=True)

    # Infra
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="rl/runs/v15_hill")
    p.add_argument("--print-every", type=int, default=10)

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

    # Load initial pool.
    print(f"Loading initial pool ({len(cfg.initial)} entries)...")
    pool: dict[str, nn.Module] = {}
    for spec in cfg.initial:
        name, path, head = _parse_spec(spec)
        if not Path(path).exists():
            print(f"  skip {name}: {path} not found")
            continue
        try:
            pool[name] = _load_generic(path, device, head=head)
            print(f"  loaded {name:>12s} ← {path}")
        except Exception as e:
            print(f"  skip {name}: {e}")
    print(f"Pool size: {len(pool)}")

    run = _init_wandb(cfg)
    global_t0 = time.time()

    summaries = []
    snake_idx = 0
    budget_sec = cfg.hours_budget * 3600

    while (time.time() - global_t0) < budget_sec:
        run_dir = out_root / f"snake_{snake_idx}"
        print(f"\n=== SNAKE {snake_idx} | pool={len(pool)} | "
              f"budget {(time.time()-global_t0)/3600:.2f}h/{cfg.hours_budget}h ===")
        summary = train_one_snake(
            snake_idx=snake_idx, cfg=cfg, device=device,
            pool=pool, seed=cfg.seed + 1000 * snake_idx,
            run_dir=run_dir, wandb_run=run, global_t0=global_t0,
        )
        summaries.append(summary)
        if summary["promoted"]:
            # Add frozen snapshot to pool.
            snap = _ActorQWrapper(build_actor_critic(obs_dim(cfg.snake_length), cfg.hidden_size).to(device))
            # Load from the saved checkpoint to make the snapshot self-contained.
            ckpt_path = run_dir / "promoted.pt"
            state = torch.load(ckpt_path, map_location=device)
            snap.ac.load_state_dict(state["ac"])
            snap.eval()
            for pp in snap.parameters():
                pp.requires_grad_(False)
            pool[f"hill_s{snake_idx}"] = snap
            print(f"  ✔ promoted. new pool size: {len(pool)}")
        else:
            worst = summary.get("last_eval") or {}
            if worst:
                wn = min(worst.items(), key=lambda kv: kv[1]["win_rate"])
                print(f"  ✗ not promoted (best min-win {summary['best_min_win']*100:.0f}%, "
                      f"hardest=`{wn[0]}` {wn[1]['win_rate']*100:.0f}%)")
        (out_root / "summaries.json").write_text(json.dumps(summaries, indent=2))
        snake_idx += 1

    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
    print(f"\nDone. {snake_idx} snakes attempted, {sum(s['promoted'] for s in summaries)} promoted.")


if __name__ == "__main__":
    main()
