"""Multi-lineage league PPO trainer for Snake-Pong (v16).

Five+ independent lineages train concurrently, fighting each other's
admitted snapshots from a shared opponent pool. A snapshot is admitted only
if its **mean** greedy win rate across pool members is >= --promote-threshold
(robust to a single brittle bad-matchup, unlike v16's strict per-pair gate).
Admission uses cheap rolling rollout stats as a pre-trigger, then a
confirmation tournament with safety filter on both sides.

Frozen deployed tiers (easy/medium/hard/master/insane) are loaded as
eval-only benchmarks — never opponents — so wandb shows an objective
progress curve against the deployed ladder. Tier checkpoints are resolved
to .pt files when present; otherwise reconstructed from the deployed ONNX
files (so tier eval works in environments where only the ONNX is checked
in).

Mandatory deployment-correctness invariants (no flag to disable):
  - Learner samples actions from a Categorical over safety-masked logits.
  - The same per-step legal mask is reapplied in PPO's evaluate_actions
    so the recomputed Categorical matches the masked sampling distribution.
  - Opponent argmax is taken over masked logits in rollout, admission
    tournament, and tier eval.

PPO follows Huang et al. "37 Implementation Details": orthogonal init,
Adam eps=1e-5, LR linear-anneal per cycle, GAE, advantage normalization,
clipped surrogate, value-loss clipping, entropy bonus, global grad clip,
debug metrics, and separate actor/critic.

Outputs (under --out-dir):
  lineage_{i}/snap_u{update}.pt    — periodic, unconditional
  pool/{lineage_i_gen_j}.pt        — admitted-to-shared-pool snapshots
  summaries.json                   — per-cycle stats
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .gym_env import obs_dim
from .ppo_model import ActorCritic, build_actor_critic
from .ppo_rollout import PPORollout
from .ppo_utils import approx_kl_k3, compute_gae, explained_variance, linear_lr
from .vec_tournament import run_tournament


# ── Tier eval set ─────────────────────────────────────────────────────────────
# (.pt source first, ONNX fallback). The ONNX files are the deployed tiers and
# are guaranteed present for any deployment-tracking install.
TIER_PATHS: dict[str, tuple[str, str]] = {
    "easy":   ("rl/runs/v9/play_h4_final.pt",        "models/snake-pong-easy.onnx"),
    "medium": ("rl/runs/v5/best.pt",                 "models/snake-pong-medium.onnx"),
    "hard":   ("rl/runs/v12/snap_u1400.pt",          "models/snake-pong-hard.onnx"),
    "master": ("rl/runs/v14/snap_final.pt",          "models/snake-pong-master.onnx"),
    "insane": ("rl/runs/v15_hill/snake_0/failed.pt", "models/snake-pong-insane.onnx"),
}
TIER_HEADS: dict[str, Optional[int]] = {
    "easy": 4, "medium": None, "hard": None, "master": None, "insane": None,
}


# ── ONNX → torch reconstruction (small MLPs only) ─────────────────────────────
class _DuelingFromONNX(nn.Module):
    """Dueling DQN: Q = V + (A - A.mean()). Matches DuelingQNetwork.forward."""

    def __init__(self, trunk: nn.Module, value_head: nn.Module, adv_head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.value_head = value_head
        self.adv_head = adv_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=1, keepdim=True)


def _build_mlp_from_inits(
    inits: dict[str, torch.Tensor], prefix: str, use_tanh: bool,
    trailing_activation: bool = False,
) -> nn.Sequential:
    """Reconstruct a `Linear-Act-…-Linear[-Act]` stack from named ONNX
    initializers.

    Looks for keys `{prefix}.{i}.weight` / `.bias` (sorted by `i`); inserts the
    chosen activation between consecutive Linears. With `trailing_activation`,
    also appends one after the last Linear — required for the dueling
    `trunk` whose output feeds into both heads as already-activated features.
    """
    indices = sorted({
        int(k[len(prefix) + 1:].split(".")[0])
        for k in inits if k.startswith(prefix + ".") and ".weight" in k
    })
    layers: list[nn.Module] = []
    act_cls = nn.Tanh if use_tanh else nn.ReLU
    for i, idx in enumerate(indices):
        w = inits[f"{prefix}.{idx}.weight"]
        b = inits[f"{prefix}.{idx}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        if i < len(indices) - 1:
            layers.append(act_cls())
    if trailing_activation:
        layers.append(act_cls())
    return nn.Sequential(*layers)


def _load_onnx_as_torch(path: str, device: torch.device) -> nn.Module:
    """Reconstruct the small MLP encoded in `path` as an nn.Module on `device`.

    Handles: PPO actor (Tanh activations, prefix `net.`), DQN single-head
    extraction (ReLU, prefix `net.`), and dueling DQN (trunk + value_head +
    adv_head).
    """
    import onnx
    import onnx.numpy_helper as nph

    m = onnx.load(path)
    inits = {
        ini.name: torch.from_numpy(nph.to_array(ini).copy())
        for ini in m.graph.initializer
    }
    ops = [n.op_type for n in m.graph.node]
    use_tanh = "Tanh" in ops
    is_dueling = (
        any(n.startswith("value_head.") for n in inits) and
        any(n.startswith("adv_head.") for n in inits)
    )
    if is_dueling:
        # Trunk: Linear-ReLU-Linear-ReLU (output feeds heads as activated features).
        # Heads:  Linear-ReLU-Linear (final layer outputs raw V/A).
        trunk = _build_mlp_from_inits(
            inits, prefix="trunk", use_tanh=False, trailing_activation=True,
        )
        value_head = _build_mlp_from_inits(inits, prefix="value_head", use_tanh=False)
        adv_head = _build_mlp_from_inits(inits, prefix="adv_head", use_tanh=False)
        net: nn.Module = _DuelingFromONNX(trunk, value_head, adv_head)
    else:
        net = _build_mlp_from_inits(inits, prefix="net", use_tanh=use_tanh)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _load_tier_actor_state(name: str) -> Optional[dict[str, torch.Tensor]]:
    """Return a state_dict matching the `Actor` module (keys: net.{0,2,4}...)
    loaded from a deployed PPO tier ONNX file. Returns None if tier isn't a
    PPO actor (e.g. dueling DQN — can't transfer to a different arch)."""
    if name not in TIER_PATHS:
        return None
    _, onnx_path = TIER_PATHS[name]
    if not Path(onnx_path).exists():
        return None
    import onnx
    import onnx.numpy_helper as nph
    m = onnx.load(onnx_path)
    inits = {ini.name: torch.from_numpy(nph.to_array(ini).copy())
             for ini in m.graph.initializer}
    if any(k.startswith("trunk.") for k in inits):
        return None  # dueling DQN — wrong shape for ppo actor
    has_tanh = any(n.op_type == "Tanh" for n in m.graph.node)
    if not has_tanh:
        return None  # ReLU DQN — different activation
    return {k: v for k, v in inits.items() if k.startswith("net.")}


def _load_tier(name: str, device: torch.device) -> Optional[nn.Module]:
    """Resolve a deployed tier name to an nn.Module. Tries .pt then ONNX.
    Returns None if neither source is present (tier eval is then skipped for
    this name)."""
    pt_path, onnx_path = TIER_PATHS[name]
    pt_full = Path(pt_path)
    if pt_full.exists():
        try:
            from .big_tournament import _load_generic
            return _load_generic(str(pt_full), device, head=TIER_HEADS[name])
        except Exception as e:
            print(f"[tier:{name}] failed to load .pt ({e}); trying ONNX.")
    onnx_full = Path(onnx_path)
    if onnx_full.exists():
        return _load_onnx_as_torch(str(onnx_full), device)
    return None


# ── Wandb-friendly opponent wrapper ───────────────────────────────────────────
class _ActorQWrapper(nn.Module):
    """Adapt an ActorCritic to Q-net opponent plumbing — argmax(logits) is
    the same as argmax(softmax(logits)) so the in-rollout greedy opponent
    selection matches PPO's deterministic policy."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.ac.actor(obs)


# ── Lineage state ─────────────────────────────────────────────────────────────
@dataclass
class Lineage:
    """One independently-initialized PPO trainer (its own AC, optimizer,
    rollout state) inside the league round-robin."""

    idx: int
    ac: ActorCritic
    opt: torch.optim.Optimizer
    rollout: PPORollout
    rng: np.random.Generator
    seed: int
    updates_done: int = 0
    gen: int = 0  # number of times this lineage's snapshot has been admitted
    roll_wins: dict[str, deque] = field(default_factory=dict)
    roll_losses: dict[str, deque] = field(default_factory=dict)
    # "offensive" (draw_reward < 0) or "defensive" (draw_reward > 0) — drives
    # different play styles via asymmetric outcome rewards.
    style: str = "balanced"
    draw_reward: float = 0.0


# ── Wandb helpers ─────────────────────────────────────────────────────────────
def _init_wandb(cfg: argparse.Namespace):
    try:
        import wandb
    except ImportError:
        return None
    if cfg.wandb_mode == "disabled":
        return None
    try:
        run = wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity,
            name=cfg.wandb_run_name, mode=cfg.wandb_mode,
            config=vars(cfg), dir=cfg.out_dir,
        )
        # Each lineage's metrics use its own update counter as x-axis. Without
        # this, the default monotonic step interleaves all lineages so they
        # show as disconnected segments instead of overlapping curves.
        for i in range(cfg.n_lineages):
            wandb.define_metric(f"lin{i}/update", hidden=True)
            wandb.define_metric(f"lin{i}/*", step_metric=f"lin{i}/update")
            wandb.define_metric(f"videos/lin{i}/*", step_metric=f"lin{i}/update")
        # Tier-eval metrics share a cycle-counter axis.
        wandb.define_metric("eval/cycle", hidden=True)
        wandb.define_metric("eval/*", step_metric="eval/cycle")
        # Pool stats indexed on cycle for clean overlay.
        wandb.define_metric("pool/cycle", hidden=True)
        wandb.define_metric("pool/*", step_metric="pool/cycle")
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}); proceeding without.")
        return None


def _log_video(run, frames: np.ndarray, key: str, lineage_idx: int,
               update: int, fps: int = 10) -> None:
    """Upload a video; tagged with the lineage's update counter so the
    `lin{i}/update` step_metric places it on the per-lineage x-axis."""
    if run is None:
        return
    try:
        import wandb
        run.log({
            f"lin{lineage_idx}/update": update,
            key: wandb.Video(frames, fps=fps, format="gif"),
        })
    except Exception as e:
        print(f"[wandb] video upload failed ({key}): {e}")


# ── Save helpers ──────────────────────────────────────────────────────────────
def _save_snapshot(
    ac: ActorCritic, cfg: argparse.Namespace, update: int, path: Path,
    extra: dict | None = None,
) -> None:
    payload = {
        "ac": ac.state_dict(),
        "q_net": ac.state_dict(),  # alias for legacy tooling
        "iter": update,
        "config": {**vars(cfg), "model_arch": "ppo_ac"},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _freeze_actor_snapshot(ac: ActorCritic, device: torch.device) -> _ActorQWrapper:
    """Deep-copy the AC, wrap its actor as a Q-net, freeze gradients."""
    snap = copy.deepcopy(ac).eval().to(device)
    for p in snap.parameters():
        p.requires_grad_(False)
    return _ActorQWrapper(snap).eval()


# ── Build a lineage ───────────────────────────────────────────────────────────
def _build_lineage(idx: int, cfg: argparse.Namespace, device: torch.device) -> Lineage:
    seed = cfg.seed + cfg.seed_stride * idx
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Style assignment: first n_offensive lineages get offensive reward
    # (penalize draws); remaining lineages get defensive reward (reward
    # draws). Magnitude is shared via cfg.draw_reward_magnitude.
    if idx < cfg.n_offensive:
        style = "offensive"
        draw_r = -cfg.draw_reward_magnitude
    elif cfg.draw_reward_magnitude > 0.0:
        style = "defensive"
        draw_r = +cfg.draw_reward_magnitude
    else:
        style = "balanced"
        draw_r = 0.0

    D = obs_dim(cfg.snake_length)
    ac = build_actor_critic(obs_dim=D, hidden=cfg.hidden_size).to(device)

    # Optional warm-start: load a deployed tier's actor weights into this
    # lineage's actor. Critic is left at orthogonal-init (it converges fast
    # against a strong initial policy).
    if cfg.init_from_tier:
        state = _load_tier_actor_state(cfg.init_from_tier)
        if state is None:
            raise SystemExit(f"--init-from-tier={cfg.init_from_tier} unavailable "
                             f"or not a PPO-Tanh actor (only ppo_ac tiers supported)")
        ac.actor.load_state_dict({k: v.to(device) for k, v in state.items()})
        # Per-lineage symmetry-breaking perturbation. Without this, all
        # lineages would start with identical actor weights and only diverge
        # via env/opp sampling. The σ=1e-3 noise is tiny vs master weight
        # magnitudes (~0.1–1.0) so master's policy is preserved, but
        # gradients differ from step 1.
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        with torch.no_grad():
            for p in ac.actor.parameters():
                p.add_(torch.randn(p.shape, generator=g, device=device) * 1e-3)
        if idx == 0:
            print(f"   [init] all lineages warm-started from tier '{cfg.init_from_tier}'"
                  f" + per-lineage σ=1e-3 perturbation")

    # Optional torch.compile — fuses Linear/Tanh kernels in the tiny MLPs;
    # the big win for an H100 launch-bound rollout. reduce-overhead uses
    # CUDA graphs internally so per-step launches are amortized.
    if cfg.compile:
        ac.actor = torch.compile(ac.actor, mode="reduce-overhead", dynamic=True)
        ac.critic = torch.compile(ac.critic, mode="reduce-overhead", dynamic=True)

    # fused=True bundles all parameter updates into a single CUDA kernel —
    # ~10-30% update-phase win on Ampere+.
    opt = torch.optim.Adam(
        ac.parameters(), lr=cfg.lr, eps=cfg.adam_eps, fused=(device.type == "cuda"),
    )

    rollout = PPORollout(
        n_envs=cfg.n_envs, rollout_steps=cfg.rollout_steps, obs_dim=D,
        device=device, snake_length=cfg.snake_length,
        snake_multiplier=cfg.snake_multiplier, max_steps=cfg.max_steps,
        interp_ball=cfg.interp_ball_obs,
        shape_coef=0.0, shape_gamma=cfg.gamma,
        bf16=cfg.bf16,
        learner_safety_filter=True,  # forced — matches deployment
        rng=rng,
        win_reward=1.0, draw_reward=draw_r, loss_reward=-1.0,
    )
    print(f"   [lineage {idx}] style={style:<10} draw_reward={draw_r:+.2f}  seed={seed}")
    return Lineage(
        idx=idx, ac=ac, opt=opt, rollout=rollout, rng=rng, seed=seed,
        style=style, draw_reward=draw_r,
    )


# ── Per-lineage training chunk ────────────────────────────────────────────────
def _ensure_roll(L: Lineage, name: str, maxlen: int) -> None:
    if name not in L.roll_wins:
        L.roll_wins[name] = deque(maxlen=maxlen)
        L.roll_losses[name] = deque(maxlen=maxlen)


def _run_lineage_chunk(
    L: Lineage,
    cycle_updates: int,
    pool: dict[str, nn.Module],
    cfg: argparse.Namespace,
    device: torch.device,
    lineage_dir: Path,
    wandb_run,
    global_t0: float,
) -> dict:
    """Run `cycle_updates` PPO updates on `L`, sampling opponents from the
    shared pool (or random if pool empty / sampled by `opp_random_prob`).
    Saves a periodic snapshot every `cfg.snap_every_updates`. Returns rolling
    stats ready for the admission gate."""
    cfg_target_kl = cfg.target_kl
    batch_size = cfg.n_envs * cfg.rollout_steps
    mb_size = batch_size // cfg.n_minibatches

    chunk_t0 = time.time()
    completed_total = 0

    # Maxlen sized to capture a healthy sample per opponent at steady state.
    pool_size_now = max(1, len(pool))
    roll_max = max(300, 10 * cfg.check_every // pool_size_now)

    indices_full = torch.arange(batch_size, device=device)

    # Self-play opponent: a frozen snapshot of L's own current actor, taken
    # once at chunk start ("latest version of itself"). Always included in
    # the opponent rotation alongside every pool member, so each update has
    # a uniform-over-{pool ∪ self} opponent. Frozen per-chunk avoids chasing
    # a moving target within a single chunk's PPO updates.
    self_snap: nn.Module = _freeze_actor_snapshot(L.ac, device)

    for u_in_chunk in range(cycle_updates):
        # Per-lineage LR anneal across the full per-lineage budget. We don't
        # know that budget exactly (driven by wall clock), so we anneal across
        # cfg.lr_anneal_updates as a soft target.
        lr_now = linear_lr(L.updates_done, cfg.lr_anneal_updates, cfg.lr)
        for g in L.opt.param_groups:
            g["lr"] = lr_now

        # Opponent: random with prob opp_random_prob (annealed if requested),
        # otherwise uniform over {pool members ∪ self_snap}. Self is always
        # a candidate, so the learner never goes a chunk without facing the
        # latest of itself.
        if cfg.random_prob_anneal_updates > 0:
            random_prob = max(0.0, linear_lr(
                L.updates_done, cfg.random_prob_anneal_updates,
                cfg.opp_random_prob,
            ))
        else:
            random_prob = cfg.opp_random_prob
        pool_names = list(pool.keys())
        candidates = pool_names + ["__self__"]
        if float(L.rng.random()) < random_prob:
            opp_name = "__random__"
            L.rollout.set_opponent(None, 0.0, opp_safety_filter=True)
        else:
            opp_name = candidates[int(L.rng.integers(0, len(candidates)))]
            opp_module = self_snap if opp_name == "__self__" else pool[opp_name]
            L.rollout.set_opponent(
                opp_module, cfg.opponent_epsilon, opp_safety_filter=True,
            )
        _ensure_roll(L, opp_name, roll_max)

        # ── Rollout ──────────────────────────────────────────────────────────
        t_rollout = time.time()
        completed = L.rollout.collect(L.ac)
        t_rollout = time.time() - t_rollout
        completed_total += len(completed)

        # Tag completed eps with the active opponent so admission has accurate
        # rolling stats. Loss-rate excludes draws and truncations, matching
        # the strict definition `won + lost + draw + trunc = 1`.
        for e in completed:
            won = bool(e["won"])
            term = e["terminal"]
            scorer = e["scorer"]
            L.roll_wins[opp_name].append(1.0 if won else 0.0)
            lost_ep = (not won) and (term != "truncated") and (scorer != 3)
            L.roll_losses[opp_name].append(1.0 if lost_ep else 0.0)

        # ── GAE bootstrap ────────────────────────────────────────────────────
        with torch.no_grad():
            last_value = L.rollout.bootstrap_value(L.ac)
        advs, rets = compute_gae(
            L.rollout.rewards, L.rollout.values, L.rollout.dones,
            last_value, L.rollout.next_done,
            cfg.gamma, cfg.gae_lambda,
        )

        flat = L.rollout.flat_views()
        b_obs = flat["obs"]
        b_actions = flat["actions"]
        b_logp = flat["logprobs"]
        b_values = flat["values"]
        b_advs = advs.reshape(-1)
        b_rets = rets.reshape(-1)
        b_legal = flat.get("legal")  # forced safety filter is on

        # ── PPO update ───────────────────────────────────────────────────────
        t_update = time.time()
        kls: list[float] = []
        clips: list[float] = []
        pg_losses: list[float] = []
        v_losses: list[float] = []
        ent_losses: list[float] = []
        stopped_early_at = -1

        for epoch in range(cfg.n_epochs):
            perm = indices_full[torch.randperm(batch_size, device=device)]
            for s in range(0, batch_size, mb_size):
                idx = perm[s: s + mb_size]
                mb_obs = b_obs[idx]
                mb_act = b_actions[idx]
                mb_old_logp = b_logp[idx]
                mb_old_v = b_values[idx]
                mb_adv = b_advs[idx]
                mb_ret = b_rets[idx]
                mb_legal = b_legal[idx] if b_legal is not None else None

                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    new_logp, ent, new_v = L.ac.evaluate_actions(mb_obs, mb_act, legal=mb_legal)
                new_logp = new_logp.float()
                ent = ent.float()
                new_v = new_v.float()
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

                L.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(L.ac.parameters(), cfg.max_grad_norm)
                L.opt.step()

                with torch.no_grad():
                    kls.append(float(approx_kl_k3(logratio).item()))
                    clips.append(float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()))
                pg_losses.append(float(pg.item()))
                v_losses.append(float(v_loss.item()))
                ent_losses.append(float(ent_loss.item()))

            if cfg_target_kl is not None and float(np.mean(kls[-cfg.n_minibatches:])) > cfg_target_kl:
                stopped_early_at = epoch + 1
                break

        t_update = time.time() - t_update
        ev = explained_variance(b_values, b_rets)
        L.updates_done += 1

        # ── Periodic snapshot (independent of pool admission) ────────────────
        if cfg.snap_every_updates and L.updates_done % cfg.snap_every_updates == 0:
            snap_path = lineage_dir / f"snap_u{L.updates_done}.pt"
            _save_snapshot(L.ac, cfg, L.updates_done, snap_path)

        # ── Log ──────────────────────────────────────────────────────────────
        if wandb_run is not None and (L.updates_done % cfg.log_every_updates == 0):
            log_metrics = {
                # Per-lineage step_metric — every lin{i}/* metric is plotted
                # against this so all 8 lineages overlap on the same x-axis.
                f"lin{L.idx}/update": L.updates_done,
                f"lin{L.idx}/lr": lr_now,
                f"lin{L.idx}/pool_size": len(pool),
                f"lin{L.idx}/budget_h": (time.time() - global_t0) / 3600,
                f"lin{L.idx}/rollout_sec": t_rollout,
                f"lin{L.idx}/update_sec": t_update,
                f"lin{L.idx}/episodes_per_update": len(completed),
                f"lin{L.idx}/train/approx_kl": float(np.mean(kls)) if kls else 0.0,
                f"lin{L.idx}/train/clipfrac": float(np.mean(clips)) if clips else 0.0,
                f"lin{L.idx}/train/pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
                f"lin{L.idx}/train/v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
                f"lin{L.idx}/train/entropy": float(np.mean(ent_losses)) if ent_losses else 0.0,
                f"lin{L.idx}/train/explained_variance": ev,
                f"lin{L.idx}/train/stopped_early_at": stopped_early_at,
            }
            for n, dq in L.roll_wins.items():
                if not dq or n == "__random__":
                    # Skip random — not a meaningful curve and clutters the dashboard.
                    continue
                log_metrics[f"lin{L.idx}/roll/{n}/win"] = float(np.mean(dq))
                log_metrics[f"lin{L.idx}/roll/{n}/loss"] = float(np.mean(L.roll_losses[n]))
            try:
                # No explicit step — wandb uses its own monotonic counter for
                # storage ordering, while the chart UI uses the per-lineage
                # step_metric set in _init_wandb.
                wandb_run.log(log_metrics)
            except Exception:
                pass

        if cfg.print_every and L.updates_done % cfg.print_every == 0:
            roll_str = ""
            if pool_names:
                wins_now = [
                    float(np.mean(L.roll_wins[n])) if L.roll_wins.get(n) else float("nan")
                    for n in pool_names
                ]
                roll_str = f" min_win={min(w for w in wins_now if not np.isnan(w)) if wins_now else 0.0:.2f}"
            print(
                f"  [L{L.idx}/u{L.updates_done}/pool{len(pool)}] "
                f"opp={opp_name} ent={float(np.mean(ent_losses)) if ent_losses else 0:.3f} "
                f"kl={float(np.mean(kls)) if kls else 0:.4f} "
                f"ev={ev:.2f} eps={len(completed)}{roll_str} "
                f"r={t_rollout:.2f}s u={t_update:.2f}s "
                f"budget={(time.time() - global_t0) / 3600:.2f}h"
            )

        # Hard wall-clock guard (in case a single chunk straddles the budget).
        if (time.time() - global_t0) > cfg.hours_budget * 3600:
            break

    return {
        "lineage_idx": L.idx,
        "updates_done": L.updates_done,
        "completed_episodes": completed_total,
        "chunk_sec": time.time() - chunk_t0,
    }


# ── Pool admission ────────────────────────────────────────────────────────────
def _maybe_admit(
    L: Lineage,
    pool: dict[str, nn.Module],
    cfg: argparse.Namespace,
    device: torch.device,
    pool_dir: Path,
    wandb_run,
) -> dict:
    """Try to admit a frozen snapshot of `L.ac` into the shared pool.

    1. Empty pool → admit unconditionally (seed).
    2. Otherwise, gate on rolling stats; confirm via tournament; admit if
       mean_win_rate (averaged over pool members) >= cfg.promote_threshold.
    """
    out = {"admitted": False, "reason": "", "min_win_rate": None, "mean_win_rate": None}

    # Cap: drop oldest pool member if we overflow.
    if cfg.max_pool_size and len(pool) >= cfg.max_pool_size:
        # FIFO over admission order (insertion order is preserved in Python dicts)
        first_key = next(iter(pool))
        del pool[first_key]
        out["evicted"] = first_key

    snap = _freeze_actor_snapshot(L.ac, device)
    snap_name = f"lineage_{L.idx}_gen_{L.gen}"

    if not pool:
        pool[snap_name] = snap
        L.gen += 1
        out["admitted"] = True
        out["reason"] = "seed"
        out["snap_name"] = snap_name
        path = pool_dir / f"{snap_name}.pt"
        _save_snapshot(L.ac, cfg, L.updates_done, path, extra={"admitted": True})
        return out

    # No-gate mode: every snapshot enters the pool unconditionally.
    if cfg.no_admission_gate:
        pool[snap_name] = snap
        L.gen += 1
        out["admitted"] = True
        out["reason"] = "no-admission-gate"
        out["snap_name"] = snap_name
        path = pool_dir / f"{snap_name}.pt"
        _save_snapshot(L.ac, cfg, L.updates_done, path, extra={"admitted": True})
        return out

    # Pre-trigger: rolling stats. Require enough samples per current pool member.
    enough = all(
        len(L.roll_wins.get(n, deque())) >= max(30, cfg.games_per_check // 3)
        for n in pool
    )
    if not enough:
        out["reason"] = "insufficient rolling data"
        return out
    soft_thresh = cfg.promote_threshold * 0.95
    rolling_mean = float(np.mean([
        float(np.mean(L.roll_wins[n])) for n in pool
    ]))
    if rolling_mean < soft_thresh:
        out["reason"] = f"rolling_mean={rolling_mean:.2f} < {soft_thresh:.2f}"
        return out

    # Confirmation tournament — safety filter on both sides.
    models = {snap_name: snap, **pool}
    res = run_tournament(
        models, games_per_pair=cfg.games_per_check, device=device, seed=L.seed + L.updates_done,
        snake_length=cfg.snake_length, snake_multiplier=cfg.snake_multiplier,
        max_steps=cfg.max_steps, interp_ball=cfg.interp_ball_obs,
        safety_filter=True, show_progress=False,
    )
    names = res["names"]
    wr = res["win_rate"]
    i0 = names.index(snap_name)
    pool_idxs = [names.index(n) for n in pool]
    win_rates = [float(wr[i0, j]) for j in pool_idxs]
    min_wr = min(win_rates) if win_rates else 0.0
    mean_wr = float(np.mean(win_rates)) if win_rates else 0.0
    out["min_win_rate"] = min_wr
    out["mean_win_rate"] = mean_wr
    out["per_opponent"] = dict(zip(pool, win_rates))

    if mean_wr >= cfg.promote_threshold:
        pool[snap_name] = snap
        L.gen += 1
        out["admitted"] = True
        out["reason"] = (
            f"mean_wr={mean_wr:.2f} >= {cfg.promote_threshold:.2f} "
            f"(min={min_wr:.2f})"
        )
        out["snap_name"] = snap_name
        path = pool_dir / f"{snap_name}.pt"
        _save_snapshot(L.ac, cfg, L.updates_done, path, extra={"admitted": True})
    else:
        out["reason"] = (
            f"mean_wr={mean_wr:.2f} < {cfg.promote_threshold:.2f} "
            f"(min={min_wr:.2f})"
        )

    if wandb_run is not None:
        try:
            log = {
                f"lin{L.idx}/update": L.updates_done,
                f"lin{L.idx}/admission/min_win_rate": min_wr,
                f"lin{L.idx}/admission/mean_win_rate": mean_wr,
                f"lin{L.idx}/admission/admitted": int(out["admitted"]),
                f"lin{L.idx}/admission/pool_size_after": len(pool),
            }
            for opp_name, opp_wr in out["per_opponent"].items():
                log[f"lin{L.idx}/admission/vs/{opp_name}"] = opp_wr
            wandb_run.log(log)
        except Exception:
            pass
    return out


# ── Tier eval (lineages × frozen tiers) ───────────────────────────────────────
def _eval_one_lineage_vs_tiers(
    L: Lineage,
    tiers: dict[str, nn.Module],
    cfg: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run a small tournament: this lineage's current actor vs each tier.

    Returns {tier: {"win": w, "loss": l, "draw": d}} — fractions summing to 1.

    Cheap (~5s on H100): 6 models × 6 × games_per_pair = ~288 games batched
    into one vec env. Logged at the end of each lineage chunk so the per-
    lineage `lin{i}/update` x-axis fills in 8x more often than cycle-level
    eval (which still runs once per cycle for the cross-lineage matrix view).
    """
    if not tiers:
        return {}
    models: dict[str, nn.Module] = {"__learner__": _ActorQWrapper(L.ac).eval(), **tiers}
    res = run_tournament(
        models, games_per_pair=cfg.tier_eval_games, device=device, seed=seed,
        snake_length=cfg.snake_length, snake_multiplier=cfg.snake_multiplier,
        max_steps=cfg.max_steps, interp_ball=cfg.interp_ball_obs,
        safety_filter=True, show_progress=False,
    )
    names = res["names"]
    wins = res["wins"]
    losses = res["losses"]
    draws = res["draws"]
    i0 = names.index("__learner__")
    out: dict[str, dict[str, float]] = {}
    for t in tiers:
        j = names.index(t)
        total = max(int(wins[i0, j]) + int(losses[i0, j]) + int(draws[i0, j]), 1)
        out[t] = {
            "win": float(wins[i0, j]) / total,
            "loss": float(losses[i0, j]) / total,
            "draw": float(draws[i0, j]) / total,
        }
    return out


def _eval_vs_tiers(
    lineages: list[Lineage],
    tiers: dict[str, nn.Module],
    cfg: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[int, dict[str, dict[str, float]]]:
    """Run a tournament with each lineage's CURRENT actor + every tier net.
    Returns {lineage_idx: {tier: {"win": w, "loss": l, "draw": d}}}."""
    if not tiers:
        return {}
    learner_models: dict[str, nn.Module] = {}
    for L in lineages:
        learner_models[f"__lin{L.idx}__"] = _ActorQWrapper(L.ac).eval()
    models = {**learner_models, **tiers}
    res = run_tournament(
        models, games_per_pair=cfg.tier_eval_games, device=device, seed=seed,
        snake_length=cfg.snake_length, snake_multiplier=cfg.snake_multiplier,
        max_steps=cfg.max_steps, interp_ball=cfg.interp_ball_obs,
        safety_filter=True, show_progress=False,
    )
    names = res["names"]
    wins = res["wins"]
    losses = res["losses"]
    draws = res["draws"]
    out: dict[int, dict[str, dict[str, float]]] = {}
    for L in lineages:
        i0 = names.index(f"__lin{L.idx}__")
        row: dict[str, dict[str, float]] = {}
        for t in tiers:
            j = names.index(t)
            total = max(int(wins[i0, j]) + int(losses[i0, j]) + int(draws[i0, j]), 1)
            row[t] = {
                "win": float(wins[i0, j]) / total,
                "loss": float(losses[i0, j]) / total,
                "draw": float(draws[i0, j]) / total,
            }
        out[L.idx] = row
    return out


# ── Video recording (CPU rollout, every N lineage-updates) ────────────────────
def _ac_to_cpu_actor(ac: ActorCritic) -> nn.Module:
    """Deep-copy the actor and move it to CPU, frozen, for use by the
    Python-loop video recorder. Avoids mutating the training AC."""
    cpu = build_actor_critic(
        obs_dim=ac.actor.net[0].in_features,
        hidden=ac.actor.net[0].out_features,
    ).to("cpu")
    # `ac.actor` may be a torch.compile-wrapped module; deepcopy of the AC
    # does not preserve the compile wrapping (torch.compile wraps lazily).
    cpu_ac_state = {
        k: v.detach().to("cpu").clone() for k, v in ac.state_dict().items()
    }
    cpu.load_state_dict(cpu_ac_state)
    cpu.eval()
    for p in cpu.parameters():
        p.requires_grad_(False)
    return cpu.actor


def _module_to_cpu(net: nn.Module) -> nn.Module:
    """Snapshot an nn.Module to CPU for the slow record_episode loop. Tier
    nets are immutable, so we cache these once at startup."""
    out = copy.deepcopy(net).to("cpu").eval()
    for p in out.parameters():
        p.requires_grad_(False)
    return out


def _record_videos_for_lineage(
    L: Lineage,
    cpu_tiers: dict[str, nn.Module],
    cfg: argparse.Namespace,
    seed: int,
    wandb_run,
) -> None:
    """Record an episode of the lineage's current actor vs each tier and
    one self-play episode. Logged to wandb as gif videos under
    `videos/lin{idx}/{opponent}`. Runs entirely on CPU; intended to be cheap
    relative to the training cycle (~1s per episode for the small MLP)."""
    if wandb_run is None:
        return
    from .render import record_episode

    cpu_actor = _ac_to_cpu_actor(L.ac)

    def _q_fn(net: nn.Module):
        def fn(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0)
                q = net(t)
                if q.dim() == 3:  # (1, K, A) ensemble
                    q = q.mean(dim=1)
                return q.squeeze(0).cpu().numpy()
        return fn

    def _action_fn(net: nn.Module):
        qf = _q_fn(net)
        return lambda obs: int(np.argmax(qf(obs)))

    targets: list[tuple[str, nn.Module]] = list(cpu_tiers.items())
    # Self-play: lineage vs a frozen copy of itself (current weights).
    self_copy = copy.deepcopy(cpu_actor).eval()
    targets.append(("self", self_copy))

    for tname, opp in targets:
        try:
            frames, info = record_episode(
                _action_fn(cpu_actor), _action_fn(opp),
                snake_length=cfg.snake_length,
                snake_multiplier=cfg.snake_multiplier,
                max_steps=cfg.max_steps,
                interp_ball=cfg.interp_ball_obs,
                seed=seed,
                safety_filter=True,
                learner_q_fn=_q_fn(cpu_actor),
                opponent_q_fn=_q_fn(opp),
            )
            side_label = "L" if int(info.get("learner_side", 1)) == 1 else "R"
            key = f"videos/lin{L.idx}/vs_{tname}_side{side_label}"
            _log_video(wandb_run, frames, key, L.idx, L.updates_done)
        except Exception as e:
            print(f"[video] lin{L.idx} vs {tname} failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()

    # League topology
    p.add_argument("--n-lineages", type=int, default=5)
    p.add_argument("--cycle-updates", type=int, default=200,
                   help="Updates per lineage per cycle. Round-robin scheduler.")
    p.add_argument("--seed-stride", type=int, default=1000,
                   help="Seed offset per lineage (lineage i uses seed = seed + i * seed_stride).")

    # Pool admission
    p.add_argument("--promote-threshold", type=float, default=0.7,
                   help="Min mean greedy win rate (averaged across pool members) "
                        "to admit a lineage's snapshot to the shared pool.")
    p.add_argument("--no-admission-gate", action="store_true", default=False,
                   help="Skip the win-rate gate; admit every chunked snapshot "
                        "into the pool unconditionally.")
    p.add_argument("--init-from-tier", type=str, default=None,
                   choices=("easy", "medium", "hard", "master", "insane"),
                   help="Warm-start every lineage's actor from this deployed "
                        "PPO tier ONNX (critic stays at random init). Lineages "
                        "diverge via different seeds + opponent samples + "
                        "minibatch noise.")
    p.add_argument("--n-offensive", type=int, default=0,
                   help="First N lineages get offensive reward (1, "
                        "-draw_magnitude, -1). Remaining lineages get defensive "
                        "reward (1, +draw_magnitude, -1).")
    p.add_argument("--draw-reward-magnitude", type=float, default=0.0,
                   help="Magnitude of the asymmetric draw reward. 0 disables "
                        "(symmetric (1, 0, -1)) (default).")
    p.add_argument("--random-prob-anneal-updates", type=int, default=0,
                   help="Anneal --opp-random-prob linearly to 0 over this many "
                        "per-lineage updates. 0 (default) keeps it constant.")
    p.add_argument("--check-every", type=int, default=200,
                   help="Per-lineage update cadence between admission attempts.")
    p.add_argument("--games-per-check", type=int, default=8,
                   help="Confirmation tournament games per pair. Greedy "
                        "policies are deterministic over a 4-condition ball-"
                        "start space; 8 covers both learner-side assignments.")
    p.add_argument("--max-pool-size", type=int, default=64)
    p.add_argument("--snap-every-updates", type=int, default=200,
                   help="Per-lineage cadence for periodic disk snapshots "
                        "(separate from pool admission). 0 disables.")

    # Eval / videos
    p.add_argument("--eval-every-cycles", type=int, default=5)
    p.add_argument("--tier-eval-games", type=int, default=8)
    p.add_argument("--video-every-updates", type=int, default=100,
                   help="Per-lineage update cadence for recording one "
                        "wandb video per (lineage × tier) plus self-play. "
                        "0 disables.")

    # Opponent sampling
    p.add_argument("--opp-random-prob", type=float, default=0.15)
    p.add_argument("--opponent-epsilon", type=float, default=0.0)

    # PPO / env
    p.add_argument("--n-envs", type=int, default=131072)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-minibatches", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-anneal-updates", type=int, default=4000,
                   help="Soft target horizon for per-lineage LR linear anneal.")
    p.add_argument("--adam-eps", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--snake-length", type=int, default=4)
    p.add_argument("--snake-multiplier", type=int, default=2)
    p.add_argument("--interp-ball-obs", action="store_true", default=True)
    p.add_argument("--no-interp-ball-obs", dest="interp_ball_obs", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile actor + critic with reduce-overhead. "
                        "On H100 this is the biggest single perf knob; off "
                        "by default for the smoke path which exercises many "
                        "fresh shapes.")

    # Budget / infra
    p.add_argument("--hours-budget", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="rl/runs/v16_league")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--log-every-updates", type=int, default=1)

    # Wandb
    p.add_argument("--wandb-project", default="snake-pong-rl")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    p.add_argument("--wandb-run-name", default=None)

    cfg = p.parse_args()
    device = torch.device(cfg.device)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pool_dir = out_root / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Sanity: batch_size must divide n_minibatches.
    assert (cfg.n_envs * cfg.rollout_steps) % cfg.n_minibatches == 0, (
        f"n_envs*rollout_steps={cfg.n_envs * cfg.rollout_steps} "
        f"must divide n_minibatches={cfg.n_minibatches}"
    )

    print(f"== v16 league trainer ==  device={device}  hours_budget={cfg.hours_budget}")
    print(f"   {cfg.n_lineages} lineages, cycle_updates={cfg.cycle_updates}, "
          f"n_envs={cfg.n_envs}, rollout_steps={cfg.rollout_steps}, "
          f"hidden={cfg.hidden_size}, bf16={cfg.bf16}, compile={cfg.compile}")

    # Build lineages.
    lineages = [_build_lineage(i, cfg, device) for i in range(cfg.n_lineages)]
    for L in lineages:
        (out_root / f"lineage_{L.idx}").mkdir(parents=True, exist_ok=True)

    # Tiers (eval-only).
    tiers: dict[str, nn.Module] = {}
    cpu_tiers: dict[str, nn.Module] = {}
    for tname in TIER_PATHS:
        net = _load_tier(tname, device)
        if net is None:
            print(f"   [tier:{tname}] NOT FOUND — skipping eval slot")
            continue
        tiers[tname] = net
        cpu_tiers[tname] = _module_to_cpu(net)
        print(f"   [tier:{tname}] loaded ({sum(p.numel() for p in net.parameters())/1e3:.1f}k params)")
    print(f"   tiers loaded: {len(tiers)}/{len(TIER_PATHS)}")

    # Wandb.
    run = _init_wandb(cfg)

    # Shared pool starts empty.
    pool: dict[str, nn.Module] = {}

    summaries: list[dict] = []
    cycle_idx = 0
    global_t0 = time.time()

    while (time.time() - global_t0) < cfg.hours_budget * 3600:
        cycle_t0 = time.time()
        print(
            f"\n=== CYCLE {cycle_idx}  budget={(time.time() - global_t0) / 3600:.2f}h"
            f"/{cfg.hours_budget}h  pool={len(pool)} ==="
        )
        cycle_summary: dict = {"cycle": cycle_idx, "lineages": []}

        for L in lineages:
            chunk_summary = _run_lineage_chunk(
                L, cycle_updates=cfg.cycle_updates, pool=pool,
                cfg=cfg, device=device, lineage_dir=out_root / f"lineage_{L.idx}",
                wandb_run=run, global_t0=global_t0,
            )
            cycle_summary["lineages"].append(chunk_summary)
            if (time.time() - global_t0) > cfg.hours_budget * 3600:
                break

            adm = _maybe_admit(L, pool, cfg, device, pool_dir, run)
            if adm.get("admitted"):
                print(
                    f"  ✔ L{L.idx} admitted as {adm.get('snap_name')} "
                    f"({adm.get('reason')})"
                )
            else:
                if adm.get("mean_win_rate") is not None:
                    print(
                        f"  ✗ L{L.idx} admission failed: "
                        f"mean_wr={adm['mean_win_rate']:.2f} "
                        f"min_wr={adm['min_win_rate']:.2f}  ({adm.get('reason')})"
                    )
            chunk_summary["admission"] = {
                k: v for k, v in adm.items()
                if k in (
                    "admitted", "reason", "min_win_rate", "mean_win_rate",
                    "snap_name", "per_opponent",
                )
            }

            # Per-lineage tier eval — fills in the lin{i}/eval/* curves on
            # the per-lineage update axis 8× per cycle (not just at cycle
            # boundary). Cheap: ~5s for one tournament with 6 models.
            if tiers:
                t_le = time.time()
                lin_tier_eval = _eval_one_lineage_vs_tiers(
                    L, tiers, cfg, device,
                    seed=cfg.seed + cycle_idx * cfg.n_lineages + L.idx,
                )
                t_le = time.time() - t_le
                chunk_summary["lineage_tier_eval"] = lin_tier_eval
                if run is not None:
                    try:
                        payload = {f"lin{L.idx}/update": L.updates_done}
                        for t, rates in lin_tier_eval.items():
                            payload[f"lin{L.idx}/eval/{t}/win_rate"] = rates["win"]
                            payload[f"lin{L.idx}/eval/{t}/loss_rate"] = rates["loss"]
                            payload[f"lin{L.idx}/eval/{t}/draw_rate"] = rates["draw"]
                        payload[f"lin{L.idx}/eval/sec"] = t_le
                        run.log(payload)
                    except Exception:
                        pass
                print(
                    f"  📊 L{L.idx} tier-eval ({t_le:.1f}s): " +
                    "  ".join(
                        f"{t}={r['win']*100:.0f}/{r['draw']*100:.0f}/{r['loss']*100:.0f}"
                        for t, r in lin_tier_eval.items()
                    ) + "  (W/D/L %)"
                )

            # Per-lineage video recording (every cfg.video_every_updates).
            if (cfg.video_every_updates and run is not None
                    and L.updates_done % cfg.video_every_updates == 0):
                t_video = time.time()
                _record_videos_for_lineage(
                    L, cpu_tiers, cfg, seed=cfg.seed + cycle_idx,
                    wandb_run=run,
                )
                print(f"  📹 L{L.idx} videos uploaded ({time.time() - t_video:.1f}s)")

        # Tier eval (one tournament covers all lineages × all tiers in one shot).
        if tiers and (cycle_idx % cfg.eval_every_cycles == 0):
            t_eval = time.time()
            tier_eval = _eval_vs_tiers(
                lineages, tiers, cfg, device, seed=cfg.seed + cycle_idx + 1,
            )
            t_eval = time.time() - t_eval
            tier_names = list(tiers.keys())
            print(
                f"  [tier-eval cycle {cycle_idx}] {t_eval:.1f}s  "
                f"lineage → tier  W/D/L %:"
            )
            header = "      | " + " | ".join(f"{t:>10}" for t in tier_names) + " |"
            sep = "      |" + "|".join(["-" * 12 for _ in tier_names]) + "|"
            print(header)
            print(sep)
            for lidx, row in tier_eval.items():
                cells = " | ".join(
                    f"{row[t]['win']*100:>3.0f}/{row[t]['draw']*100:>2.0f}/{row[t]['loss']*100:>3.0f}"
                    for t in tier_names
                )
                print(f"   L{lidx} | {cells} |")
            cycle_summary["tier_eval"] = tier_eval
            if run is not None:
                try:
                    payload: dict[str, float | int] = {}
                    for lidx, row in tier_eval.items():
                        for t, rates in row.items():
                            payload[f"eval/lin{lidx}/{t}/win_rate"] = rates["win"]
                            payload[f"eval/lin{lidx}/{t}/loss_rate"] = rates["loss"]
                            payload[f"eval/lin{lidx}/{t}/draw_rate"] = rates["draw"]
                    payload["eval/tier_eval_sec"] = t_eval
                    payload["eval/cycle"] = cycle_idx
                    payload["pool/cycle"] = cycle_idx
                    payload["pool/size"] = len(pool)
                    run.log(payload)
                except Exception:
                    pass

        cycle_summary["pool_size"] = len(pool)
        cycle_summary["wall_sec"] = time.time() - cycle_t0
        summaries.append(cycle_summary)
        (out_root / "summaries.json").write_text(json.dumps(summaries, indent=2))
        cycle_idx += 1

    print(f"\nDone. {cycle_idx} cycles. Pool size: {len(pool)}.")
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
