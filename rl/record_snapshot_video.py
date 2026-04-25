"""Record a safety-filtered video of a single snapshot against a chosen
opponent, save as a GIF next to the checkpoint.

Usage:
    python -m rl.record_snapshot_video \
        --checkpoint rl/runs/v16/snap_u1500.pt \
        --out rl/runs/v16/videos/u1500.gif

By default the opponent is uniform-random (with safety filter) so videos
are comparable across snapshots. Use --opponent <path> to pit against
another checkpoint instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .big_tournament import _load_generic
from .gym_env import obs_dim
from .ppo_model import build_actor_critic
from .render import record_episode, save_gif


def _build_learner_q_fn(ckpt_path: str, device: torch.device):
    """Return a callable: np.ndarray(obs) -> np.ndarray(4 logits/Q)."""
    model = _load_generic(ckpt_path, device)

    @torch.no_grad()
    def q_fn(obs):
        t = torch.from_numpy(obs).unsqueeze(0).to(device)
        out = model(t)
        if out.dim() == 3:
            out = out.mean(dim=1)
        return out[0].detach().cpu().numpy()
    return q_fn


def _random_q_fn():
    """Uniform random: returns the same Q for every action, so after safety
    mask the argmax is over whichever legal action comes first — equivalent
    to the env's random opponent but safety-filtered."""
    def q_fn(obs):
        return np.zeros(4, dtype=np.float32)
    return q_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Snapshot to play as learner.")
    p.add_argument("--opponent", default=None,
                   help="Optional opponent checkpoint. Defaults to safety-filtered random.")
    p.add_argument("--out", required=True, help="Output .gif path.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--snake-length", type=int, default=4)
    p.add_argument("--snake-multiplier", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--interp-ball-obs", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--learner-side", default="random")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    device = torch.device(args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    learner_q = _build_learner_q_fn(args.checkpoint, device)
    opp_q = _build_learner_q_fn(args.opponent, device) if args.opponent else _random_q_fn()

    # Dummy int-returning callables (record_episode requires them even when
    # safety_filter=True — they're kept as a fallback for the unmasked path).
    def learner_int(obs):
        return int(np.asarray(learner_q(obs)).argmax())
    def opp_int(obs):
        return int(np.asarray(opp_q(obs)).argmax())

    frames, info = record_episode(
        learner_int, opp_int,
        snake_length=args.snake_length,
        snake_multiplier=args.snake_multiplier,
        max_steps=args.max_steps,
        interp_ball=args.interp_ball_obs,
        seed=args.seed,
        learner_side=args.learner_side,
        safety_filter=True,
        learner_q_fn=learner_q,
        opponent_q_fn=opp_q,
    )
    save_gif(frames, args.out, fps=args.fps)
    print(f"wrote {args.out}  ({len(frames)} frames, info={info})")


if __name__ == "__main__":
    main()
