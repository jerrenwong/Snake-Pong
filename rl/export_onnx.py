"""Export a trained Q-network checkpoint to ONNX for in-browser inference.

Usage:
    python -m rl.export_onnx \
        --checkpoint rl/runs/v9/play_h4_final.pt \
        --head 4 \
        --out models/snake-pong-v9-h4.onnx

For bootstrapped nets you MUST pass --head to select a specific head. The
resulting ONNX model has signature (1, obs_dim) → (1, n_actions).

Metadata (snake_length, snake_multiplier, interp_ball_obs) is baked into
the filename only; the JS loader reads a sibling JSON sidecar with the
same stem for those values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .gym_env import obs_dim
from .models import (
    BootstrappedDuelingQNetwork,
    BootstrappedQNetwork,
    DuelingQNetwork,
    IndependentEnsembleQNetwork,
    QNetwork,
    build_q_net,
)
from .ppo_model import build_actor_critic


def _load_ppo_actor(ckpt: dict, cfg: dict) -> nn.Module:
    """Load a PPO ActorCritic checkpoint and return just the actor wrapped as
    a (B, D) → (B, A) module. Argmax of logits = greedy policy action, which
    is what the in-browser loader uses.
    """
    L = cfg.get("snake_length", 4)
    hidden = cfg.get("hidden_size", 64)
    ac = build_actor_critic(obs_dim=L * 4 + 4, hidden=hidden).eval()
    ac.load_state_dict(ckpt.get("ac") or ckpt["q_net"])
    for p_ in ac.parameters():
        p_.requires_grad_(False)
    return ac.actor.eval()


def extract_single_head_model(base: nn.Module, head_idx: int) -> nn.Module:
    """Build an equivalent *single-head* module that computes only head `head_idx`.

    Serializing the full bootstrapped net wastes weights for the 4 unused heads
    (≈80% of parameter memory). We instead extract the trunk + the one head and
    wrap it as a fresh `QNetwork` / `DuelingQNetwork`. Weights are copied by
    reference; no retraining.
    """
    if isinstance(base, BootstrappedQNetwork):
        out = QNetwork(obs_dim=1, n_actions=base.n_actions)
        # Replace `out.net` with trunk + chosen head, reusing original modules.
        trunk = base.trunk                 # Linear-ReLU-Linear-ReLU
        head = base.heads[head_idx]        # Linear-ReLU-Linear
        out.net = nn.Sequential(*trunk, *head)
        return out.eval()

    if isinstance(base, BootstrappedDuelingQNetwork):
        out = DuelingQNetwork(obs_dim=1, n_actions=base.n_actions)
        out.trunk = base.trunk
        out.value_head = base.value_heads[head_idx]
        out.adv_head = base.adv_heads[head_idx]
        return out.eval()

    if isinstance(base, IndependentEnsembleQNetwork):
        out = QNetwork(obs_dim=1, n_actions=base.n_actions)
        out.net = base.nets[head_idx]
        return out.eval()

    raise TypeError(f"Unsupported multi-head arch for extraction: {type(base).__name__}")


class MeanReduceWrapper(nn.Module):
    """Wrap a (B, K, A) multi-head model so it returns (B, A) = mean over heads.

    Used for `--reduce mean` export: we keep all K heads' weights and average
    their Q-values at inference time, matching the "mean" reduction policy
    used by vec_tournament / play_server when no specific head is chosen.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.base(obs)
        if q.dim() == 3:
            return q.mean(dim=1)
        return q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--head", type=int, default=None,
                   help="Which head to export. For multi-head archs pass either --head N or --reduce mean.")
    p.add_argument("--reduce", choices=["mean"], default=None,
                   help="Reduction over heads (multi-head archs only). Currently only 'mean' is supported.")
    p.add_argument("--out", required=True, help="Output .onnx path.")
    args = p.parse_args()
    if args.head is not None and args.reduce is not None:
        raise SystemExit("Pass at most one of --head / --reduce.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["q_net"]
    cfg = ckpt.get("config", {}) or {}
    arch = cfg.get("model_arch", "mlp")
    L = cfg.get("snake_length", 4)
    mult = cfg.get("snake_multiplier", 1)
    interp = cfg.get("interp_ball_obs", False)
    hidden = cfg.get("hidden_size", 256)
    n_heads = cfg.get("n_heads", 5)

    is_multi = arch in ("bootstrapped", "bootstrapped_dueling", "independent_ensemble")
    is_ppo = arch == "ppo_ac"
    if is_multi and args.head is None and args.reduce is None:
        raise SystemExit(
            f"Checkpoint has arch={arch} (multi-head). "
            f"Pass --head N (0..{n_heads - 1}) or --reduce mean."
        )
    if is_ppo and (args.head is not None or args.reduce is not None):
        raise SystemExit("PPO actor is single-output; don't pass --head or --reduce.")

    D = obs_dim(L)
    if is_ppo:
        model = _load_ppo_actor(ckpt, cfg)
        head_desc = "_actor"
    else:
        q = build_q_net(arch, D, n_heads=n_heads, hidden=hidden).eval()
        q.load_state_dict(state)
        for p_ in q.parameters():
            p_.requires_grad_(False)

        if is_multi:
            if args.head is not None:
                model = extract_single_head_model(q, args.head)
                head_desc = f"_h{args.head}"
            else:
                model = MeanReduceWrapper(q).eval()
                head_desc = "_mean"
        else:
            model = q
            head_desc = ""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export. Use the legacy tracer (`dynamo=False`) so all initializers are
    # inlined into the .onnx file — the dynamo/torch.export path defaults to
    # writing weights to a sibling `.onnx.data` file, which complicates serving.
    dummy = torch.zeros(1, D, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["obs"],
        output_names=["q"],
        dynamic_axes={"obs": {0: "batch"}, "q": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Sidecar JSON with metadata
    sidecar = out_path.with_suffix(".json")
    meta = {
        "source_checkpoint": args.checkpoint,
        "arch": arch,
        "head": args.head,
        "reduce": args.reduce,
        "snake_length": L,
        "snake_multiplier": mult,
        "interp_ball_obs": interp,
        "hidden_size": hidden,
        "obs_dim": D,
        "n_actions": 4,
        "input_name": "obs",
        "output_name": "q",
        "description": f"{arch}{head_desc}, L={L}, mult={mult}",
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    print(f"Wrote {sidecar}")

    # Verify by running ONNX model once
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        test_obs = torch.zeros(1, D, dtype=torch.float32).numpy()
        out = sess.run(["q"], {"obs": test_obs})[0]
        py_out = model(torch.from_numpy(test_obs)).detach().numpy()
        err = abs(out - py_out).max()
        print(f"Verification: ONNX vs PyTorch max|Δ| = {err:.6f}")
        if err > 1e-4:
            print("WARNING: large discrepancy — export may be incorrect.")
    except ImportError:
        print("onnxruntime not installed; skipping verification.")


if __name__ == "__main__":
    main()
