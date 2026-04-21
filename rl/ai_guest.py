"""Run a trained Snake-Pong DQN as an online guest (player 2).

Usage:
    # 1) start Node server:   node server.js
    # 2) in browser, host a room and get a 5-letter code (e.g. XY3KM)
    # 3) run this script:
    python -m rl.ai_guest --code XY3KM --checkpoint rl/runs/v3/latest.pt

    # If running remotely, tunnel ws://localhost:3000 via ssh -L 3000:localhost:3000
    # Or pass --url ws://<server-ip>:3000

The Q-net observes the board from P2's egocentric view (mirrored). Actions
are mirrored back to real-board directions before sending.
"""
from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
import torch
import websockets

from .dqn import build_q_net
from .env import COLS, ROWS
from .gym_env import _mirror_action, obs_dim

# Egocentric action → real-board (dx, dy) for the LEFT player (no mirror).
ACTION_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right


def _build_obs_from_state(
    state: dict,
    as_player: int,
    target_len: int,
    interp_ball: bool = False,
    phase: int = 0,
    mult: int = 1,
) -> np.ndarray:
    """Build the same flat vector obs as `rl/gym_env.py::_build_obs`, but
    from the raw JS-shaped state dict sent over the wire.

    Args:
        target_len: snake length the model was trained on. If the live
            game uses a different length, we truncate (keep head-first) or
            pad with the tail segment to match.
        interp_ball: if True, apply phase-interpolated ball position and
            scale velocity by 1/mult (matches training-time obs for
            checkpoints trained with --interp-ball-obs).
        phase: substep phase ∈ [0, mult); 0 = ball just moved.
        mult: snake_multiplier.
    """
    mirror = as_player == 2
    own = state["s2" if as_player == 2 else "s1"]
    opp = state["s1" if as_player == 2 else "s2"]
    nx, ny = COLS - 1, ROWS - 1

    def encode_body(body):
        if len(body) >= target_len:
            body = body[:target_len]
        else:
            body = list(body) + [body[-1]] * (target_len - len(body))
        out = np.empty(target_len * 2, dtype=np.float32)
        for i, c in enumerate(body):
            x, y = c["x"], c["y"]
            xm = (nx - x) if mirror else x
            out[2 * i] = xm / nx
            out[2 * i + 1] = y / ny
        return out

    own_body = encode_body(own["body"])
    opp_body = encode_body(opp["body"])
    ball = state["ball"]

    bx_f, by_f = float(ball["x"]), float(ball["y"])
    vx_f, vy_f = float(ball["vx"]), float(ball["vy"])
    if interp_ball and mult > 1:
        frac = phase / mult
        bx_f = bx_f + frac * vx_f
        by_f = by_f + frac * vy_f
        vx_f = vx_f / mult
        vy_f = vy_f / mult

    bx_out = (nx - bx_f) if mirror else bx_f
    vx_out = -vx_f if mirror else vx_f
    out_ball = np.array(
        [bx_out / nx, by_f / ny, vx_out, vy_f],
        dtype=np.float32,
    )
    return np.concatenate([own_body, opp_body, out_ball], axis=0)


def _load_q(checkpoint: str, device: torch.device):
    """Returns (q_net, snake_length_trained, snake_multiplier_trained, interp_ball).

    Reads saved config for arch / length / multiplier / obs-format. Defaults
    match the old pre-config behavior so v3-era checkpoints still load.
    """
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "q_net" in ckpt:
        state = ckpt["q_net"]
        cfg = ckpt.get("config", {}) or {}
        snake_length = cfg.get("snake_length", 4)
        snake_multiplier = cfg.get("snake_multiplier", 1)
        arch = cfg.get("model_arch", "mlp")
        interp_ball = cfg.get("interp_ball_obs", False)
    else:
        state = ckpt
        snake_length = 4
        snake_multiplier = 1
        arch = "mlp"
        interp_ball = False
    q = build_q_net(arch, obs_dim(snake_length)).to(device).eval()
    q.load_state_dict(state)
    return q, snake_length, snake_multiplier, interp_ball


async def run(url: str, code: str, checkpoint: str, device_str: str, verbose: bool) -> None:
    device = torch.device(device_str)
    q, target_len, _target_mult, _interp = _load_q(checkpoint, device)
    last_sent: tuple[int, int] | None = None

    print(f"[ai_guest] loading model OK. connecting to {url} as guest in room {code}…")
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "join", "code": code.upper()}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type")

            if t == "error":
                print(f"[ai_guest] server error: {msg.get('reason')}")
                return
            if t == "joined":
                print("[ai_guest] joined. Waiting for host to start a game…")
                continue
            if t == "opponent_left":
                print("[ai_guest] host left. exiting.")
                return
            if t != "relay":
                continue

            payload = msg.get("payload") or {}
            if payload.get("type") != "state":
                continue
            if payload.get("phase") != "playing":
                continue
            if not (payload.get("s1") and payload.get("s2") and payload.get("ball")):
                continue

            obs = _build_obs_from_state(payload, as_player=2, target_len=target_len)
            with torch.no_grad():
                q_vals = q(torch.from_numpy(obs).unsqueeze(0).to(device))
                action = int(q_vals.argmax(dim=1).item())

            # Egocentric action → real-board action (P2 is mirrored)
            real_action = _mirror_action(action)
            dx, dy = ACTION_DELTAS[real_action]

            # Host rejects direction reversals; don't bother re-sending same dir.
            if (dx, dy) == last_sent:
                continue
            last_sent = (dx, dy)
            await ws.send(json.dumps({
                "type": "relay",
                "payload": {"type": "input", "dx": dx, "dy": dy},
            }))
            if verbose:
                print(f"[ai_guest] action={action} → dx={dx} dy={dy}  "
                      f"ball=({payload['ball']['x']},{payload['ball']['y']}) "
                      f"vxvy=({payload['ball']['vx']},{payload['ball']['vy']})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:3000",
                   help="WebSocket URL of the Node server.")
    p.add_argument("--code", required=True, help="5-letter room code from host.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args.url, args.code, args.checkpoint, args.device, args.verbose))


if __name__ == "__main__":
    main()
