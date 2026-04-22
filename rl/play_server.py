"""All-in-one server: static files + WebSocket relay + built-in AI guest.

Replaces the Node `server.js` for the purpose of playing against the trained
DQN agent. The user opens the served HTML, clicks "Online" → "Host", and the
server auto-joins as the guest (P2) using the provided checkpoint. No second
human is needed.

Usage:
    python -m rl.play_server --checkpoint rl/runs/v3/latest.pt --port 3000

Then in a browser (tunnel the port if remote):
    http://localhost:3000

Protocol matches src/network.js / server.js:
- {"type":"host"}          host-client creates a room
- {"type":"hosted",code}   server replies with a room code
- server auto-joins as guest, forwards state/input through the relay
- {"type":"relay",payload} mutually forwarded between host and (AI) guest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import random
import string
from pathlib import Path

import numpy as np
import torch
from aiohttp import WSMsgType, web

from .ai_guest import ACTION_DELTAS, _build_obs_from_state, _load_q
from .gym_env import _mirror_action


SNAKE_PONG_DIR = Path(__file__).resolve().parents[1]


class GameRoom:
    def __init__(self, code: str):
        self.code = code
        self.host_ws: web.WebSocketResponse | None = None
        self.last_sent_input: tuple[int, int] | None = None
        # Phase tracking: last ball (x, y) seen; if unchanged at next tick,
        # phase increments; on change, resets to 0.
        self.last_ball_xy: tuple[int, int] | None = None
        self.phase: int = 0
        # Bootstrapped DQN: which head this room uses (fixed for room lifetime)
        self.active_head: int | None = None


def gen_code(existing: set[str]) -> str:
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        code = "".join(random.choices(chars, k=5))
        if code not in existing:
            return code


async def serve_static(request: web.Request) -> web.Response:
    path = request.match_info.get("path", "") or "index.html"
    # prevent traversal
    path = path.replace("..", "")
    full = SNAKE_PONG_DIR / path
    if not full.exists() or not full.is_file():
        return web.Response(status=404, text="Not found")
    ctype, _ = mimetypes.guess_type(str(full))
    return web.FileResponse(full, headers={"Content-Type": ctype or "application/octet-stream"})


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    app = request.app
    rooms: dict[str, GameRoom] = app["rooms"]
    model_specs: list[dict] = app["model_specs"]
    device: torch.device = app["device"]
    verbose: bool = app["verbose"]
    room: GameRoom | None = None

    async for msg in ws:
        if msg.type != WSMsgType.TEXT:
            continue
        try:
            data = json.loads(msg.data)
        except Exception:
            continue
        t = data.get("type")

        if t == "host":
            code = gen_code(set(rooms.keys()))
            room = GameRoom(code)
            room.host_ws = ws
            rooms[code] = room
            # Priority: explicit client-supplied head > server head_mode > None (mean).
            client_head = data.get("head")
            head_mode = request.app["head_mode"]
            source = None
            if client_head is not None:
                try:
                    room.active_head = int(client_head)
                    source = f"client(head={room.active_head})"
                except (ValueError, TypeError):
                    pass
            if room.active_head is None:
                if head_mode == "random_per_room":
                    for spec in model_specs:
                        m = spec["model"]
                        k = getattr(m, "n_heads", None)
                        if k is not None and k > 1:
                            room.active_head = random.randrange(k)
                            source = f"random_per_room(head={room.active_head})"
                            break
                elif head_mode.startswith("fixed_"):
                    try:
                        room.active_head = int(head_mode.split("_", 1)[1])
                        source = f"fixed(head={room.active_head})"
                    except Exception:
                        room.active_head = 0
                        source = "fixed(default=0)"
            if source is None:
                source = "mean"

            await ws.send_json({"type": "hosted", "code": code})
            await ws.send_json({"type": "guest_joined"})
            print(f"[server] hosted room {code} — AI guest joined ({source})")

        elif t == "join":
            # Shouldn't happen in this setup (the AI is the only guest).
            await ws.send_json({"type": "error", "reason": "AI guest server — host only."})

        elif t == "relay":
            # Host is sending us a message; we're both the relay and the AI guest.
            if room is None:
                continue
            payload = data.get("payload") or {}
            if payload.get("type") == "state":
                await handle_state(room, payload, model_specs, device, verbose)
            elif payload.get("type") in ("pause", "resume"):
                # No-op for AI — nothing to pause on guest side.
                pass

    # close: clean up room
    if room is not None:
        rooms.pop(room.code, None)
        print(f"[server] room {room.code} closed.")
    return ws


def _pick_model(specs: list[dict], actual_len: int, actual_mult: int) -> dict:
    """Pick the model spec whose (length, mult) best matches the live game.
    Exact match preferred; otherwise nearest in L1 distance over (length, mult).
    """
    # Exact match
    for s in specs:
        if s["length"] == actual_len and s["mult"] == actual_mult:
            return s
    # Nearest by (|Δlen|, |Δmult|)
    return min(
        specs,
        key=lambda s: (abs(s["length"] - actual_len), abs(s["mult"] - actual_mult)),
    )


def _infer_mult(payload: dict) -> int:
    """snake_multiplier = ballTickMs / tickMs (rounded)."""
    tick = payload.get("tickMs") or 0
    ball = payload.get("ballTickMs") or 0
    if tick <= 0:
        return 1
    return max(1, int(round(ball / tick)))


async def handle_state(
    room: GameRoom,
    payload: dict,
    model_specs: list[dict],
    device: torch.device,
    verbose: bool,
) -> None:
    if payload.get("phase") != "playing":
        return
    if not (payload.get("s1") and payload.get("s2") and payload.get("ball")):
        return

    actual_len = len(payload["s2"].get("body", []))
    actual_mult = _infer_mult(payload)
    spec = _pick_model(model_specs, actual_len, actual_mult)
    q_net = spec["model"]

    # Phase tracking: detect if the ball moved since the last state message.
    ball = payload["ball"]
    xy = (ball["x"], ball["y"])
    if room.last_ball_xy is None or xy != room.last_ball_xy:
        room.phase = 0
    else:
        room.phase = min(room.phase + 1, max(1, actual_mult) - 1)
    room.last_ball_xy = xy

    obs = _build_obs_from_state(
        payload, as_player=2, target_len=spec["length"],
        interp_ball=spec["interp_ball"], phase=room.phase, mult=actual_mult,
    )
    with torch.no_grad():
        q_vals = q_net(torch.from_numpy(obs).unsqueeze(0).to(device))
        # Bootstrapped nets output (1, K, A) — pick per room.active_head, else mean.
        if q_vals.dim() == 3:
            if room.active_head is not None and 0 <= room.active_head < q_vals.size(1):
                q_vals = q_vals[:, room.active_head, :]
            else:
                q_vals = q_vals.mean(dim=1)
        action = int(q_vals.argmax(dim=1).item())
    real_action = _mirror_action(action)
    dx, dy = ACTION_DELTAS[real_action]

    if (dx, dy) == room.last_sent_input:
        return
    room.last_sent_input = (dx, dy)
    if room.host_ws is not None and not room.host_ws.closed:
        await room.host_ws.send_json({
            "type": "relay",
            "payload": {"type": "input", "dx": dx, "dy": dy},
        })
        if verbose:
            print(f"[ai] game(len={actual_len},mult={actual_mult},phase={room.phase}) "
                  f"→ model({spec['label']}: len={spec['length']},mult={spec['mult']},"
                  f"interp={spec['interp_ball']}) action={action} dx={dx} dy={dy}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Checkpoint path. Pass multiple times to enable length-based routing: "
                        "--checkpoint rl/runs/v3/latest.pt --checkpoint rl/runs/v4/checkpoint_iter78000.pt")
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--head-mode", type=str, default="random_per_room",
                   help="For Bootstrapped nets: 'mean' (avg all heads), "
                        "'random_per_room' (random head per room, fixed for lifetime), "
                        "or 'fixed_N' for a specific head index.")
    args = p.parse_args()

    device = torch.device(args.device)
    model_specs: list[dict] = []
    for ckpt_path in args.checkpoint:
        q_net, trained_len, trained_mult, interp_ball = _load_q(ckpt_path, device)
        model_specs.append({
            "path": ckpt_path,
            "label": ckpt_path,
            "length": trained_len,
            "mult": trained_mult,
            "interp_ball": interp_ball,
            "model": q_net,
        })
        print(f"[server] loaded {ckpt_path} → length={trained_len} mult={trained_mult} interp_ball={interp_ball}")

    app = web.Application()
    app["rooms"] = {}
    app["model_specs"] = model_specs
    app["head_mode"] = args.head_mode
    app["device"] = device
    app["verbose"] = args.verbose

    # JS client opens ws://host/ (no path), so root must handle both HTTP and WS.
    app.router.add_get("/", root_handler)
    app.router.add_get("/{path:.*}", serve_static)

    print(f"[server] serving Snake-Pong at http://{args.host}:{args.port}")
    print(f"[server] loaded {len(model_specs)} model(s):")
    for s in model_specs:
        print(f"           length={s['length']} mult={s['mult']}  {s['path']}")
    print(f"[server] device: {device}")
    print(f"[server] open the URL, click Online → Host; the AI joins automatically.")
    web.run_app(app, host=args.host, port=args.port, print=None)


async def root_handler(request: web.Request) -> web.StreamResponse:
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await ws_handler(request)
    return web.FileResponse(SNAKE_PONG_DIR / "index.html")


if __name__ == "__main__":
    main()
