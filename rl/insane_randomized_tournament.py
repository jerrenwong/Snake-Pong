"""Randomized-INSANE round-robin tournament.

A "randomized INSANE" agent with parameter p plays the INSANE policy, but on
each decision it has a probability p of consulting the policy on a y-flipped
(top/bottom-mirrored) observation and remapping the chosen action back. The
policy is otherwise unchanged — same ONNX session, just fed a flipped obs and
its output translated through VFLIP_ACTION = [1, 0, 2, 3] (UP↔DOWN).

Tournament: agents for p ∈ {0.1, 0.2, 0.3, 0.4, 0.5}, every pair plays
`--games` games (default 100), with sides alternated each game for fairness.

Usage:
    .venv/bin/python rl/insane_randomized_tournament.py --games 100
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

_env_spec = importlib.util.spec_from_file_location(
    "_snake_env", str(Path(__file__).parent / "env.py"),
)
_env_mod = importlib.util.module_from_spec(_env_spec)
sys.modules["_snake_env"] = _env_mod
_env_spec.loader.exec_module(_env_mod)
ACTION_DELTAS = _env_mod.ACTION_DELTAS
COLS = _env_mod.COLS
ROWS = _env_mod.ROWS
SnakePongGame = _env_mod.SnakePongGame

MIRROR_ACTION = np.array([0, 1, 3, 2], dtype=np.int64)   # x-flip: L↔R
VFLIP_ACTION  = np.array([1, 0, 2, 3], dtype=np.int64)   # y-flip: U↔D

P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]


def build_obs(game, as_player: int, meta: dict, yflip: bool = False) -> np.ndarray:
    L = meta["snake_length"]
    mult = meta["snake_multiplier"]
    interp = meta["interp_ball_obs"]
    nx, ny = COLS - 1, ROWS - 1
    mirror = (as_player == 2)
    own = game.s2 if as_player == 2 else game.s1
    opp = game.s1 if as_player == 2 else game.s2

    def encode(body):
        if len(body) >= L:
            body = body[:L]
        else:
            body = list(body) + [body[-1]] * (L - len(body))
        out = np.empty(L * 2, dtype=np.float32)
        for i, (x, y) in enumerate(body):
            xm = (nx - x) if mirror else x
            ym = (ny - y) if yflip else y
            out[2 * i]     = xm / nx
            out[2 * i + 1] = ym / ny
        return out

    own_b = encode(own.body)
    opp_b = encode(opp.body)
    bx, by = float(game.ball.x), float(game.ball.y)
    vx, vy = float(game.ball.vx), float(game.ball.vy)
    if interp and mult > 1:
        frac = game.phase / mult
        bx += frac * vx
        by += frac * vy
        vx /= mult
        vy /= mult
    bx_o = (nx - bx) if mirror else bx
    vx_o = -vx if mirror else vx
    by_o = (ny - by) if yflip else by
    vy_o = -vy if yflip else vy
    tail = np.array([bx_o / nx, by_o / ny, vx_o, vy_o], dtype=np.float32)
    return np.concatenate([own_b, opp_b, tail], axis=0)


def legal_actions_real(game, as_player: int) -> np.ndarray:
    """Legal mask in the engine-action frame (post-mirror, post-vflip)."""
    own = game.s2 if as_player == 2 else game.s1
    opp = game.s1 if as_player == 2 else game.s2
    head_x, head_y = own.body[0]
    cur_dx, cur_dy = own.dx, own.dy
    self_blocked = set(own.body[:-1])
    opp_blocked = set(opp.body)
    legal = np.zeros(4, dtype=bool)
    for ego in range(4):
        real_a = MIRROR_ACTION[ego] if as_player == 2 else ego
        rdx, rdy = ACTION_DELTAS[real_a]
        if rdx == -cur_dx and rdy == -cur_dy:
            rdx, rdy = cur_dx, cur_dy
        nx_, ny_ = head_x + rdx, head_y + rdy
        if nx_ < 0 or nx_ >= COLS or ny_ < 0 or ny_ >= ROWS:
            continue
        if (nx_, ny_) in self_blocked or (nx_, ny_) in opp_blocked:
            continue
        legal[ego] = True
    return legal


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_insane(models_dir: Path):
    onnx_path = models_dir / "snake-pong-insane.onnx"
    meta_path = models_dir / "snake-pong-insane.json"
    meta = json.loads(meta_path.read_text())
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess, meta


def randomized_action(sess, meta, game, as_player: int, p: float, rng) -> int:
    """Randomized-INSANE: with prob p, query policy on y-flipped obs and remap."""
    yflip = (rng.random() < p)
    obs = build_obs(game, as_player, meta, yflip=yflip)[None, :]
    logits = sess.run([meta["output_name"]], {meta["input_name"]: obs})[0][0]

    legal_real = legal_actions_real(game, as_player)
    if yflip:
        # Action a in the flipped frame corresponds to VFLIP[a] in the real frame.
        legal_query = legal_real[VFLIP_ACTION]
    else:
        legal_query = legal_real

    if legal_query.any():
        masked = np.where(legal_query, logits, -np.inf)
        a_query = int(np.argmax(masked))
    else:
        a_query = int(np.argmax(logits))

    a_ego = int(VFLIP_ACTION[a_query]) if yflip else a_query
    real_a = int(MIRROR_ACTION[a_ego]) if as_player == 2 else a_ego
    return real_a


def play_game(sess, meta, p_a, p_b, a_is_p1, max_steps, seed):
    """One game between agent A (param p_a) and agent B (param p_b).
    Returns 'A' / 'B' / 'D'."""
    rng = np.random.default_rng(seed)
    L = meta["snake_length"]
    M = meta["snake_multiplier"]
    game = SnakePongGame(snake_length=L, snake_multiplier=M, seed=seed)

    for _ in range(max_steps):
        if a_is_p1:
            a1 = randomized_action(sess, meta, game, 1, p_a, rng)
            a2 = randomized_action(sess, meta, game, 2, p_b, rng)
        else:
            a1 = randomized_action(sess, meta, game, 1, p_b, rng)
            a2 = randomized_action(sess, meta, game, 2, p_a, rng)
        res = game.step(a1, a2)
        if game.done:
            scorer = res.scorer
            if scorer is None or scorer == 0:
                return "D"
            a_won = (scorer == 1) if a_is_p1 else (scorer == 2)
            return "A" if a_won else "B"
    return "D"


def play_matchup(sess, meta, p_a, p_b, n_games, max_steps, base_seed):
    a_w = b_w = d = 0
    for g in range(n_games):
        a_is_p1 = (g % 2 == 0)
        res = play_game(sess, meta, p_a, p_b, a_is_p1, max_steps, base_seed + g)
        if res == "A":
            a_w += 1
        elif res == "B":
            b_w += 1
        else:
            d += 1
    return a_w, b_w, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--p-values", type=float, nargs="+", default=P_VALUES)
    args = ap.parse_args()

    sess, meta = load_insane(Path(args.models_dir))
    ps = args.p_values
    n = len(ps)

    wins   = np.zeros((n, n), dtype=np.int32)
    losses = np.zeros((n, n), dtype=np.int32)
    draws  = np.zeros((n, n), dtype=np.int32)

    print(f"Randomized-INSANE round-robin: p ∈ {ps}, {args.games} games per pair\n")
    for i in range(n):
        for j in range(i + 1, n):
            seed = args.seed + 1000 * (i + 1) + j
            a_w, b_w, dr = play_matchup(
                sess, meta, ps[i], ps[j], args.games, args.max_steps, seed,
            )
            wins[i, j],   wins[j, i]   = a_w, b_w
            losses[i, j], losses[j, i] = b_w, a_w
            draws[i, j],  draws[j, i]  = dr, dr
            print(f"  p={ps[i]:.1f} vs p={ps[j]:.1f}: "
                  f"{a_w}-{b_w}-{dr}  (W-L-D for p={ps[i]:.1f})")

    # Win-rate matrix (W / (W+L), draws excluded).
    print("\n=== Win-rate matrix (row vs col, draws excluded) ===")
    header = "       " + "  ".join(f"p={p:.1f}" for p in ps) + "    total"
    print(header)
    totals_w = wins.sum(axis=1)
    totals_l = losses.sum(axis=1)
    totals_d = draws.sum(axis=1)
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("  --  ")
            else:
                w, l = wins[i, j], losses[i, j]
                wr = w / (w + l) if (w + l) else float("nan")
                row.append(f"{wr:5.2f} ")
        tw, tl, td = totals_w[i], totals_l[i], totals_d[i]
        twr = tw / (tw + tl) if (tw + tl) else float("nan")
        print(f"p={ps[i]:.1f}  " + "  ".join(row)
              + f"   {tw}W-{tl}L-{td}D  ({twr:.2%})")

    # Win-loss-draw counts table.
    print("\n=== Raw W-L-D per matchup (row vs col) ===")
    print("       " + "  ".join(f"  p={p:.1f}   " for p in ps))
    for i in range(n):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("   --    ")
            else:
                cells.append(f"{wins[i,j]:>2}-{losses[i,j]:>2}-{draws[i,j]:>2}")
        print(f"p={ps[i]:.1f}  " + "  ".join(cells))

    # Final standings.
    print("\n=== Standings (by total wins, draws excluded) ===")
    order = sorted(range(n), key=lambda i: (-totals_w[i], totals_l[i]))
    print(f"{'rank':>4}  {'p':>5}  {'W':>5}  {'L':>5}  {'D':>5}  {'win%':>7}")
    for rank, i in enumerate(order, 1):
        tw, tl, td = totals_w[i], totals_l[i], totals_d[i]
        twr = tw / (tw + tl) if (tw + tl) else float("nan")
        print(f"{rank:>4}  {ps[i]:>5.1f}  {tw:>5}  {tl:>5}  {td:>5}  {twr:>6.2%}")


if __name__ == "__main__":
    main()
