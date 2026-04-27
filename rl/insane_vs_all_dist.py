"""For each opponent tier (easy, medium, hard, master, insane, boss),
self-play INSANE vs that opponent and record the rank-ordered action
probabilities for INSANE only (softmax over LEGAL actions).

Output: a single PNG with rows = opponent, cols = rank-1..4.

Usage:
    .venv/bin/python rl/insane_vs_all_dist.py --games 200 --plot rl/insane_vs_all.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Import env.py without the rl/__init__.py side-effects.
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

MIRROR_ACTION = np.array([0, 1, 3, 2], dtype=np.int64)
TIERS = ["easy", "medium", "hard", "master", "insane", "boss"]


def build_obs(game, as_player: int, meta: dict) -> np.ndarray:
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
            out[2 * i]     = xm / nx
            out[2 * i + 1] = y / ny
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
    tail = np.array([bx_o / nx, by / ny, vx_o, vy], dtype=np.float32)
    return np.concatenate([own_b, opp_b, tail], axis=0)


def legal_actions(game, as_player: int) -> np.ndarray:
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


def load_tier(tier: str, models_dir: Path):
    onnx_path = models_dir / f"snake-pong-{tier}.onnx"
    meta_path = models_dir / f"snake-pong-{tier}.json"
    meta = json.loads(meta_path.read_text())
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess, meta


def greedy_legal_action(sess, meta, game, as_player: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Returns (real_engine_action, sorted_legal_probs_desc, legal_mask).
    Uses safety-mask + argmax for action selection (matches in-browser play).
    """
    obs = build_obs(game, as_player, meta)[None, :]
    logits = sess.run([meta["output_name"]], {meta["input_name"]: obs})[0][0]
    legal = legal_actions(game, as_player)
    if legal.any():
        masked = np.where(legal, logits, -np.inf)
        probs = softmax(masked)
        legal_idx = np.where(legal)[0]
        sorted_legal_probs = np.sort(probs[legal_idx])[::-1]
        a_ego = int(legal_idx[np.argmax(probs[legal_idx])])
    else:
        sorted_legal_probs = np.zeros(0)
        a_ego = int(np.argmax(logits))
    real_a = int(MIRROR_ACTION[a_ego]) if as_player == 2 else a_ego
    return real_a, sorted_legal_probs, legal


def play_matchup(insane, opp, opp_name, n_games, max_steps, base_seed):
    """Play `insane` (P2) vs `opp` (P1) and collect per-step rank-ordered probs
    for INSANE's decisions only. Returns dict[rank] -> list[float]."""
    insane_sess, insane_meta = insane
    opp_sess, opp_meta = opp
    L = insane_meta["snake_length"]
    M = insane_meta["snake_multiplier"]

    rank_probs = {r: [] for r in range(4)}
    legal_hist = np.zeros(5, dtype=np.int64)
    total_decisions = 0

    rng = np.random.default_rng(base_seed)
    for g in range(n_games):
        game = SnakePongGame(
            snake_length=L, snake_multiplier=M,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        for _ in range(max_steps):
            # P1 = opponent, P2 = INSANE.
            a1, _, _ = greedy_legal_action(opp_sess, opp_meta, game, 1)
            a2, sorted_probs, legal = greedy_legal_action(
                insane_sess, insane_meta, game, 2,
            )
            n_legal = int(legal.sum())
            legal_hist[n_legal] += 1
            total_decisions += 1
            for r, pv in enumerate(sorted_probs):
                rank_probs[r].append(float(pv))
            game.step(a1, a2)
            if game.done:
                break
    return rank_probs, legal_hist, total_decisions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--plot", default="rl/insane_vs_all.png")
    p.add_argument(
        "--opponents", nargs="+", default=TIERS,
        help="Which tiers to use as opponent (P1). INSANE always plays P2.",
    )
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    insane = load_tier("insane", models_dir)

    matchups = {}
    for opp_name in args.opponents:
        print(f"  loading {opp_name} …")
        opp = load_tier(opp_name, models_dir)
        print(f"  playing INSANE vs {opp_name.upper()} ({args.games} games)…")
        rank_probs, legal_hist, total = play_matchup(
            insane, opp, opp_name, args.games, args.max_steps,
            args.seed + hash(opp_name) % 10000,
        )
        matchups[opp_name] = {
            "rank_probs": rank_probs,
            "legal_hist": legal_hist,
            "total": total,
        }

    # ── Text summary ────────────────────────────────────────────────────────
    print()
    print(f"{'opponent':>8}  {'decisions':>10}  "
          + "  ".join(f"{'top'+str(r+1)+'_mean':>12}" for r in range(4)))
    for name in args.opponents:
        d = matchups[name]
        means = []
        for r in range(4):
            v = d["rank_probs"][r]
            means.append(np.mean(v) if v else float("nan"))
        print(f"{name:>8}  {d['total']:>10d}  "
              + "  ".join(f"{m:>12.4f}" for m in means))

    print()
    print("Legal-action count split (% of decisions):")
    print(f"{'opponent':>8}  " + "  ".join(f"{k+'leg':>8}" for k in ["0","1","2","3","4"]))
    for name in args.opponents:
        h = matchups[name]["legal_hist"]
        tot = max(1, int(h.sum()))
        pcts = [100.0 * int(h[k]) / tot for k in range(5)]
        print(f"{name:>8}  " + "  ".join(f"{v:>7.2f}%" for v in pcts))

    # ── Plot ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(args.opponents)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 2.6 * n_rows + 1.2),
                             sharex=True)
    if n_rows == 1:
        axes = axes[None, :]
    bins = np.linspace(0.0, 1.0, 51)

    for i, name in enumerate(args.opponents):
        d = matchups[name]
        for r in range(4):
            ax = axes[i, r]
            vals = np.array(d["rank_probs"][r], dtype=np.float64)
            if vals.size:
                ax.hist(vals, bins=bins, color="#3a7", edgecolor="#125", alpha=0.85)
                ax.axvline(vals.mean(), color="#c33", lw=1.2,
                           label=f"μ={vals.mean():.3f}")
                ax.axvline(np.median(vals), color="#36c", lw=1.2, ls="--",
                           label=f"med={np.median(vals):.3f}")
                ax.legend(fontsize=7, loc="upper center")
            ax.set_yscale("log")
            ax.set_xlim(0, 1)
            if i == 0:
                ax.set_title(f"Rank {r+1}", fontsize=11)
            if r == 0:
                ax.set_ylabel(f"INSANE vs\n{name.upper()}\n(n={d['total']})",
                              fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("softmax prob (legal-only)", fontsize=9)

    fig.suptitle(
        f"INSANE action probability per rank, by opponent  "
        f"({args.games} games each, log-y)",
        y=0.995, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.plot, dpi=140)
    print(f"\nSaved → {args.plot}")


if __name__ == "__main__":
    main()
