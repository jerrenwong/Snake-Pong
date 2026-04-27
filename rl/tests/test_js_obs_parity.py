"""Verify that the JS in-browser obs builder produces identical observations
to the Python training-time obs builder.

Generates random game states (randomized snake bodies, ball, phase), writes
them to a temp JSON file, runs a Node subprocess to compute the JS obs, and
compares elementwise to the Python reference.

Run with:
    python -m rl.tests.test_js_obs_parity
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from rl.env import COLS, ROWS


def _build_obs_from_state(
    state: dict,
    as_player: int,
    target_len: int,
    interp_ball: bool,
    phase: int,
    mult: int,
) -> np.ndarray:
    """Build the egocentric observation from a serialized game state.

    Mirrors the in-browser JS encoder: own snake first (mirrored on x for P2),
    then opponent snake, then ball (x, y, vx, vy) with optional sub-step
    interpolation when `interp_ball` is True and `mult > 1`.
    """
    nx, ny = COLS - 1, ROWS - 1
    mirror = (as_player == 2)
    own = state["s2"]["body"] if mirror else state["s1"]["body"]
    opp = state["s1"]["body"] if mirror else state["s2"]["body"]

    def encode(body):
        body = list(body)
        if len(body) >= target_len:
            body = body[:target_len]
        else:
            body = body + [body[-1]] * (target_len - len(body))
        out = np.empty(target_len * 2, dtype=np.float32)
        for i, p in enumerate(body):
            x, y = int(p["x"]), int(p["y"])
            xm = (nx - x) if mirror else x
            out[2 * i]     = xm / nx
            out[2 * i + 1] = y / ny
        return out

    own_b = encode(own)
    opp_b = encode(opp)
    ball = state["ball"]
    bx, by = float(ball["x"]), float(ball["y"])
    vx, vy = float(ball["vx"]), float(ball["vy"])
    if interp_ball and mult > 1:
        frac = phase / mult
        bx += frac * vx
        by += frac * vy
        vx /= mult
        vy /= mult
    bx_o = (nx - bx) if mirror else bx
    vx_o = -vx if mirror else vx
    tail = np.array([bx_o / nx, by / ny, vx_o, vy], dtype=np.float32)
    return np.concatenate([own_b, opp_b, tail], axis=0)

TESTS_DIR = Path(__file__).parent
JS_RUNNER = TESTS_DIR / "parity_js_obs.mjs"

# Match the checkpoint we ship for in-browser play.
META = {
    "snake_length": 4,
    "snake_multiplier": 2,
    "interp_ball_obs": True,
}


def _random_body(rng: np.random.Generator, length: int) -> list[dict]:
    # Any contiguous walk in bounds; doesn't need to be a legal game position
    # — we just care that both obs encoders handle the exact coordinates the
    # same way.
    hx = int(rng.integers(1, COLS - 1))
    hy = int(rng.integers(1, ROWS - 1))
    dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    body = []
    for i in range(length):
        body.append({"x": int(max(0, min(COLS - 1, hx - i * dx))),
                     "y": int(max(0, min(ROWS - 1, hy - i * dy)))})
    return body


def _random_case(rng: np.random.Generator) -> dict:
    L = META["snake_length"]
    mult = META["snake_multiplier"]
    s1 = {"body": _random_body(rng, L)}
    s2 = {"body": _random_body(rng, L)}
    ball = {
        "x": int(rng.integers(0, COLS)),
        "y": int(rng.integers(0, ROWS)),
        "vx": int(rng.choice([-1, 1])),
        "vy": int(rng.choice([-1, 1])),
    }
    phase = int(rng.integers(0, mult))
    return {"s1": s1, "s2": s2, "ball": ball, "phase": phase, "meta": META}


def _py_obs(case: dict) -> np.ndarray:
    state = {"s1": case["s1"], "s2": case["s2"], "ball": case["ball"]}
    return _build_obs_from_state(
        state, as_player=2,
        target_len=META["snake_length"],
        interp_ball=META["interp_ball_obs"],
        phase=case["phase"],
        mult=META["snake_multiplier"],
    )


def main() -> int:
    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        print("SKIP: node not available on PATH")
        return 0

    rng = np.random.default_rng(0)
    n_cases = 200
    cases = [_random_case(rng) for _ in range(n_cases)]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cases, f)
        cases_path = f.name

    try:
        proc = subprocess.run(
            [node, str(JS_RUNNER), cases_path],
            capture_output=True, text=True, check=False,
        )
    finally:
        Path(cases_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        print("node runner failed:")
        print(proc.stderr)
        return 1

    js_lines = proc.stdout.strip().split("\n")
    assert len(js_lines) == n_cases, f"expected {n_cases} lines, got {len(js_lines)}"

    max_diff = 0.0
    worst_idx = -1
    mismatches = 0
    for i, (case, line) in enumerate(zip(cases, js_lines)):
        py = _py_obs(case)
        js = np.array([float(x) for x in line.split()], dtype=np.float64)
        diff = np.abs(py.astype(np.float64) - js).max()
        if diff > max_diff:
            max_diff = diff
            worst_idx = i
        if diff > 1e-6:
            mismatches += 1
            if mismatches <= 3:
                print(f"  case {i}: max|Δ|={diff:.3e}")
                print(f"    py: {py.tolist()}")
                print(f"    js: {js.tolist()}")

    print(f"Ran {n_cases} parity cases. max|Δ| = {max_diff:.3e} (case {worst_idx}).")
    if mismatches > 0:
        print(f"FAIL: {mismatches} cases with |Δ| > 1e-6")
        return 1
    print("OK: JS obs exactly matches Python obs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
