# Snake Pong

A two-player game combining Snake and Pong. Each player controls a snake that acts as their paddle — maneuver your snake to deflect the ball past your opponent. Includes a trained AI opponent with five difficulty tiers.

## Credits

Thanks to **Mehmet Can Bastemir** for improvement suggestions and for bravely fighting the AI snakes.

## Running Locally

**Clone the repo:**

```bash
git clone https://github.com/jerrenwong/Snake-Pong.git
cd Snake-Pong
```

The page uses ES modules and `fetch()`, which browsers block on `file://`.
You need to serve the folder over HTTP — any static server works.

**Option 1 — Python 3** (almost always installed):

```bash
python3 -m http.server 8000
```

**Option 2 — Node.js** (if you have `npx` available):

```bash
npx --yes http-server -p 8000 .
```

Then open **<http://localhost:8000/>** in any modern browser (Chrome, Firefox, Safari, Edge).

The first click on the AI button briefly pulls `onnxruntime-web` from a CDN (cached after the first load). After that the game runs fully offline.

Two other ports of call:

- `npm install && node server.js` — starts a WebSocket relay for online play on top of the same static server (the page itself works the same way).
- `python -m rl.play_server --checkpoint <path>` — the training-era server that also serves the page but plays as the AI opponent itself (useful when prototyping new checkpoints).

## How to Play

### Controls

| Player | Left | Right | Up | Down |
|--------|------|-------|----|------|
| P1 (blue, left side) | `A` | `D` | `W` | `S` |
| P2 (orange, right side) | `←` | `→` | `↑` | `↓` |

**ESC** — pause / resume

### Objective

Deflect the ball so it exits through your opponent's side. First to reach the Win Score wins.

### Losing a point

- The ball exits through **your side** (you failed to deflect it)
- Your snake hits the **outer wall**, the **center divider**, or **itself**
- Your snake **collides head-on** with the opponent's snake

When a snake dies mid-rally the opposing player scores a point and a new round starts automatically.

## Settings (⚙ gear icon)

| Setting | Effect |
|---------|--------|
| **Snake Length** | Starting length of each snake (3–20 segments) |
| **Ball Speed** | Ball ticks per second (slider 1–10) |
| **Snake Speed** | Snake moves 1×, 2×, or 3× per ball tick |
| **Win Score** | Points needed to win the match (1–20) |
| **Power-ups** | Toggle optional power-up mode ON / OFF |

## Power-ups (optional)

Enable in Settings. When active, colored orbs spawn on the field every ~15 seconds on average (5s cooldown after each spawn, then geometric distribution with 10s expected wait). Collect a power-up by moving your snake's head onto it.

| Symbol | Color | Effect | Duration |
|--------|-------|--------|----------|
| **+5** | Green | Your snake grows 5 cells longer | 10s |
| **⚡** | Yellow | Ball moves 2× faster while in your opponent's half | 10s |
| **>>** | Pink | Your snake moves 2× faster | 10s |

A timer bar at the top corner of the canvas shows the remaining duration for each active effect. If you collect the same power-up type while it's already active, the timer resets.

## Rules

- Snakes cannot reverse 180° — only left/right turns relative to current direction.
- The ball bounces off the top/bottom walls and off any segment of either snake.
- The center divider is lethal to snakes; the ball passes through it freely.
- Power-up effects expire when the timer runs out; snake length returns to pre-boost size on expiry.

## Project Structure

```
snake_pong/
├── index.html        — layout, CSS, HTML
└── src/
    ├── constants.js  — grid dimensions, wall positions
    ├── logic.js      — snake & ball physics, collision detection
    ├── renderer.js   — canvas drawing (background, snakes, ball, power-ups, effect bars)
    ├── input.js      — keyboard event → semantic callbacks
    ├── audio.js      — Web Audio API chiptune BGM + SFX (no external files)
    ├── powerups.js   — power-up definitions, spawn logic
    ├── ai_local.js   — onnxruntime-web inference + JS-side safety filter
    ├── ai_local_obs.js — bit-identical port of the Python obs builder
    └── main.js       — game loop, state machine, UI wiring
```

---

## AI Opponent

The single-player mode runs a trained policy entirely in your browser via
[onnxruntime-web](https://github.com/microsoft/onnxruntime). The menu offers
five difficulty tiers — **EASY**, **MEDIUM**, **HARD**, **MASTER**,
**INSANE** — each backed by a different checkpoint produced by ~20 hours of
training experiments described below. A safety filter applied client-side
prevents the AI from ever voluntarily walking into a wall, its own body, or
the opponent (illegal actions are masked before argmax).

## Training Journey

The project's training code lives in `rl/`. Six generations of agents were
trained, each addressing a specific failure mode of the previous.

### DQN era (v3 → v11)

We started with vanilla DQN, then iterated through Dueling, Bootstrapped
(K parallel Q-heads sharing a trunk), Bootstrapped Dueling, and finally
Independent Ensemble (K full Q-nets, no shared trunk). Each variant patched
a specific issue:

- **v3 → v5 (Dueling)**: separate value & advantage heads, marginal lift.
- **v8 (Bootstrapped Dueling)**: K=5 heads with bootstrap masks for
  uncertainty-aware exploration.
- **v9 (Bootstrapped, head 4 picked)**: longer training, single-head export
  for tournament use.
- **v11 (Independent Ensemble, mean reduction)**: 5 fully independent nets
  averaged at inference. The mean-policy was the strongest DQN we got, but
  it stalemated against itself in self-play (mutual-defense plateau), and
  later snapshots regressed under FIFO opponent-pool rotation
  (catastrophic forgetting).

What we learned: DQN self-play with a small opponent pool converges to
mutually defensive policies that never score on each other, capping skill
at "good defender" rather than "good attacker."

### PPO self-play (v12)

Switched to PPO following the
[37 implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
checklist. Separate actor/critic MLPs (Tanh, orthogonal init), bf16 autocast
on Ampere, GAE(λ=0.95), clipped value loss, per-minibatch advantage norm,
KL-based early stop. 16 384 vec envs × 128 rollout steps × 5 000 updates ≈
2.5 h on a 4090. v12@1400 was the first checkpoint to sweep every prior
DQN baseline 100 % — it's the model behind the **HARD** tier.

### Iterated PPO experts (v13)

Six independent fresh-init runs against a fixed top-3 opponent set
(strongest v12 / v11 / v9). Each run trained until it beat all three at
≥ 95 % win rate. The result: a small ensemble of diverse experts seeded
later tournaments and stress-tested specific tactical weaknesses.

### bf16 + safety-filtered opponents (v14)

Added bf16 autocast to the actor / critic forward in both rollout and
update phases (~2× speedup on the update phase) and applied the same
in-browser safety filter to every opponent during training. The result is
a policy trained against opponents that never voluntarily suicide — closer
to actual deployment conditions and harder to exploit cheaply. v14_final
became the **MASTER** tier.

### King-of-the-hill PPO (v15)

Strict-100 %-wins promotion criterion against a 14-member pool spanning
v5 → v14. Each new snake is trained from scratch; only added to the pool
if it beats every member 100 % over a fresh greedy tournament. Three
attempts, none passed strict, but **snake 0 won the post-run all-vs-all
at +80.7 % net** — better than v14_final (+77.6 %) head-to-head. This is
the **INSANE** tier.

## Tournament & Tier Selection

`rl/big_tournament.py` runs an all-vs-all bracket on the GPU via
`rl/vec_tournament.py`. It batches `M × M × games_per_pair` games into one
`TorchVectorSnakePongGame` instance and groups action selection by model so
each ONNX runs once per slot per step. The same safety filter the browser
applies is enabled on both sides for fairness. The five checkpoints chosen
for the tier picker correspond to clear skill plateaus on that ranking.

## Engineering: sync-free GPU rollout

The first PPO rollout was bottlenecked by host-device syncs in the inner
loop (`int(mask.sum().item())`, `mask.any()` used as control flow,
per-step `.cpu().numpy()` for episode bookkeeping). The refactor in
`rl/ppo_rollout.py`:

- Records episode-end events into preallocated GPU tensors during the loop
  and bulk-transfers once at the end.
- Uses `torch.where`-driven resets (always O(N)) instead of indexed writes
  that need a sync to compute their size.
- Same change applied to `TorchVectorSnakePongGame._init_mask`.

Net result: ~26× rollout-throughput improvement at v12's config and
genuinely sync-free interaction between rollout and the rest of the loop.

## In-browser inference

`src/ai_local.js` loads an ONNX policy via onnxruntime-web (WASM backend
from a CDN), runs it in the page, and feeds actions into the existing local
game engine. Two subtleties:

- **Obs parity.** `src/ai_local_obs.js` rebuilds the egocentric, mirrored,
  phase-interpolated observation vector that training used. A parity test
  (`rl/parity_test.py` and the JS test under `rl/tests/`) verifies the JS
  obs is bit-identical to the Python builder.
- **Safety filter.** `_legalActionsP2` in `src/ai_local.js` checks the
  immediate consequence of each of the four ego actions and masks any that
  would land the AI's head in a wall, its own body, or the opponent. The
  argmax then picks among the survivors. The same logic is implemented
  GPU-batched (`_safety_mask_ego_subset` in `rl/vec_tournament.py`) and
  applied symmetrically during training rollouts and tournament evaluation.

## Repo layout (training)

```
rl/
├── env.py             — single-game numpy snake-pong sim
├── env_torch.py       — N-vector GPU sim (TorchVectorSnakePongGame)
├── gym_env.py         — gymnasium wrapper, obs builder
├── models.py          — DQN architectures (mlp, dueling, bootstrapped, ensemble)
├── ppo_model.py       — PPO actor/critic
├── ppo_rollout.py     — sync-free on-policy GPU rollout
├── ppo_train.py       — PPO trainer (v12, v14)
├── ppo_iterated.py    — iterated experts trainer (v13)
├── ppo_hill.py        — king-of-the-hill trainer (v15)
├── selfplay.py        — opponent pool, scripted bots, benchmark set
├── eval.py            — single-env evaluator
├── vec_tournament.py  — GPU-batched all-vs-all bracket
├── big_tournament.py  — leaderboard runner across all checkpoints
└── export_onnx.py     — checkpoint → ONNX (single head / mean reduce / PPO actor)
```
