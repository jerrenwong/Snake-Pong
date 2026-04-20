# Snake-Pong RL (self-play DQN)

Trains a DQN agent to play Snake-Pong against copies of itself (self-play).

## Layout

- `env.py` — headless Python port of `src/logic.js` (no power-ups, 1-tick-per-step).
- `gym_env.py` — single-agent `gymnasium` wrapper. Observations are
  *egocentric* (board is mirrored for player 2) so one policy plays both sides.
- `dqn.py` — MLP Q-network, replay buffer, Double-DQN loss.
- `selfplay.py` — reservoir-sampled opponent pool + fixed benchmark set.
- `eval.py` — evaluation utilities (used for benchmark evals).
- `render.py` — PIL frame renderer + episode-to-gif recorder.
- `train.py` — training loop with wandb integration.
- `play.py` — evaluate / visualize a checkpoint (ANSI or GIF).

## Install

Requires Python 3.10+, `torch`, `gymnasium`, `numpy`, `Pillow`, `imageio`,
`wandb`.

```bash
pip install torch gymnasium numpy Pillow imageio wandb
```

## Train

Set your wandb API key and start training:

```bash
export WANDB_API_KEY=<your_key>
python -m rl.train --iters 500 --episodes-per-iter 20 --out-dir rl/runs/v1
```

For local-only / no-network runs:

```bash
python -m rl.train --wandb-mode disabled --out-dir rl/runs/local
# or
WANDB_MODE=offline python -m rl.train --out-dir rl/runs/offline
```

Useful knobs:

- `--device cuda|cpu`
- `--snapshot-every N` — snapshot to opponent pool every N iterations
- `--pool-size N` — reservoir pool capacity (default 30)
- `--opponent-random-prob P` — probability of sampling a uniform-random
  opponent instead of a snapshot (helps against self-play collapse)
- `--epsilon-decay-steps N` — linear ε schedule over env steps
- `--eval-every N` / `--eval-episodes K` — benchmark evaluation cadence
- `--video-every N` — upload sample gifs every N iters
- `--video-opponents a,b,c` — which benchmarks to film (default
  `random,snap_latest`)

Training writes `train_log.jsonl`, periodic checkpoints, and `latest.pt`
to `--out-dir`, and (when wandb is enabled) logs metrics, uploads sample
gifs, and registers checkpoint artifacts.

## Catastrophic-forgetting mitigations

Self-play can "forget" how to beat early-training strategies when the
opponent pool rolls over. We mitigate this two ways:

1. **Reservoir-sampled opponent pool** (`selfplay.py::OpponentPool`). Every
   historical snapshot has equal probability of remaining in the pool,
   regardless of age — so training never fully stops seeing old styles.
   One slot is reserved for the most recent snapshot so the learner is
   always challenged at its current skill level. Sampling during matches
   is uniform over the pool.
2. **Frozen benchmark set** (`selfplay.py::BenchmarkSet`). A separate,
   immutable collection of opponents: `random`, `snap_first`, `snap_25`,
   `snap_50`, `snap_75` (captured at 25/50/75% training milestones), and
   rolling `snap_latest`. Every `--eval-every` iters we run
   `--eval-episodes` episodes against each and log win/loss/draw rates.
   **A drop in `eval/snap_first/win_rate` over time is the signal that
   forgetting is happening.**

## Play / evaluate

Self-play (default opponent = same checkpoint, no exploration):

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt --episodes 100
```

Versus uniform-random opponent (sanity check):

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt --vs-random --episodes 200
```

Render in terminal:

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt --render --sleep 0.08
```

Save a gif of one episode:

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt --render-gif out.gif
```

Versus a different checkpoint:

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt \
                  --vs-checkpoint rl/runs/v1/checkpoint_iter00100.pt \
                  --episodes 200
```

## Play / evaluate

Self-play (default opponent = same checkpoint, no exploration):

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt --episodes 100
```

Versus uniform-random opponent (sanity check):

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt --vs-random --episodes 200
```

Render in terminal:

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt --render --sleep 0.08
```

Versus a different checkpoint:

```bash
python -m rl.play --checkpoint rl/runs/v1/latest.pt \
                  --vs-checkpoint rl/runs/v1/checkpoint_iter00100.pt \
                  --episodes 200
```

## Reward

Zero-sum terminal reward:

- `+1` learner scored (opponent's ball miss, or opponent died)
- `-1` learner was scored on (learner's miss, or learner died)
- `0` both died, or episode truncated

No shaping. Episode ends on any score / death or after `--max-steps`.

## Observation

Per-agent egocentric flat vector (board mirrored for player 2):

- Own snake body, head-first, as `(x, y)` pairs normalized to `[0, 1]`
  (`snake_length * 2` values).
- Opponent body, same format (`snake_length * 2` values).
- Ball: `(x, y, vx, vy)` — position normalized, velocity in `{-1, 0, 1}`.

Total dimension: `4 * snake_length + 4` (= 20 for default `snake_length=4`).
Snake length is fixed (no power-ups), so no padding / masking needed. The
Q-network is a 3-hidden-layer MLP.

## Action

Discrete(4): `0=up, 1=down, 2=left, 3=right`. Reversals are ignored (as in
the game).
