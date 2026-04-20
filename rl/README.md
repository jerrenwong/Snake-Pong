# Snake-Pong RL (self-play DQN)

Trains a DQN agent to play Snake-Pong against copies of itself (self-play).

## Layout

- `env.py` — headless Python port of `src/logic.js` (no power-ups, 1-tick-per-step).
- `gym_env.py` — single-agent `gymnasium` wrapper. Observations are
  *egocentric* (board is mirrored for player 2) so one policy plays both sides.
- `dqn.py` — CNN+MLP Q-network, replay buffer, Double-DQN loss.
- `selfplay.py` — opponent pool: frozen snapshots + random fallback.
- `train.py` — training loop.
- `play.py` — evaluate / visualize a checkpoint.

## Install

Requires Python 3.10+, `torch`, `gymnasium`, `numpy`.

```bash
pip install torch gymnasium numpy
```

## Train

```bash
python -m rl.train --iters 500 --episodes-per-iter 20 --out-dir rl/runs/v1
```

Useful knobs:

- `--device cuda|cpu`
- `--snapshot-every N` — snapshot to opponent pool every N iterations
- `--pool-size N` — max frozen opponents kept
- `--opponent-random-prob P` — probability of sampling a uniform-random
  opponent instead of a snapshot (helps against self-play collapse)
- `--epsilon-decay-steps N` — linear ε schedule over env steps

Training writes `train_log.jsonl`, periodic checkpoints, and `latest.pt` to
`--out-dir`.

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
