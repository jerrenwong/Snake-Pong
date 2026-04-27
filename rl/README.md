# Snake-Pong RL (self-play PPO)

Trains a PPO actor-critic to play Snake-Pong against copies of itself
(self-play), with a reservoir-sampled opponent pool and a frozen benchmark
set for eval. Trained policies are exported to ONNX and served in-browser
as the tiered AI ladder (`easy → medium → hard → master → insane → boss`).

## Layout

### Core env + networks
- `env.py` — headless Python port of `src/logic.js` (no power-ups, 1-tick-per-step).
- `env_torch.py` — GPU-vectorized env (N parallel games on device).
- `gym_env.py` — single-agent `gymnasium` wrapper. Observations are
  *egocentric* (board is mirrored for player 2) so one policy plays both sides.
- `actions.py` — single-obs action helpers (`greedy_action`, `epsilon_greedy_action`).
- `models.py` — Q-network architectures (`QNetwork`, `DuelingQNetwork`,
  `BootstrappedQNetwork`, `IndependentEnsembleQNetwork`, `build_q_net`).
- `ppo_model.py` — `ActorCritic` network and adapters.
- `render.py` — PIL frame renderer + episode-to-gif recorder.

### Training
- `ppo_train.py` — primary PPO trainer (self-play + benchmark eval + wandb).
- `ppo_iterated.py` — iterated-PPO variant (frozen-opponent stages).
- `ppo_hill.py` — hill-climbing PPO variant.
- `ppo_rollout.py` — GPU-resident PPO rollout buffer.
- `ppo_utils.py` — GAE, KL approximations, LR scheduling.
- `selfplay.py` — reservoir-sampled opponent pool + frozen benchmark set.
- `eval.py` — single-env evaluation utilities (used for benchmark evals).
- `vec_rollout_gpu.py` — GPU obs-batch and Q-action helpers (shared by PPO
  rollout, det-eval, and tournament).

### Evaluation / tournaments
- `vec_tournament.py` — vectorized round-robin tournament runner.
- `big_tournament.py` — multi-checkpoint tournament with mixed-arch loading.
- `det_eval.py` — deterministic evaluator (forced initial conditions).
- `parity_test.py` — verifies Python env matches the JS engine.
- `record_snapshot_video.py` — render gifs of arbitrary checkpoints.

### Analysis
- `insane_vs_all_dist.py` — INSANE action-distribution sweep vs every tier.
- `insane_randomized_tournament.py` — round-robin of "randomized INSANE"
  agents (with prob *p* the agent queries the policy on a y-flipped board).

### Deployment
- `export_onnx.py` — export checkpoints to ONNX + JSON metadata for
  in-browser inference (`models/snake-pong-<tier>.onnx`).

## Install

Requires Python 3.10+, `torch`, `gymnasium`, `numpy`, `Pillow`, `imageio`,
`wandb`, `onnx`, `onnxruntime`.

```bash
pip install torch gymnasium numpy Pillow imageio wandb onnx onnxruntime
```

## Train

```bash
export WANDB_API_KEY=<your_key>
python -m rl.ppo_train --iters 500 --out-dir rl/runs/v17
```

For local-only / no-network runs:

```bash
python -m rl.ppo_train --wandb-mode disabled --out-dir rl/runs/local
# or
WANDB_MODE=offline python -m rl.ppo_train --out-dir rl/runs/offline
```

Useful knobs (full list via `python -m rl.ppo_train --help`):

- `--device cuda|cpu`
- `--snapshot-every N` — snapshot to opponent pool every N iterations
- `--pool-size N` — reservoir pool capacity
- `--opponent-random-prob P` — probability of sampling a uniform-random
  opponent instead of a snapshot (helps against self-play collapse)
- `--eval-every N` / `--eval-episodes K` — benchmark evaluation cadence
- `--video-every N` — upload sample gifs every N iters

Training writes `train_log.jsonl`, periodic checkpoints, and `latest.pt`
to `--out-dir`.

## Catastrophic-forgetting mitigations

Self-play can "forget" how to beat early-training strategies when the
opponent pool rolls over. We mitigate this two ways:

1. **Reservoir-sampled opponent pool** (`selfplay.py::OpponentPool`). Every
   historical snapshot has equal probability of remaining in the pool,
   regardless of age. One slot is reserved for the most recent snapshot.
2. **Frozen benchmark set** (`selfplay.py::BenchmarkSet`). Immutable
   opponents: `random`, `snap_first`, `snap_25/50/75`, rolling
   `snap_latest`. A drop in `eval/snap_first/win_rate` over time is the
   forgetting signal.

## Tournaments / evaluation

Round-robin between checkpoints:

```bash
python -m rl.big_tournament --checkpoints rl/runs/v17/snap_*.pt --games 200
```

Deterministic head-to-head with forced initial conditions:

```bash
python -m rl.det_eval --a rl/runs/v17/best.pt --b rl/runs/v16/best.pt
```

Randomized-INSANE perturbation analysis (uses ONNX models in `models/`):

```bash
python rl/insane_randomized_tournament.py --games 1000 --p-values 0.0 0.1 0.2 0.3 0.4 0.5
```

## Export to ONNX

```bash
python -m rl.export_onnx --checkpoint rl/runs/v17/best.pt \
                         --tier insane --out-dir models/
```

Produces `models/snake-pong-<tier>.onnx` and `models/snake-pong-<tier>.json`.

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

## Action

Discrete(4): `0=up, 1=down, 2=left, 3=right`. Reversals are ignored (as in
the game).
