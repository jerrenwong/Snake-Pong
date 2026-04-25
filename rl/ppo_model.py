"""Actor-critic networks for PPO.

Follows Huang et al. "The 37 Implementation Details of PPO" recommendations for
state-vector MLP envs:
  - Separate actor and critic (no shared trunk).
  - Tanh activations.
  - Orthogonal init: gain=sqrt(2) for hidden layers, 0.01 for policy head
    (keeps step-1 action dist near uniform so KL doesn't explode), 1.0 for
    value head. All biases zero.

Shapes:
  forward(obs: (B, D)) on Actor   → logits (B, A)
  forward(obs: (B, D)) on Critic  → value  (B, 1)

Actor's (B, A) forward is drop-in compatible with the existing DQN opponent
plumbing (`rl/vec_rollout_gpu._batched_q_actions_gpu`) — argmax of logits is
the deterministic PPO policy action, so tournaments and opponent-pool
rollouts Just Work.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical


N_ACTIONS = 4
HIDDEN = 64


def _orth_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    """MLP policy: obs → logits over discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = HIDDEN):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            _orth_init(nn.Linear(obs_dim, hidden), math.sqrt(2)), nn.Tanh(),
            _orth_init(nn.Linear(hidden, hidden), math.sqrt(2)), nn.Tanh(),
            _orth_init(nn.Linear(hidden, n_actions), 0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """MLP value net: obs → V(s)."""

    def __init__(self, obs_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            _orth_init(nn.Linear(obs_dim, hidden), math.sqrt(2)), nn.Tanh(),
            _orth_init(nn.Linear(hidden, hidden), math.sqrt(2)), nn.Tanh(),
            _orth_init(nn.Linear(hidden, 1), 1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Container holding separate Actor + Critic. Single optimizer runs over
    both sets of parameters (via ActorCritic.parameters()).
    """

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = HIDDEN):
        super().__init__()
        self.actor = Actor(obs_dim, n_actions, hidden)
        self.critic = Critic(obs_dim, hidden)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action, log_prob, value)."""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(obs)

    @torch.no_grad()
    def act_greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic argmax action — for evaluation and opponent play."""
        return self.actor(obs).argmax(dim=-1)

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor,
        legal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log_prob, entropy, value for given (obs, action) batch.
        Used inside the PPO update loop to form the ratio / entropy / value
        loss terms. Gradients flow through the returned tensors.

        If `legal` (bool mask, shape (..., n_actions)) is provided, illegal
        actions are set to −inf before building the Categorical so the
        recomputed distribution matches the masked sampling distribution used
        during rollout. When an entry has *no* legal action (fully trapped)
        the raw logits are kept so Categorical stays valid.
        """
        logits = self.actor(obs)
        if legal is not None:
            any_legal = legal.any(dim=-1, keepdim=True)
            masked = logits.masked_fill(~legal, float("-inf"))
            logits = torch.where(any_legal, masked, logits)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs)
        return log_prob, entropy, value


class ActorOpponentAdapter(nn.Module):
    """Thin wrapper making an ActorCritic plug into DQN-era opponent plumbing.

    `rl/vec_rollout_gpu._batched_q_actions_gpu` expects a module whose
    `forward(obs)` returns `(B, A)` values that it then argmaxes. Logits work
    just as well — argmax(logits) == argmax(softmax(logits)). We also strip
    any `n_heads` attribute so callers know this is a single-output module.
    """

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.ac.actor(obs)


def build_actor_critic(obs_dim: int, hidden: int = HIDDEN) -> ActorCritic:
    return ActorCritic(obs_dim=obs_dim, n_actions=N_ACTIONS, hidden=hidden)
