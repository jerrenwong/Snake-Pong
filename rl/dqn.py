"""Backward-compat re-exports. The actual code now lives in:
  rl.models    — QNetwork, DuelingQNetwork, BootstrappedQNetwork,
                 BootstrappedDuelingQNetwork, build_q_net, N_ACTIONS
  rl.replay    — Transition, ReplayBuffer, GpuReplayBuffer
  rl.losses    — compute_loss, compute_loss_bootstrapped
  rl.actions   — greedy_action, epsilon_greedy_action, obs_to_tensor

Old code doing `from rl.dqn import ...` continues to work.
"""
from .actions import epsilon_greedy_action, greedy_action, obs_to_tensor
from .losses import compute_loss, compute_loss_bootstrapped
from .models import (
    BootstrappedDuelingQNetwork,
    BootstrappedQNetwork,
    DuelingQNetwork,
    N_ACTIONS,
    QNetwork,
    build_q_net,
)
from .replay import GpuReplayBuffer, ReplayBuffer, Transition

__all__ = [
    "N_ACTIONS",
    "QNetwork", "DuelingQNetwork",
    "BootstrappedQNetwork", "BootstrappedDuelingQNetwork",
    "build_q_net",
    "Transition", "ReplayBuffer", "GpuReplayBuffer",
    "compute_loss", "compute_loss_bootstrapped",
    "greedy_action", "epsilon_greedy_action", "obs_to_tensor",
]
