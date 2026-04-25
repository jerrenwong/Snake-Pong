"""GAE, Schulman k3 KL, and misc helpers for PPO training."""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,       # (T, N) float32
    values: torch.Tensor,        # (T, N) float32 — V(obs_t)
    dones: torch.Tensor,         # (T, N) bool / float — episode ended AFTER action_t
    last_value: torch.Tensor,    # (N,) float32 — V(next_obs_{T-1})
    last_done: torch.Tensor,     # (N,) bool / float
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(λ) backward recursion. Returns (advantages, returns), both (T, N).

    Semantic:
        delta_t = r_t + γ·V(s_{t+1})·(1 - done_{t+1}) - V(s_t)
        A_t     = delta_t + γ·λ·(1 - done_{t+1})·A_{t+1}
        return_t = A_t + V(s_t)

    Where `done_{t+1}` is the "what happened after taking action_t" mask (the
    next slot's done). For t=T-1 we use `last_value` / `last_done`. Truncation
    should NOT be treated as a terminal (bootstrap through it) — the caller
    is expected to only flag environmental terminations in `dones`.
    """
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    dones_f = dones.to(rewards.dtype)
    last_done_f = last_done.to(rewards.dtype)

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            next_nonterminal = 1.0 - last_done_f
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones_f[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        adv[t] = lastgaelam

    returns = adv + values
    return adv, returns


def approx_kl_k3(logratio: torch.Tensor) -> torch.Tensor:
    """Schulman k3 unbiased estimator: E[(ratio - 1) - log ratio].

    logratio = new_logp - old_logp, so ratio = exp(logratio). Always ≥ 0 in
    expectation; small means the policy moved little in this minibatch.
    """
    ratio = logratio.exp()
    return ((ratio - 1.0) - logratio).mean()


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """1 - Var(returns - values) / Var(returns). 1.0 is perfect fit, 0.0 is
    no better than predicting the mean. Can go negative if worse than mean.
    """
    var_y = returns.var(unbiased=False)
    if var_y.item() < 1e-12:
        return float("nan")
    return float(1.0 - (returns - values).var(unbiased=False) / var_y)


def linear_lr(update_idx: int, total_updates: int, lr0: float) -> float:
    """Linear anneal from lr0 → 0 across total_updates.

    update_idx is 0-based; at the LAST update (update_idx = total_updates - 1)
    the returned lr is 1 / total_updates · lr0 (small but nonzero).
    """
    frac = 1.0 - (update_idx / max(1, total_updates))
    return lr0 * frac
