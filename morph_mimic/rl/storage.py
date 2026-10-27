"""Rollout storage for the mixed reference/goal-reaching PPO (dual critics).

Each env belongs statically to one regime (reference-following or
goal-reaching). GAE uses the env's own critic; advantages are normalized
separately per regime (paper Stage 3 description).
"""

from __future__ import annotations

import torch


class RolloutStorage:
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        is_ref: torch.Tensor,
        critic_m_dim: int = 353,
        critic_fs_dim: int = 298,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.is_ref = is_ref.to(device)  # (E,) bool

        s = (num_steps, num_envs)
        self.obs = torch.zeros((*s, obs_dim), device=device)
        self.critic_obs_m = torch.zeros((*s, critic_m_dim), device=device)
        self.critic_obs_fs = torch.zeros((*s, critic_fs_dim), device=device)
        self.actions = torch.zeros((*s, action_dim), device=device)
        self.action_means = torch.zeros((*s, action_dim), device=device)
        self.action_stds = torch.ones((*s, action_dim), device=device)
        self.has_policy_statistics = False
        self.log_probs = torch.zeros(*s, device=device)
        self.rewards = torch.zeros(*s, device=device)
        self.dones = torch.zeros(*s, device=device)
        self.values_m = torch.zeros(*s, device=device)
        self.values_fs = torch.zeros(*s, device=device)
        self.returns = torch.zeros(*s, device=device)
        self.advantages = torch.zeros(*s, device=device)
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values_m: torch.Tensor,
        values_fs: torch.Tensor,
        critic_obs_m: torch.Tensor,
        critic_obs_fs: torch.Tensor,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ):
        i = self.step
        self.obs[i] = obs
        self.critic_obs_m[i] = critic_obs_m
        self.critic_obs_fs[i] = critic_obs_fs
        self.actions[i] = actions
        if action_mean is not None and action_std is not None:
            self.action_means[i] = action_mean
            self.action_stds[i] = action_std
            self.has_policy_statistics = True
        self.log_probs[i] = log_probs
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.values_m[i] = values_m
        self.values_fs[i] = values_fs
        self.step += 1

    def compute_returns(self, last_values_m: torch.Tensor, last_values_fs: torch.Tensor,
                        gamma: float, lam: float):
        """GAE per regime; timeout rewards already include pre-reset bootstrap.

        Every reset remains a GAE boundary, including time limits, so advantages
        cannot propagate from the next independently initialized episode.
        """
        advantage = torch.zeros(self.num_envs, device=self.device)
        next_m = last_values_m
        next_fs = last_values_fs
        is_ref = self.is_ref
        for i in reversed(range(self.num_steps)):
            done = self.dones[i]
            v = torch.where(is_ref, self.values_m[i], self.values_fs[i])
            nv = torch.where(is_ref, next_m, next_fs)
            delta = self.rewards[i] + gamma * nv * (1.0 - done.float()) - v
            advantage = delta + gamma * lam * (1.0 - done.float()) * advantage
            self.advantages[i] = advantage
            next_m = self.values_m[i]
            next_fs = self.values_fs[i]
        self.returns = self.advantages + torch.where(
            is_ref.unsqueeze(0), self.values_m, self.values_fs
        )
        # separate advantage normalization per regime (Stage 3)
        for mask in (is_ref, ~is_ref):
            adv = self.advantages[:, mask]
            if adv.numel() == 0:
                continue
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            self.advantages[:, mask] = adv

    def minibatch_indices(self, num_minibatches: int):
        """Yield flat indices for minibatches over the full (steps x envs) buffer."""
        size = self.num_steps * self.num_envs
        if size < 1 or num_minibatches < 1:
            raise ValueError("Rollout size and minibatch count must be positive")
        num_minibatches = min(size, num_minibatches)
        idx = torch.randperm(size, device=self.device)
        batch_size = size // num_minibatches
        for i in range(num_minibatches):
            start, end = i * batch_size, (i + 1) * batch_size if i < num_minibatches - 1 else size
            mb = idx[start:end]
            env_i = mb % self.num_envs
            yield mb, env_i
