"""PPO update with dual critics, KL-adaptive learning rate, and warmup.

Implements the paper's Table S6 settings:
- clip range 0.2, value-loss coefficient 1.0, max grad norm 1.0
- 5 epochs, 8 minibatches per epoch
- tracker + reference-following critic: lr 1e-4, KL-adaptive (target 0.02/0.01)
- goal-reaching critic: fixed lr 1e-3, warmup 500 iterations (no actor
  updates from goal samples, no value clipping during warmup)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.distributions as D
import torch.nn as nn

from morph_mimic.rl.storage import RolloutStorage


@dataclass
class PPOCfg:
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    epochs: int = 5
    num_minibatches: int = 8
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    entropy_coef: float = 0.005
    target_kl: float = 0.01
    init_lr: float = 1e-4
    kl_min_lr: float = 1e-6
    kl_max_lr: float = 1e-2
    goal_critic_lr: float = 1e-3
    warmup_iterations: int = 500
    init_noise_std: float = 1.0
    min_noise_std: float = 0.01
    max_noise_std: float = 1.0


class PPO:
    def __init__(
        self,
        actor: nn.Module,
        critic_m: nn.Module,
        critic_fs: nn.Module,
        cfg: PPOCfg,
        device: torch.device,
    ):
        self.actor = actor
        self.critic_m = critic_m
        self.critic_fs = critic_fs
        self.cfg = cfg
        self.device = device
        if not (0 < cfg.min_noise_std <= cfg.init_noise_std <= cfg.max_noise_std):
            raise ValueError("Noise standard deviations must satisfy 0 < min <= initial <= max")
        self.log_std = nn.Parameter(
            torch.full((23,), torch.log(torch.tensor(cfg.init_noise_std)), device=device)
        )
        self.actor_lr = cfg.init_lr
        self.critic_m_lr = cfg.init_lr
        self.critic_fs_lr = cfg.goal_critic_lr

        params = list(actor.parameters()) + list(critic_m.parameters()) + [self.log_std]
        self.opt_main = torch.optim.Adam(params, lr=cfg.init_lr, eps=1e-8)
        self.opt_fs = torch.optim.Adam(critic_fs.parameters(), lr=cfg.goal_critic_lr, eps=1e-8)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def constrain_noise(self):
        """Project the parameter so clamping cannot strand it with zero gradients."""
        self.log_std.clamp_(math.log(self.cfg.min_noise_std), math.log(self.cfg.max_noise_std))

    def act(self, obs: torch.Tensor):
        with torch.no_grad():
            mean = self.actor(obs)
            std = self.log_std.exp().clamp(self.cfg.min_noise_std, self.cfg.max_noise_std)
            dist = D.Normal(mean, std.unsqueeze(0))
            self.action_mean = mean
            self.action_std = std.unsqueeze(0).expand_as(mean)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std = self.log_std.exp().clamp(self.cfg.min_noise_std, self.cfg.max_noise_std)
        return D.Normal(mean, std.unsqueeze(0))

    # ------------------------------------------------------------------
    def update(self, storage: RolloutStorage, iteration: int) -> dict:
        cfg = self.cfg
        obs = storage.obs.flatten(0, 1)
        critic_obs_m = storage.critic_obs_m.flatten(0, 1)
        critic_obs_fs = storage.critic_obs_fs.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        old_log_probs = storage.log_probs.flatten(0, 1)
        advantages = storage.advantages.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        is_ref = storage.is_ref.repeat(storage.num_steps)  # (S*E,)
        warmup = iteration < cfg.warmup_iterations
        if storage.has_policy_statistics:
            old_means = storage.action_means.flatten(0, 1)
            old_stds = storage.action_stds.flatten(0, 1)
        else:
            # Compatibility for callers that collect with an unchanged actor but
            # have not yet recorded distribution parameters in storage.
            with torch.no_grad():
                old_dist = self.evaluate(obs)
                old_means, old_stds = old_dist.loc.detach(), old_dist.scale.detach()

        totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        count = 0
        parameters = list(self.actor.parameters()) + list(self.critic_m.parameters()) + list(self.critic_fs.parameters()) + [self.log_std]
        for _ in range(cfg.epochs):
            for mb, env_i in storage.minibatch_indices(cfg.num_minibatches):
                mb_ref = is_ref[mb]
                dist = self.evaluate(obs[mb])
                new_log_probs = dist.log_prob(actions[mb]).sum(-1)
                ratio = (new_log_probs - old_log_probs[mb]).exp()
                adv = advantages[mb]
                surr1, surr2 = ratio * adv, ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * adv
                # goal samples excluded from the actor loss during warmup
                actor_mask = torch.ones_like(adv)
                if warmup:
                    actor_mask[~mb_ref] = 0.0
                policy_loss = -torch.min(surr1, surr2) * actor_mask
                entropy = dist.entropy().sum(-1) * actor_mask
                n_actor = int(actor_mask.sum().clamp(min=1))
                policy_loss = policy_loss.sum() / n_actor - cfg.entropy_coef * entropy.sum() / n_actor

                # critics: each env sample trains its own critic
                vm = self.critic_m(critic_obs_m[mb])
                vfs = self.critic_fs(critic_obs_fs[mb])
                v_target = returns[mb]
                v_pred = torch.where(mb_ref, vm, vfs)
                old_value = torch.where(
                    mb_ref, storage.values_m.flatten(0, 1)[mb], storage.values_fs.flatten(0, 1)[mb]
                )
                v_clip = old_value + (v_pred - old_value).clamp(-cfg.clip_range, cfg.clip_range)
                value_error = (v_pred - v_target).pow(2)
                clipped_error = torch.maximum(value_error, (v_clip - v_target).pow(2))
                # Only the newly introduced goal critic skips clipping during
                # warmup. Reference samples always train with clipped values.
                value_loss = torch.where(warmup & ~mb_ref, value_error, clipped_error).mean()
                loss = policy_loss + cfg.value_loss_coef * value_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")

                self.opt_main.zero_grad()
                self.opt_fs.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    parameters, cfg.max_grad_norm, error_if_nonfinite=True,
                )
                self.opt_main.step()
                self.opt_fs.step()
                self.constrain_noise()
                if not torch.stack([torch.isfinite(p).all() for p in parameters]).all():
                    raise FloatingPointError("Nonfinite PPO parameters")
                n = mb.numel()
                totals["policy_loss"] += policy_loss.item() * n
                totals["value_loss"] += value_loss.item() * n
                totals["entropy"] += entropy.mean().item() * n
                count += n

        # KL-adaptive learning rate for actor + reference critic
        # Evaluate the final policy against the distributions that generated the
        # rollout. Gaussian KL is nonnegative and uses every rollout sample.
        with torch.no_grad():
            eligible = is_ref if warmup else torch.ones_like(is_ref)
            kl_sum = obs.new_zeros(())
            eligible_count = eligible.sum()
            chunk = max(1, obs.shape[0] // max(1, cfg.num_minibatches))
            for start in range(0, obs.shape[0], chunk):
                sl = slice(start, start + chunk)
                current = self.evaluate(obs[sl])
                old = D.Normal(old_means[sl], old_stds[sl])
                kl_sum += D.kl_divergence(old, current).sum(-1).clamp_min(0)[eligible[sl]].sum()
            kl = kl_sum / eligible_count.clamp_min(1)
            if not torch.isfinite(kl):
                raise FloatingPointError("Nonfinite PPO KL divergence")
        if kl > 1e-4:
            factor = max(0.5, min(2.0, cfg.target_kl / max(kl.item(), 1e-8)))
            self.actor_lr = max(cfg.kl_min_lr, min(cfg.kl_max_lr, self.actor_lr * factor))
            self.critic_m_lr = self.actor_lr
            for g in self.opt_main.param_groups:
                g["lr"] = self.actor_lr
        stats = {key: value / max(1, count) for key, value in totals.items()}
        stats.update(kl=kl.item(), actor_lr=self.actor_lr)
        return stats

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_m": self.critic_m.state_dict(),
                "critic_fs": self.critic_fs.state_dict(),
                "log_std": self.log_std.detach(),
                "actor_lr": self.actor_lr,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_m.load_state_dict(ckpt["critic_m"])
        self.critic_fs.load_state_dict(ckpt["critic_fs"])
        self.log_std.data.copy_(ckpt["log_std"])
        self.constrain_noise()
        self.actor_lr = ckpt.get("actor_lr", self.cfg.init_lr)
        for g in self.opt_main.param_groups:
            g["lr"] = self.actor_lr
