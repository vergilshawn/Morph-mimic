"""Stage presets for the ACP training pipeline.

Env counts: the paper uses 12,000 (Stage 1, 3x RTX 4090), 4,800 (Stage 2,
2x 4090) and 8,400 (Stage 3, 4x 4090). Single-GPU defaults are scaled for
one RTX 5090 (32 GB); adjust --num_envs for your hardware.
"""

from __future__ import annotations

from morph_mimic.envs.acp_env import AcpEnvCfg
from morph_mimic.rl.ppo import PPOCfg
from morph_mimic.rl.runner import RunnerCfg


def stage1_env_cfg(num_envs: int = 8192) -> AcpEnvCfg:
    cfg = AcpEnvCfg()
    cfg.stage = 1
    cfg.ref_env_ratio = 1.0
    cfg.csi_enabled = True
    cfg.eta_obs = 0.0
    cfg.eta_ref_obs = 0.0
    cfg.eta_rew = 0.0
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 15.0
    return cfg


def stage2_env_cfg(num_envs: int = 4096) -> AcpEnvCfg:
    cfg = AcpEnvCfg()
    cfg.stage = 2
    cfg.ref_env_ratio = 1.0
    cfg.csi_enabled = False
    cfg.eta_obs = 0.0
    cfg.eta_ref_obs = 0.0
    cfg.eta_rew = 0.0
    cfg.scene.num_envs = num_envs
    return cfg


def stage3_env_cfg(num_envs: int = 4096, ref_env_ratio: float = 0.3) -> AcpEnvCfg:
    cfg = AcpEnvCfg()
    cfg.stage = 3
    cfg.ref_env_ratio = ref_env_ratio
    cfg.csi_enabled = False
    cfg.eta_obs = 1.0
    cfg.eta_ref_obs = 1.0
    cfg.eta_rew = 1.0
    cfg.scene.num_envs = num_envs
    return cfg


def runner_cfg(stage: int) -> RunnerCfg:
    cfg = RunnerCfg()
    ppo = PPOCfg()
    if stage == 1:
        ppo.target_kl = 0.02
        ppo.entropy_coef = 0.006
    elif stage == 2:
        ppo.target_kl = 0.01
        ppo.entropy_coef = 0.005
    else:
        ppo.target_kl = 0.01
        ppo.entropy_coef = 0.002
    cfg.ppo = ppo
    return cfg
