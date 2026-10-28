"""Explicit scope for nominal skill learning versus the implemented ACP randomization.

The partial profile is intentionally named: not all paper Table S2
randomizations are implemented. Nominal pretraining establishes physical
tracking first and does not claim domain robustness or a complete reproduction.
"""

from __future__ import annotations


PROFILES = ("acp_partial_dr", "nominal_pretrain")


def configure_training_profile(cfg, profile: str):
    if profile not in PROFILES:
        raise ValueError(f"unknown training profile {profile!r}")
    cfg.training_profile = profile
    if profile == "nominal_pretrain":
        cfg.dr_torso_mass = (0.0, 0.0)
        cfg.dr_gain_scale = (1.0, 1.0)
        cfg.dr_joint_bias_nonankle = cfg.dr_joint_bias_ankle = 0.0
        cfg.dr_delay = (0, 0)
        for key in vars(cfg):
            if key.startswith("obs_noise_") or key.endswith("scan_noise"):
                setattr(cfg, key, 0.0)
    return cfg
