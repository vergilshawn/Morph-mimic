"""Build Isaac Lab ArticulationCfg objects from robot specs.

- g1: configured nucleus USD by default; local MJCF for scene-bound references
- mojia: local URDF imported via UrdfFileCfg (cached USD conversion)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec

ROBOT_PRIM_PATH = "/World/envs/env_.*/Robot"


def make_robot_cfg(spec: RobotSpec, use_usd: bool = True, prim_path: str = ROBOT_PRIM_PATH) -> ArticulationCfg:
    actuators = {
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=dict(zip(spec.joint_names, spec.stiffness)),
            damping=dict(zip(spec.joint_names, spec.damping)),
            effort_limit_sim=dict(zip(spec.joint_names, spec.effort_limits)),
            velocity_limit_sim=dict(zip(spec.joint_names, spec.vel_limits)),
            armature=dict(zip(spec.joint_names, spec.armature)),
        )
    }
    if spec.name == "g1" and use_usd:
        from isaaclab_assets.robots.unitree import G1_29DOF_CFG

        cfg = G1_29DOF_CFG.copy()
        cfg.prim_path = prim_path
        cfg.spawn.activate_contact_sensors = True
        cfg.actuators = actuators
        cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        cfg.init_state.joint_pos = dict(zip(spec.joint_names, spec.default_pose))
        return cfg

    urdf_path = spec.runtime_assets.get("urdf")
    if not urdf_path:
        mjcf_path = spec.runtime_assets.get("mjcf")
        if not mjcf_path:
            raise ValueError(f"{spec.name} has no local URDF or MJCF; use its configured USD asset")
        mjcf = PROJECT_ROOT / mjcf_path
        if not mjcf.is_file():
            raise ValueError(f"MJCF not found: {mjcf}")
        from morph_mimic.utils.explicit_mjcf import checked_explicit_mjcf

        mjcf = checked_explicit_mjcf(mjcf, PROJECT_ROOT / ".cache" / "mjcf" / spec.name)
        # AppLauncher does not enable the MJCF command extension by default.
        # Load it synchronously after SimulationApp startup, before spawning
        # invokes MjcfConverter/MJCFCreateImportConfig.
        from isaacsim.core.utils.extensions import enable_extension

        if not enable_extension("isaacsim.asset.importer.mjcf"):
            raise RuntimeError("Could not enable the Isaac Sim MJCF importer extension")
        # The importer mishandles nested MJCF defaults (capsules become spheres).
        # Import a hashed explicit derivative verified against the source in
        # MuJoCo. Resolve names after startup; PhysX equivalence still requires
        # runtime geometry, inertia and FK checks.
        return ArticulationCfg(
            prim_path=prim_path,
            # The MJCF importer also creates an auxiliary worldBody with an
            # articulation-root schema. Select the robot's physical free root.
            articulation_root_prim_path=f"/{spec.base_link}/{spec.base_link}",
            spawn=sim_utils.MjcfFileCfg(
                asset_path=str(mjcf),
                usd_dir=str(PROJECT_ROOT / ".cache" / "usd" / f"{spec.name}_mjcf"),
                make_instanceable=True,
                fix_base=False,
                import_inertia_tensor=True,
                import_sites=True,
                self_collision=True,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8), rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=dict(zip(spec.joint_names, spec.default_pose)),
            ),
            soft_joint_pos_limit_factor=0.95,
            actuators=actuators,
        )
    urdf = PROJECT_ROOT / urdf_path
    assert urdf.exists(), f"URDF not found: {urdf}"
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            usd_dir=str(PROJECT_ROOT / ".cache" / "usd" / spec.name),
            make_instanceable=True,
            fix_base=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        soft_joint_pos_limit_factor=0.95,
        actuators=actuators,
    )
