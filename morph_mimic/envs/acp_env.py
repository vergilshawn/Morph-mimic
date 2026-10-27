"""ACP contact-tracking / goal-reaching environment (arXiv 2610.31608).

Single DirectRLEnv covering all three training stages:
- Stage 1: reference-anchored contact tracking (oracle contacts, CSI curriculum)
- Stage 2: planner co-training (eta_obs / eta_rew annealing)
- Stage 3: mixed reference-following + goal-reaching rollouts (dual critics)

Robot: Unitree G1 (29-DOF asset, 23 controlled) or mojia (23-DOF).
Control: 50 Hz policy, 200 Hz joint PD. Terrain: procedural tiles
(reference-anchored flat/rough + goal-reaching suite). Perception:
RayCaster grid scan (17x11 for the planner, 9x7 central subset for the
tracker). Contacts: rigid contact sensor on the nine contact bodies.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from morph_mimic.envs.terrains import (
    BeamCfg,
    BoxCfg,
    FlatCfg,
    GapCfg,
    PalletCfg,
    PlatformCfg,
    RoughCfg,
    ReferenceSceneCfg,
    SlopeCfg,
    StairsCfg,
    StonesCfg,
    TERRAIN_GROUP,
    UnevenClimbCfg,
    VaultCfg,
)
from morph_mimic.utils.motion import MotionDataset
from morph_mimic.utils.specs import RobotSpec
from morph_mimic.utils.transforms import (
    encode_goal,
    quat_apply,
    quat_diff_rad,
    to_body_yaw_frame,
    wrap_to_pi,
    yaw_from_quat,
    yaw_quat,
)

# reward kernels (paper Table 3 caption)
SIGMA_P, SIGMA_R, SIGMA_V, SIGMA_W, SIGMA_Q, SIGMA_QD, SIGMA_C = 0.5, 1.0, 1.0, 3.0, 2.0, 10.0, 1.0

# record layout (see _record_step): 75 + 3 + 561 + 3 + 1 + 9 + 3 + 27
RECORD_FIELDS = 682
REC_PRO, REC_GOAL, REC_PTS = 0, 75, 78
REC_BASE, REC_YAW, REC_FORCE, REC_VEL, REC_BODY = 78 + 561, 78 + 561 + 3, 78 + 561 + 4, 78 + 561 + 4 + 9, 78 + 561 + 4 + 9 + 3

_GRAV = (0.0, 0.0, -1.0)


def build_terrain_cfg(cfg: "AcpEnvCfg") -> TerrainImporterCfg:
    """Procedural terrain: rows = difficulty levels, cols = terrain types."""
    size = cfg.terrain_size
    sub = {
        "flat": FlatCfg(),
        "rough": RoughCfg(),
        "stairs": StairsCfg(),
        "platform": PlatformCfg(),
        "gap": GapCfg(),
        "beam": BeamCfg(),
        "pallet": PalletCfg(),
        "stones_grid": StonesCfg(layout="grid"),
        "stones_paired": StonesCfg(layout="paired"),
        "stones_double": StonesCfg(layout="double"),
        "box": BoxCfg(),
        "vault": VaultCfg(),
        "slope": SlopeCfg(),
        "uneven_climb": UnevenClimbCfg(),
    }
    if cfg.reference_scene_path is not None:
        from morph_mimic.utils.reference_scene import reference_scene_grid_shape

        if (cfg.ref_env_ratio != 1.0 or cfg.stage not in (1, 2)
                or cfg.terrain_types != ("reference_scene",)
                or cfg.ref_terrain_types != ("reference_scene",) or cfg.num_terrain_levels != 1):
            raise ValueError("fixed reference scenes require one bound clip in Stage 1/2 reference environments")
        rows, cols = reference_scene_grid_shape(cfg.scene.num_envs)
        sub["reference_scene"] = ReferenceSceneCfg(
            sidecar_path=cfg.reference_scene_path, fingerprint=cfg.reference_scene_fingerprint,
        )
    else:
        rows, cols = cfg.num_terrain_levels, len(cfg.terrain_types)
    gen = TerrainGeneratorCfg(
        size=size,
        curriculum=True,
        num_rows=rows,
        num_cols=cols,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={t: sub[t] for t in cfg.terrain_types},
    )
    return TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=gen,
        use_terrain_origins=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


@configclass
class AcpEnvCfg(DirectRLEnvCfg):
    # environment
    decimation: int = 4  # physics 200 Hz -> control 50 Hz
    episode_length_s: float = 15.0
    action_space: int = 23
    observation_space: int = 468

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0)

    # terrain layout
    terrain_size: tuple[float, float] = (9.0, 6.0)
    num_terrain_levels: int = 10
    terrain_types: tuple = ("flat", "rough", "stairs", "platform", "gap", "beam", "pallet",
                            "stones_grid", "stones_paired", "stones_double", "box", "vault", "slope")
    ref_terrain_types: tuple = ("flat", "rough")
    reference_scene_path: str | None = None
    reference_scene_fingerprint: str = ""
    oracle_contact_frame: str = "reference_query_to_actual_body_yaw_v1"

    # env mix
    ref_env_ratio: float = 1.0

    # scans
    scan_spacing: float = 0.1
    planner_scan_shape: tuple[int, int] = (17, 11)
    tracker_scan_shape: tuple[int, int] = (9, 7)
    scan_max_distance: float = 4.0

    # robot
    robot: ArticulationCfg = ArticulationCfg(prim_path="/World/envs/env_.*/Robot")

    # stage config
    stage: int = 1
    eta_obs: float = 0.0  # planner-input probability for goal-reaching envs
    eta_ref_obs: float = 0.0  # planner-input probability for reference envs
    eta_rew: float = 0.0
    csi_enabled: bool = True
    csi_init_phase: float = 0.9
    csi_adjust: float = 0.05
    csi_success_root_error_m: float = 0.3  # engineering completion gate
    critic_reference_frame: str = "scene_local"

    # goal / curriculum
    goal_scale_radius: float = 0.5
    ref_goal_scale_radius: float = 0.3
    ref_goal_lookahead: tuple[float, float] = (0.5, 1.5)
    success_horiz: float = 0.5
    success_vert: float = 0.1
    promote_ema: float = 0.2
    promote_threshold: float = 0.5
    demote_horiz: float = 1.0
    min_goal_dist: float = 2.0
    max_goal_dist: float = 4.5

    # termination
    min_height_ref: float = 0.30
    min_height_goal: float = 0.45
    grace_steps: int = 75  # 1.5 s
    reference_grace_steps: int = 5  # 0.1 s settling; not a paper-specified value
    max_reference_root_error_m: float = 0.5  # engineering tracking termination

    # contact detection
    contact_force_threshold: float = 1.0
    contact_history: int = 10  # physics steps

    # rewards (Table 3)
    w_track_base_pos: float = 3.0
    w_track_base_vel: float = 1.0
    w_track_joint_pos: float = 2.5
    w_track_joint_vel: float = 0.5
    w_track_contact_label: float = 5.0
    w_track_contact_pos_base: float = 2.0
    w_track_contact_pos_global: float = 1.0
    w_track_planned_contact: float = 1.5
    w_move2goal: float = 40.0
    w_standatgoal: float = 20.0
    w_action_rate: float = 0.1
    w_action_rate_2nd: float = 5e-2
    w_joint_torque: float = 1e-5
    w_torque_limits: float = 1e-3
    w_dof_pos_limits: float = 10.0
    w_dof_vel_limits: float = 1.0
    w_contact_acc: float = 0.01
    w_flat_orientation: float = 10.0
    w_stumbling: float = 10.0
    w_slippage: float = 10.0
    w_ang_vel_xy: float = 0.1

    # action
    action_scale: float = 0.25
    action_clip: float = 20.0  # normalized controller input bound (engineering safeguard)

    # domain randomization (Table S2)
    dr_torso_mass: tuple[float, float] = (-2.0, 4.0)
    dr_gain_scale: tuple[float, float] = (0.9, 1.1)
    dr_joint_bias_nonankle: float = 0.01
    dr_joint_bias_ankle: float = 0.03
    dr_delay: tuple[int, int] = (0, 1)  # 0-20 ms in control steps
    obs_noise_angvel: float = 0.05
    obs_noise_gravity: float = 0.0125
    obs_noise_jpos: float = 0.01
    obs_noise_jvel: float = 1.5
    planner_scan_noise: float = 0.03
    tracker_scan_noise: float = 0.1

    # episode recording for hindsight buffers
    record_max_steps: int = 500

    # offline contact-labeling replay mode (tracks the reference with PD,
    # no policy; used by scripts/label_contacts.py)
    replay: bool = False
    evaluation: bool = False
    evaluation_start_phase: float = 0.0
    evaluation_record_trajectory: bool = False


class AcpEnv(DirectRLEnv):
    cfg: AcpEnvCfg

    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg: AcpEnvCfg,
        robot_spec: RobotSpec,
        motion_dataset: MotionDataset | None = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        if cfg.evaluation:
            if not 0.0 <= cfg.evaluation_start_phase < 1.0:
                raise ValueError("evaluation_start_phase must be in [0, 1)")
            # Full-clip evaluation uses nominal dynamics and clean observations.
            cfg.csi_enabled = False
            cfg.dr_torso_mass = (0.0, 0.0)
            cfg.dr_gain_scale = (1.0, 1.0)
            cfg.dr_joint_bias_nonankle = cfg.dr_joint_bias_ankle = 0.0
            cfg.dr_delay = (0, 0)
            cfg.obs_noise_angvel = cfg.obs_noise_gravity = 0.0
            cfg.obs_noise_jpos = cfg.obs_noise_jvel = 0.0
            cfg.planner_scan_noise = cfg.tracker_scan_noise = 0.0
        self.robot_spec = robot_spec
        self.motion_data = motion_dataset
        if getattr(motion_dataset, "scene_bindings", {}):
            from morph_mimic.scripts.common import configure_reference_scene

            if hasattr(cfg, "terrain"):
                raise ValueError("a bound reference scene cannot use a preconfigured terrain importer")
            configure_reference_scene(cfg, motion_dataset, cfg.scene.num_envs)
        self.L, self.W = cfg.planner_scan_shape
        self.LW = self.L * self.W
        self.TL, self.TW = cfg.tracker_scan_shape

        if not hasattr(cfg, "terrain"):
            cfg.terrain = build_terrain_cfg(cfg)

        super().__init__(cfg, render_mode, **kwargs)

        # ---- contact links (asset data is available now that the sim started)
        import os as _os2
        if _os2.environ.get("MORPH_DEBUG_BODIES"):
            print("[debug] robot bodies:", self.robot.data.body_names)
            print("[debug] robot joints:", self.robot.data.joint_names)
        self._resolve_contact_links()

        # Scene sensors initialize through Isaac Lab's simulation-play callbacks.
        if self.contact_sensor is not None:
            sensor_names = self.contact_sensor.body_names
            self.contact_sensor_idx = torch.tensor(
                [sensor_names.index(name) for name in self.contact_link_names],
                dtype=torch.long, device=self.device,
            )

        # ---- joint bookkeeping (asset order)
        asset_names = list(self.robot.data.joint_names)
        aliases = (getattr(robot_spec, "asset_aliases", None) or {}).get("joints", {})

        def resolve_joint(nm):
            if nm in asset_names:
                return asset_names.index(nm)
            if nm in aliases and aliases[nm] in asset_names:
                return asset_names.index(aliases[nm])
            norm = nm.replace("_joint", "")
            for i, an in enumerate(asset_names):
                if an.replace("_joint", "") == norm:
                    return i
            raise KeyError(f"joint {nm} (alias {aliases.get(nm)}) not found in asset joints")

        self.controlled_idx = torch.tensor(
            [resolve_joint(nm) for nm in robot_spec.controlled_joint_names],
            dtype=torch.long, device=self.device,
        )
        # held = every remaining joint of the asset (USD variants have extra DoFs)
        self.held_idx = torch.tensor(
            [i for i in range(len(asset_names)) if i not in self.controlled_idx.tolist()],
            dtype=torch.long, device=self.device,
        )

        # per-asset-joint spec arrays (asset joints absent from the spec get
        # zero defaults and passive gains)
        spec_idx_of_asset = []
        for an in asset_names:
            idx = None
            for si, sn in enumerate(robot_spec.joint_names):
                if sn == an or aliases.get(sn) == an:
                    idx = si
                    break
            spec_idx_of_asset.append(idx)
        mapped = torch.tensor([i is not None for i in spec_idx_of_asset], dtype=torch.bool, device=self.device)
        mapped_idx = torch.tensor([i if i is not None else 0 for i in spec_idx_of_asset], dtype=torch.long, device=self.device)
        self.default_pose = torch.zeros(len(asset_names), device=self.device)
        self.default_pose[mapped] = robot_spec.default_pose_t.to(self.device)[mapped_idx[mapped]]
        self.joint_limits = torch.full((len(asset_names), 2), 100.0, device=self.device)
        self.joint_limits[:, 0] = -100.0
        self.joint_limits[mapped] = robot_spec.joint_pos_limits.to(self.device)[mapped_idx[mapped]]
        self.vel_limits = torch.full((len(asset_names),), 100.0, device=self.device)
        self.vel_limits[mapped] = robot_spec.vel_limits_t.to(self.device)[mapped_idx[mapped]]
        kp = torch.zeros(len(asset_names), device=self.device)
        kd = torch.zeros(len(asset_names), device=self.device)
        kp[mapped] = robot_spec.pd_gains.to(self.device)[mapped_idx[mapped], 0]
        kd[mapped] = robot_spec.pd_gains.to(self.device)[mapped_idx[mapped], 1]
        self._pd_gains_asset = torch.stack((kp, kd), dim=-1)
        all_ids = torch.arange(self.robot.num_joints, device=self.device)
        self.robot.write_joint_stiffness_to_sim(kp, joint_ids=all_ids)
        self.robot.write_joint_damping_to_sim(kd, joint_ids=all_ids)
        self._sync_actuator_gains(kp, kd)

        # ---- terrain mapping (after importer init)
        self.terrain_origins = torch.as_tensor(self.terrain.terrain_origins, device=self.device).float()
        self._init_terrain_mapping()

        # ---- per-env bookkeeping
        self.is_ref_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_ref_env[: int(self.num_envs * self.cfg.ref_env_ratio)] = True
        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_duration = torch.ones(self.num_envs, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.init_phase = torch.zeros(self.num_envs, device=self.device)
        self.csi_min_phase = torch.full((self.num_envs,), self.cfg.csi_init_phase, device=self.device)
        self.ref_lookahead = torch.zeros(self.num_envs, device=self.device)

        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_yaw = torch.zeros(self.num_envs, device=self.device)
        self.goal_height = torch.full((self.num_envs,), 0.75, device=self.device)
        self.success_ema = torch.zeros(self.num_envs, device=self.device)
        self.ep_reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ep_min_goal_dist = torch.full((self.num_envs,), 1e9, device=self.device)

        # contact targets (9 x 3) fed to the tracker
        self.contact_targets = torch.zeros(self.num_envs, 9, 3, device=self.device)
        self.contact_target_flags = torch.zeros(self.num_envs, 9, device=self.device)
        self.predicted_contacts = torch.zeros(self.num_envs, 9, 3, device=self.device)
        self.predicted_occupancy = torch.zeros(self.num_envs, 9, device=self.device)
        self.ref_contact_flags = torch.zeros(self.num_envs, 9, device=self.device)

        # action / proprio history
        self.actions = torch.zeros(self.num_envs, 23, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, 2, 23, device=self.device)
        self.action_delay = torch.randint(self.cfg.dr_delay[0], self.cfg.dr_delay[1] + 1, (self.num_envs,), device=self.device)
        self.proprio_history = torch.zeros(self.num_envs, 5, 75, device=self.device)

        # contact windows
        self.contact_duration = torch.zeros(self.num_envs, 9, device=self.device)
        self.contact_body_vel = torch.zeros(self.num_envs, 9, 3, device=self.device)
        self.prev_contact_body_vel = torch.zeros(self.num_envs, 9, 3, device=self.device)

        # episode recording (CPU staging)
        self.ep_record = (
            torch.zeros(self.num_envs, self.cfg.record_max_steps, RECORD_FIELDS)
            if self._record_enabled else None
        )
        self.ep_record_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.finished_episodes: list[dict] = []

        self._setup_domain_randomization()

        # clip body-index cache for the nine contact links
        self._contact_clip_body_idx: dict[tuple[int, int], int | None] = {}

    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        # MJCF imports add an articulation-root namespace (Robot/pelvis),
        # whereas the stock USD has bodies directly below Robot.
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        source_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        rigid_prims = [prim for prim in Usd.PrimRange(source_root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        parents = {str(prim.GetPath().GetParentPath()) for prim in rigid_prims}
        if len(parents) != 1:
            raise ValueError("robot bodies must share one USD namespace for ray/contact sensors")
        body_parent = next(iter(parents))
        self.robot_body_root_pattern = body_parent.replace("/env_0/", "/env_.*/")
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        if self.cfg.reference_scene_path is not None:
            # Each robot gets a distinct copy of the source scene. The scene's
            # reported origins must agree with motion, ray and physics origins.
            self.terrain.env_origins = self.terrain.terrain_origins.reshape(-1, 3)[:self.num_envs].clone()
            self.scene._terrain = self.terrain

        # Sensors are created before cloning. ContactSensor filters select the
        # opposing bodies, not the reporting bodies; map its body order after init.
        body_aliases = (getattr(self.robot_spec, "asset_aliases", None) or {}).get("bodies", {})
        base_link = body_aliases.get(self.robot_spec.base_link, self.robot_spec.base_link)
        ray_cfg = RayCasterCfg(
            prim_path=f"{self.robot_body_root_pattern}/{base_link}",
            mesh_prim_paths=["/World/terrain"],
            pattern_cfg=GridPatternCfg(
                resolution=self.cfg.scan_spacing,
                size=(self.cfg.scan_spacing * (self.L - 1), self.cfg.scan_spacing * (self.W - 1)),
            ),
            ray_alignment="yaw",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.12)),
            max_distance=self.cfg.scan_max_distance,
            update_period=0.0,
        )
        self.ray_caster = RayCaster(ray_cfg)
        contact_cfg = ContactSensorCfg(
            prim_path=f"{self.robot_body_root_pattern}/.*",
            update_period=0.0,
            history_length=self.cfg.contact_history,
        )
        import os as _os
        if _os.environ.get("MORPH_DEBUG_BODIES"):
            self.contact_sensor = None
        else:
            self.contact_sensor = ContactSensor(contact_cfg)
            self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.sensors["ray_caster"] = self.ray_caster

        self.scene_contact_sensors = []
        if self.cfg.reference_scene_path is not None:
            body_names = [prim.GetName() for prim in rigid_prims]
            contact_names, _ = self.robot_spec.resolve_contact_bodies(body_names)
            # PhysX filtered reporting supports one reporter per sensor. A
            # separate view for each anatomy excludes robot self collisions.
            for i, name in enumerate(contact_names):
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"{self.robot_body_root_pattern}/{name}", update_period=0.0,
                    history_length=self.cfg.contact_history,
                    filter_prim_paths_expr=["/World/terrain/terrain/mesh"],
                ))
                self.scene_contact_sensors.append(sensor)
                self.scene.sensors[f"scene_contact_{i}"] = sensor

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # terrain / env mapping
    def _init_terrain_mapping(self):
        n_ref = int(self.num_envs * self.cfg.ref_env_ratio)
        tnames = list(self.cfg.terrain_types)
        ref_cols = np.array([tnames.index(t) for t in self.cfg.ref_terrain_types if t in tnames], dtype=np.int64)
        goal_cols = np.array([i for i, t in enumerate(tnames) if t not in self.cfg.ref_terrain_types])
        if (n_ref and len(ref_cols) == 0) or (n_ref < self.num_envs and len(goal_cols) == 0):
            raise ValueError("terrain_types must include a terrain for each active reference/goal environment group")
        types = np.zeros(self.num_envs, dtype=np.int64)
        ref_ids = np.arange(n_ref)
        goal_ids = np.arange(n_ref, self.num_envs)
        types[ref_ids] = ref_cols[ref_ids % max(1, len(ref_cols))]
        types[goal_ids] = goal_cols[goal_ids % max(1, len(goal_cols))]
        self.terrain_type_id = torch.from_numpy(types).to(self.device)
        self.terrain_level = torch.randint(0, self.cfg.num_terrain_levels, (self.num_envs,), device=self.device)
        self.terrain_level[:n_ref] = 0
        # group id per env (for style masks / buffer partitions)
        group_of = {g: i for i, g in enumerate(["dense_foot", "sparse_foot", "dense_whole_body", "sparse_whole_body"])}
        self.terrain_group_id = torch.tensor(
            [group_of[TERRAIN_GROUP[tnames[t]]] for t in types], device=self.device
        )
        # deterministic spawn offsets within a tile
        slot = torch.arange(self.num_envs, device=self.device) % 16
        self.cell_offset = torch.stack(
            (slot % 4 * 1.8 - 2.7, slot // 4 * 1.8 - 2.7, torch.zeros(self.num_envs, device=self.device)), dim=-1
        )
        if self.cfg.reference_scene_path is not None:
            # The scene and motion share their exact transform; shifting only
            # a robot within the tile would destroy box/hand alignment.
            self.cell_offset.zero_()
            self.reference_scene_origins = self.terrain_origins.reshape(-1, 3)[:self.num_envs].clone()
            if (self.reference_scene_origins.shape != (self.num_envs, 3)
                    or len(torch.unique(self.reference_scene_origins, dim=0)) != self.num_envs):
                raise ValueError("bound scenes require a distinct terrain origin per environment")

    def env_origin(self) -> torch.Tensor:
        """Tile origin of each env: terrain_origins[level, type]."""
        if self.cfg.reference_scene_path is not None:
            return self.reference_scene_origins
        o = self.terrain_origins[self.terrain_level, self.terrain_type_id]
        return o

    def _resolve_contact_links(self):
        """Map spec contact-body candidates to asset body names (9 links)."""
        body_names = self.robot.body_names
        names, motion_candidates = self.robot_spec.resolve_contact_bodies(body_names)
        idxs = [body_names.index(name) for name in names]
        self.contact_link_names = names
        self.contact_motion_candidates = motion_candidates
        self.contact_link_idx = torch.tensor(idxs, dtype=torch.long, device=self.device)

    def _sync_actuator_gains(self, stiffness: torch.Tensor, damping: torch.Tensor):
        # Isaac Lab's write_joint_*_to_sim does not update actuator models.
        # Their gains also determine the torque estimates used by rewards.
        for actuator in self.robot.actuators.values():
            ids = actuator.joint_indices
            actuator.stiffness[:] = stiffness[..., ids]
            actuator.damping[:] = damping[..., ids]

    def _setup_domain_randomization(self):
        rng = torch.Generator(device=self.device).manual_seed(0)
        lo, hi = self.cfg.dr_torso_mass
        self.dr_mass_delta = (lo + (hi - lo) * torch.rand(self.num_envs, generator=rng, device=self.device)).clamp(min=-4.0)
        g_lo, g_hi = self.cfg.dr_gain_scale
        self.dr_gain = (g_lo + (g_hi - g_lo) * torch.rand(self.num_envs, generator=rng, device=self.device))[:, None]
        # torso mass offset (Table S2)
        body_names = list(self.robot.data.body_names)
        torso_name = {"g1": "torso_link", "mojia": "lumbar_yaw"}[self.robot_spec.name]
        aliases = (getattr(self.robot_spec, "asset_aliases", None) or {}).get("bodies", {})
        torso_name = aliases.get(torso_name, torso_name)
        if torso_name not in body_names:
            raise ValueError(f"Torso body {torso_name!r} is missing from the robot asset")
        self.torso_body_idx = body_names.index(torso_name)
        # (mass / gain application deferred to the first reset: the physx
        # view is only available after the simulation starts)
        self._dr_dynamics_applied = False
        # joint zero-position bias (ankles larger, Table S2)
        is_ankle = torch.tensor(["ankle" in nm for nm in self.robot_spec.controlled_joint_names], device=self.device)
        bias = torch.where(is_ankle, self.cfg.dr_joint_bias_ankle, self.cfg.dr_joint_bias_nonankle)
        self.dr_joint_bias = (2 * bias * torch.rand(self.num_envs, 23, generator=rng, device=self.device) - bias)
        self.dr_scan_bias = torch.zeros(self.num_envs, self.LW, device=self.device)

    # ------------------------------------------------------------------
    # reference sampling (vectorized)
    def _ref_sample(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        out = {
            "root_pos": torch.zeros(len(ids), 3, device=self.device),
            "root_quat": torch.zeros(len(ids), 4, device=self.device),
            "root_lin_vel": torch.zeros(len(ids), 3, device=self.device),
            "root_ang_vel": torch.zeros(len(ids), 3, device=self.device),
            "joint_pos": torch.zeros(len(ids), 23, device=self.device),
            "joint_vel": torch.zeros(len(ids), 23, device=self.device),
        }
        if self.motion_data is None or len(ids) == 0:
            out["root_quat"][:, 0] = 1.0
            return out
        s = self.motion_data.sample(self.clip_id[ids], self.phase[ids])
        # rebase x/y: displacement from the clip's first frame placed at the tile origin
        clips = self.motion_data.clips
        first = torch.stack(
            [clips[n].root_pos[0] for n in self.motion_data.clip_names], dim=0
        )  # (C, 3)
        first[:, 2] = 0.0  # rebase horizontal motion, preserve reference height
        origin = self.env_origin()[ids] + self.cell_offset[ids]
        for k in out:
            if k == "root_pos":
                disp = s[k] - first[self.clip_id[ids]]  # (K, 3)
                out[k] = origin + disp
            elif k == "root_quat":
                out[k] = s[k]
                out[k][out[k].norm(dim=-1) < 0.5, 0] = 1.0
            else:
                out[k] = s[k]
        return out

    def _ref_goal_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        base_pos = self.robot.data.root_pos_w[env_ids]
        base_yaw = yaw_from_quat(self.robot.data.root_quat_w[env_ids])
        clips = self.motion_data.clips
        fut_phase = self.phase[env_ids] + self.ref_lookahead[env_ids] / self.clip_duration[env_ids]
        s = self.motion_data.sample(self.clip_id[env_ids], fut_phase.clamp(0.0, 1.0 - 1e-6))
        first = torch.stack([clips[n].root_pos[0] for n in self.motion_data.clip_names], dim=0)
        first[:, 2] = 0.0
        origin = self.env_origin()[env_ids] + self.cell_offset[env_ids]
        fut_pos = origin + s["root_pos"] - first[self.clip_id[env_ids]]
        fut_yaw = yaw_from_quat(s["root_quat"])
        return encode_goal(base_pos, base_yaw, fut_pos, fut_yaw, self.cfg.ref_goal_scale_radius)

    def goal_command(self) -> torch.Tensor:
        base_pos = self.robot.data.root_pos_w
        base_yaw = yaw_from_quat(self.robot.data.root_quat_w)
        g = torch.zeros(self.num_envs, 3, device=self.device)
        ref_ids = self.is_ref_env.nonzero().squeeze(-1)
        if len(ref_ids):
            g[ref_ids] = self._ref_goal_command(ref_ids)
        goal_ids = (~self.is_ref_env).nonzero().squeeze(-1)
        if len(goal_ids):
            g[goal_ids] = encode_goal(
                base_pos[goal_ids], base_yaw[goal_ids], self.goal_pos[goal_ids], self.goal_yaw[goal_ids],
                self.cfg.goal_scale_radius,
            )
        return g

    # ------------------------------------------------------------------
    # scans
    def _scan_points_body_yaw(self, noisy: bool = True) -> torch.Tensor:
        pos = self.ray_caster.data.ray_hits_w  # (E, LW, 3)
        base_pos = self.robot.data.root_pos_w
        base_yaw = yaw_from_quat(self.robot.data.root_quat_w)
        # A missed downward ray returns inf. Preserve its grid x/y and place
        # the missing surface at the maximum scan depth to keep network inputs finite.
        ray_xy = quat_apply(yaw_quat(base_yaw)[:, None, :], self.ray_caster.ray_starts)
        fallback = base_pos[:, None, :] + ray_xy
        fallback[..., 2] = base_pos[:, None, 2] - self.cfg.scan_max_distance
        valid = torch.isfinite(pos).all(dim=-1, keepdim=True)
        pos = torch.where(valid, pos, fallback)
        pts = to_body_yaw_frame(pos, base_pos[:, None, :], base_yaw[:, None])
        if noisy:
            pts[..., 2] += self.dr_scan_bias
            pts[..., 2] += torch.randn(self.num_envs, self.LW, device=self.device) * self.cfg.planner_scan_noise
        return pts

    def _tracker_scan(self) -> torch.Tensor:
        pts = self._scan_points_body_yaw()
        x0, y0 = (self.L - self.TL) // 2, (self.W - self.TW) // 2
        idx = torch.arange(self.LW, device=self.device).view(self.L, self.W)[x0 : x0 + self.TL, y0 : y0 + self.TW].flatten()
        z = pts[:, idx, 2] - self.robot.data.root_pos_w[:, 2:3]
        z = z + (torch.rand(self.num_envs, 1, device=self.device) * 2 - 1) * self.cfg.tracker_scan_noise
        return z

    def get_planner_obs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = self._scan_points_body_yaw()
        heights = points[..., 2].view(self.num_envs, self.L, self.W)
        pg = torch.cat((self.proprio_history.flatten(1), self.goal_command()), dim=-1)
        return points, heights, pg

    def set_predicted_contacts(self, contacts: torch.Tensor, occupancy: torch.Tensor):
        self.predicted_contacts = contacts.detach().to(self.device)
        self.predicted_occupancy = occupancy.detach().to(self.device)

    # ------------------------------------------------------------------
    # contacts
    def _cs(self):
        assert self.contact_sensor is not None, "contact sensor disabled (MORPH_DEBUG_BODIES)"
        return self.contact_sensor

    def _contact_measured(self) -> torch.Tensor:
        hist = self._contact_force_history()  # (E, H, 9, 3)
        fz = hist[..., 2].abs()
        return (fz.max(dim=1).values > self.cfg.contact_force_threshold).float()

    def _contact_force_history(self) -> torch.Tensor:
        if getattr(self, "scene_contact_sensors", []):
            return torch.cat([sensor.data.force_matrix_w_history.sum(dim=-2)
                              for sensor in self.scene_contact_sensors], dim=2)
        return self._cs().data.net_forces_w_history[:, :, self.contact_sensor_idx]

    def _contact_forces_w(self) -> torch.Tensor:
        if getattr(self, "scene_contact_sensors", []):
            return torch.cat([sensor.data.force_matrix_w.sum(dim=-2)
                              for sensor in self.scene_contact_sensors], dim=1)
        return self._cs().data.net_forces_w[:, self.contact_sensor_idx]

    def _contact_body_positions_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.contact_link_idx]

    def _contact_body_positions_body_yaw(self) -> torch.Tensor:
        p = self._contact_body_positions_w()
        return to_body_yaw_frame(
            p, self.robot.data.root_pos_w[:, None, :], yaw_from_quat(self.robot.data.root_quat_w)[:, None]
        )

    # reference contacts (instantaneous flags c_ref) from label files
    def _update_ref_contact_flags(self):
        self.ref_contact_flags.zero_()
        if self.motion_data is None or not self.motion_data.contact_flags:
            return
        for cid, name in enumerate(self.motion_data.clip_names):
            cf = self.motion_data.contact_flags.get(name)
            if cf is None:
                continue
            m = (self.clip_id == cid) & self.is_ref_env
            if not m.any():
                continue
            idx = (self.phase[m] * (cf.shape[0] - 1)).long().clamp(0, cf.shape[0] - 1)
            self.ref_contact_flags[m] = cf[idx]

    def _oracle_contacts(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Express fixed reference contact anchors in the current robot frame.

        Geometry labels store XY relative to the *discrete reference query
        pose*, with absolute Z. Reconstruct that world anchor before applying
        the current robot's translation/yaw; otherwise a tracking error moves
        the requested contact along with the robot. Legacy replay labels did
        not retain their generating query poses and cannot be inverted exactly.
        """
        targets = torch.zeros(len(env_ids), 9, 3, device=self.device)
        flags = torch.zeros(len(env_ids), 9, device=self.device)
        if self.motion_data is None or not self.motion_data.contact_targets:
            return targets, flags
        for cid, name in enumerate(self.motion_data.clip_names):
            if name not in self.motion_data.contact_targets:
                continue
            tg, fl = self.motion_data.contact_targets[name]
            m = self.clip_id[env_ids] == cid
            if not m.any():
                continue
            sel = env_ids[m]
            idx = (self.phase[sel] * (tg.shape[0] - 1)).long().clamp(0, tg.shape[0] - 1)
            frame = getattr(self.motion_data, "contact_target_frames", {}).get(
                name, "legacy_query_body_yaw_absolute_z_unverified")
            if frame == "reference_query_body_yaw_absolute_z":
                clip = self.motion_data.clips[name]
                reference_yaw = yaw_from_quat(clip.root_quat[idx])[:, None]
                cos_yaw, sin_yaw = torch.cos(reference_yaw), torch.sin(reference_yaw)
                local = tg[idx]
                world = local.clone()
                origin = self.env_origin()[sel] + self.cell_offset[sel]
                offset = origin + clip.root_pos[idx] - clip.root_pos[0]
                world[..., 0] = cos_yaw * local[..., 0] - sin_yaw * local[..., 1] + offset[:, None, 0]
                world[..., 1] = sin_yaw * local[..., 0] + cos_yaw * local[..., 1] + offset[:, None, 1]
                world[..., 2] = local[..., 2] + origin[:, None, 2]
                actual = to_body_yaw_frame(
                    world, self.robot.data.root_pos_w[sel, None, :],
                    yaw_from_quat(self.robot.data.root_quat_w[sel])[:, None],
                )
                targets[m] = torch.where(fl[idx, :, None].bool(), actual, torch.zeros_like(actual))
            else:
                if name in getattr(self.motion_data, "scene_bindings", {}):
                    raise ValueError("bound scene oracle requires known reference-query contact target coordinates")
                targets[m] = tg[idx]
            flags[m] = fl[idx]
        return targets, flags

    # reference contact-body positions from clip kinematics
    def _ref_contact_pos(self, body_yaw: bool):
        """(E, 9, 3) reference contact-body positions.

        body_yaw=True: in the reference body-yaw frame; else world.
        """
        e = self.num_envs
        out = torch.zeros(e, 9, 3, device=self.device)
        if self.motion_data is None:
            return out
        for b in range(9):
            for cid, clip in enumerate(self.motion_data.clips.values()):
                key = (cid, b)
                if key not in self._contact_clip_body_idx:
                    bi = next(
                        (idx for name in self.contact_motion_candidates[b] if (idx := clip.body_idx(name)) is not None),
                        None,
                    )
                    if bi is None:
                        raise ValueError(f"Missing contact body {self.contact_motion_candidates[b]} in motion {clip.name}")
                    self._contact_clip_body_idx[key] = bi
                bi = self._contact_clip_body_idx[key]
                if bi is None:
                    continue
                m = (self.clip_id == cid) & self.is_ref_env
                if not m.any():
                    continue
                p_w = clip.sample_body_pos(self.phase[m], bi)
                first_xy = clip.root_pos[0].clone()
                first_xy[2] = 0.0
                origin = self.env_origin()[m] + self.cell_offset[m]
                if body_yaw:
                    ref = self.motion_data.sample(self.clip_id[m], self.phase[m])
                    out[m, b] = to_body_yaw_frame(p_w, ref["root_pos"], yaw_from_quat(ref["root_quat"]))
                    out[m, b, 2] += origin[:, 2]
                else:
                    out[m, b] = origin + p_w - first_xy
        return out

    # ------------------------------------------------------------------
    # action / physics
    def _pre_physics_step(self, actions: torch.Tensor):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite policy action")
        if self.cfg.action_clip <= 0:
            raise ValueError("action_clip must be positive")
        self.prev_actions[:, 1] = self.prev_actions[:, 0]
        self.prev_actions[:, 0] = self.actions
        self.actions = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip).clone()

        # advance reference phase
        if self.motion_data is not None:
            ref = self.is_ref_env
            if ref.any():
                if self.cfg.replay or self.cfg.evaluation:
                    intervals = (self.clip_duration[ref] / self.step_dt).round().clamp(min=1)
                    self.phase[ref] = (self.init_phase[ref] +
                                       (self.episode_length_buf[ref] + 1).float() / intervals).clamp(max=1.0)
                else:
                    self.phase[ref] += self.step_dt / self.clip_duration[ref].clamp(min=1e-3)
        if self.cfg.replay:
            self._replay_jpos = self._ref_sample()["joint_pos"]
        self._update_ref_contact_flags()

    def _apply_action(self):
        targets = self.default_pose.unsqueeze(0).repeat(self.num_envs, 1)
        if self.cfg.replay:
            targets[:, self.controlled_idx] = self._replay_jpos
        else:
            action_history = torch.stack((self.actions, self.prev_actions[:, 0]), dim=1)
            delayed = action_history[torch.arange(self.num_envs, device=self.device), self.action_delay]
            targets[:, self.controlled_idx] = (
                self.default_pose[self.controlled_idx].unsqueeze(0) + self.cfg.action_scale * delayed
            )
        self.robot.set_joint_position_target(targets, joint_ids=torch.arange(self.robot.num_joints, device=self.device))

    # ------------------------------------------------------------------
    # observations
    def _compute_proprio(self) -> torch.Tensor:
        ang_vel = self.robot.data.root_ang_vel_b
        gz = quat_apply(
            torch.cat((self.robot.data.root_quat_w[:, :1], -self.robot.data.root_quat_w[:, 1:]), dim=-1),
            torch.tensor(_GRAV, device=self.device).repeat(self.num_envs, 1),
        )
        q = self.robot.data.joint_pos[:, self.controlled_idx] - self.default_pose[self.controlled_idx]
        qd = self.robot.data.joint_vel[:, self.controlled_idx]
        ang_vel = ang_vel + (2 * torch.rand(self.num_envs, 3, device=self.device) - 1) * self.cfg.obs_noise_angvel
        gz = gz + (2 * torch.rand(self.num_envs, 3, device=self.device) - 1) * self.cfg.obs_noise_gravity
        q = q + (2 * torch.rand(self.num_envs, 23, device=self.device) - 1) * self.cfg.obs_noise_jpos
        qd = qd + (2 * torch.rand(self.num_envs, 23, device=self.device) - 1) * self.cfg.obs_noise_jvel
        return torch.cat((ang_vel, gz, q, qd, self.actions), dim=-1)

    def _get_observations(self, record: bool = True) -> dict:
        proprio = self._compute_proprio()
        self.proprio_history[:, :-1] = self.proprio_history[:, 1:].clone()
        self.proprio_history[:, -1] = proprio
        self._update_contact_targets()
        scan = self._tracker_scan()
        goal = self.goal_command()
        obs = torch.cat((self.proprio_history.flatten(1), scan, self.contact_targets.flatten(1), goal), dim=-1)
        if record and self._record_enabled:
            initial = ((self.episode_length_buf == 0) & (self.ep_record_step == 0)).nonzero().squeeze(-1)
            if len(initial):
                self._record_step(proprio, goal, initial)
        return {"policy": obs}

    def refresh_observations(self) -> dict:
        """Re-read observations without stepping (after new planner outputs)."""
        self._update_contact_targets()
        obs = torch.cat(
            (self.proprio_history.flatten(1), self._tracker_scan(), self.contact_targets.flatten(1), self.goal_command()),
            dim=-1,
        )
        return {"policy": obs}

    def get_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(Vm obs (E, 353), Vfs obs (E, 298)) at the current state.

        Vm (reference-following critic): clean proprio (75) + full 17x11
        height scan (187) + base linear velocity (3) + privileged reference
        streams: 52-d kinematic vector, 27-d scene-local reference contact positions,
        9-d reference contact flags. Vfs (goal-reaching critic): clean
        proprio + heights + base lin vel + goal g (3) + d_g (1) + dz (1) +
        normalized episode time (1) + planner contacts (27).
        """
        e = self.num_envs
        # clean proprio (no observation noise)
        ang_vel = self.robot.data.root_ang_vel_b
        gz = quat_apply(
            torch.cat((self.robot.data.root_quat_w[:, :1], -self.robot.data.root_quat_w[:, 1:]), dim=-1),
            torch.tensor(_GRAV, device=self.device).repeat(e, 1),
        )
        q = self.robot.data.joint_pos[:, self.controlled_idx] - self.default_pose[self.controlled_idx]
        qd = self.robot.data.joint_vel[:, self.controlled_idx]
        proprio = torch.cat((ang_vel, gz, q, qd, self.actions), dim=-1)
        heights = self._planner_heights_relative()
        base_vel = self.robot.data.root_link_lin_vel_b

        # reference streams (privileged)
        ref = self._ref_sample()
        ref_kin = torch.cat(
            (ref["root_lin_vel"], ref["root_ang_vel"], ref["joint_pos"], ref["joint_vel"]), dim=-1
        )  # (E, 52)
        p_ref_w = self._ref_contact_pos(False)  # (E, 9, 3)
        if self.cfg.critic_reference_frame != "scene_local":
            raise ValueError("critic_reference_frame must be scene_local")
        # Copies of the same scene must not change their value inputs solely
        # because the terrain grid places them at different world coordinates.
        p_ref_local = p_ref_w - (self.env_origin() + self.cell_offset)[:, None, :]
        vm = torch.cat(
            (proprio, heights, base_vel, ref_kin, p_ref_local.flatten(1), self.ref_contact_flags), dim=-1
        )

        # goal-reaching streams
        base_pos = self.robot.data.root_pos_w
        d_g = (base_pos[:, :2] - self.goal_pos[:, :2]).norm(dim=-1, keepdim=True)
        dz = (self.goal_height - base_pos[:, 2]).unsqueeze(-1)
        t_norm = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1)
        p_hat = torch.where(
            (self.predicted_occupancy > 0.5).unsqueeze(-1),
            self.predicted_contacts,
            torch.zeros_like(self.predicted_contacts),
        )
        vfs = torch.cat(
            (proprio, heights, base_vel, self.goal_command(), d_g, dz, t_norm, p_hat.flatten(1)), dim=-1
        )
        return vm, vfs

    def _planner_heights_relative(self) -> torch.Tensor:
        """Full 17x11 height scan relative to the base (unnoised, for critics)."""
        pts = self._scan_points_body_yaw(noisy=False)
        return pts[..., 2] - self.robot.data.root_pos_w[:, 2:3]

    def _update_contact_targets(self):
        """Stochastic contact-input selection (paper Eq. 3 input schedule).

        Goal envs use planner outputs with probability eta_obs; reference
        envs with probability eta_ref_obs; otherwise the oracle (reference
        labels) for ref envs and zeros for goal envs.
        """
        eta_goal = torch.rand(self.num_envs, device=self.device) < self.cfg.eta_obs
        eta_ref = torch.rand(self.num_envs, device=self.device) < self.cfg.eta_ref_obs
        oracle_t, oracle_f = self._oracle_contacts(torch.arange(self.num_envs, device=self.device))
        use_planner = (eta_goal & ~self.is_ref_env) | (eta_ref & self.is_ref_env)
        use_oracle = ~eta_ref & self.is_ref_env
        self.contact_targets = torch.where(
            use_planner.unsqueeze(-1).unsqueeze(-1),
            self.predicted_contacts,
            torch.where(use_oracle.unsqueeze(-1).unsqueeze(-1), oracle_t, torch.zeros_like(self.predicted_contacts)),
        )
        self.contact_target_flags = (
            (self.predicted_occupancy > 0.5).float() * use_planner.float().unsqueeze(-1)
            + (oracle_f > 0.5).float() * use_oracle.float().unsqueeze(-1)
        )
        self.contact_targets = torch.where(
            self.contact_target_flags.unsqueeze(-1).bool(),
            self.contact_targets,
            torch.zeros_like(self.contact_targets),
        )

    def _record_step(self, proprio: torch.Tensor, goal: torch.Tensor, env_ids: torch.Tensor | None = None):
        pts = self._scan_points_body_yaw()
        base = self.robot.data.root_pos_w
        base_yaw = yaw_from_quat(self.robot.data.root_quat_w)
        fz = self._contact_forces_w()[..., 2].abs()
        vel = self.robot.data.root_link_lin_vel_w
        body = self._contact_body_positions_w().flatten(1)
        rec = torch.cat((proprio, goal, pts.flatten(1), base, base_yaw.unsqueeze(-1), fz, vel, body), dim=-1)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        ids = ids[self.ep_record_step[ids] < self.cfg.record_max_steps]
        step = self.ep_record_step[ids]
        self.ep_record[ids.cpu(), step.cpu()] = rec[ids].detach().cpu()
        self.ep_record_step[ids] += 1

    # ------------------------------------------------------------------
    # rewards (Table 3)
    def _get_rewards(self) -> torch.Tensor:
        e = self.num_envs
        w = self.cfg
        is_ref = self.is_ref_env.float()
        is_goal = 1.0 - is_ref

        ref = self._ref_sample()
        ref_pos, ref_quat, ref_lin_vel, ref_ang_vel = (
            ref["root_pos"], ref["root_quat"], ref["root_lin_vel"], ref["root_ang_vel"],
        )
        ref_jpos, ref_jvel = ref["joint_pos"], ref["joint_vel"]
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        base_vel = self.robot.data.root_link_lin_vel_b
        base_ang = self.robot.data.root_ang_vel_b
        chi_mov = (ref_lin_vel.norm(dim=-1) > 0.1).float()

        # task: reference tracking
        d_pos = (base_pos - ref_pos).pow(2).sum(-1)
        theta_e = quat_diff_rad(base_quat, ref_quat)
        r_base_pos = torch.exp(-chi_mov * d_pos / SIGMA_P**2 - theta_e**2 / SIGMA_R**2)
        d_vel = (self.robot.data.root_link_lin_vel_w - ref_lin_vel).pow(2).sum(-1)
        d_ang = (self.robot.data.root_ang_vel_w - ref_ang_vel).pow(2).sum(-1)
        r_base_vel = torch.exp(-d_vel / SIGMA_V**2 - d_ang / SIGMA_W**2)
        d_jpos = wrap_to_pi(self.robot.data.joint_pos[:, self.controlled_idx] - ref_jpos).pow(2).sum(-1)
        d_jvel = (self.robot.data.joint_vel[:, self.controlled_idx] - ref_jvel).pow(2).sum(-1)
        r_jpos = torch.exp(-d_jpos / SIGMA_Q**2)
        r_jvel = torch.exp(-d_jvel / SIGMA_QD**2)

        c_meas = self._contact_measured()
        r_contact_label = (c_meas * (2 * self.ref_contact_flags - 1)).sum(-1) / 9.0

        p_exec_b = self._contact_body_positions_body_yaw()
        p_ref_b, p_ref_w = self._ref_contact_pos(True), self._ref_contact_pos(False)
        p_exec_w = self._contact_body_positions_w()
        r_contact_pos_base = torch.exp(-(p_exec_b - p_ref_b).pow(2).sum(-1).sum(-1) / SIGMA_C**2)
        r_contact_pos_global = torch.exp(-chi_mov * (p_exec_w - p_ref_w).pow(2).sum(-1).sum(-1) / SIGMA_C**2)

        m_b = self.contact_target_flags
        d_plan = (p_exec_b - self.contact_targets).pow(2).sum(-1)
        r_planned = torch.exp(-(m_b * d_plan).sum(-1) / (SIGMA_C**2 * m_b.sum(-1).clamp(min=1.0)))

        # task: goal reaching (goal envs only)
        goal = self.goal_command()
        d_g = (base_pos[:, :2] - self.goal_pos[:, :2]).norm(dim=-1)
        dz = self.goal_height - base_pos[:, 2]
        v_toward = (base_vel[:, :2] * goal[:, :2]).sum(-1) / goal[:, :2].norm(dim=-1).clamp(min=1e-6)
        stair_col = self.cfg.terrain_types.index("stairs") if "stairs" in self.cfg.terrain_types else -1
        vmax = torch.where(self.terrain_type_id == stair_col, 0.8, 2.0)
        g_xy = (d_g < 0.5).float()
        g_h = (dz.abs() < 0.2).float()
        g_psi = (wrap_to_pi(self.goal_yaw - yaw_from_quat(base_quat)).abs() < 0.5).float()
        g_v = ((v_toward > 0.1) & (base_vel[:, :2].norm(dim=-1) <= vmax)).float()
        g_z = torch.ones_like(g_xy)  # get-up variant: 1[v_z sign(dz) > 0.1 or |dz| < 0.2]
        r_move2goal = (1 - g_xy) * g_v + g_xy * g_z
        d_foot = 1.0 - c_meas[:, [6, 7]].sum(-1) / 2.0
        gz_base = quat_apply(
            torch.cat((base_quat[:, :1], -base_quat[:, 1:]), dim=-1),
            torch.tensor(_GRAV, device=self.device).repeat(e, 1),
        )
        d_tilt = 1.0 - gz_base[:, 2] ** 2
        d_q = (self.robot.data.joint_pos[:, self.controlled_idx] - self.default_pose[self.controlled_idx]).abs().sum(-1) / 23.0
        r_standatgoal = g_xy * g_h * g_psi * torch.exp(-(d_foot + d_tilt + d_q) / 3.0)

        # regularization
        r_action_rate = (self.actions - self.prev_actions[:, 0]).pow(2).sum(-1)
        r_action_2nd = (self.actions - 2 * self.prev_actions[:, 0] + self.prev_actions[:, 1]).pow(2).sum(-1)
        tau_applied = self.robot.data.applied_torque[:, self.controlled_idx]
        tau_computed = self.robot.data.computed_torque[:, self.controlled_idx]
        r_joint_torque = tau_applied.pow(2).sum(-1)
        r_torque_limits = (tau_computed - tau_applied).abs().sum(-1)
        q = self.robot.data.joint_pos[:, self.controlled_idx]
        lo, hi = self.joint_limits[self.controlled_idx, 0], self.joint_limits[self.controlled_idx, 1]
        r_dof_pos = ((lo - q).clamp(min=0) + (q - hi).clamp(min=0)).sum(-1).clamp(max=30.0)
        qd = self.robot.data.joint_vel[:, self.controlled_idx]
        r_dof_vel = (qd.abs() - 0.9 * self.vel_limits[self.controlled_idx]).clamp(min=0, max=1.0).sum(-1)

        # style
        self.contact_duration = torch.where(c_meas > 0, self.contact_duration + self.step_dt, 0.0)
        self.prev_contact_body_vel = self.contact_body_vel
        self.contact_body_vel = self.robot.data.body_link_lin_vel_w[:, self.contact_link_idx]
        acc = (self.contact_body_vel - self.prev_contact_body_vel).norm(dim=-1).sum(-1) / self.step_dt
        r_contact_acc = torch.minimum(torch.full((e,), 1e4, device=self.device), acc)
        torso_quat = self.robot.data.body_quat_w[:, self.torso_body_idx]
        g_torso = quat_apply(
            torch.cat((torso_quat[:, :1], -torso_quat[:, 1:]), dim=-1),
            torch.tensor(_GRAV, device=self.device).repeat(e, 1),
        )
        r_flat = g_torso[:, :2].pow(2).sum(-1)
        hist = self._contact_force_history()
        f_xy_max = hist[..., :2].norm(dim=-1).max(dim=1).values
        f_z_max = hist[..., 2].abs().max(dim=1).values
        r_stumbling = (f_xy_max > 1.1 * f_z_max + 1.0).float().sum(-1)
        r_slippage = ((self.contact_duration > 0.026) & (self.contact_body_vel.norm(dim=-1) > 0.5)).float().sum(-1)
        r_angvel_xy = base_ang[:, :2].pow(2).sum(-1)

        # style-term terrain masks
        dense = (self.terrain_group_id == 0) | (self.terrain_group_id == 2)
        foot = (self.terrain_group_id == 0) | (self.terrain_group_id == 1)
        m_flat, m_stumble, m_slip, m_ang = dense.float(), dense.float(), foot.float(), dense.float()

        total = (
            w.w_track_base_pos * r_base_pos * is_ref
            + w.w_track_base_vel * r_base_vel * is_ref
            + w.w_track_joint_pos * r_jpos * is_ref
            + w.w_track_joint_vel * r_jvel * is_ref
            + w.w_track_contact_label * (1.0 - self.cfg.eta_rew) * r_contact_label * is_ref
            + w.w_track_contact_pos_base * (1.0 - self.cfg.eta_rew) * r_contact_pos_base * is_ref
            + w.w_track_contact_pos_global * (1.0 - self.cfg.eta_rew) * r_contact_pos_global * is_ref
            + w.w_track_planned_contact * (self.cfg.eta_rew * is_ref + is_goal) * r_planned
            + w.w_move2goal * r_move2goal * is_goal
            + w.w_standatgoal * r_standatgoal * is_goal
            - w.w_action_rate * r_action_rate
            - w.w_action_rate_2nd * r_action_2nd
            - w.w_joint_torque * r_joint_torque
            - w.w_torque_limits * r_torque_limits
            - w.w_dof_pos_limits * r_dof_pos
            - w.w_dof_vel_limits * r_dof_vel
            - w.w_contact_acc * r_contact_acc
            - w.w_flat_orientation * m_flat * r_flat * is_goal
            - w.w_stumbling * m_stumble * r_stumbling * is_goal
            - w.w_slippage * m_slip * r_slippage * is_goal
            - w.w_ang_vel_xy * m_ang * r_angvel_xy * is_goal
        )
        return total * self.step_dt

    # ------------------------------------------------------------------
    # termination
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv resets completed environments before observations. Capture
        # the post-physics state here so the terminal frame belongs to its clip.
        if self._record_enabled:
            self._record_step(self._compute_proprio(), self.goal_command())
        base_z = self.robot.data.root_pos_w[:, 2]
        min_z = torch.where(self.is_ref_env, self.cfg.min_height_ref, self.cfg.min_height_goal)
        grace_steps = torch.where(self.is_ref_env, self.cfg.reference_grace_steps, self.cfg.grace_steps)
        grace = self.episode_length_buf > grace_steps
        ref = self._ref_sample()
        root_error = (self.robot.data.root_pos_w - ref["root_pos"]).norm(dim=-1)
        reference_error_exceeded = self.is_ref_env & (root_error > self.cfg.max_reference_root_error_m)
        died = ((base_z < min_z) | reference_error_exceeded) & grace
        d_g = (self.robot.data.root_pos_w[:, :2] - self.goal_pos[:, :2]).norm(dim=-1)
        dz = (self.robot.data.root_pos_w[:, 2] - self.goal_height).abs()
        reached = (d_g < self.cfg.success_horiz) & (dz < self.cfg.success_vert) & ~self.is_ref_env
        self.ep_reached_goal |= reached
        self.ep_min_goal_dist = torch.minimum(
            self.ep_min_goal_dist, torch.where(~self.is_ref_env, d_g, torch.full_like(d_g, 1e9))
        )
        ref_done = self.is_ref_env & (self.phase >= (1.0 if self.cfg.replay or self.cfg.evaluation else 0.99))
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = died | reached
        truncated = time_out | ref_done
        self.extras.pop("timeout_bootstrap", None)
        timeout_ids = (time_out & ~terminated & ~ref_done).nonzero().squeeze(-1)
        if len(timeout_ids):
            vm, vfs = self.get_critic_obs()
            self.extras["timeout_bootstrap"] = {
                "env_ids": timeout_ids.detach().clone(),
                "critic_m": vm[timeout_ids].detach().clone(),
                "critic_fs": vfs[timeout_ids].detach().clone(),
            }
        if self.cfg.evaluation:
            self.extras["final_tracking"] = {
                "root_squared_error_m2": (self.robot.data.root_pos_w - ref["root_pos"]).square().sum(-1).detach().clone(),
                "joint_squared_error_rad2": (self.robot.data.joint_pos[:, self.controlled_idx] - ref["joint_pos"]).square().detach().clone(),
                "body_squared_error_m2": (self._contact_body_positions_w() - self._ref_contact_pos(False)).square().sum(-1).detach().clone(),
                "root_orientation_squared_error_rad2": quat_diff_rad(self.robot.data.root_quat_w, ref["root_quat"]).square().detach().clone(),
                "phase": self.phase.detach().clone(),
                "clip_id": self.clip_id.detach().clone(),
                "terminated": terminated.detach().clone(),
                "truncated": truncated.detach().clone(),
                "reached_end": ref_done.detach().clone(),
            }
            if self.cfg.evaluation_record_trajectory:
                data = self.robot.data
                self.extras["final_tracking"].update({
                    "root_pos_w": data.root_pos_w.detach().clone(),
                    "ref_root_pos_w": ref["root_pos"].detach().clone(),
                    "root_quat_w": data.root_quat_w.detach().clone(),
                    "ref_root_quat_w": ref["root_quat"].detach().clone(),
                    "joint_pos": data.joint_pos[:, self.controlled_idx].detach().clone(),
                    "joint_ref_pos": ref["joint_pos"].detach().clone(),
                    "joint_target_pos": data.joint_pos_target[:, self.controlled_idx].detach().clone(),
                    "joint_vel": data.joint_vel[:, self.controlled_idx].detach().clone(),
                    "joint_ref_vel": ref["joint_vel"].detach().clone(),
                    "applied_torque": data.applied_torque[:, self.controlled_idx].detach().clone(),
                    "computed_torque": data.computed_torque[:, self.controlled_idx].detach().clone(),
                    "joint_effort_limit": data.joint_effort_limits[:, self.controlled_idx].detach().clone(),
                    "measured_contact_flags": self._contact_measured().detach().clone(),
                    "ref_contact_flags": self.ref_contact_flags.detach().clone(),
                    "contact_force_w": self._contact_forces_w().detach().clone(),
                    "died": died.detach().clone(),
                    "time_out": time_out.detach().clone(),
                    "termination_height_threshold_m": min_z.detach().clone(),
                    "grace_active": (~grace).detach().clone(),
                    "reference_error_exceeded": reference_error_exceeded.detach().clone(),
                    "termination_root_error_threshold_m": torch.full_like(root_error, self.cfg.max_reference_root_error_m),
                })
        return terminated, truncated

    # ------------------------------------------------------------------
    # reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        self._finish_episodes(env_ids)
        super()._reset_idx(env_ids)

        if not getattr(self, "_dr_dynamics_applied", True):
            # per-env dynamics randomization (Table S2), applied once at spawn
            all_ids = torch.arange(self.robot.num_joints, device=self.device)
            try:
                masses = self.robot.root_physx_view.get_masses().clone()
                masses[:, self.torso_body_idx] = (
                    masses[:, self.torso_body_idx] + self.dr_mass_delta.to(masses.device)
                ).clamp(min=0.5)
                self.robot.root_physx_view.set_masses(masses, self.robot._ALL_INDICES.cpu())
            except Exception as exc:
                raise RuntimeError("Could not apply torso mass randomization") from exc
            self.robot.write_joint_stiffness_to_sim(
                self._pd_gains_asset[:, 0].unsqueeze(0).repeat(self.num_envs, 1) * self.dr_gain,
                joint_ids=all_ids,
            )
            self.robot.write_joint_damping_to_sim(
                self._pd_gains_asset[:, 1].unsqueeze(0).repeat(self.num_envs, 1) * self.dr_gain,
                joint_ids=all_ids,
            )
            self._sync_actuator_gains(
                self._pd_gains_asset[:, 0].unsqueeze(0) * self.dr_gain,
                self._pd_gains_asset[:, 1].unsqueeze(0) * self.dr_gain,
            )
            self._dr_dynamics_applied = True

        # ---- reference envs: sample clip + phase
        ref_ids = env_ids[self.is_ref_env[env_ids]]
        if self.motion_data is not None and len(ref_ids):
            if not self.cfg.replay and not self.cfg.evaluation:
                self.clip_id[ref_ids] = torch.randint(0, len(self.motion_data.clip_names), (len(ref_ids),), device=self.device)
            durs = torch.tensor(
                [(c.num_frames - 1) / c.fps if self.cfg.replay or self.cfg.evaluation else c.duration_s for c in self.motion_data.clips.values()],
                device=self.device,
            )
            self.clip_duration[ref_ids] = durs[self.clip_id[ref_ids]]
            if self.cfg.replay or self.cfg.evaluation:
                self.init_phase[ref_ids] = self.cfg.evaluation_start_phase if self.cfg.evaluation else 0.0
            elif self.cfg.stage == 1 and self.cfg.csi_enabled:
                self.init_phase[ref_ids] = torch.rand(len(ref_ids), device=self.device) * (
                    0.99 - self.csi_min_phase[ref_ids]
                ) + self.csi_min_phase[ref_ids]
            elif self.cfg.stage == 2:
                self.init_phase[ref_ids] = torch.rand(len(ref_ids), device=self.device) * 0.98
            else:
                self.init_phase[ref_ids] = 0.0
            self.phase[ref_ids] = self.init_phase[ref_ids]
            if self.cfg.evaluation:
                self.ref_lookahead[ref_ids] = sum(self.cfg.ref_goal_lookahead) / 2.0
            else:
                self.ref_lookahead[ref_ids] = torch.rand(len(ref_ids), device=self.device) * (
                    self.cfg.ref_goal_lookahead[1] - self.cfg.ref_goal_lookahead[0]
                ) + self.cfg.ref_goal_lookahead[0]

        # ---- goal envs: sample goal
        goal_ids = env_ids[~self.is_ref_env[env_ids]]
        if len(goal_ids):
            self._sample_goals(goal_ids)

        # ---- place the robot
        ref = self._ref_sample(env_ids)
        origin = self.env_origin()[env_ids] + self.cell_offset[env_ids]
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] = ref["root_pos"]
        default_root[:, 3:7] = ref["root_quat"]
        default_root[:, 7:10] = ref["root_lin_vel"]
        default_root[:, 10:13] = ref["root_ang_vel"]
        goal_mask = ~self.is_ref_env[env_ids]
        spawn = origin.clone()
        default_root[goal_mask, :3] = spawn[goal_mask] + torch.tensor([0.0, 0.0, 0.75], device=self.device)
        default_root[goal_mask, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        default_root[goal_mask, 7:13] = 0.0

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.controlled_idx] = ref["joint_pos"]
        joint_vel[:, self.controlled_idx] = ref["joint_vel"]
        joint_pos[goal_mask] = self.default_pose.unsqueeze(0).repeat(int(goal_mask.sum()), 1)
        joint_vel[goal_mask] = 0.0

        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_link_velocity_to_sim(default_root[:, 7:13], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ---- reset buffers
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.proprio_history[env_ids] = 0.0
        self.contact_targets[env_ids] = 0.0
        self.contact_target_flags[env_ids] = 0.0
        self.predicted_contacts[env_ids] = 0.0
        self.predicted_occupancy[env_ids] = 0.0
        self.ep_reached_goal[env_ids] = False
        self.ep_min_goal_dist[env_ids] = 1e9
        self.ep_record_step[env_ids] = 0
        self.contact_duration[env_ids] = 0.0
        self.contact_body_vel[env_ids] = 0.0
        self.prev_contact_body_vel[env_ids] = 0.0
        self.action_delay[env_ids] = torch.randint(
            self.cfg.dr_delay[0], self.cfg.dr_delay[1] + 1, (len(env_ids),), device=self.device
        )
        self.dr_scan_bias[env_ids] = (0.0 if self.cfg.evaluation else
                                    torch.randn(len(env_ids), self.LW, device=self.device) * 0.05)

        self._update_ref_contact_flags()
        self._update_contact_targets_for_reset(env_ids)

    def _update_contact_targets_for_reset(self, env_ids):
        eta_goal = torch.rand(len(env_ids), device=self.device) < self.cfg.eta_obs
        eta_ref = torch.rand(len(env_ids), device=self.device) < self.cfg.eta_ref_obs
        oracle_t, oracle_f = self._oracle_contacts(env_ids)
        use_oracle = ~eta_ref & self.is_ref_env[env_ids]
        use_planner = (eta_goal & ~self.is_ref_env[env_ids]) | (eta_ref & self.is_ref_env[env_ids])
        self.contact_targets[env_ids] = torch.where(
            use_planner.unsqueeze(-1).unsqueeze(-1),
            self.predicted_contacts[env_ids],
            torch.where(use_oracle.unsqueeze(-1).unsqueeze(-1), oracle_t, torch.zeros_like(self.predicted_contacts[env_ids])),
        )
        self.contact_target_flags[env_ids] = (
            (self.predicted_occupancy[env_ids] > 0.5).float() * use_planner.float().unsqueeze(-1)
            + (oracle_f > 0.5).float() * use_oracle.float().unsqueeze(-1)
        )
        self.contact_targets[env_ids] = torch.where(
            self.contact_target_flags[env_ids].unsqueeze(-1).bool(),
            self.contact_targets[env_ids],
            torch.zeros_like(self.contact_targets[env_ids]),
        )

    def _sample_goals(self, env_ids):
        tnames = self.cfg.terrain_types
        origin = self.env_origin()[env_ids]
        lvl = self.terrain_level[env_ids].float() / max(1, self.cfg.num_terrain_levels - 1)
        d = torch.rand(len(env_ids), device=self.device) * (self.cfg.max_goal_dist - self.cfg.min_goal_dist) + self.cfg.min_goal_dist
        x = origin[:, 0] + d
        y = origin[:, 1] + (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.5
        z = torch.zeros(len(env_ids), device=self.device)
        yaw = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.8
        for i, env_id in enumerate(env_ids.tolist()):
            tname = tnames[self.terrain_type_id[env_id]]
            o = origin[i]
            if tname == "stairs":
                z[i] = 0.06 + lvl[i] * 0.12
                x[i] = o[0] + min(4.0 + lvl[i] * 2.0, self.cfg.terrain_size[0] - 2.0)
            elif tname == "platform":
                z[i] = 0.15 + lvl[i] * 0.45
            elif tname == "box":
                z[i] = 0.4 + lvl[i] * 0.6
            elif tname in ("vault", "gap"):
                x[i] = o[0] + 3.5
        self.goal_pos[env_ids] = torch.stack((x, y, z), dim=-1)
        self.goal_yaw[env_ids] = yaw
        self.goal_height[env_ids] = z + 0.75

    def _finish_episodes(self, env_ids):
        """Emit summaries for CSI in every stage, with optional hindsight records."""
        if not len(env_ids):
            return
        reference = self._ref_sample(env_ids)
        root_errors = (self.robot.data.root_pos_w[env_ids] - reference["root_pos"]).norm(dim=-1)
        for position, env_id in enumerate(env_ids.tolist()):
            if int(self.episode_length_buf[env_id]) == 0:
                continue
            n = int(self.ep_record_step[env_id].clamp(max=self.cfg.record_max_steps).item())
            episode = {
                    "env_id": int(env_id),
                    "clip_id": int(self.clip_id[env_id]),
                    "is_ref": bool(self.is_ref_env[env_id]),
                    "init_phase": float(self.init_phase[env_id]),
                    "end_phase": float(self.phase[env_id]),
                    "reached_goal": bool(self.ep_reached_goal[env_id]),
                    "min_goal_dist": float(self.ep_min_goal_dist[env_id]),
                    "terrain_type": int(self.terrain_type_id[env_id]),
                    "terrain_level": int(self.terrain_level[env_id]),
                    "goal_pos": self.goal_pos[env_id].detach().cpu().clone(),
                    "terminated": bool(self.reset_terminated[env_id]),
                    "truncated": bool(self.reset_time_outs[env_id]),
                    "root_position_error_m": float(root_errors[position]),
                }
            if self._record_enabled:
                episode["record"] = self.ep_record[env_id, :n].clone()
            self.finished_episodes.append(episode)

    # ------------------------------------------------------------------
    # curriculum (called by the runner after draining finished episodes)
    def update_curriculum(self, finished: list[dict]):
        if self.cfg.evaluation:
            return
        for ep in finished:
            eid = ep["env_id"]
            if ep["is_ref"]:
                if self.cfg.stage == 1 and self.cfg.csi_enabled:
                    if (ep["end_phase"] >= 0.99 and not ep.get("terminated", True)
                            and ep.get("root_position_error_m", float("inf")) <= self.cfg.csi_success_root_error_m):
                        self.csi_min_phase[eid] = max(0.0, self.csi_min_phase[eid] - self.cfg.csi_adjust)
                    else:
                        self.csi_min_phase[eid] = min(0.99, self.csi_min_phase[eid] + self.cfg.csi_adjust)
            else:
                success = 1.0 if ep["reached_goal"] else 0.0
                self.success_ema[eid] = (1 - self.cfg.promote_ema) * self.success_ema[eid] + self.cfg.promote_ema * success
                if success and self.success_ema[eid] > self.cfg.promote_threshold:
                    self.terrain_level[eid] = min(self.cfg.num_terrain_levels - 1, self.terrain_level[eid] + 1)
                    self.success_ema[eid] = 0.0
                elif ep["min_goal_dist"] > self.cfg.demote_horiz:
                    self.terrain_level[eid] = max(0, self.terrain_level[eid] - 1)

    # ------------------------------------------------------------------
    @property
    def _record_enabled(self) -> bool:
        return (self.cfg.stage >= 2 or self.cfg.replay) and not self.cfg.evaluation
