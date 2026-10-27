# 单条 G1 箱攀：数据、场景、标签与训练管线

当前目标是先训练出能够从起点完成整条 G1 箱攀的 Stage 1 策略。数据、场景、标签和多环境复制已接通，1024 环境的新基线已启动。当前采用后台自动分段训练与验收：每训练到下一个 1000 轮边界，从 phase 0 评估完整动作；连续三个不同 checkpoint 达标即提前停止，最多训练到总计 20000 轮。无需人工逐段续训或盯日志；训练仍在进行，尚未获得完整箱攀成功的验收结果。

## 当前制品与证据

| 制品 | 用途与状态 |
| --- | --- |
| `datasets/g1_box_climb_smoke/climb_18_z_scale_1.0.*` | 首个格式/场景试样，281 帧、50 Hz、5.6 s。固定六关节后重新计算 FK；纯关节 PD 回放完整记录，但 root 最大误差 5.345 m，未通过跟踪验收。 |
| `datasets/g1_box_climb_retarget/climb_03_z_scale_1.0_scene_fit_0000_0501.*` | 原始 0–10 s、501 帧的第一版场景适配。环境接触检查通过后，独立自碰检查发现最大穿透 4.116 cm，71 帧超过 2 cm，已拒绝。 |
| `datasets/g1_box_climb_retarget/diagnostics/scene_only_contacts_rejected_self_collision.npz` | 第一版已隔离的历史标签；位于诊断目录，不能作为正式训练标签使用。 |
| `datasets/g1_box_climb_seed_v2/` | 加入自碰约束的 10 s 候选，501 帧、50 Hz。环境最大穿透 4.425 mm、自碰 5.191 mm，均通过独立 2 cm 门禁；已生成参考几何标签，尚不代表物理跟踪通过。 |
| `runs/g1_box_climb_stage1_v3_1024/` | 当前 Stage 1 新基线，1024 个独立场景副本、seed 0。首段 1000 轮、每 100 轮保存；后台在每个 1000 轮边界自动完整评估并决定续训或停止，总轮数上限 20000。新建网络和优化器，未恢复旧五轮 checkpoint；首段配置和源码哈希见 [launch_manifest.json](../runs/g1_box_climb_stage1_v3_1024/launch_manifest.json)。 |

原始数据来自 [OmniRetarget 数据集](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)，固定 revision `135259130158f1cb1799e4ab9b6566c938285fc9`。下载清单及哈希见 [外部数据清单](../datasets/external/README.md)；原始动作、URDF、mesh 和迁入的机器人资产均保留，派生制品写入独立目录或 `.cache/`。

[Drake 实证记录](../datasets/g1_box_climb_smoke/diagnostics/drake_joint_order.json)使用隔离安装的官方 Drake 1.51.1，通过 `Parser`、`GetPositionNames()` 和每个 joint 的 `position_start()` 确认：36 列是 **WXYZ 四元数、XYZ 平移、源 spherehand URDF 声明顺序的 29 个关节**。机器人单独解析和加入原地形/固定地面两种情况均一致；原始首、中、末帧的具名关节角及 root XYZ 误差为 0。没有运行上游可视化脚本；作者生成时的 Drake 版本未提供。

## 标签和模型约定

[ACP 论文](https://arxiv.org/html/2610.31608v1)的参考接触由重定向工具的模拟接触几何提取，Stage 2 hindsight 则来自策略实际 rollout 的接触力。工程分别提供：

- `label_reference_geometry`：参考几何标签。使用实际 collision 几何和场景关联；1 cm 接触距离、2 cm 环境/自碰门限是本项目的工程阈值，并非论文公布的验收阈值。
- `label_contacts`：完整物理回放的力记录和 hindsight。沿用垂直力大于 1 N、至少 10 个控制步、未来 75 步（50 Hz 下 1.5 s）的处理；回放失败只保留诊断，不生成有效力标签。

九列在工程中统一为左右手、左右肘、左右膝、左右脚、骨盆，并保存具名列顺序。几何接近不证明能够支撑受力；未训练的纯关节 PD 失败也不能改写为成功的物理标签。

这里的“手”是接触部位名称，不表示灵巧手、手指关节或抓握。论文 Fig. 1 的实机采用无手指的臂端；正文 Fig. 5A 区分低箱直接迈上去与高箱借助双臂末端支撑。九列是可选接触通道，并不要求每条动作都使用全部部位。几何检查默认允许仅有双足接触的箱体动作；只有明确指定 `--require-hand-contact` 时才要求持续臂端支撑。

当前模型的 `hands` 实际对应 `left/right_wrist_yaw_link` 上的固定胶囊碰撞体，没有手指自由度。OmniRetarget 源模型则使用固定连接的半球形末端网格，两者并非同一几何；原论文实机末端的精确碰撞形状和质量尚未对齐。下文 99.733% 接触保留率是在本地 MJCF 上对源关节角与派生轨迹做比较，**不能证明保留了源半球模型的接触，更不证明已匹配 ACP 实机**。运行时资产审计只验证本地 MJCF 与其 Isaac 导入结果一致。

当前工程使用 **29 个实体关节、23 个控制关节、六个 nominal held 关节**。论文明确策略输出 23 个关节指令，但这一点不能单独证明作者采用了本工程相同的 29 轴锁六轴方案。held 关节变更后必须重建全身 FK、速度和接触，不能只删六列。场景适配使用有界的工程优化，并分别核验环境穿透、自碰、接触保留和速度约束。

场景参考自动使用本地规范化 MJCF。其显式缓存消除 importer 对默认值/命名的歧义；真实 Isaac smoke 已确认 29 joints、30 bodies、187 条扫描射线和 468 维 actor observation，并完成 20 步。默认 G1 USD 含手指，共 43 joints，不用于认证本场景参考。不同物理后端使用同一源模型，仍需分别验收动力学。

场景文件绑定 motion SHA256、源 URDF/mesh 哈希、地面和坐标变换 fingerprint；标签绑定源动作与场景，checkpoint 继续绑定模型、动作、标签及场景。动作与全部碰撞表面共享首帧 root XY 平移，禁止独立 spawn offset。当前支持 **一个 clip 的多环境复制、Stage 1/2、reference ratio=1**；每个环境绑定独立 tile 中的同一份原始场景，机器人、扫描与接触坐标共享该 tile 原点。tile 副本属于运行布局，不改变原始场景 fingerprint。仍拒绝混入其他 clip、Stage 3、flat 替换及丢失场景绑定的通用裁剪/投影。

Isaac 中每个解剖部位独立建立环境过滤 sensor，过滤目标为 `/World/terrain/terrain/mesh`，排除自碰对标签/奖励的污染。该接口报告法向接触力；不把它称为包含全部切向摩擦的受力测量。

## 当前训练的坐标、课程与兼容性

- **Oracle 坐标**：几何标签保存在离散查询帧的参考 root-yaw 坐标中，Z 为绝对高度。运行时先用该查询帧的参考 root/yaw 还原固定世界锚点，再应用首帧 XY rebase 和 tile 原点，最后转入当前机器人的 root-yaw 坐标。标签索引使用离散帧，避免在两个采样帧之间错误移动锚点。新接口标记为 `oracle_contact_frame=reference_query_to_actual_body_yaw_v1`，无效目标保持零。
- **旧标签**：已有参考几何标签可通过生成方法识别坐标语义。旧版物理回放标签未保存生成时的实际 query root/yaw，不能假设它与参考完全重合；未知坐标的 scene-bound 标签拒绝训练，无场景旧数据保留原行为并发出警告。后续应为物理标签补存查询姿态，目前不把缺失信息补写为参考姿态。
- **CSI 工程门限**：reference 环境的初始化宽限为 5 个控制步（0.1 s）；宽限结束后 root 位置误差超过 0.5 m 或触发最低高度条件即物理终止。只有到达动作尾部 `phase >= 0.99`、未物理终止且终止帧 root 误差不超过 0.3 m，才按成功向前扩展课程。这些门限属于本项目的工程设置，并非论文公布值。尾段课程完成率不能替代从 phase 0 开始的整段成功率。
- **Critic 与 PPO**：reference critic 的参考接触位置减去 tile 平移，保持场景局部坐标；参考运动学流使用 root 速度与关节量，不引入 tile 的世界 XY 位置，避免同一动作因副本位置不同而产生不同输入。PPO 的 reference 样本始终使用 value clipping；只允许新加入的 goal critic 在其 warmup 期间跳过该裁剪。

上述 oracle 和 critic 观测语义已纳入 checkpoint 契约。旧五轮 checkpoint 仅保留作历史管线诊断，当前实现会拒绝把它当成兼容模型继续训练或评估。此次修复没有修改 `seed_v2` 动作、接触标签、原始数据及其哈希；正式训练从新的网络和优化器开始。

## 可复现命令

以下参数已逐项核对本机 `--help`。生成脚本拒绝覆盖：已有制品可直接进入检查；重新生成时将输出路径改为新的空目录。

```bash
cd /media/user/DATA/Morph_mimic
REFERENCE_PY=/home/user/miniconda3/bin/python
ISAAC_PY=/home/user/miniconda3/envs/env_isaaclab_2/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

`REFERENCE_PY` 是本机有 MuJoCo/SciPy 的 CPU 数据处理环境；`ISAAC_PY` 用于 Isaac/PhysX 和训练。GPU 命令在 Codex 文件系统沙箱内需以工具的 `require_escalated` 执行，普通沙箱无法可靠访问 GPU；它不是脚本的 CLI 参数。

### 1. 转换样本并准备本地模型缓存

```bash
"$REFERENCE_PY" -m morph_mimic.scripts.import_omniretarget \
  --clip climb_18_z_scale_1.0 \
  --output-dir datasets/g1_box_climb_smoke

"$REFERENCE_PY" -m morph_mimic.scripts.prepare_runtime_mjcf --robot g1
```

输出保留 `raw_reference/` 的 29 DOF 对照、转换 provenance 和 `.scene.json`；转换本身不写接触标签。显式模型缓存位于 `.cache/mjcf/g1/`，Isaac USD 转换缓存位于 `.cache/usd/g1_mjcf/`。

### 2. 生成带场景约束的 10 s 候选

```bash
BOX_DIR=datasets/g1_box_climb_seed_v2
BOX_CLIP=climb_03_z_scale_1.0_scene_fit_0000_0501

"$REFERENCE_PY" -m morph_mimic.scripts.retarget_scene_reference \
  --clip climb_03_z_scale_1.0 --start-s 0 --duration-s 10 \
  --max-nfev 30 --output-dir "$BOX_DIR"
```

v2 加入自碰处理，保留原模型有效 collision masks、excludes 和 parent/weld 过滤。优化目标和独立验收同时生效；不能通过修改门限、屏蔽碰撞或删掉失败帧使候选过关。

### 3. 独立几何检查，合格后才写参考标签

```bash
"$REFERENCE_PY" -m morph_mimic.scripts.check_reference_self_collision \
  --motion "$BOX_DIR/$BOX_CLIP.npz" \
  --scene "$BOX_DIR/$BOX_CLIP.scene.json" \
  --raw-reference "$BOX_DIR/raw_reference/$BOX_CLIP.npz" \
  --output "$BOX_DIR/diagnostics/self_collision_recheck.json"

"$REFERENCE_PY" -m morph_mimic.scripts.label_reference_geometry \
  --robot g1 --motion "$BOX_DIR/$BOX_CLIP.npz" \
  --scene "$BOX_DIR/$BOX_CLIP.scene.json" \
  --compare-raw "$BOX_DIR/raw_reference/$BOX_CLIP.npz" \
  --report "$BOX_DIR/diagnostics/geometry_recheck.json" --write-labels
```

两个检查都必须通过，并核对 retarget 报告的接触保留与速度约束。成功标签位于动作旁的 `*_contacts.npz`；已有正式标签时重检应去掉 `--write-labels`。任何失败均保留报告并返回非零状态，诊断 NPZ 不能手动移回正式标签路径。

### 4. 物理基线与 Isaac 环境检查

```bash
"$REFERENCE_PY" -m morph_mimic.scripts.validate_scene_reference \
  --robot g1 --motion "$BOX_DIR/$BOX_CLIP.npz" \
  --scene "$BOX_DIR/$BOX_CLIP.scene.json" \
  --output "$BOX_DIR/diagnostics/pd_recheck.json" \
  --trajectory "$BOX_DIR/diagnostics/pd_recheck_trajectory.npz"

"$ISAAC_PY" tests/smoke_env.py --robot g1 --stage 1 \
  --num_envs 1 --steps 20 --device cuda:0 --headless --no-use_usd \
  --motion_dir "$BOX_DIR" --clips "$BOX_CLIP"

"$ISAAC_PY" -m morph_mimic.scripts.label_contacts \
  --robot g1 --device cuda:0 --headless \
  --motion_dir "$BOX_DIR" --clips "$BOX_CLIP" \
  --episode_length_s 20 --record_max_steps 2000 \
  --diagnose-only --report "$BOX_DIR/diagnostics/isaac_pd_recheck.json"
```

CPU PD 基线只初始化一次 free root，不持续覆盖 root 或加 root 助力。Isaac 诊断使用 `--diagnose-only` 保留参考几何标签。两者分别报告根部误差、关节误差和完整覆盖情况；环境 smoke 只证明模型、传感器和控制接口能够运行。

### 5. 多环境复制审计与当前 Stage 1 训练

32 和 1024 环境的真实 GPU 复制审计均已通过，分别见 [32 环境报告](../runs/g1_box_climb_diagnostics_20261027/scene_replication_32.json) 与 [1024 环境报告](../runs/g1_box_climb_diagnostics_20261027/scene_replication_1024.json)。审计检查独立场景原点、状态/参考平移一致性、扫描、过滤接触信号、20 步有限状态和局部 reset 隔离；报告明确 `physical_skill_validated=false`，不代表零动作能够完成箱攀。

需要重做复制检查时使用新的报告路径，例如：

```bash
"$ISAAC_PY" -m morph_mimic.scripts.audit_scene_replication \
  --robot g1 --device cuda:0 \
  --motion-dir "$BOX_DIR" --clips "$BOX_CLIP" \
  --num-envs 32 --steps 20 \
  --out runs/g1_box_climb_diagnostics_20261027/scene_replication_32_recheck.json
```

该审计脚本默认 headless。需要 1024 环境检查时把 `--num-envs` 改为 `1024`，同时使用新的输出文件名。

只对已通过几何、自碰、接触保留和速度检查、且有有效 scene-bound 标签的候选开始训练。以下记录当前首段训练的启动参数，`1000` 是第一个训练边界；后续由后台管理程序自动严格恢复同一运行，无需人工重复启动。独立复跑应另选新的 `--log_dir`，避免混合现有运行记录：

```bash
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
"$ISAAC_PY" -m morph_mimic.scripts.train_stage1 \
  --robot g1 --device cuda:0 --headless --num_envs 1024 \
  --motion_dir "$BOX_DIR" --clips "$BOX_CLIP" \
  --max_iterations 1000 --save_interval 100 --seed 0 \
  --log_dir runs/g1_box_climb_stage1_v3_1024
```

首段日志为 [training.log](../runs/g1_box_climb_stage1_v3_1024/training.log)，启动时的冻结源码哈希与门限见 [launch_manifest.json](../runs/g1_box_climb_stage1_v3_1024/launch_manifest.json)。这份初始清单记录首段 1000 轮，不代表整个自动任务的最终预算。后台等待首段正常完成后，保持 1024 环境、seed 0、同一动作/标签/场景和训练契约，严格恢复网络、优化器、课程与随机状态，按总轮数 2000、3000 等边界继续。不会借续训修改原始数据、放宽验收门限或切换旧 checkpoint。

### 6. 后台自动验收、续训与停止

每到总计 1000 轮的边界，后台保存该边界的独立 checkpoint，校验契约后自动进行确定性的完整动作评估：`start_phase=0`、单场景、标称动力学和干净观测。评估必须覆盖整段并正常完成；局部起点、短预算、异常中断和尾段 CSI 完成率均不能替代这一检查。

| 自动验收项 | 每个完整评估必须满足 |
| --- | --- |
| 整段完成 | `completed=true`，从 phase 0 开始，无物理失败或预算截断 |
| root 位置 RMSE | `root_position_rmse_m <= 0.15` |
| root 最大位置误差 | `root_position_max_error_m <= 0.30` |
| 关节角 RMSE | `joint_rmse_rad <= 0.30` |
| root 朝向 RMSE | `root_orientation_rmse_rad <= 0.50` |

这些是本项目为当前 clip、模型与场景设置的工程验收标准，并非论文公布的成功阈值。每项指标都必须有限，checkpoint 与评估的模型、数据和观测契约必须一致。连续 **三个不同训练边界的 checkpoint** 全部满足表中条件，才将自动任务标记为通过并提前停止；中间一次未达标会重置连续通过计数，继续训练。

总轮数上限为 **20000 轮**，不是额外增加 20000 轮。按每轮 24 个控制步、1024 环境计，最多采集 **491520000 个环境控制步**。达到上限仍未取得连续三次通过时，结束并明确报告未通过；训练、评估或校验失败同样保留错误与制品，不把任务标记成功。平均奖励上涨、训练正常退出和用完预算均不构成验收通过。

后台状态与制品统一位于 `runs/g1_box_climb_stage1_v3_1024/`：

| 路径 | 内容 |
| --- | --- |
| `supervisor_status.json` | 当前阶段、总预算、已完成轮数、连续通过计数、评估结果和停止原因；作为后台任务状态入口。 |
| `supervisor.log` | 自动训练管理程序的运行日志。 |
| `progress.log` | 独立只读跟随器写入的实时进度，自动跟随各训练段日志，并显示整段评估、连续达标次数与最终停止原因。 |
| `logs/` | 后续各训练段和评估子进程的日志。 |
| `evaluations/` | 各边界从 phase 0 开始的完整评估报告与轨迹。 |
| `checkpoints/` | 各训练边界的独立 checkpoint，避免后续 `final.pt` 覆盖已评估权重。 |

当前无需人工滚动监督。后台最终状态决定本轮是否通过，相关阈值、检查点和评估报告均可复查；训练进行中不预填成功率，也不声称已经复现技能。

需要查看实时日志时，在项目根目录运行：

```bash
tail -n 30 -F runs/g1_box_climb_stage1_v3_1024/progress.log
```

跟随器已在后台运行，不占用 GPU，不改变训练。`Ctrl+C` 只退出日志查看。日志出现 `STOP validated` 表示连续三个不同 checkpoint 达标后已自动停止；`STOP budget_exhausted` 表示达到 20000 轮仍未通过验收；`STOP failed` 表示异常停止，日志会给出原因。通常无需手动停止训练。`supervisor_status.json` 的 `current_iteration` 是已完成检查点轮数，`progress.log` 另外显示当前训练日志中的实时轮次。

也可直接在终端运行只读跟随器：

```bash
python -m morph_mimic.scripts.watch_training \
  --run-dir runs/g1_box_climb_stage1_v3_1024 --follow
```

不要固定跟随首段 `training.log` 或 `supervisor.log` 来判断实时进度：前者只覆盖首个 1000 轮，后者主要记录管理程序结束与异常；后续子进程日志位于 `logs/`。

## 当前新增验收记录

| 项目 | 结果与制品 |
| --- | --- |
| 32 / 1024 环境场景复制 | 两项真实 GPU 审计均通过，分别有 32 / 1024 个独立原点；扫描、参考坐标、接触传感器和局部 reset 隔离通过。各执行 20 步，未验证完整物理技能。见上文两份审计报告。 |
| 新版 oracle / critic / PPO 与 CSI | 已修复并通过相关自动化检查；当前新运行使用更新后的观测契约和工程终止/完成门限。源码哈希固定在启动清单。 |
| 1024 环境 Stage 1 与后台验收 | `g1_box_climb_stage1_v3_1024` 已启动，seed 0；首段 1000 轮，后续每 1000 轮自动完整评估并续训，连续三个不同 checkpoint 达标提前结束，最多总计 20000 轮。最终结果以 `supervisor_status.json` 和对应评估制品为准，目前未验收通过。 |

## 历史验收记录：旧五轮管线检查

以下是修复观测坐标与训练终止规则之前的记录，保留原报告以便追溯；不作为当前代码的训练结果，也不使用旧 checkpoint 重跑当前 recipe。

| 项目 | 结果与制品 |
| --- | --- |
| v2 候选生成、环境/自碰/接触保留/速度门禁 | 已通过；原始几何接触召回 99.733%，root 最大修正 3.75 cm，anchors RMS 3.67 cm，最大关节速度 9.51 rad/s。见 [retarget 报告](../datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501.retarget.json) 与 [自碰检查](../datasets/g1_box_climb_seed_v2/diagnostics/self_collision_check.json)。 |
| GPU 资产、FK 与传感器 | 已通过。29 关节/30 body/33 碰撞形状匹配；最大质量差 6.37e-7 kg、惯量差 1.09e-8 kg·m²，四姿态 FK 最大位置差 5.15e-7 m。见 [运行时资产审计](../datasets/g1_box_climb_retarget/diagnostics/isaac_runtime_audit.json)。该审计使用显式传送姿态，只验证模型，不代表动力学跟踪成功。 |
| 完整物理 PD 回放 | 未通过。Isaac 在相位 0.152 终止，记录 77/501 帧，根部最大误差 1.2255 m；没有覆盖参考几何标签或生成有效受力标签。见 [回放报告](../datasets/g1_box_climb_seed_v2/diagnostics/isaac_pd_replay.json)。 |
| 五轮 PPO 与 checkpoint | 已完成，120 个控制步；最终权重及记录的 TensorBoard 指标均有限，主优化器执行 200 次更新，场景 fingerprint 已写入 checkpoint。见 [final.pt](../runs/g1_box_climb_pipelinecheck_v2/final.pt)。 |
| 从 phase 0 的真实策略评估 | 未完成箱攀：76 步时记录物理终止，相位 0.152，根部最大误差 1.2411 m，关节 RMSE 0.3279 rad。旧代码使用 75 步宽限，76 步是宽限结束后报告失败的时刻，不能解释为此前一直稳定、到第 76 步才开始跌倒。见 [历史独立评估报告](../runs/g1_box_climb_pipelinecheck_v2/evaluation.json)。报告中的 `evaluation_is_full=true` 表示该次评估观察到终止，完成率仍为 0。 |
| 历史自动化检查 | 当时 Isaac Python 全套 95 passed、18 skipped、114 subtests passed；默认 Python 的 MuJoCo/SciPy 相关检查 47 项通过。这些是当时的验收计数，不能代替当前实现的新测试和 GPU 审计。 |

旧管线状态见 [validation_status.json](../datasets/g1_box_climb_seed_v2/validation_status.json)，历史执行日志保存在 [旧训练日志目录](../runs/g1_box_climb_pipelinecheck_v2/logs/)。每环境独立场景复制已实现并通过上述 GPU 审计；当前任务是完成新 Stage 1 训练及从起点的技能验收。Stage 2 只接收成功回录轨迹，历史失败 PD 和失败短训练轨迹不能入库。

相关入口：[数据来源](DATA_SOURCES.md)、[复现计划](REPRODUCTION_PLAN.md)、[Drake 顺序证据](../datasets/g1_box_climb_smoke/diagnostics/drake_joint_order.json)、[v1 自碰拒绝报告](../datasets/g1_box_climb_retarget/diagnostics/self_collision_check.json)。
