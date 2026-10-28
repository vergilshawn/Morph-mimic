# G1 箱攀管线与统一检查

目标是从真实起点完成整条箱攀，并在箱顶持续支撑。当前训练目录为 `runs/g1_box_climb_contact_teacher_v3_1024`，已启动并正常迭代。最终通过 1024 环境、240 个带噪控制步压力检查，以及短训练 5→6 轮保存恢复和完整评估链路。

2026-10-28 已核对 V3 第 3000 轮单次全部通过，第 3500 轮仍完成整段箱攀、旧箱顶门禁 50/50 通过，但新增接触保持未过：双足参考姿态超出 0.35rad，11.50–11.54s 三个端点的 CoM 支撑余量低于 5mm。成功候选 3000 权重仍保留；这不是连续三次验收完成，也不代表完整 ACP 多技能结果。权重和详细记录仅在本机。

## 数据与控制

参考为 `datasets/g1_box_climb_contact_v3/climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3`，551 帧、50 Hz、11 秒。原 `g1_box_climb_seed_v2` 的 501 帧时序数组保持原字节，追加 50 帧平滑 settle；接触标签针对完整新序列重新生成。追加段是本地工程补充，详见 [数据说明](../datasets/g1_box_climb_contact_v3/README.md)。

原末帧的贴近箱顶与静止稳定目标存在冲突：同一 5mm 几何候选口径下，CoM 支撑余量由 −5.20cm 改为 +3.204cm。此数值仅证明新目标的几何潜力，不证明存在真实受力平衡解；训练验收仍要求额外 50 个真实控制端点通过。[冲突证据与论文口径](PAPER_METRIC_AUDIT.md)

同目录 `.npz`、`_contacts.npz`、`.scene.json` 绑定动作、标签、场景和哈希。标签是几何候选，`physics_validated=false`，不能当成成功物理回放。G1 为 29 轴物理模型、23 个策略输出，其余六轴 held nominal。四元数 WXYZ，速度原点和 body 名称按输入契约核对；既有 policy 接触帧为根中心 XY/yaw、绝对 Z，表面与 body 原点分开保存。

## 两个检查入口

在项目根目录运行；默认检查不启动仿真或训练：

```bash
conda activate env_isaaclab_2
python -m morph_mimic.scripts.check \
  --motion-dir datasets/g1_box_climb_contact_v3 \
  --clip climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3
python -m morph_mimic.scripts.inspect --help
```

`check` 按基础回归 → 资产 → 参考/标签/场景契约顺序执行，失败即停；日志位于 `runs/_reports/checks/<UTC时间>/`，`latest.json` 指向最近结果。默认回归包含当前控制所需的 272 项检查；数据转换、Stage 2/3 planner、辅助基线、诊断及 CPU MuJoCo 检查按 `--suite` 选择。七个合并模块的分工见 [测试说明](../tests/README.md)。

```bash
# 临时选择检查；仍按依赖顺序执行
python -m morph_mimic.scripts.check --only regression --pytest-args='-k contact -v'
python -m morph_mimic.scripts.check --only regression --suite reference
python -m morph_mimic.scripts.check --only regression --suite basic planner
python -m morph_mimic.scripts.check --only regression --suite all
python -m morph_mimic.scripts.check --only assets --robot mojia

# 当前数据的额外 CPU 自碰检查；使用有 MuJoCo 的 Python
/home/user/miniconda3/bin/python -m morph_mimic.scripts.check \
  --only reference --self-collision \
  --motion-dir datasets/g1_box_climb_contact_v3 \
  --clip climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3
```

新版保存检查点出现后，严格验收通过同一入口调用；以下以第 500 轮为例。必须指定 `teacher-v3` 和新参考，不能把旧 ACP/站立/v2 权重作为 v3 严格续训：

```bash
python -m morph_mimic.scripts.check --gpu-only \
  --checkpoint-kind teacher-v3 \
  --checkpoint runs/g1_box_climb_contact_teacher_v3_1024/checkpoints/iter000500.pt \
  --motion-dir datasets/g1_box_climb_contact_v3 \
  --clip climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3
```

报告区分评估正常完成与技能达标；技能失败保留证据并返回非零退出码。短 GPU 步进只证明链路可运行，不证明站立或箱攀。`--checkpoint-kind acp/standing/teacher-v2/teacher-v3` 分别使用对应契约。

`inspect` 的专项功能包括资产、场景、held 轴、坐标、自碰、接触、分段评估、视频和播放。GUI 策略播放需入口支持相应模型契约；当前 V3 优先使用已保存物理轨迹离线渲染，避免把 584 维教师交给默认 ACP 播放器。

## 新版教师与验收

V3 为名义动力学特权教师：584 维 actor 输入，`reference_q + 0.25 × residual`。显式读取旧 v2 的 `checkpoints/iter002000.pt`，仅转入 actor；新 critic、优化器、探索噪声和迭代进度从零开始。这是新实验初始化，不是原训练严格续训，也没有转入独立站立模型。

基础 PPO 与 Table 3 奖励保留，另有常开的世界根 XYZ/高度/双足位置项。新增工程项为参考表面姿态、法向距离、有效 stance 的实测切向滑动、非预期身体接触和仅在保持时开启的支撑稳定奖励。实际 normal 接触点与 friction patch 分别从 PhysX 读取，覆盖全身 30 个刚体并计算力臂矩；独立扭转/滚动接触偶矩及机器人自接触力仍不可得。

训练采用分层 RSI：至少 15% 起点，约 35% 覆盖 3.2–4.5 秒承重转换，其余覆盖整段。初始化仅发生在 episode 边界，过程中无锁根、姿态传送或卡脚暂停。运动中允许参考手、膝等多部位支撑，不能全程强制双足或直立。

配置为 1024 环境、seed 29、最多 15000 轮、每 500 轮评估。每次从 phase 0 完整运行 **11 秒参考 + 1 秒额外真实保持，共 600 个控制步**。所有条件均需通过：

- 完整执行、无物理失败或预算截断；根位置 RMSE≤0.15m、最大误差≤0.30m、关节 RMSE≤0.30rad、根朝向 RMSE≤0.50rad、held 最大偏离≤0.01rad。
- 50 个额外保持端点全部通过旧箱顶门禁：根高度/倾斜、双足箱顶支撑中心余量、足底高度及双足当前正向 Fz 各≥20N。
- 50 个端点还全部通过全身实测门禁：无额外身体承重、有效实际受力支撑凸包、CoM 内侧余量与速度、双足姿态；传感器健康和容量检查通过。

连续三个不同检查点通过才自动停止为 `validated`；达到预算仍不成功为 `budget_exhausted`，执行异常为 `failed`。站立辅助任务、局部片段和到达参考片尾均不能代替该验收。最终启动检查见 [readiness.json](../runs/_reports/preflight/contact_surface_v3_readiness.json)。正式运行冻结了 32 个源码文件；输入与源码哈希漂移会拒绝继续训练。足部探索曾产生 33 个接触点，最终导出容量设为每刚体 128 点，并保留溢出、映射和力重建健康门禁。

查看正式训练：

```bash
tail -n 30 -F runs/g1_box_climb_contact_teacher_v3_1024/progress.log
```

## 查看真实轨迹

检查点评估轨迹出现后，可离线生成对照视频；以下路径是第 500 轮示例：

```bash
/home/user/miniconda3/bin/python -m morph_mimic.scripts.inspect render \
  --trace runs/g1_box_climb_contact_teacher_v3_1024/evaluations/iter000500.npz \
  --motion datasets/g1_box_climb_contact_v3/climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3.npz \
  --output-dir runs/g1_box_climb_contact_teacher_v3_1024/visualizations/iter000500 \
  --hold-seconds 0 \
  --ffmpeg /home/user/miniconda3/envs/env_isaaclab_2/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
```

渲染只重建已记录状态，不重新执行策略；失败后的定格不是平衡。独立评估分别报告控制开始时真正收到的 body-origin 指令与实测 onset 落点误差、参考表面事件代理、接触有无/覆盖和箱顶保持，缺失事件不记为零误差。当前是单动作教师，未完成论文多 seed/多 episode 基准，详见 [指标审计](PAPER_METRIC_AUDIT.md)。

旧 v2 已标记 `superseded` 并停在 3000 轮；2000 轮单次通过旧门禁，2500/3000 轮失败，未达到连续三次通过。其权重与评估仍保留作基线。更早失败运行的必要证据集中在 `runs/_reports/baselines/`，清理映射见本机 `runs/_reports/cleanup.json`。新的奖励、数据或物理配置必须使用独立目录并重新检查，不原地修改运行中的冻结输入。

本机基础 Python 未安装 ffmpeg，示例显式指定 Isaac 环境内的打包二进制；其他机器换成自己的 ffmpeg。已有 [短流程对比视频](../runs/_reports/preflight/teacher_v3_preview/policy_vs_reference.mp4)：真实 600 步/12 秒，2 倍慢放，无额外定格。它通过旧箱顶门禁和实际 CoM 支撑检查，但左足末端姿态约 0.42rad，超过 0.35rad，因此新增验收正确返回失败；正式训练继续优化。
