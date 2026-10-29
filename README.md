# Morph Mimic

基于 Isaac Lab 的 ACP 复现工程，包含从 supermimic 迁移的 G1 和 mojia 资产。当前在解决单任务箱攀的控制、接触与稳定性，**完整 ACP 多技能复现尚未完成**。

**参考教师 `runs/g1_box_climb_contact_teacher_v3_1024` 已在第 6000 轮成功停止。** 第 5000、5500、6000 轮三个不同检查点均完成 600 个真实控制步、箱顶保持和接触稳定各 50/50，状态为 `validated`。这验证的是单动作、名义动力学的参考增强教师；下一阶段为独立的 468 维学生课程，入口见 [课程说明](docs/BOX_CLIMB_PIPELINE.md#教师验收后的学生课程)。

2026-10-28 的第 3000 轮曾单次通过，第 3500 轮部分指标退步；后续 5000/5500/6000 连续达标。最终三次根位置 RMSE 分别为 5.59/5.99/6.64cm；三个检查点不是三个独立 seed，尚未完成论文多技能/多种子统计。

V3 仍是 584 维特权参考教师，采用 `reference_q + 0.25 × residual` 与名义动力学。新参考保留原 501 帧，追加 50 帧本地 settle，形成 11 秒动作，再执行 1 秒真实物理保持，共 600 个控制步。新增全身 30 个刚体的真实 normal/friction、作用点与力臂矩，接触表面姿态/法向/stance slip 奖励，以及保持阶段的实际支撑凸包、CoM 与稳定门禁；运动中允许参考手、膝等多部位支撑。

从旧 v2 第 2000 轮**仅初始化 actor 权重**，critic、优化器、探索噪声与迭代进度重新开始。旧 v2 已在第 3000 轮后停止并标记 `superseded`：2000 轮通过旧门禁，2500/3000 轮失败，未达到连续三次验收；不能称 v2 已验证成功。其权重和评估仍保留。独立站立模型的成功也不代表箱攀成功。

在项目根目录使用两个检查入口：

```bash
conda activate env_isaaclab_2
python -m morph_mimic.scripts.check \
  --motion-dir datasets/g1_box_climb_contact_v3 \
  --clip climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3
python -m morph_mimic.scripts.inspect --help
```

`check` 顺序执行基础回归 → 资产 → 动作/标签/场景契约，失败即停，默认不启动 GPU 或训练。`tests/` 已由 47 个模块合并为 7 个主题模块；专项用 `--only regression --suite <组名>` 选择，组名包括 `reference`、`planner`、`auxiliary`、`diagnostics`、`physics`、`all`，可继续用 `--pytest-args` 筛选。详见 [测试分组与命令](tests/README.md)。`inspect` 汇总专项诊断、轨迹渲染和播放，用 `inspect 子命令 --help` 查看参数。

Git 仓库包含源码、测试和机器人模型资产；动作、标签、训练权重、评估轨迹与启动验收记录保留在本机。新克隆尚不能直接启动同一训练，数据重建与启动汇总入口的缺口见 [复现边界](docs/REPRODUCTION_PLAN.md#仓库范围与复现边界)。可先安装依赖并运行基础回归；缺失的可选后端或本机夹具会明确跳过：

```bash
python -m pip install -e '.[test]'
python -m morph_mimic.scripts.check --only regression --suite basic
```

查看训练进度：

```bash
tail -n 30 -F runs/g1_box_climb_acp_student_v1_1024/progress.log
```

| 路径 | 用途 |
| --- | --- |
| `datasets/g1_box_climb_contact_v3/` | 新版 551 帧参考、几何接触标签、绑定场景及来源记录 |
| `datasets/g1_box_climb_seed_v2/` | 保留的 10 秒训练参考；新参考前 501 帧的来源 |
| `datasets/external/` | OmniRetarget、PHP、VideoMimic 独有源动作和地形 |
| `runs/g1_box_climb_contact_teacher_v3_1024/` | 6000 轮已验收的新版教师；冻结保留 |
| `runs/g1_box_climb_acp_student_v1_1024/` | 已启动的 468 维学生 DAgger；教师混合概率逐步降至零，再独立验收 |
| `runs/g1_box_climb_reference_teacher_v2_2048/` | 已停止的 v2 基线权重与评估 |
| `runs/g1_standing_aux_v1_1024/` | 已验收站立模型与证据 |
| `runs/_reports/` | 统一检查、预检、论文口径审计与历史证据 |

源码、测试、机器人模型资产和小型数据说明随 Git 保存；动作、标签、训练权重、缓存与运行制品在本机保留。新增 settle 和稳定门禁是本地工程补充，不是作者原始数据；短预检通过也不能证明箱攀已学会。当前限制、原文指标与工程指标的区别见 [论文指标审计](docs/PAPER_METRIC_AUDIT.md)。

详见 [箱攀管线与命令](docs/BOX_CLIMB_PIPELINE.md)、[复现路线](docs/REPRODUCTION_PLAN.md)、[新版数据说明](datasets/g1_box_climb_contact_v3/README.md)、[数据来源](docs/DATA_SOURCES.md) 和 [第三方许可](THIRD_PARTY_NOTICES.md)。训练中的输入与冻结源码不得原地改写。

本机另有早期第 6 轮短流程视频 `runs/_reports/preflight/teacher_v3_preview/policy_vs_reference.mp4`（旧 v2 actor 初始化，2 倍慢放，无额外片尾定格）。该早期模型实际完成 600 个控制步，但左足姿态误差超标；它不代表后续正式训练的最新结果。
