# ACP 复现状态与后续路线

截至 2026-10-28，资产、场景绑定、几何标签、PPO 与三阶段原型已具备；完整 ACP 多技能复现仍未完成。目标为 [ACP](https://leggedrobotics.github.io/ACP/)，[MGMT](https://wholebodylocomotion.github.io/)用于前序设计对照，OmniRetarget 提供公开箱攀动作和场景来源。原文与本地指标区别见 [论文指标审计](PAPER_METRIC_AUDIT.md)。

## 当前状态

| 部分 | 状态 |
| --- | --- |
| 资产 | G1/mojia 从 supermimic 迁移，模型、来源清单和哈希保留；原始动作尚不等于完整配对多技能训练集 |
| 新版参考 | `datasets/g1_box_climb_contact_v3`，551 帧/11 秒；原 501 帧保留，另加 50 帧本地 settle，几何标签与场景重新绑定 |
| 新版训练入口 | `runs/g1_box_climb_contact_teacher_v3_1024`；已通过启动检查，正在正式训练 |
| V3 正式评估 | 已核对 3000 轮单次全部通过；3500 轮仍完成箱攀，但足姿与三端点 CoM 余量失败，尚未连续三次验收 |
| V3 控制 | 584 维特权教师、参考关节加残差、名义动力学，属于独立工程实验，非有限观测 ACP 学生 |
| 接触与稳定 | 全身 30 刚体真实 normal/friction、作用点/力臂矩、表面姿态/法向、stance 滑动、实际支撑凸包与 CoM；终点稳定单独验收 |
| V2 基线 | 已在 3000 轮后 `superseded`；2000 轮通过旧门禁，2500/3000 轮失败，未达连续三次，权重/评估仍在 |
| 站立辅助 | 250/500/750 轮均完整站稳 10 秒；其成绩不能替代箱攀，权重未用于本次初始化 |
| 检查入口 | `check` 依次执行必要检查，`inspect` 汇总专项诊断与渲染 |

V3 从旧 v2 第 2000 轮仅加载 actor，critic、优化器、探索噪声、迭代从零开始；不是旧训练严格续训。最终 1024 环境、240 步带噪压力测试、5→6 轮严格恢复和完整评估链路已通过，已启动最多 15000 轮、每 500 轮验收的训练。连续三个不同检查点从起点完成 11 秒参考与额外 1 秒真实箱顶保持，且跟踪、受力、支撑和稳定均达标才成功。细则见 [箱攀管线](BOX_CLIMB_PIPELINE.md#新版教师与验收)。

原末帧与静止保持的几何目标有冲突，因此另建 settle 数据，而未改写原 501 帧。新几何目标、短物理检查或单个旧检查点通过都不能证明新策略已经完成任务。最终准备状态由 `runs/_reports/preflight/contact_surface_v3_readiness.json` 记录，不能用先前 debug 快照代替最终源码的启动检查。

## 仓库范围与复现边界

当前 Git 仓库是源码、机器人模型资产与方法说明的快照，尚不是下载即可完整重跑的实验包。`.gitignore` 排除了动作、标签、所有训练权重、`runs/` 及派生缓存；正文中的这些路径均指本机记录。V3 还需要原始配对动作/场景、场景拟合后的 seed_v2、稳定终点、settle 后动作与新标签、显式 v2 actor 初始化权重，以及针对最终源码/输入的启动验收汇总。

原始来源和库存见 [数据来源](DATA_SOURCES.md) 与 `datasets/external/download_inventory.json`。导入、场景拟合、稳定目标优化、追加 settle、重新标注的实现已保留，但把这些步骤及其参数串为可公开重建相同输入的入口仍待补齐；不能用其他派生动作或权重直接替换冻结输入后继续原实验。

`train_reference_teacher_v3 preflight` 仅生成单项物理检查报告，不能直接作为 `supervise --readiness-report`。后者需要真实汇总 CPU 回归、1024 环境压力测试、保存恢复和完整评估的记录，并绑定源码及输入哈希；当前汇总尚无独立 CLI。已有运行必须保留冻结输入，新实验使用独立运行目录和重新核验的汇总。

新克隆可安装 `python -m pip install -e '.[test]'`，先运行 `python -m morph_mimic.scripts.check --only regression --suite basic`。默认完整 `check` 还会验证本机数据，须在准备数据后运行；可选依赖或本机夹具缺失造成的跳过不等于物理技能通过。

## 日常操作

```bash
conda activate env_isaaclab_2
python -m morph_mimic.scripts.check \
  --motion-dir datasets/g1_box_climb_contact_v3 \
  --clip climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3
python -m morph_mimic.scripts.inspect --help

# 正式启动后查看
tail -n 30 -F runs/g1_box_climb_contact_teacher_v3_1024/progress.log
```

默认检查不启动 GPU；模型验收显式使用 `--gpu-only --checkpoint-kind teacher-v3` 并指定新动作。已有轨迹优先离线渲染，避免用默认 468 维 ACP 播放器加载特权教师。命令和退出状态见 [管线说明](BOX_CLIMB_PIPELINE.md)。

## 推进顺序

| 顺序 | 工作与通过要求 |
| --- | --- |
| 1 | 新版单任务教师完整箱攀；严格起点跟踪、额外 50 个真实保持端点和全身支撑稳定达标 |
| 2 | 有限观测 ACP Stage 1 学生；独立验收，不能以教师成绩代替 |
| 3 | 扩充走路、起身、楼梯、跨栏、梯子；完整来源/场景/变换/目标/九部位接触与逐动作物理验证 |
| 4 | 多技能 Stage 1；oracle 接触跟踪，分技能报告成功率、接触误差和失败原因 |
| 5 | Stage 2 联训；成功 rollout hindsight 入库，planner 真正更新，替代 oracle 后仍完整执行 |
| 6 | Stage 3 与泛化；可达目标场景混训、双 critic、旧技能保持、留出地形和组合任务评估 |
| 7 | mojia、感知与实机；各自核验动力学/接触、LiDAR 建图、噪声/延迟和硬件结果 |

仍需补齐完整多场景分配、多技能数据、论文随机化接线、参考/hindsight/body-origin 与 planner 地形表面候选的接口一致性，以及原文多 seed/多 episode 统计协议。当前已分别实现指令—实际 onset 落点指标与参考表面代理；单教师指标不能视作论文 Fig.7 统计复现。完整动量稳定证明和独立自由接触偶矩也不在当前测量范围。

原始长动作应按完整技能事件切片，并按来源分组划分验证集；不能直接把 100 秒以上整段接入 15 秒回合。六轴 held 约束需要一致投影和 FK/速度重算，有场景动作还需场景约束重定向，不能只删六列沿用旧标签。

跨阶段迁移与同阶段续训须区分；V3 actor-only 初始化尤其不能称严格恢复。恢复学习状态后重新开始仿真，未恢复完整 PhysX 状态，不保证逐步轨迹一致。修改奖励、数据、观测或物理配置须建立独立实验，重新执行检查后冻结源码与输入。
