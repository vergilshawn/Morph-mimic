# Morph Mimic

基于 Isaac Lab 的 ACP 论文复现工程，包含从 supermimic 迁移的 G1 和 mojia 资产。当前已完成资产核验、训练通路修复与小规模真实训练检查，尚未通过完整技能训练及论文实验验收。

Git 仓库包含源码、测试、文档、机器人模型、mesh、控制配置和来源清单。大规模动作数据、下载归档、派生标签、模型缓存和 `runs/` 下的权重、日志与运行状态保留在本机，不随 Git 提交。新克隆需按 [数据指南](docs/DATA_SOURCES.md) 和 [箱攀管线](docs/BOX_CLIMB_PIPELINE.md) 准备数据；下文提到的 checkpoint 与训练状态路径是本机实验记录。第三方资产来源及现有许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

已实际接入一条 10 秒箱攀参考，完成场景约束重定向、九部位几何标签、本地 G1 的 Isaac 资产审计，以及 32/1024 个独立场景的 GPU 复制验收。数据位于 `datasets/g1_box_climb_seed_v2`。当前训练目录是 `runs/g1_box_climb_stage1_v3_1024`；使用 1024 环境，上限 20000 轮，每 1000 轮自动从起点评估整段动作，连续三个 checkpoint 达到工程验收门限后提前结束。完成状态以该目录的 `supervisor_status.json` 为准，尚未宣称箱攀技能成功。旧五轮权重仅保留作历史诊断，与修复后的接触输入坐标不兼容。命令、验收门限与证据见 [单条箱攀管线](docs/BOX_CLIMB_PIPELINE.md)。

[复现状态、待办与执行顺序](docs/REPRODUCTION_PLAN.md)记录了验收标准、命令和已知差距；[与 supermimic 的工程补强对照](docs/SUPERMIMIC_COMPARISON.md)说明了本轮精度与验证改进；[资产审计结果](docs/asset_audit.json)保存了源文件一致性检查。

在项目根目录、已安装 Isaac Lab 的环境运行：

```bash
conda activate env_isaaclab_2
OMP_NUM_THREADS=1 python -m pytest -q tests
```

恢复完整本地数据后，可运行 `python -m morph_mimic.scripts.audit_assets --verify-hashes --deep` 检查迁移清单。仅克隆代码时，与未下载数据或可选仿真依赖有关的测试会明确跳过；完整资产审计仍需要清单中的本地数据。

已有 `datasets/g1_23dof_smoke/walk1_subject1.npz`，为原动作的 0–8 秒窗口，共 401 帧，已将 6 个保持关节设为 nominal 并重算全身体运动学。原始裁剪保留在 `datasets/g1_smoke`，不能直接作为 23 关节控制参考。此样例没有接触标签，也没有经过物理跟踪验收。可访问 NVIDIA GPU 后，先执行文档中的单动作回放与标注，再启动小规模 Stage 1。

所有 Sim 入口先启动 SimulationApp，再导入 Isaac Lab 环境。建议使用 `python -m morph_mimic.scripts.<script>`；`--help` 无需启动 Sim。训练入口支持 `--motion_dir`、`--clips`、`--max_iterations`、`--save_interval`、`--resume` 和 `--continue_run`。

确定性整段策略评估使用 `python -m morph_mimic.scripts.evaluate`，支持 Stage 1/2 绑定场景或无场景的平地参考跟踪，并可保存终止前的物理诊断轨迹。无场景的 G1 新参考可经 `project_reference` 做关节一致性投影；有场景的参考应使用场景约束重定向入口，保留场景和标签绑定。投影额外需要 MuJoCo，可安装可选依赖 `.[reference]`，也可使用本机已有 MuJoCo 的 Python 环境。

已下载 PHP、OmniRetarget 和 VideoMimic 的动作与场景数据，保存在 `datasets/external/`，具体数量、校验与格式差异见 [下载清单](datasets/external/README.md)。这些源数据尚需格式适配、场景绑定和接触标注。箱攀、楼梯、跨栏与梯子的来源及接入顺序见 [数据指南](docs/DATA_SOURCES.md)；本地 BONES-SEED 已有四类动作候选。
