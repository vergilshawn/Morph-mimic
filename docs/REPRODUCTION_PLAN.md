# ACP 论文复现状态与执行路线

当前工程已具备机器人资产、接触规划器、跟踪控制器和三阶段训练原型。继续推进的首要工作是验证 G1 单个短动作的动力学跟踪和接触标签，再接入已有的动作与场景配对资源。已下载 PHP、OmniRetarget、VideoMimic 三个来源的原始数据，见 [下载清单](../datasets/external/README.md)；其余公开源与本地候选已整理在 [数据来源与接入顺序](DATA_SOURCES.md)。现有代码和 CPU 测试不能证明论文中的越障能力已经复现。

状态日期为 2026 年 10 月 27 日。复现主目标是 [Humanoid Mobility via Attention based Contacts](https://arxiv.org/html/2610.31608v1)，对应 [ACP 项目](https://leggedrobotics.github.io/ACP/)。[MGMT](https://wholebodylocomotion.github.io/)可用于理解数据整理与跟踪控制的前序设计；它的全身动作扩散生成器属于另一条方法路线，ACP 的主线应实现接触规划器。

后续已实际完成一条箱攀参考的场景约束重定向、几何/自碰验收、九部位参考标签和 GPU PPO，并进行了匹配场景的物理回放与策略评估。单 clip 的 1024 环境复制已验收，当前后台进行最多 20000 轮的 Stage 1 训练与自动整段评估。**训练链路通过，完整技能仍未通过验收**。最新状态、场景限制及执行命令见 [单条箱攀管线](BOX_CLIMB_PIPELINE.md)；下文原平地样例仍可用于独立基线调试。

## 当前基础和阻断点

| 部分 | 已核实状态 | 对下一步的影响 |
| --- | --- | --- |
| 资产迁移 | 源目录 `/media/user/DATA/supermimic/data` 中 G1 81 文件、mojia 42 文件、原始动作 81 文件与当前对应资产 SHA256 一致 | 无需重复复制 1.5 GB 数据 |
| 参考动作 | 两机器人各 40 段，约 2.45 小时；G1 441140 帧，mojia 441120 帧 | 可做走路等基线，不能据此宣称覆盖论文任务 |
| 接触标签 | 原始 40 段动作，两机器人均为 0/40 | 参考训练默认拒绝缺失标签的数据 |
| 动作时长 | 原始片段长 102.18–273.12 秒 | 必须先按完整技能切短，避免课程与成功轨迹入库失效 |
| 参考地形 | 已支持一个 scene-bound 动作/环境的原始碰撞场景；其余保留程序地形 | 扩展多环境/多场景分配前先让单个箱攀策略完成整段 |
| G1 运行资产 | 已切换本机 `G1_29DOF_CFG`，23 关节控制，其余保持；启用接触报告并使用源控制参数 | USD 与本地 MJCF 的质量、惯量、碰撞几何一致性仍需验收 |
| mojia | 模型与动作迁移完整，manifest 标记未通过物理跟踪验证 | G1 链路稳定后重新验证 mojia 的动力学与接触定义 |
| 训练成果 | 当前没有经过验收的策略或论文成功率结果 | 不能从脚本齐全推断复现完成 |

长动作是硬阻断：Stage 2 的参考轨迹入库要求从前 10% 开始，至少到达 90%。最短动作 102.18 秒也至少需要约 81.74 秒，而默认回合只有 15 秒。因此原始整段动作无法入库，规划器监督数据会一直为空。Stage 1 的课程也无法逐步扩展到完整长动作。

当前短片段样例为 `datasets/g1_23dof_smoke/walk1_subject1.npz`，来源、SHA256 与裁剪信息见同目录 `.projection.json`。派生参考已将 6 个保持关节设为 nominal 并重算全身体运动学，没有接触标签；原始裁剪保留在 `datasets/g1_smoke`。它是链路调试输入，是否适合训练要由回放验收决定。

## 首先完成 G1 单动作验收

以下命令在项目根目录运行。先进入已经安装 Isaac Lab 的环境：

```bash
conda activate env_isaaclab_2
python -m morph_mimic.scripts.audit_assets --verify-hashes --deep
OMP_NUM_THREADS=1 python -m pytest -q tests
```

本次 CPU 验证环境为 Python 3.11.15、PyTorch 2.7.0+cu128、Isaac Lab 0.54.3、Isaac Sim 5.1.0.0。早期 CPU 测试见 [supermimic 对照报告](SUPERMIMIC_COMPARISON.md)，其中环境与几何检查使用 AST/Tensor 替身。后续确认普通沙箱内 CUDA 不可见，而工具沙箱外可访问 RTX 5090；已实际完成局部模型/场景/传感器审计和五轮训练。完整动力学跟踪及训练收敛仍未通过，最新证据见箱攀管线文档。

在能访问 NVIDIA GPU 的会话中先运行少量环境的构造与步进检查：

```bash
PYTHONPATH=. python tests/smoke_env.py \
  --robot g1 --num_envs 4 --steps 20 --stage 1 --device cuda:0 --headless \
  --motion_dir datasets/g1_23dof_smoke --clips walk1_subject1
```

这一步允许缺少接触标签，用于检查传感器、有限数值及观测维度。它不验收接触跟踪能力。

随后对现有 8 秒样例做完整 PD 回放、物理误差检查与接触标注：

```bash
python -m morph_mimic.scripts.label_contacts \
  --robot g1 --motion_dir datasets/g1_23dof_smoke --clips walk1_subject1 \
  --episode_length_s 12 --record_max_steps 700 --device cuda:0 --headless
```

标注必须有完整 401 帧、从相位 0 到完整终点、没有失败终止且包含有限数值。脚本还检查根位置最大误差与关节 RMSE，默认门槛分别为 0.3 米和 0.5 弧度，属于可调整的工程验收门槛，不是论文报告的阈值。应结合视频和接触可视化进一步收紧；不能为得到标签而直接放宽失败条件。标注拒绝时，先处理关节映射、惯量、碰撞形状、PD 参数或参考数据问题。

标注脚本在无 sidecar 时按轨迹范围扩大平地，适用于平地基线；有 sidecar 时自动导入匹配的碰撞场景，并过滤环境接触。楼梯与箱攀不能用平地模式代替。参考几何标签和物理受力回放标签已分为不同入口，见箱攀管线文档。

接触标注成功后进行五个迭代的优化通路检查：

```bash
python -m morph_mimic.scripts.train_stage1 \
  --robot g1 --motion_dir datasets/g1_23dof_smoke --clips walk1_subject1 \
  --flat_reference --num_envs 16 --max_iterations 5 --save_interval 0 \
  --log_dir runs/g1_smoke_stage1 --device cuda:0 --headless
```

五个迭代只验收 rollout、PPO、日志和 checkpoint。要把策略作为 Stage 2 的起点，还需延长训练，确认 CSI 能扩展至动作开头，整段完成率提高，并检查动作质量与接触误差。

回放这个 Stage 1 checkpoint 可使用 GUI：

```bash
python -m morph_mimic.scripts.play \
  --robot g1 --stage 1 --resume runs/g1_smoke_stage1/final.pt \
  --motion_dir datasets/g1_23dof_smoke --clips walk1_subject1 \
  --flat_reference --num_envs 1 --max_steps 500 --no-headless
```

增加新的裁剪样例时使用不同输出目录，工具拒绝覆盖已有文件：

```bash
python -m morph_mimic.scripts.prepare_motion \
  --robot g1 --clip walk1_subject1 --start-s 8 --duration-s 8 \
  --output-dir datasets/g1_walk_second
```

正式片段应按完整技能事件确定边界，而不是机械地等长切割。训练默认检查参考片段是否短于回合上限。

## 三阶段推进和验收

| 次序 | 工作 | 通过标准 |
| --- | --- | --- |
| G1 平地基线 | 站立与走路短片段、完整 replay、接触标注、Stage 1 | 姿态和坐标正确；标签覆盖全帧；CSI 确实退至开头；整段跟踪成功 |
| 多技能数据 | 起身、楼梯、箱攀、跨栏、梯子、横梁和踏石等逐项加入 | 每段有对应场景、初始变换、目标、有效接触、来源和物理验证结果 |
| Stage 1 多技能 | 用 oracle 接触训练单个接触跟踪器 | 分技能报告完成率、根/关节/接触误差及失败原因 |
| Stage 2 | 从 Stage 1 初始化，成功轨迹 hindsight 入库，监督 planner，与 tracker 联训 | buffer 非空且更新；留出轨迹的 occupancy 精确率/召回率、接触位置误差、支撑面命中率改善；替换 oracle 后仍能完整执行 |
| Stage 3 | 可达目标的程序化场景与参考场景混训，双 critic、成功轨迹反馈 | 验证 Stage 2/3 对比、原技能保持率、目标成功率及未见地形泛化 |
| mojia 扩展 | 重新标注、重定向检查与独立控制参数验收 | 在其自身模型上通过以上步骤，不能直接以 G1 通过代替 |
| 感知与实机 | LiDAR 建图、噪声与延迟、部署和实机验证 | 将仿真真值高度图与真实感知分开报告，单独验收 |

Stage 2/3 入口要求 `--resume`。跨阶段默认只迁移网络与探索噪声，重新开始该阶段的训练进度；同阶段继续使用 `--continue_run --resume ...`，恢复迭代、优化器和 replay buffer。新 checkpoint 已记录数据与物理契约、RNG 和课程状态；恢复学习状态后会重新开始仿真，未恢复完整 PhysX 状态，因此不保证轨迹逐步完全一致。

所有实验先从 1–16 个环境验证，再按实际显存和 CPU 内存扩展。默认 8192/4096 个环境是已有预设，不是本机容量验收结果。当前实现没有复现论文多 GPU 分布式训练通路。

## 正式复现仍需补全的实现

已建立单场景的 `clip → scene → transform → contacts` 绑定，并扩展动作类别；下一步需要支持多场景与多环境分配、显式技能目标及更大规模采样。随机选择 flat/rough 无法表达参考手脚应接触的箱面，程序化地形通过也不代表目标可达或与动作对齐。

完整奖励与消融需要继续对照论文 Table 3 和补充材料。本轮已修正超时 bootstrap 和 torso 姿态项。尚需关注起身任务的竖直进展门控、goal 目标高度/支撑面的可达性、校准偏置随机化的实际应用，以及奖励分项日志。当前 replay 质量分数使用竖直力与根竖直速度的代理指标，尚不等同于论文的正机械功质量标准。

FiLM 的调制方向还需按论文图与文字核对。完整随机化、未见组合/干扰地形、LiDAR 神经建图、部署延迟与硬件实验仍未完成。截至此次核查，ACP/MGMT 项目页未显示已核实的官方训练代码或数据下载入口；不能据此断言作者没有公开资源。

## 此轮代码修复

- 训练与回放入口在 SimulationApp 启动后导入 Isaac Lab，统一传递设备、处理 reset 返回值并支持小规模迭代。
- rollout 保存各自的 critic 观测，PPO 分别使用 353/298 维输入；修复单样本优势归一化及小 batch 分配。
- 修复九接触点传感器映射、RayCaster hits、缺失扫描值、WXYZ 共轭、根高度和参考坐标、动作延迟与动作变化惩罚。
- 修复 Stage 1 oracle 输入、CSI 完成摘要、历史重复推进、奖励 timestep 与越限张量；Stage 3 使用 planner 接触并切换接触奖励。
- 修复 attention 二次 softmax，保留网络权重布局；MLP baseline 现在返回可解码接触点。
- 修复 hindsight 历史窗口、CPU/GPU replay 索引、布尔监督标签和梯度累积归一化，接通 checkpoint 恢复。
- 修复地形 mesh 嵌套列表、箱体底面/高度语义、台阶与 landing 高度，以及被填平的 gap。
- 标注过程按固定环境关联动作，拒绝失败或短回放，严格校验标签 shape/FPS/来源，原子保存。

修改前文件保存在 `/tmp/morph_mimic_before/`，仅作为本机临时备份，不随 Git 发布。源码、测试、文档与必要机器人模型已纳入 Git；数据、模型缓存、训练权重和运行状态仍保留在本机。

## G1 参考数据的一致性要求

原始 G1 动作包含 29 个运动关节，当前策略只控制 23 个，其余 6 个目标保持 nominal。直接删去 6 列动作会保留不一致的手臂与腕部参考。新的加载器会拒绝这种数据，并要求在新目录生成一致参考：

```bash
python -m morph_mimic.scripts.project_reference \
  --robot g1 --source-dir datasets/g1_walk_second \
  --output-dir datasets/g1_23dof_walk_second
```

该命令需要 MuJoCo；本机默认 Python 已有该依赖，`env_isaaclab_2` 当前未安装。投影保留受控关节轨迹和根轨迹，重算全部身体位置、四元数与原点速度，并丢弃旧接触标签。仍需用 Isaac Lab 完整回放重新标注。

训练后使用独立评估入口，而非把 GUI 播放当完整动作评价：

```bash
python -m morph_mimic.scripts.evaluate \
  --robot g1 --stage 1 --resume runs/g1_smoke_stage1/final.pt \
  --motion-dir datasets/g1_23dof_smoke --clips walk1_subject1 \
  --out runs/g1_reference_evaluation.json
```

评估固定首帧、关闭随机化，使用实际策略动作，输出完成情况与根位置、关节、九身体点、根朝向误差。短预算会标记为不完整；从训练动作得到的指标不等同于留出动作或未知地形泛化。Stage 3 目标评估尚需单独建设。
