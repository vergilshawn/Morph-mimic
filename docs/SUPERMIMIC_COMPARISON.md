# supermimic 工程补强与 ACP 适用修改

supermimic 中最值得用于本项目的补强，是机器人与数据约定、参考运动学验证、独立物理评估和训练记录。本轮已经落实了会改变训练目标或数值可靠性的明确修复；全帧可行性审计、任务数据配对和泛化评估仍需继续建设。

检查对象为本地 `/media/user/DATA/supermimic` 的 mojia 分支，包括其 G1 对照审计。本轮没有修改该项目的文件或分支。其 G1 成绩、Mojia 参考限制和 CPU/GPU 接触差异不能直接作为 ACP 结果。

## 比较与实施状态

| supermimic 的补强或检查方法 | 当前 ACP 项目的对应问题 | 本轮处理 |
| --- | --- | --- |
| 完整 joint/body/state 约定及命名 FK 检查 | G1 仅控制 23 个关节，参考身体位置却来自原 29 个运动关节 | 新增 nominal held 参考投影，重算完整 FK 与原点速度，默认拒绝不一致输入 |
| 明确根、torso 与 distal tracking point | root/COM 速度混用；mojia torso 质量随机化可能落在 base | 统一使用 link-origin 速度 API；torso 明确映射到 G1 torso_link、mojia lumbar_yaw |
| 受力、接触点速度及滑移诊断 | 标签九列只检查形状，列交换仍可能静默通过 | 校验 hands/elbows/knees/feet/pelvis 的逐列候选名及 alias；真实接触点滑移指标仍待实现 |
| timeout 与跌倒分开处理 | 所有 truncated 都终止价值 bootstrap | 人工超时使用 reset 前 critic 观测；失败、目标完成和参考结束不 bootstrap |
| loss、梯度与参数有限值检查 | NaN 可污染优化器与 checkpoint | PPO/planner 增加非有限值阻断；动作执行使用统一有限值检查与 ±20 界限 |
| robot/data/physics checkpoint contract 和原子保存 | 同名机器人换数据、模型或物理配置仍可静默恢复；双文件存在错配风险 | 保存并校验配置与 hash；不可变 buffer generation 加 SHA256，再原子提交主 checkpoint；恢复 RNG 与课程 |
| 首帧开始、关闭噪声的独立全动作评价 | play 有随机起点，只打印 reward/base 高度 | 新增 Stage 1/2 平地策略评估、四类误差指标及完整/失败/预算截断状态 |
| 全帧 FK、碰撞、穿地、抬升审计 | deep 资产检查主要是 hash、shape 与 finite | 已补完整主数组/根姿态契约和投影数值验证；完整碰撞/可行性工具尚待移植 |

另修复了 ACP 自身的两个 PPO 问题：探索噪声参数越界后因 clamp 导数为零而停滞，以及用最后一个 minibatch 的有符号 KL 估计调整学习率。现在更新后直接约束 log_std 参数，并使用全 rollout 的解析 Gaussian KL，统计聚合全部优化 batch。这些是 ACP 实现修复，不是 SuperMimic 算法的移植。

## G1 参考偏差的数值证据

对原始 40 段 G1 动作各取 8 帧，共 320 个状态，用命名关节与 MuJoCo 重算 FK。完整 29 关节状态与缓存身体位置的最大欧氏误差约为 1.26 微米。将 6 个保持关节改为当前控制器的 nominal 后，左腕、右腕最大位移分别为 23.64 厘米、20.84 厘米；左右肘最大位移约为 16.76 厘米、13.04 厘米。

这些差异说明参考几何与控制约束确实不一致。完整检查参数、源文件 SHA256 和结果见 [数值证据](reference_contract_evidence.json)。该检查只验证源 MJCF 运动学，不能证明运行时 USD、受力接触或跟踪策略正确。

新工具 `morph_mimic.scripts.project_reference` 会：

- 保留原根轨迹和 23 个受控关节轨迹，将 6 个保持关节的位置设为 nominal、速度设为零。
- 按名称映射全部物理关节与身体，重算 FK、身体原点世界线速度和世界角速度。
- 保留完整物理状态、源/模型/依赖/配置 hash，输出到新目录，拒绝覆盖和继承旧接触标签。

已生成 `datasets/g1_23dof_smoke/walk1_subject1.npz`，401 帧、8 秒，来源见 `.projection.json`。原始资产和 `datasets/g1_smoke` 保留不变。后续标注、训练与评估应使用新目录。

MuJoCo 的 free-root 角速度使用 root-local 轴，`mj_objectVelocity(BODY)` 的线速度参考点是 COM；投影工具显式处理了局部/世界转换及 COM 到 body-origin 的平移。测试还用独立积分后的位姿差分验证身体线角速度，没有仅用同一个 FK 输出互相比对。

## 物理约定与实验边界

本机 Isaac Lab 的 `root_lin_vel_w/b` 是质心速度别名，而参考数据存的是基座坐标原点速度。当前环境已改用 `root_link_lin_vel_w/b`、`write_root_link_velocity_to_sim`；接触身体位置与速度也统一到 link origin。torso 质量随机化使用显式身体名称，torso 姿态惩罚按 torso 坐标系重力计算。

`action_scale=0.25` 保持 ACP 当前默认设置。±20 动作界限是借鉴 supermimic 的工程补充，应在实验记录中保留；PPO 仍保存原始采样动作及其概率，仿真执行端负责限制。没有将原 G1 分关节尺度、SuperMimic 的世界模型/BPTT、29 动作网络或历史窗口直接替换进 ACP。

supermimic 使用显式、每物理步计算并限制力矩的 PD；当前 ACP 使用 PhysX 隐式执行器。即使 kp/kd/effort 相同，也不能认定两个后端轨迹等价。checkpoint 已记录实际执行器与物理配置；隐式执行器的 computed/applied torque 估计不能当成实机实测扭矩。远程 USD 只记录标识符及导入配置，无法在此环境核验其远程内容 hash。

## checkpoint 与评估方式

新 checkpoint 保存所选动作和标签 hash、关节/九身体顺序、源资产 hash、物理与控制时步、重力、solver、碰撞与执行器配置，以及训练配置、seed、RNG 和课程。恢复同阶段要求语义一致；跨阶段允许更换该阶段的训练设置，已存在的 planner 类型仍需一致。评价允许训练数据的明确子集并关闭噪声，不允许同名动作或标签被替换。

buffer 先保存为不可变 generation 并计算 SHA256，主 checkpoint 再原子提交对应文件名与 hash。加载时拒绝缺失或错配 buffer。未提交成功时可能留下未引用 generation；可在确认没有 checkpoint 引用后清理。旧 checkpoint 没有 contract 时会明确标记兼容性未验证。当前恢复学习状态后重新开始仿真，不保证 PhysX 轨迹逐步续接。

新的 `evaluate` 入口使用真实 actor/planner 控制，固定从首帧开始并纳入自动 reset 前的终态，报告：

- 根位置欧氏 RMSE，即 `sqrt(mean(sum_xyz(error²)))`，不是每轴 RMS。
- 所有受控关节的 RMSE。
- 九身体位置整体及逐身体 RMSE。
- 根朝向测地角 RMS。

是否完整执行、是否失败、是否预算截断分别记录。完成率与跟踪精度分开报告，不能用“没有摔倒”替代精度。当前仅覆盖 Stage 1/2 的平地参考任务；Stage 3 目标任务、非平地配对场景、真实接触点质量和实机仍需独立评价。

## 后续仍有价值的补强

1. **全帧参考可行性审计。** supermimic 已记录 Mojia 的残余腿部自碰撞和较大的地板高度修正。当前资产深检通过不代表这些问题消失；应输出 FK、速度差分、自碰撞、穿地和抬升的逐片段报告。复核阈值不能自动变成过滤规则或接触力标签。
2. **Mojia 虚拟手尖。** supermimic 用 elbow_yaw 局部 x 方向 +0.1384 米定义手尖。ACP 若采用这一适配，接触目标、标签、奖励与速度定义必须一起修改并重新标注。当前仍使用现有物理接触身体定义。
3. **按来源和地形分组的验证集。** 同一原始长动作切出的窗口应使用来源 hash 分组；不能随机拆相邻帧后声称未知动作泛化。两个项目当前都没有完整的保留动作/subject/terrain 协议。
4. **完整随机化与目标场景。** 当前 joint calibration bias 虽有采样但没有进入执行或观测，仍未兑现相应随机化；需要明确标定约定后接线。起身竖直进展、可达目标采样、实际正机械功 replay 质量及奖励分项日志也尚需完成。
5. **周期性独立评估与 best checkpoint。** 新入口提供了可量化报告，但尚未自动接入异步评估、best checkpoint、持续回退保护和 Ctrl+C 时的可靠保存。

RunningNorm、训练集配比、显式 PD 后端和虚拟手尖都可以成为独立消融；不应静默混入默认配置后把结果归因于 ACP。

## 本轮验证与源码位置

在 `env_isaaclab_2` 中，完整测试为 **64 passed、2 skipped、87 subtests passed**。两个跳过项是该环境未安装 MuJoCo 的投影测试；使用默认 Python 的 `python -m unittest tests.test_reference_projection tests.test_contact_labels -q`，相关 **12 项测试全部通过**，包括独立 FK/速度数值验证。CLI 帮助和编译检查通过。真实 Isaac Sim/GPU 训练尚未验收。

主要来源为 supermimic 的 `docs/ROBOT_CONTRACT.md`、`docs/DATA.md`、`robot_control/dataset_audit.py`、`runtime/physics.py`、`runtime/collector.py`、`runtime/train.py`、`runtime/evaluation_worker.py`、`runtime/evaluate.py`、`tests/test_robot_contract.py` 和 `tests/test_guards.py`。这些文件用于比较工程约定；supermimic 的既有成绩不属于本工程的验证结果。
