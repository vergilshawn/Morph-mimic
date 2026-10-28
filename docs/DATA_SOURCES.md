# ACP 动作与地形数据来源及接入顺序

数据来源于 2026 年 10 月 27 日核实，本地保留范围于 2026 年 10 月 28 日清理后更新。已核实 ACP 的来源表，以及 PHP、OmniRetarget、VideoMimic 等公开资源的实际数据入口。本机另有 BONES-SEED 的箱攀、楼梯、跨越和梯子动作。当前差距应细分为动作筛选、格式与模型适配、场景配对、接触重标和物理验收，不能再笼统理解为这些技能没有数据。

已实际下载 **OmniRetarget 的两个交互子集及模型、PHP 的五个训练包与原子技能库、VideoMimic captures 包**，源格式数据、模型、许可和下载报告保存在 `datasets/external/`。历史下载总量约 643 MB；清理后保留的外部文件约 1.43 GB。9 个 ZIP/TAR.GZ 归档中的 **1219 个有效文件**已逐文件与解压副本核对 SHA256，随后删除重复归档，释放约 554 MB；macOS 元数据不作为动作或场景数据保留。各目录原有 `download_report.json` 和统一 `download_inventory.json` 是历史下载凭据，其下载字节数不表示当前磁盘占用。清理详情见 [cleanup.json](../runs/_reports/cleanup.json)，逐文件核验见 [data_archive_verification.json](../runs/_reports/data_archive_verification.json)，下载范围和配对检查见 [本地数据清单](../datasets/external/README.md)。大部分仍未批量导入或通过物理验收。机器可读来源表见 [resource_registry.json](../references/data_sources/resource_registry.json)，四个本地候选的文件位置、帧数与 hash 见 [local_motion_candidates.json](../references/data_sources/local_motion_candidates.json)。

其中 OmniRetarget 的 `climb_03` 已完成一个 0–10 秒派生样例，位于 `datasets/g1_box_climb_seed_v2`：场景、自碰及接触保留检查通过，已接入训练，完整箱攀物理验收尚未通过。另保留 `g1_box_climb_omniretarget_baseline_inputs_v1` 的原始 29 轴对照输入，以及 `g1_standing_reference_v1` 的工程静态站立参考。旧格式试样、被拒绝的重定向候选和平地调试裁剪已删除；有效诊断转存于 `runs/_reports`，测试使用临时生成的夹具。其余源数据尚未批量导入。详见 [单条箱攀管线](BOX_CLIMB_PIPELINE.md)。

## ACP 原论文实际使用的数据

[论文补充材料 Table S1](https://arxiv.org/html/2610.31608v1)列出：

| 技能 | 原始来源 | 说明 |
| --- | --- | --- |
| 箱攀 | OmniRetarget | ACP 未再次做其自身的重定向，另外标注接触 |
| 楼梯、梯子、坐下 | Mixamo | ACP 自行重定向并标接触 |
| 走路、起身 | LAFAN | ACP 自行重定向并标接触 |
| 跨栏或撑手翻越 | Kimodo | 已有 G1 候选，ACP 再标接触；原版 prompt/seed/约束与轨迹文件未在论文中给出 |
| 稀疏地形 | 表中记为 AME policy | 来自策略轨迹；需要另核对作者实际策略版本和导出流程 |

表中 Retargeted 与 Contact labeled 的含义是“是否使用 ACP 作者的方法处理”，不能把 Yes 理解成公开源数据已经含有本工程所需的九部位受力接触标签。上述来源也不等于作者最终精选的 13.2 分钟数据包已经整体公开。

前序 [MGMT 数据构建章节](https://arxiv.org/html/2604.17335v1#S3.SS1)说明其约 5 分钟初始动作来自自拍录像和公共数据，随后增强到约 1 小时。它将重定向动作与人工放置场景配对，再训练 tracker 并回录物理轨迹。这是本项目值得采用的数据整理路线；本次尚未确认 MGMT 最终完整训练库的官方下载入口。

## 优先获取的动作与场景配对资源

| 资源 | 实际公开内容 | 对本项目的作用 |
| --- | --- | --- |
| [PHP 官方训练包](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1) | G1 运动与地形对：locomotion 100 对、低/高速 step 56/60 对、低/高速 climb-76 各 40 对 | 先用箱攀和上落台阶建立场景绑定；现成五包没有独立跨栏包 |
| [PHP 原子技能库](https://raw.githubusercontent.com/amazon-far/php_parkour/main/release-assets.json) | 约 39.9 MB 数据库包，含 hurdle、cat/dash/speed vault 等与对应地形 JSON/OBJ | 生成或导出跨栏、撑手翻越的动作与场景 |
| [OmniRetarget 数据](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/tree/main) | G1 29DOF，robot-terrain.zip 约 38.3 MB，配套 URDF/OBJ；另有物体交互子集 | ACP 自己使用的箱攀来源，适合先接入一个 climb 场景 |
| [VideoMimic](https://github.com/hongsukchoi/VideoMimic) | 官方下载脚本与动作配置；楼梯 G1 H5 轨迹和背景 OBJ 显式配对 | 补楼梯上下与扫描场景 |
| [PARC](https://github.com/mshoe/PARC) | [官方数据](https://huggingface.co/datasets/mxucg/PARC)含运动、身体接触和高度场，涵盖箱攀、vault、上下楼梯等 | 后续做场景一致的机器人重定向和物理增强；不是原生 G1 轨迹 |

PHP 的[运动匹配文档](https://github.com/amazon-far/php_parkour/blob/main/motion_matching/README.md)明确输出 50 FPS `*_motion.npz`、箱体描述 `*_terrain.npy` 与 `*_terrain.obj`。下载器使用 [release-assets.json](https://raw.githubusercontent.com/amazon-far/php_parkour/main/release-assets.json)中的归档和逐文件 SHA256。现成箱攀包可以跳过其 motion matching，但仍需适配本工程数据与控制模型。

OmniRetarget 的 [README](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/blob/main/README.md)介绍了 NPZ 的 fps 和 qpos 字段；本次实际检查还发现 112 条包含 `human_joints[T,53,3]`，但上游未说明 53 点名称、顺序与坐标语义，暂不能直接用于 mojia 重定向。无物体时 qpos 为 36 维，其中浮基顺序是 **四元数 WXYZ 在前、XYZ 平移在后**，再接 29 关节。应按源模型显式解析关节顺序。其 [visualize.py](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/blob/main/visualize.py)用动作前缀和高度后缀查找 terrain URDF；下载的 145 条 robot-terrain 轨迹全部匹配，148 条 robot-object-terrain 中 138 条匹配，另 10 条原始 Take 命名无法明确配对。186 个 URDF 的 796 个网格引用均存在。必须保留 OBJ/URDF 的坐标与缩放，不能仅依据技能名称随便生成一个箱子。

对 mojia 更有帮助的是 Holosoma 的[原始 climbing 演示](https://github.com/amazon-far/holosoma/tree/main/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb)：含人体关节点和场景，能从人体源重新重定向；这通常比从 G1 关节角转给另一种形态更合适。

VideoMimic 的[下载脚本](https://github.com/hongsukchoi/VideoMimic/blob/main/simulation/data/download_videomimic_data.sh)给出 videomimic_captures.zip 的 Google Drive ID。其[动作配置](https://github.com/hongsukchoi/VideoMimic/blob/main/simulation/videomimic_gym/resources/data_config/human_motion_list_123_motions.yaml)明确将 `retarget_poses_g1.h5` 与 `background_mesh.obj` 对应，楼梯条目使用 60 FPS override。代码许可与数据许可应分别核对；代码 MIT 不能自动证明所有原始视频和扫描数据具有同一许可。

下载后确认 captures 包含 123 对 H5/OBJ，其中 4 条楼梯均能与官方配置精确对应。全部动作均为 23 关节，数值字段有限。官方配置共 49 条按目录 basename 精确匹配、74 条按唯一子字符串匹配，覆盖 123 个不同目录；还需回放验收文件名关联。全库 H5 的 fps 属性为 30 或 15，官方配置覆盖值为 60 或 6；4 条楼梯均为 H5 30 / override 60。导入时必须明确选择有来源记录的时间约定，不能仅根据文件属性重采样。完整配对状态见本地下载报告。

## 本机已经存在的四种候选

根目录为 `/home/user/Beyondmimic/Dataset-mimic/G1_bones-seed`。以下四个文件及同名 SOMA Proportional BVH 均已检查存在，CSV 帧数与 BVH 一致：

| 技能 | G1 CSV 相对路径 | 帧数 |
| --- | --- | ---: |
| 50 cm 箱攀 | `g1/csv/230320/come_up_50cm_box_R_001__A290.csv` | 318 |
| 撑手跨越 | `g1/csv/240327/vault_over_R_001__A533.csv` | 463 |
| 梯子起步 | `g1/csv/230323/ladder_climbing_up_start_R_001__A300.csv` | 840 |
| 楼梯上行 | `g1/csv/230323/stairs_climbing_jog_up_start_R_102__A301.csv` | 431 |

源 README 标注 120 FPS，对应 BVH 的 Frame Time 为 0.008333 秒，四条时长都适合整理成短技能。原始人体动作在 `soma_proportional/bvh/` 对应目录，可用于 mojia 重定向；元数据在 `metadata/seed_metadata_v004.csv`，另有 temporal labels 可用于确定完整动作边界。

这些文件尚无明确配套的箱体网格、梯子参数与九部位接触标签。CSV 中高度约 78、角度约 90 的原始数值不能直接当作米和弧度；单位、Euler 旋转顺序、轴向、G1 模型版本与 50 Hz 重采样仍需按源导出约定核实。候选清单记录文件位置和 hash，没有复制原始动作。

[BONES-SEED 官方许可](https://bones.studio/info/seed-license)允许符合条件的机器人策略训练，但采用专门的学术机构及合资格 startup 条件，不能视作无条件公开数据许可。应按实际使用主体和用途核对条款，原始数据不要直接放进公开仓库。

本地 `/home/user/Parkour/InstinctLab` 与 `hiking-in-the-wild_Data&Model.zip` 还包含程序地形、G1 跑酷参考和策略资产；它们适合借用地形实现，当前没有核实到逐参考片段的原场景绑定。`/media/user/DATA/Terrain` 同样有场地 USD，但只有场景不等于动作与场景配对。

## 梯子与生成数据的处理路线

[LadderMan](https://ladderman-robot.github.io/)已经专门研究 G1 感知爬梯。论文采用单条示范、多种梯子几何和接触约束迁移，适合作为梯子数据构造的方法参考。不过其[官方仓库说明](https://github.com/amazon-far/LadderMan/blob/main/README.md)目前仍写训练代码即将发布，本次未确认完整参考轨迹下载。不能把网站演示等同于数据已拿到。

梯子的首批动作可以从本地 BONES-SEED 选择完整上/下梯片段，结合手脚落点拟合梯子几何，再做固定控制关节约束下的 IK 与物理跟踪修正。初版应明确保存人工拟合的角度、阶距、梯宽和标注不确定性。支持梯子的碰撞场景应保留真实横杆三维结构，而非用平地近似。

[Kimodo 官方实现](https://github.com/nv-tlabs/kimodo)及 [G1 模型](https://huggingface.co/nvidia/Kimodo-G1-RP-v1)可以生成跨栏候选并施加手脚空间约束。它是候选生成器，不是 ACP 原版轨迹的替代下载链接；官方也说明 [G1 后处理存在限制](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html)。默认脚部接触输出不能直接代替九部位受力接触。

Mixamo 是 ACP 的真实来源，但[Adobe 社区 FAQ](https://community.adobe.com/questions-696/mixamo-faq-licensing-royalties-ownership-eula-and-tos-589400?lang=en)列出了机器学习训练限制。该社区说明不能代替对实际适用条款和作者授权的核实，因此不能仅凭“免费下载”就把它作为默认可训练数据来源。

## 本项目的接入顺序

1. 首批使用 **OmniRetarget 一个箱攀场景、PHP 一个箱攀/跨栏技能、VideoMimic 一条楼梯、本地 BONES-SEED 一条完整梯子动作**，先把每种技能的一对动作和场景做正确。
2. 增加场景清单：每条记录至少保存 source/group、motion、scene 文件与 hash、场景变换、FPS、机器人模型与控制 profile、接触定义和验证状态。
3. 在对应三维场景中做模型/关节约束一致的重定向或优化。G1 原始 29DOF 改为 23 控制时，nominal held 投影仅保证 FK 一致，**不能保证箱沿或梯级接触仍被保持**；要进一步做场景约束优化。
4. 重采样到 50 Hz，重算 FK/速度，在匹配场景中检查环境接触与自碰，提取参考几何标签供 Stage 1 起步；另做物理 replay 或 tracker 回录，完整验收后才能把实际受力接触作为有效 hindsight 标签。两种标签分别记录来源，不能相互冒充。
5. 每种技能稳定后再增强地形，并按源动作及地形 seed 分组评估。继续采用 ACP 的三阶段方法，不需要为使用 PHP 数据而切换成 PHP 的整套训练算法。

当前已增加 **reference clip 到 scene 的绑定与导入**，单条参考场景的每环境独立副本已用于并行训练。`label_reference_geometry` 检查真实碰撞几何并写参考标签；`label_contacts` 检测到 sidecar 后导入匹配场景，无 sidecar 时保留平地基线。把箱攀或爬梯文件单独丢进 motions 目录而省略场景，仍会破坏交互意义。
