# 已下载的公开动作与场景数据

下载日期：2026-10-27；保留范围更新：2026-10-28。目录为 `/media/user/DATA/Morph_mimic/datasets/external`。以下三个来源已真实下载、解压并检查，保留源格式文件、模型、许可和来源报告。OmniRetarget 的一条箱攀动作已生成派生参考并用于本项目训练，其余数据尚未批量导入。

| 本地目录 | 已下载内容 | 历史下载体积（十进制） | 完整性依据 |
| --- | --- | ---: | --- |
| [omniretarget](omniretarget/LOCAL_README.md) | robot-terrain 145 条、robot-object-terrain 148 条，以及配套机器人、地形和椅子模型 | 142.5 MB | 333 个上游文件全部通过 Hugging Face LFS SHA-256 或 Git blob SHA-1 校验；ZIP CRC 通过 |
| [php_parkour](php_parkour/download_report.json) | 296 对动作/地形，以及含 hurdle、vault 等的 19 个原子技能数据库 | 346.6 MB | 6 个归档及 680 个解压文件均通过官方 manifest 的大小和 SHA-256 校验 |
| [videomimic](videomimic/download_report.json) | 123 对 G1 H5 动作与 OBJ 场景，其中 4 条在官方配置中标为楼梯 | 153.8 MB | 下载大小与远端一致；ZIP 全部成员 CRC 通过；已记录本地 SHA-256，上游未提供预期哈希 |

原始下载共 **643,002,294 bytes**；重复归档清理后，当前保留外部文件约 **1.43 GB**。共有 712 条源轨迹，含生成或增强变体，不能当作 712 条独立真人采集动作。逐来源的 `download_report.json` 与 [download_inventory.json](download_inventory.json) 保留为历史下载凭据，下载体积与当时的压缩/解压统计不表示当前占用。

2026-10-28 清理前，已将 **9 个归档中的 1219 个有效文件**与本地解压副本逐一核对大小和 SHA256；有效文件全部一致，ZIP 读取同时校验 CRC。随后删除 OmniRetarget 的两个 ZIP、PHP 的六个 TAR.GZ 和 VideoMimic 的 captures ZIP，共 **553,778,086 bytes**。已解压的动作、场景和模型完整保留，macOS 元数据排除在有效文件之外。逐文件结果见 [data_archive_verification.json](../../runs/_reports/data_archive_verification.json)，删除列表和证据迁移路径见 [cleanup.json](../../runs/_reports/cleanup.json)。报告中记录的归档路径与哈希用于说明当时下载的对象，不表示归档仍存在。

## 数据实际包含什么

- OmniRetarget：全部 30 Hz；145 条 robot-terrain 全部有匹配场景。148 条 robot-object-terrain 中 138 条按上游命名规则找到地形/椅子，10 条原始 Take 仍缺明确映射。186 个 URDF 的 796 个 mesh 引用均存在。112 条另含人体关节点，但 53 点骨架的名称、顺序和坐标语义未公开定义。
- PHP：296 对为 locomotion 100、低速 step 56、高速 step 60、低速 climb-76 40、高速 climb-76 40，均 50 Hz。累计 181,236 帧，约 60.4 分钟，包含生成变体。step 是箱体台阶，不代表完整楼梯；hurdle、cat/dash/speed vault 在原子技能数据库中，还需解码/导出，不能当作已准备好的训练 NPZ。
- VideoMimic：每个动作目录均有 `retarget_poses_g1.h5` 和 `background_mesh.obj`，123 份 H5 均含 23 关节，数值字段均有限。官方 YAML 的 49 条按目录 basename 精确匹配，74 条按唯一子字符串匹配，共对应 123 个不同目录；这只是文件名关联，还需场景回放确认。全库 H5 的 fps 属性为 30（72 条）或 15（51 条），而官方配置为 60（122 条）或 6（1 条）。4 条楼梯均为 H5 30 / override 60。源文件保持原样，双脚布尔接触不能代替 ACP 九部位受力标签。

## 接入训练前还需完成

1. 为每条参考显式绑定场景、坐标变换、时间约定和源动作分组。未确定的映射先排除，不按相似名字猜测场景。
2. 按名称映射机器人关节、统一根位姿与 FPS。OmniRetarget 根状态为 `[qw,qx,qy,qz,x,y,z]`；PHP 为 `[x,y,z,qw,qx,qy,qz]`。G1 的 29 关节源数据需要适配本项目 23 控制关节并重算 FK，保持手脚在对应障碍表面的接触；mojia 需要另外重定向。
3. 在对应三维场景中回放并验收，生成 ACP 九部位接触标签。`label_reference_geometry` 生成参考几何标签；`label_contacts` 应通过 scene sidecar 加载对应障碍场景，物理回录验收成功后才能把真实受力作为有效训练标签。PHP 最长片段 26.52 秒，另需处理当前 15 秒回合上限。

除已生成 `datasets/g1_box_climb_seed_v2` 的单条箱攀参考外，这些数据尚未完成本项目的训练导入；完整箱攀物理验收也尚未通过，不能直接把整个 `external` 目录传给 `--motion_dir`。目前没有确认到 ACP 作者最终精选训练库的完整下载包；这里包含其箱攀来源及其他论文的补充数据。

## 版本、许可与保留范围

OmniRetarget 固定于 `135259130158f1cb1799e4ab9b6566c938285fc9`，只下载两个涉及地形的 ZIP 与模型，未下载纯 robot-object 包；数据卡声明 MIT。PHP 固定于 `c7c9ce3fc75a8754a7e403c58de3b6933254d01b`，保存了 Apache-2.0 LICENSE 与发布 manifest。VideoMimic 保存了下载脚本、动作 YAML、README 与 MIT 代码许可快照，未发现独立数据许可；代码许可不自动证明数据适用同一条款。

PARC 约 3.91 GB 的完整 ZIP 本轮未下载，仅保存公开元数据；优先采用已经获得的原生 G1 数据。本机已有 BONES-SEED，未重复下载；其中梯子候选仍需配套几何和接触标注。Kimodo 模型、Mixamo 与未确认公开的 LadderMan 完整训练数据未下载。来源与本机候选详见 [数据指南](../../docs/DATA_SOURCES.md)。
