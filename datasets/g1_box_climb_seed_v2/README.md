# G1 单条箱攀参考

当前动作是 `climb_03_z_scale_1.0_scene_fit_0000_0501`，来自 OmniRetarget 首次完整上箱的 0–10 秒，501 帧、50 Hz。场景约束优化保持六个非控制关节为 nominal，并保留完整帧序列。

- 环境最大穿透 4.425 mm，自碰最大穿透 5.191 mm。
- 原始几何接触召回 99.733%；生成的是九部位参考几何标签，未声明受力或物理跟踪通过。
- 同名 `.scene.json` 绑定源场景与哈希；`raw_reference/` 为源动作对照，`diagnostics/` 为独立检查记录。
- 作为新版 `datasets/g1_box_climb_contact_v3` 原始 501 帧的来源保留；旧 v2 已停止，新主训练为 `runs/g1_box_climb_contact_teacher_v3_1024`，V3 第 3000 轮单次通过完整新门禁，尚未完成连续三次验收。
- 早期失败管线的权重、旧派生试样和重复归档已清理；有效历史证据集中到 `runs/_reports/`，路径映射见 [清理记录](../../runs/_reports/cleanup.json)。

来源与早期管线记录见 [validation_status.json](validation_status.json)，其中历史启动状态不代表当前训练。当前状态以运行目录的 `supervisor_status.json` 为准，可复现命令见 [箱攀管线文档](../../docs/BOX_CLIMB_PIPELINE.md)。不要把 `raw_reference/` 或诊断 NPZ 当作正式训练动作；当前训练限定一个动作及其绑定场景，支持多环境独立复制。
