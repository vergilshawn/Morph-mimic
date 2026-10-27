# G1 单条箱攀参考

当前动作是 `climb_03_z_scale_1.0_scene_fit_0000_0501`，来自 OmniRetarget 首次完整上箱的 0–10 秒，501 帧、50 Hz。场景约束优化保持六个非控制关节为 nominal，并保留完整帧序列。

- 环境最大穿透 4.425 mm，自碰最大穿透 5.191 mm。
- 原始几何接触召回 99.733%；生成的是九部位参考几何标签，未声明受力或物理跟踪通过。
- 同名 `.scene.json` 绑定源场景与哈希；`raw_reference/` 为源动作对照，`diagnostics/` 为独立检查记录。
- 已用于五轮真实 Stage 1 PPO，checkpoint 为 `runs/g1_box_climb_pipelinecheck_v2/final.pt`。
- 完整箱攀仍失败：短训练策略从相位 0 开始，在相位 0.152 终止。需要继续训练与验收。

完整状态和哈希见 [validation_status.json](validation_status.json)，可复现命令见 [箱攀管线文档](../../docs/BOX_CLIMB_PIPELINE.md)。不要把 `raw_reference/` 或诊断 NPZ 当作正式训练动作；当前场景入口限定一个动作、一个环境。
