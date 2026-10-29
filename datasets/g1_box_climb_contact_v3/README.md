# G1 箱攀接触教师 V3 参考

此目录是本地工程派生数据，**不是 ACP 作者发布的原始动作**。来源是 `datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501`，其上游为 OmniRetarget 的配对箱攀动作/地形。

主 clip：`climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3`。551 帧、50Hz、11 秒；原动作的 8 个时序数组前 501 帧保持相同 dtype 和原始数组字节，追加 50 帧平滑 settle 到独立优化的终点。新标签针对完整 551 帧重新生成，不能把动作前缀保留理解成标签也保持原样。

原末帧不满足新的静止支撑目标。稳定目标工具在相同 5mm 平面候选筛选下得到原 CoM 凸包余量 −5.20cm、新值 +3.204cm；20mm 容限的旧诊断约为 −4.31cm，二者不是同一筛选口径。优化只验证几何潜力，未求得动力学平衡或真实受力解。

| 文件 | 用途 |
| --- | --- |
| `.npz` | 551 帧动作、完整命名关节/身体运动学及速度 |
| `_contacts.npz` | 九部位参考几何接触、未来 body-origin 目标、表面点与有效性 |
| `.scene.json` | 绑定场景、初始变换、来源与碰撞资产哈希 |
| `.settle.json` | 原数据哈希、逐数组前缀凭据、追加段算法与来源 |
| `diagnostics/geometry_gate.json` | 全序列场景/自碰/接触几何门禁 |

几何标签仍为 `physics_validated=false`、`forces_used=false`。九部位为双手、双肘、双膝、双足与骨盆；关节/身体列使用名称映射，不能按列位置猜测。运行时 G1 有 29 个物理轴、23 个策略输出，其余六轴 held nominal。

V3 从起点执行 550 个参考控制步，再执行 50 个额外真实物理保持控制步，共 600 步/12 秒。追加 settle 与最后的物理保持是两个阶段，不能用参考末帧、定格视频或几何检查代替实测保持。

检查此数据：

```bash
conda activate env_isaaclab_2
python -m morph_mimic.scripts.check \
  --motion-dir datasets/g1_box_climb_contact_v3 \
  --clip climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3
```

教师运行目录为 `runs/g1_box_climb_contact_teacher_v3_1024`，已在 6000 轮自动停止：5000/5500/6000 连续三次均通过完整起点与物理保持。该成绩属于 584 维参考增强教师，后续 468 维学生仍须独立验收。旧 v2 第 2000 轮仅提供显式 actor 初始化，教师的 critic、优化器、噪声和迭代从零开始。

输入被正式运行冻结后不得原地改写。派生动作/标签/诊断保持本机，只有本说明随 Git 保存。方法边界见 [论文指标审计](../../docs/PAPER_METRIC_AUDIT.md)，控制和验收见 [箱攀管线](../../docs/BOX_CLIMB_PIPELINE.md)。
