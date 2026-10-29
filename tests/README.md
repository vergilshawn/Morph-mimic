# 测试分组

47 个测试模块合并为 7 个主题模块，公共运行入口为 `morph_mimic.scripts.check`。原脚本源码已合并并删除，未通过动态加载保留另一套散落脚本。共享的临时数据生成留在 `reference_fixtures.py`，选择和排序集中在 `conftest.py`。

| 顺序 | 模块 | 合并内容 |
| --- | --- | --- |
| 1 | `test_foundation.py` | 坐标变换、资产/held 约束、标签格式、惯量；选配 planner/replay/MJCF |
| 2 | `test_reference.py` | 动作/场景契约、几何标签、场景接触；选配导入、裁剪、投影、重定向、自碰与 settle FK |
| 3 | `test_contacts.py` | 作用点/表面、力缓冲、接触跟踪与事件、支撑稳定 |
| 4 | `test_control.py` | 环境张量、CSI/RSI、观测/噪声、critic 坐标、终止与退出边界、地形 |
| 5 | `test_training.py` | PPO/保存恢复/数值保护、训练配置、教师 v1/v2/v3 控制契约与自动验收 |
| 6 | `test_evaluation.py` | 轨迹/接触指标、完整起点评估、真实箱顶保持；选配坐标诊断/FK |
| 7 | `test_workflows.py` | 统一入口、参数隔离、训练状态；选配分段诊断 |

默认基础检查保留 v1/v2 的残差目标、参考坐标和箱顶保持等控制测试，因为当前 v3 仍继承这些实现；旧版本监督流程和独立站立验收属于 `auxiliary`。

在项目根目录、具备 pytest 的 Python 环境中执行：

```bash
# 默认：297 项基础回归 → 当前机器人资产 → 当前数据契约，按表中顺序执行
python -m morph_mimic.scripts.check

# 仅基础回归
python -m morph_mimic.scripts.check --only regression

# 按需调用；可一次选多个组
python -m morph_mimic.scripts.check --only regression --suite reference
python -m morph_mimic.scripts.check --only regression --suite basic planner
python -m morph_mimic.scripts.check --only regression --suite diagnostics auxiliary

# 全部回归；安装可选 MuJoCo/SciPy 后可执行全部物理专项
python -m morph_mimic.scripts.check --only regression --suite all
python -m morph_mimic.scripts.check --only regression --suite physics

# 临时细筛，不增加新脚本
python -m morph_mimic.scripts.check --only regression --suite all --pytest-args='-k contact -v'
```

| `--suite` | 用例数 | 范围 |
| --- | ---: | --- |
| `basic` | 297 | 当前训练必要的数学、输入、控制、接触、PPO、评估和入口回归，含学生动作/历史/架构与课程门禁 |
| `reference` | 51 | 完整数据处理/场景/FK 回归，包含部分基础用例 |
| `planner` | 12 | Stage 2/3 的 planner、attention、hindsight/replay 与优化器保护 |
| `auxiliary` | 6 | 独立站立与旧教师监督流程 |
| `diagnostics` | 14 | 共同参考系审计及分段诊断，包含一个 FK 用例 |
| `physics` | 24 | 依赖 MuJoCo/SciPy 的 CPU 编译、FK、投影、自碰和物理审计 |
| `all` | 356 | 全部；多组选中的相同用例只执行一次 |

组之间可以重叠。默认基础组不选择 MuJoCo 专项；可选依赖缺失会明确跳过，跳过不代表验证通过。普通 `python -m pytest tests` 仍运行全部用例，也支持重复的 `--suite basic --suite planner`。这些测试验证代码契约和数值行为；机器人技能仍由保存检查点的完整物理评估判定。

整理时只移除了一个重复的 MLP 形状用例，将其中两个形状断言并入原有的接触解码行为测试。原 330 项变为 329 项，再增加两项入口分组/顺序回归，共 331 项。2026-10-29 学生课程另增加 25 项有效回归，当前 356 项，仍集中在原七个主题模块。没有按旧训练的成败删除仍被继承实现使用的测试。
