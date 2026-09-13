# PVS 论文基线与评价契约

日期：2026-09-09

状态：评价接口、sidecar、场景统计和 AABB+Ray MLP 正式入口已完成。五个场景的
AABB+Ray MLP 扫描与三种子 `40 x 900` 训练均已完成；HKUST、Sponza、Big City 和
Viking Village 已生成三种子 frozen-test 结果，IFCBench 尚需补齐 seed02/03 的独立
test JSON 与 score sidecar。

## 目的

本次改动把正式 test 读取、统一指标、分数存储和论文表格生成固定为同一套契约。正式数据只从 `/mnt/sda/rhyang/slm` 读取，新增结果写入当前 worktree 的 `neural_instance_culling/**/out`。

## 代码变更

- `evaluate_pvs.py` 增加冻结 test replay。test 必须显式指定 checkpoint 和其 calibration summary，只读取 checkpoint 自己的 calibration threshold，禁止诊断阈值和重新校准，并输出 `testRead=true`。
- `score_sidecar.py` 保存 pose index、candidate/predicted offsets、candidate/predicted IDs、float32 score、uint8 target 和 float32 visible weight。offsets 是每个 pose 的唯一行边界；同分 AP 合并相同 score 组，零 GT pose 保留且 AP 返回空值。
- `evaluate_unified_pvs_metrics.py` 使用 aggregate weighted recall 及其 bootstrap 下界作为安全工作点门，支持 frozen threshold、同分 AP、零 GT pose 和 sidecar；test 不允许 threshold scan、抽样或候选截断。
- `scene_statistics.py` 统计场景范围、原型/展开三角形、GLB 字节分布、实例到 GLB 复用率，以及四个 split 的 candidate、GT、visible weight 分布。
- `train_aabb_ray_baseline.py` 与 `run_aabb_ray_baseline.py` 注册 18 维 AABB+ray/MVP MLP，采用 pose-balanced BCE、单侧 RVL recall guard 和共享困难尾部分离目标。扫描固定为两个 LR 的 `6 x 300`，确认训练固定为三种子 `40 x 900`，每个成员保存 stdout/stderr。

2026-09-09 协议复核修正：统一阈值行直接提供 `TP/FP/FN/TN`，但旧汇总代码读取了不存在的 `agg_useful_cull/agg_bad_cull`，使 aggregate useful/bad cull 变成空值。现按 `TN/candidate` 与 `FN/candidate` 从混淆计数计算；calibration 阈值和 validation 配置都在严格通过 weighted recall 及 LCB `0.99` 后，先最大化 useful cull，再比较 balanced accuracy、specificity 和 precision，weighted-recall 下界只作后续平局项。若没有安全成员，诊断选择仍优先选择 weighted-recall 下界最高者。扫描 checkpoint 不重训，只需用修正后的评价代码重新完成 calibration/validation 选择。
- `generate_paper_tables.py` 只接收 `testRead=true` 且一次 test 读取的结果，生成 Table 1/2 CSV 和 Markdown。

## 运行与产物

CPU preflight 已完成并保存：

- `paper_results/preflight/hkust_v3.json`
- `paper_results/preflight/ifcbench_fantasy_metropolis.json`
- `paper_results/scene_statistics.csv`
- `paper_results/table1_scene_statistics.csv`

HKUST Full V4 frozen test 已读取全部 `684` 个 test pose，产物为：

- `paper_results/test_metrics/hkust_v3/full_v4_seed20260802.json`
- `paper_results/test_metrics/hkust_v3/full_v4_seed20260802.sidecar/manifest.json`
- sidecar candidate rows：`3,174,148`
- sidecar predicted rows：`281,263`

## 当前指标

| 指标 | HKUST Full V4 test | 含义 |
|---|---:|---|
| aggregate weighted recall | 0.996065 | 按 visible weight 汇总的 GT 找回率，画面安全主指标 |
| weighted recall 单侧 95% 下界 | 0.992348 | pose bootstrap 的安全下界 |
| aggregate recall | 0.961017 | 合并所有候选后的普通 GT 找回率 |
| aggregate precision | 0.241297 | 预测候选中真正 GT 的比例，受候选规模影响 |
| balanced accuracy | 0.946129 | recall 与 specificity 的平均值 |
| useful cull | 0.910522 | `TN / candidate`，正确剔除效率 |
| bad cull | 0.000867 | `FN / candidate`，错误剔除造成的画面风险 |
| 平均 candidate | 4640.57 | 每个 pose 的候选实例数 |
| 平均 prediction | 411.20 | 冻结阈值下每个 pose 的预测实例数 |
| aggregate AP | 0.366669 | 合并候选排序质量，使用同分组 AP |
| pose-macro AP | 0.885712 | 仅对有 GT pose 求平均的排序质量 |
| GLB 字节削减 | 0.786276 | 预测 GLB 字节相对候选 GLB 字节的削减比例 |
| 零 GT pose | 0 | 零 GT pose 仍进入集合和资源统计 |

图像 PER、miss pixel rate、wrong-ID pixel rate 和浏览器硬件延迟属于独立正式阶段，本次未实现，不能从上述集合指标推断。

## AABB 执行状态

两场景 preflight 的 split 数量均符合计划：HKUST 为 `5926/659/730/684`，IFCBench 为 `19647/2183/2712/2710`。IFCBench test 中有 `26` 个零 GT pose。

正式实验 `pvs_aabb_ray_mlp_formal_v1` 已对五个场景执行两个学习率的 `6 x 300`
扫描，并统一选择 `2e-4`，随后完成三种子 `40 x 900`。HKUST、Sponza、Big City 和
Viking Village 均已有三个种子的独立 frozen-test JSON 与 score sidecar。IFCBench
目前只有 seed20260801 的独立 frozen-test 产物；seed20260802/03 已有 checkpoint 和
calibration/validation 选择，但仍须按各自冻结阈值补写完整 test JSON 与 sidecar，不能
用汇总文件中的 inline row 代替。

## 验证

- benchmark 契约测试：`68 passed, 2 skipped`
- 相关 Python 文件编译检查通过
- Full V4 frozen test：`testRead=true`、`testEvaluationCount=1`、全部 `684` pose、sidecar offsets 覆盖完整数组
- 未修改 `train_pvs.py`、HZB、streaming 或前端。
