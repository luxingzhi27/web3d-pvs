# M7：统一下载调度评估入口

日期：2026-08-01
状态：评估器协议、独立 RankNet runner 和小型 fixture 已完成；正式主线/独立排序器 validation/calibration 长实验尚未完成。

## 变更目的

M7 需要回答一个独立于实例显示阈值的问题：在相同候选实例集合和相同资源预算下，不同的可见性、视觉效用与下载排序组合，能否更早保留高价值画面，同时减少不必要的 GLB 下载。

旧版 `evaluate_visual_utility_metrics.py` 存在三个协议问题：

1. 默认可以在 test split 扫描阈值，导致 test 被用于工作点选择；
2. 只支持 `train/val/test` 的不完整 split 入口，并且评估循环默认跳过空 pose；
3. GLB 排序无论实验声明如何，都优先取 runner 的 download score，无法区分可见性后处理、级联模型和门控方案。

本次改动把可见性阈值、实例排序分数、GLB 聚合和资源预算拆开，便于后续在同一 validation/calibration 协议上比较调度方法。

## 修改范围

仅修改或新增以下文件：

- `neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py`
- `docs/experiments/m7_unified_download_scheduling_2026-08-01.md`

没有修改 `model_runners.py`、`evaluate_unified_pvs_metrics.py`、M3/M5 文件或前端。

## 当前评估协议

### Split 与阈值

- 支持 `train`、`val`、`validation`、`calibration`、`test`。
- `val` 在存在原生 `validation` split 时解析为 `validation`，否则只作为旧数据集的 `val` 别名。
- 请求 `calibration` 但数据集没有该 split 时直接报错，不从 test 推断或临时切分 calibration。
- validation/calibration 默认遍历该 split 的全部唯一 pose；空 pose 也计入 pose 分母，不会静默丢失。
- validation/calibration 没有冻结清单时才扫描注册的 threshold grid，用于工作点诊断。
- test 必须提供由 `build_frozen_threshold_manifest.py` 生成的 `--frozen-threshold-file`，并校验 manifest schema、每个模型的 calibration/test protocol、`testEvaluationCount=1` 和 frozen test digest；test 不提供扫描开关，也不接受普通阈值映射替代 manifest。
- test 禁止有放回抽样、pose 截断、候选截断和缺失 GLB 字节插值。
- 所有 split 都使用 CSR 中保存的后退相机候选集合；评估器不把 GT 可见实例补回候选，也不重算候选。

因此正式 test 的命令只能采用类似下面的形式：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py \
  --split test \
  --frozen-threshold-file /path/to/frozen_thresholds.json \
  --models pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/benchmark/out/m7_hkust_test_frozen
```

阈值清单应由 `build_frozen_threshold_manifest.py` 根据独立 calibration 结果生成，不能从旧的 test 扫描摘要直接改名得到。只有 validation/calibration 的小 fixture 才允许使用不带 provenance 的直接阈值映射。

### 四种 score mode

| 模式 | 实例排序分数 | 状态与边界 |
|---|---|---|
| `visibility-only` | 实例可见性概率 | 已实现；用于判断专用下载头是否只复现可见性后处理 |
| `independent-utility` | 独立训练的效用/排序器输出 | runner 已实现；未注册独立 checkpoint 时仍为 `not_implemented`，评估器不会用当前 utility head 冒充它 |
| `current-cascade` | 当前级联下载头输出，必要时把 logit 转为概率 | 已实现；只在 runner 暴露 download head 时可用 |
| `visibility-gated` | 可见性概率乘以当前 utility-head 输出 | 作为 `diagnostic_proxy`；当前 utility head 本身读取可见性，因此不能解释为独立显著性头或已训练的 `visibility × salience` 模型 |

当前模型的数据流是“可见性输出 -> utility head -> download head”。因此 `current-cascade` 可以验证现有联合头，`visibility-gated` 只能验证一个后验门控诊断。要让 `independent-utility` 进入论文公平比较，必须先训练、注册并记录具有独立训练数据和相同输入预算的排序器；在此之前只能报告未实现。

### GLB 聚合

每个 score mode 都可以独立比较以下四种同一 GLB 内实例分数组合：

- `max`：取 GLB 内最高实例分数；
- `sum`：累加所有实例分数；
- `top-k`：累加最高的 `k` 个实例分数，`k` 由 `--aggregation-top-k` 注册，默认 2；
- `noisy-or`：使用 `1 - product(1 - score)`，避免大量实例简单累加造成无界分数。

排序相同分数时使用 GLB id 升序作为确定性 tie-break。真实可见效用仍按该 pose 中真实可见实例的 `log1p(visible_weights)` 求和，用于计算预算内效用召回；聚合方式只改变预测下载队列，不改变 GT。

### 字节与时间预算

`utilityAtBytes` 使用 `glbIndex.json` 指向的实际文件大小，按排序顺序选择一个严格不超过字节预算的 ranked prefix。输出包括：

- `utilityRecall`：选中 GLB 覆盖的弱可见效用比例；
- `requiredRecall`：选中真实需要 GLB 的比例；
- `selectedBytes`：实际选中字节，必须不超过预算；
- `selectedGlbCount` 和 `byteReductionVsCandidate`。

时间索引支持 `totalDecodeUploadMs`，也支持同时提供 `decodeMs` 和 `uploadMs`。两项同时存在时，评估器使用它们的和；如果同时记录总时间，还会校验总时间与分项之和一致。正式评估要求每个运行时 GLB 都有正的时间成本，重复条目、未知 schema、负值和总时间不一致都会直接失败。推荐 schema 为 `neuralstreamweb3d-glb-cost-index-v1`。

`utilityAtTime` 只有在显式提供 `--glb-time-index` 后才会计算。时间索引可以是 `times`/`costs` 的 GLB id 映射，也可以是带 `decodeMs`、`uploadMs` 或 `timeMs` 字段的 entries。没有该索引时，输出保留 `status: not_available` 和原因；脚本不会把字节除以一个假定带宽，也不会把 runner forward time 当成 GLB 解码时间。

示例参数：

```bash
--byte-budgets 1048576,5242880,10485760,20971520 \
--time-budgets-ms 100,250,500,1000 \
--glb-time-index /path/to/measured_glb_decode_upload_ms.json
```

`--budgets` 仍表示 GLB 数量预算，只作为辅助曲线，不能替代严格的 `utility@bytes` 或 `utility@time`。

## 已完成的验证

本轮没有启动真实模型、完整数据集或长时间实验，只运行了 model-free fixture 和静态检查：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py \
  --self-test
```

结果：`status: passed`。fixture 覆盖：

- 严格字节预算和显式时间预算均不超预算；
- 无时间索引时 `utilityAtTime.status=not_available`；
- 四种 score mode，其中独立 ranker 明确为 `not_implemented`；
- `max`、`sum`、`top-k`、`noisy-or` 四种 GLB 聚合；
- 冻结阈值 manifest 的模型完整性和阈值范围检查；
- 包含候选集合和 pose 计数的完整 pose-set 评估路径。

静态检查：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m py_compile neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py
git diff --check -- neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py
```

两项均通过。

另外使用临时的四 pose CSR fixture 做了主入口检查。fixture 含有独立的 `train`、`validation`、`calibration` 和 `test` split，但本次只调用前两个非 test split：

```text
validation: ok, evaluated poses=1, threshold rows=46
calibration: ok, evaluated poses=1, threshold rows=46
```

fixture 还验证了保存候选集合没有被补入 GT、空 pose 计数路径可进入、四种 GLB 聚合可以生成 `utilityAtBytes`，且每一行的 `selectedBytes` 不超过声明的预算。对缺少冻结阈值的 test 命令执行参数检查时，脚本在加载数据集之前拒绝并输出：

```text
Formal test evaluation requires --frozen-threshold-file; test threshold scanning is disabled.
```

因此本轮没有读取 test split，也没有产生 test 阈值表或 test 指标。

自测还使用带有正式 schema、provenance、`testEvaluationCount=1` 和 digest 的 fixture manifest 验证了 formal manifest loader；不带这些字段的直接映射只适用于非 test fixture。

## 未实现项与门控条件

当前文档不把 M7 说成已经通过投稿门控，原因如下：

1. 尚未在空间隔离数据上完成正式 validation/calibration 与冻结 test；本轮仅验证代码协议。
2. 独立 RankNet runner 和 train-only 训练入口已经补齐，但两个场景的 checkpoint、正式 calibration 和同容量结果仍在运行，不能提前写入主表。
3. 当前仍没有每个 GLB 的真实设备解码/上传时间索引，因此静态评估中的 `utility@time` 在真实运行中会保持不可用，直到采集可复现的设备与网络测量；本轮只修复了时间索引的解析和一致性门控。
4. M8 的最小离线轨迹、冷/温缓存、带宽并发、missing-utility integral 和无效下载字节回放已单独实现于 `evaluate_download_trajectory.py`，但它使用确定性的等份带宽槽位模型，尚未替代真实 4G/Wi-Fi trace、设备解码上传和首屏 p95 帧时间实验。详见 `m7_m8_trajectory_replay_2026-08-01.md`。
5. 图像级 miss-pixel、wrong-ID 和实例级渲染正确性不在本文件内实现，仍由 M5 管线负责，M7 只消费 weak utility 目标。

按照投稿计划，联合下载头只有在相同预算下相对最佳启发式或等容量独立排序器减少至少 15% 字节或首屏时间，并且多种子/多轨迹置信区间不跨零时，才可作为“联合调度优于后处理”的主张。否则应把下载头降级为工程组件，保留可见性安全结果，不用调度指标包装失败的独立 ranker 对照。

## 保留判断

本次代码保留为 M7 主线评估入口。它解决的是实验协议和可比较性问题，并没有提前宣称算法收益。后续正式运行必须在输出目录保存 `summary.json`、`summary.md`、冻结阈值文件路径、实际 split 解析结果、GLB 字节来源和时间索引来源；任何缺失资源或未实现模式都应保留为显式状态，而不是回退到旧的 runner download score。

## 2026-08-01 正式 HKUST validation/calibration

HKUST 主线和独立 RankNet 已在严格保存候选集合上完成完整 validation/calibration。validation 遍历
`664` 个唯一 pose，calibration 遍历 `690` 个唯一 pose；GLB 成本使用实际文件字节，时间成本没有提供，
因此 `utilityAtTime` 保持 `not_available`。可见性阈值只作为 validation/calibration 工作点诊断，不能当作
正式 test 阈值。

主线在 calibration 的安全工作点为阈值 `0.05`，weighted recall `0.990532`、useful cull
`0.881718`、bad cull `0.005222`、平均预测 `158.78`、平均候选 `5078.98`。validation 诊断工作点为
阈值 `0.03`，weighted recall `0.990822`、useful cull `0.884581`、bad cull `0.010269`、平均预测
`197.53`。独立 RankNet 不是实例可见性主模型；它只用于下载排序对照。

在 `20 MiB` 字节预算和 `max` GLB 聚合下，弱 `log1p(visible_weights)` 效用教师的结果为：

| 场景/split | score mode | 效用召回 | 字节削减 |
|---|---|---:|---:|
| HKUST validation | visibility-only | 0.9440 | 47.11% |
| HKUST validation | current-cascade | 0.9254 | 47.14% |
| HKUST validation | visibility-gated | 0.9421 | 47.08% |
| HKUST validation | independent-utility | 0.8794 | 47.05% |
| HKUST calibration | visibility-only | 0.9574 | 40.47% |
| HKUST calibration | current-cascade | 0.9379 | 40.57% |
| HKUST calibration | visibility-gated | 0.9510 | 40.44% |
| HKUST calibration | independent-utility | 0.9099 | 40.46% |

“效用召回”只表示弱教师定义下被预算前缀覆盖的可见效用比例，不能解释为真实像素覆盖率；
“字节削减”相对当前候选 GLB 字节。当前结果说明可见性-only 在该弱教师和固定字节预算下优于当前级联
与独立排序器，但没有真实解码/上传时间、网络轨迹和多种子置信区间，不能宣称联合下载头的工程或论文收益。
Metropolis 正式 M7 仍在运行。

## 2026-08-01 独立 RankNet 排序器补齐

### 变更目的

此前 `independent-utility` 只返回 `not_implemented`，因此无法判断当前下载头的收益是否来自共享可见性表征，还是仅来自一个额外的排序模型。本次加入一个明确的独立对照：它输入实例 AABB、当前相机射线/FOV 和保存的 MVP 投影特征，输出实例级弱视觉效用排序分数；输入中没有当前可见性概率，也不读取目标 GLB 三角形。

### 修改文件和协议

- `neural_instance_culling/benchmark/utility_ranker.py`：共享 18 维输入的两层 SiLU MLP 定义。
- `neural_instance_culling/benchmark/train_independent_utility_ranker.py`：只使用 train split 的候选集合训练 pairwise RankNet；正样本来自保存候选中的可见实例，负样本来自同一 pose 的保存候选不可见实例。
- `neural_instance_culling/benchmark/model_runners.py`：新增独立排序 runner 和 `independent_utility_scores` 输出字段。
- `neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py`：新增 `--independent-ranker-spec name|checkpoint`，并让 `independent-utility` 只读取独立输出字段。
- `neural_instance_culling/benchmark/evaluate_download_trajectory.py`：轨迹回放支持显式注册独立排序器。

可复现训练命令模板：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/train_independent_utility_ranker.py \
  --dataset-dir <pose-csr> --runtime-meta <runtimeVisibilityMeta.json> \
  --output-dir neural_instance_culling/model/out/m7_independent_ranknet_<scene>_spatial_fov66_seed20260801 \
  --epochs 12 --steps-per-epoch 400 --pairs-per-batch 2048 \
  --pairs-per-pose 32 --seed 20260801 --device cuda \
  > train_stdout.log 2> train_stderr.log
```

`visible_weights` 只作为弱效用教师，使用 `log1p(weight) / log1p(1,000,000)` 归一化；报告中仍不能把它称为真实像素覆盖率。排序器 checkpoint 的 schema 为 `neuralstreamweb3d-independent-utility-ranker-v1`，候选语义固定为保存的后退相机候选集合，训练不访问 test。

### 当前执行状态

- HKUST 输出目录：`model/out/m7_independent_ranknet_hkust_spatial_fov66_seed20260801/`，GPU 1，训练日志已分离保存。
- Metropolis 输出目录：`model/out/m7_independent_ranknet_metropolis_spatial_fov66_seed20260801/`，GPU 3，当前 tmux 输出同时写入 `train_combined.log`；该日志包含 stdout/stderr，后续归档时保留这一 provenance 说明。
- 两个训练均已完成：HKUST 的最佳 validation RankNet loss 为 `0.0435587`，Metropolis 为 `0.2489221`；两个
  checkpoint 均写入 `best.pt` 和 `training_summary.json`，训练只读取 train/validation，未读取 test。
- 代码回归：统一视觉效用 evaluator self-test、下载轨迹 self-test 和 benchmark unittest 全部通过。

完成训练后，正式 validation/calibration 运行应把主线模型和独立排序器并列注册，并至少比较 `visibility-only`、`independent-utility`、`current-cascade`、`visibility-gated` 四种 score mode，以及 `max`、`sum`、`top-k`、`noisy-or` 四种 GLB 聚合。独立排序器只作为下载调度 baseline，不得被当作实例可见性主模型。

## 2026-08-01 独立 RankNet validation/calibration 结果

此前一次启动把未加引号的 `name|checkpoint` 传给 shell，导致 `|checkpoint` 被解释为管道并产生
`permission denied`；失败目录保留为审计记录。随后使用环境内 Python、独立会话和带引号的参数重跑：

```bash
/home/data/rhyang/miniconda3/envs/slm_pvs/bin/python -u \
  neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py \
  --models '' \
  --independent-ranker-spec 'ranker_name|/absolute/path/to/best.pt' \
  --dataset-dir <pose-csr> --runtime-meta <runtimeVisibilityMeta.json> \
  --glb-index <glbIndex.json> --glb-root <glb-root> \
  --split validation --output-dir <output> --device cpu \
  --poses-per-batch 4 --max-candidates-per-pose 0 \
  --score-modes independent-utility \
  --glb-aggregations max,sum,top-k,noisy-or
```

四个正式输出目录均已完成，且遍历完整唯一 split：

- HKUST：`m7_ranknet_hkust_spatial_fov66_validation_20260801_retry2/`（664 pose）和
  `m7_ranknet_hkust_spatial_fov66_calibration_20260801_retry2/`（690 pose）。
- Metropolis：`m7_ranknet_metropolis_spatial_fov66_validation_20260801_retry2/`（2,088 pose）和
  `m7_ranknet_metropolis_spatial_fov66_calibration_20260801_retry2/`（2,376 pose）。

下表固定使用 `max` 聚合，数值来自独立排序器的 `utilityAtBytes`；“效用召回”是弱
`log1p(visible_weights)` 教师的召回，不是真实像素覆盖率。字节削减以该 pose 的候选 GLB 总字节为分母。

| 场景/split | 1 MiB 效用召回 / 字节削减 | 5 MiB 效用召回 / 字节削减 | 10 MiB 效用召回 / 字节削减 | 20 MiB 效用召回 / 字节削减 |
|---|---:|---:|---:|---:|
| HKUST/calibration | 0.7794 / 70.69% | 0.8512 / 50.20% | 0.8797 / 44.91% | 0.9099 / 40.46% |
| HKUST/validation | 0.6794 / 77.35% | 0.8023 / 57.55% | 0.8397 / 51.27% | 0.8794 / 47.05% |
| Metropolis/calibration | 0.2613 / 95.67% | 0.4694 / 81.82% | 0.5991 / 65.30% | 0.8063 / 42.96% |
| Metropolis/validation | 0.2709 / 96.56% | 0.4513 / 85.87% | 0.5924 / 73.50% | 0.8311 / 51.53% |

独立排序器在 HKUST 的 byte-prefix 达到效用召回约 0.999 时平均需要约 108.0 MB（validation），
在 Metropolis validation 约需要 35.8 MB，但 Metropolis 的 `requiredStatus` 为 `mixed`，不能解释为
所有 pose 都满足统一的可见性安全约束。以上结果只说明排序器能产生可复现的 GLB 预算曲线；它没有
可见性输出，评测摘要里的 `best`/阈值字段是接口诊断，不能报告为 pose visibility precision 或
visibility threshold。最终仍需等待正式主线 checkpoint 的 `visibility-only`、当前级联和可见性门控
结果，再判断是否存在同等画面安全下的调度收益。

当前 M7 质量门仍为未通过：没有真实设备解码/上传时间索引、网络 trace、paired 主线比较或最终图像
损失证据，因此不把独立 RankNet 的字节曲线写成端到端下载收益。

### 质量门状态

独立排序器的 schema、输入隔离、完整 validation/calibration 和字节曲线子门已通过；M7 的联合调度
Go 条件仍未判断。还需要正式主线 calibration-ready、固定轨迹冷/温缓存回放和 paired 结果，才能比较
安全约束下的字节/时间收益。若独立排序器不弱于当前级联头，则“联合下载头优于后处理”的主张仍需
谨慎；若级联没有至少 15% 的同效用资源收益，下载头只能作为工程组件。

## 2026-08-01 正式 baseline 回放启动记录

在正式模型 checkpoint 尚未完成前，先执行与模型无关的 M7 baseline 组合，以建立同一候选集合、
同一 GLB 文件字节和同一弱可见效用教师下的资源参照。启动配置如下：

- runner：`baseline_keep_all`、`baseline_static_frequency_train`、`baseline_camera_distance`、
  `baseline_projected_aabb_area`、`baseline_aabb_ray`、`baseline_aabb_hzb`（仅 AABB depth proxy）和
  `baseline_viewcell_bitset_train`；
- split：HKUST/Metropolis 各自完整 validation 与 calibration，禁止候选截断、GT 并集和放回抽样；
- 设备：CPU，pose batch 4；
- 字节预算：1 MiB、5 MiB、10 MiB、20 MiB；未提供 GLB 解码/上传时间索引，因此 `utilityAtTime`
  必须保持 `not_available`；
- 输出：
  - `benchmark/out/m7_baselines_hkust_spatial_fov66_validation_20260801/`
  - `benchmark/out/m7_baselines_hkust_spatial_fov66_calibration_20260801/`
  - `benchmark/out/m7_baselines_metropolis_spatial_fov66_validation_20260801/`
  - `benchmark/out/m7_baselines_metropolis_spatial_fov66_calibration_20260801/`

运行由 tmux 会话 `m7_baselines_hkust` 和 `m7_baselines_metropolis` 承载，stdout/stderr 合并
保存在各 split 的 `run.log`。本记录写入时 HKUST validation 已进入 166 个 batch 的中段，
Metropolis validation 已进入 522 个 batch 的前段；尚未有完整 `summary.json`，因此不提前报告
任何 M7 数值或准入结论。

## 2026-08-01 固定决策口径修正后的独立重跑

固定 runner 评测逻辑修正后，重新启动两个独立 calibration 输出，避免复用修正前的摘要：

- HKUST：`neural_instance_culling/benchmark/out/m7_baselines_hkust_spatial_fov66_calibration_20260801_fixedrule/`
- Metropolis：`neural_instance_culling/benchmark/out/m7_baselines_metropolis_spatial_fov66_calibration_20260801_fixedrule/`

两个任务均使用完整 calibration split、原始 `frustum_ids.bin` 候选、`--max-candidates-per-pose 0`、
CPU pose batch 4 和四个严格字节预算。启动命令采用独立 session，stdout/stderr 分离写入输出目录；
截至记录时两个进程均已确认存在，尚未完成，不能读取中间日志作为指标。任务完成后只比较新摘要，旧的
`m7_baselines_*_20260801` 输出保留为修正前诊断，不进入当前主表。

## 动态正式模型注册

为了评估新训练实验而不修改全局默认模型表，统一 M7 入口增加了显式参数：

```text
--learned-model-spec name|checkpoint|runtime_features|calibration_or_test_summary
```

例如 calibration-ready checkpoint 可以这样加入已有 baseline 组合：

```bash
--models baseline_keep_all,baseline_aabb_ray \
--learned-model-spec 'pvs_formal|/path/best.pt|/path/instance_runtime_features_fp16.bin|/path/calibration_ready_summary.json'
```

模型 runner 对 `calibration_ready_pre_test` 只接受 `testEvaluationCount=0`，对
`frozen_calibration_one_shot_test` 只接受 `testEvaluationCount=1`；两者都缺失或越界时直接失败，
不会回退到旧阈值。该入口只注册命令中明确给出的实验名，不把临时实验加入默认列表。

## 2026-08-01 聚合器缺陷与修复重跑

第一次修正口径的 Metropolis calibration 在遍历完整 594 个 calibration pose 后没有写出摘要。日志显示，
固定规则在部分 pose 的弱视觉效用预算上产生了合法的 `None`（`not_applicable`）值，预算聚合器却因为
同一字段在其他 pose 为数值而执行了 `float(None)`，任务因此在最终汇总阶段失败。该问题不改变任何候选、
GT、阈值或排序决策，但会阻止存在零效用 pose 的正式场景完成评估。

修复位于 `evaluate_visual_utility_metrics.py` 的数值聚合函数：只对实际数值取平均，保留
`utilityStatus`/`requiredStatus` 的适用性状态；不把 `None` 转换为零，也不把不可用 pose 当成有效效用。
新增回归测试 `test_budget_aggregation_ignores_non_applicable_none_values`，并完成以下验证：

```text
benchmark unittest: 30 tests, all passed
evaluate_visual_utility_metrics.py --self-test: passed
evaluate_proxy_interventions.py --self-test: passed
py_compile: passed
```

HKUST 的修复后 calibration 已完整写出摘要，包含 690 个 pose；Metropolis 的失败目录
`m7_baselines_metropolis_spatial_fov66_calibration_20260801_fixedrule/` 保留为失败审计，修复后的完整重跑
使用独立目录：
`m7_baselines_metropolis_spatial_fov66_calibration_20260801_fixedrule_retry1/`。
在该目录生成 `summary.json` 之前不读取或报告 Metropolis 数值，也不启动 test 阈值扫描。

## 2026-08-01 Metropolis baseline calibration 完成核验

修正后的目录 `m7_baselines_metropolis_spatial_fov66_calibration_20260801_fixedrule_retry3/` 已写出完整
`summary.json`，遍历全部 `2,376` 个 calibration pose。候选语义为
`stored_candidate_set_strict`，GLB 成本来自本地文件实际字节数。回放包含七个 model-free runner、四种
分数模式和四种 GLB 聚合方式；该次 model-free 输出没有注册独立 RankNet，因此其中的
`independent-utility` 仍明确标记为未实现。它与本节前面的独立 RankNet 正式结果是两个不同的评测
批次，不能混用。

| Runner | 安全工作点 | weighted recall | useful cull | bad cull | 平均预测 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| keep-all | 0.5 | 1.00000 | 0.00000 | 0.00000 | 10,030.98 | 安全控制 |
| camera distance | 0.002 | 0.99862 | 0.03092 | 0.00061 | 9,285.56 | 只有很小的有效剔除 |
| static frequency | 无 | 未达标 | 未定义 | 未定义 | 未定义 | 不能进入安全比较 |
| projected AABB area | 无 | 未达标 | 未定义 | 未定义 | 未定义 | 画面安全失败 |
| deterministic AABB+ray | 无 | 未达标 | 未定义 | 未定义 | 未定义 | L0 几何规则失败 |
| AABB depth proxy | 无 | 未达标 | 未定义 | 未定义 | 未定义 | 不是 HZB |
| train view-cell bitset | 无 | 未达标 | 未定义 | 未定义 | 未定义 | L1 查表泛化失败 |

以 `visibility-only + max` 聚合为例，达到弱效用目标的 byte-prefix 需要约 `43.99 MB`，候选 GLB 平均
总字节约 `53.61 MB`，字节削减约 `20.25%`；`sum` 聚合的对应数字约为 `36.81 MB`、`27.72%`。
这些是弱 `visible_weights` 教师下的资源排序诊断，不是像素级图像质量。20 MiB 严格预算下的 utility recall
约为 `0.855`（max）或 `0.933`（sum），说明预算本身仍是主要约束。

这次结果通过了 M7 model-free baseline 的“完整 split、严格候选、严格字节、状态显式”子门，但不通过
投稿计划中的联合调度质量门：当前没有独立 RankNet/ListNet/成本敏感排序器、真实解码时间索引或设备网络轨迹，
因此不能比较“当前级联优于后处理”。
