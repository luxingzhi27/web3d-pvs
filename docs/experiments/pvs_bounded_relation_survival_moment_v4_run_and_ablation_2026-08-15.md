# v4 完整组合快速扫描、长训与核心消融执行计划

> **计划状态（2026-08-17）**：本轮只登记后处理续跑计划，不启动扫描、训练、评测或部署进程。`scan8` 和 `formal80` 的训练产物已经完成；下一次执行只从已有 checkpoint 继续做代码校验、导出、validation、图像/资源评价、配对统计和路线报告，不得重新训练或启动并行队列。

## 当前状态

- 日期：2026-08-17
- 实验前缀：`pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4`
- 当前状态：模型、损失、训练、导出和评价代码已经完成 CUDA 单步 smoke；`scan8` 已完成 8/8 配置并冻结 `s02`；`formal80` 的 15 个成员均已完成 80 epoch。
- 执行状态：scan8 的八个成员均达到 8 epoch 并完成 calibration/validation 协议检查；validation 安全池中冻结 `s02`，其 weighted recall 为 `0.999075`、单侧 LCB 为 `0.998690`。formal80 训练产物齐全，但正式后处理曾在 FP16 生存系数融合校验处中断，当前状态为 `training_complete_evaluation_incomplete`。当前没有 v4 训练或评价进程；本轮不启动任何实验。
- 数据访问边界：参数选择只使用 `train`、`calibration` 和 `validation`，在最终模型冻结前不读取 `test`
- 默认版本边界：实验完成并形成正式结论前，不修改当前默认 checkpoint、阈值、前端资产或部署包

### 本轮交互边界

本文是执行计划和交接依据。本轮不执行计划中的命令。由于训练阶段已经完成，后续实施时只按阶段续跑缺失的后处理，不重新启动 scan8 或 formal80，也不启动并行训练队列。

计划实施前还必须完成一次代码收口：runner 的身份校验统一使用产物存在性、schema、shape、实例数、split 数量、候选/GT 语义、变体配置、checkpoint 与运行表来源一致性以及数值有限性。执行和汇报只使用这些可解释的语义检查，不增加文件摘要、代码文件指纹或其他不可解释的文件级门槛。

### 无文件哈希门控

本计划不计算、比较或汇报代码文件、checkpoint、运行包、数据文件或候选/GT 序列的哈希值、SHA 值或摘要值。后续实现收口时，活动 runner、validator、replay 复用判断、汇总脚本和报告只保留以下检查：

- 文件和目录是否存在，JSON 是否能解析，schema、字段类型和版本是否正确；
- 实例数量、特征维度、二进制文件大小、split 名称及实际 pose 数量是否一致；
- pose 顺序、候选实例 ID、真实可见 ID、权重、预测长度和实例到 GLB 映射是否按结构化字段一致；
- `visible_ids` 是否属于 `candidate_ids`，预测值、损失、梯度和导出权重是否全部有限；
- 变体、seed、epoch、checkpoint、固定阈值和运行特征表是否来自同一个成员；
- 图像、资源、bootstrap 和报告产物是否满足各自 schema、成员覆盖和数量要求。

旧运行记录中若存在摘要字段，只作为历史元数据读取，不参与新一轮执行、复用或路线判定；不为清理这些字段重新训练或重建数据集。

### 执行闸门

原始训练链已经按“代码收口 → `scan8` → 冻结配置 → `formal80`”完成。本次后续执行严格按“后处理代码收口 → 已有产物校验 → 15 个成员导出 → calibration/validation 回放 → 图像评价 → GLB 资源评价 → 10,000 次配对 bootstrap → 汇总 → 路线判定 → 正式报告”运行。不允许重新启动 scan8 或 formal80 训练，也不允许用未完成的中间目录生成路线结论。正式评价完成前不修改默认模型、默认阈值、前端资产或部署包。

### 计划确认阶段的交付边界

计划确认阶段按以下边界完成登记；实施阶段按以下边界推进：

1. 将执行顺序、固定数据口径、参数扫描、冻结规则、正式矩阵、评价指标、统计方法和失败处理写入本文；
2. 修正文档中的运行状态，使其明确记录 scan8/formal80 已完成、当前无相关进程且本轮不启动新任务；
3. 登记不依赖准入门控的完整后处理链、续跑规则、评价顺序和完成定义；
4. 后处理按固定顺序完成，不因安全门结果跳过任何变体、统计、图像或资源评价；
5. 不修改当前默认模型、默认阈值、前端部署包和既有正式结果。

### 已完成修正与待完成后处理

正式扫描启动前发现并修正了以下会影响可复现性或完整执行的实现问题：

- pose 身份记录原先直接序列化带对齐填充的结构体，重复读取同一数据可能得到不同结果；现在只序列化明确的 pose 字段，并加入重复读取回归测试。
- 安全裕度损失的辅助梯度投影在 FP32 回写后可能残留负安全点积；现在对无法由有限精度表示的冲突组执行当前步置零回退，并记录回退统计，保证不会施加安全冲突更新。
- runner 现在固定检查数据集 schema、实验身份、实例数和 train/calibration/validation pose 数量；续跑还会校验 epoch、快照周期、变体、seed、配置和成员身份。
- 扫描安全池现在显式检查 validation 的 aggregate weighted recall 和单侧置信下界均严格高于 `0.99`；冻结配置必须来自完整的八组扫描，并保留选中的 `configId`。
- 扫描按计划在第 4 和第 8 epoch 保存快照；v4 validator/exporter 不再接受已移出当前矩阵的旧变体名。
- 另修正完整评价的步数语义：`max_eval_poses=0` 现在表示遍历完整 split，而不是错误地只跑一个 batch；runner 会拒绝 calibration/validation pose 数量不符合 `168/213` 的成员。

以下协议清理属于实施阶段的前置任务；本轮只登记，不执行：

- 将 runner 和相关入口的身份检查收敛到 schema、shape、split、候选/GT、变体、checkpoint/运行表来源和有限值检查，不新增文件摘要或代码文件指纹步骤。
- 对清理后的入口执行静态编译、相关 unittest、`scan8` dry-run 和 `formal80` dry-run；dry-run 只检查任务数量和参数，不创建训练权重或评价结果。
- 复核计划文档、`AGENTS.md` 和 `docs/README.md` 的表述一致，不能留下过时的文件摘要校验门槛。

上述训练前修正已经登记，但当前工作区尚未完成最后的代码收口和验证，不能把测试或 dry-run 写成已通过。FP16 融合校验的 `0.02` 绝对误差修正属于校验容差问题，不触发重训；其余后处理入口仍需按下列计划完成。本轮只登记，不执行：

1. 将 `run_true_glb_renderer()` 的 GPU 证据采集接入正式渲染生命周期：Chrome 启动前、运行中和结束后分别保存 `nvidia-smi` 与 `nvidia-smi pmon`，并把 `gpuBackend`、`gpuGate`、启动参数和三阶段证据写入图像结果。正式模式没有完整 NVIDIA Vulkan/ANGLE 硬件证据时，该图像工作点标记失败，不能把软件渲染混入正式均值。
2. 让汇总脚本完整接收 `--resource-root`，读取 `resource/summary.json`，校验 `requestedSplit=validation` 和 15 个成员覆盖，并把资源指标回填到成员级和顶层汇总。
3. 让 replay 复用、checkpoint 校验、运行包校验和汇总只使用结构化语义，不以文件摘要、代码提交标识或任何哈希字段作为门控。旧产物中的此类字段只可被忽略，不能触发重建或失败。

后处理收口完成并通过计划中的测试后，才允许继续导出和评价；默认模型、默认阈值、前端资产和部署包在正式结论前保持不变。

本文档是当前唯一执行计划。旧的 `pilot12`、双种子 `refine24`、六变体正式矩阵和保持度数置乱对照不再进入运行链，也不保留兼容入口。当前下一次执行只处理 formal80 的后处理，不重新执行 scan8 或 formal80 训练。

## 一、执行目标

本轮不再追加新结构，直接验证已经实现的完整论文组合：

1. **共享分层遮挡关系先验与逐实例校准生存场**：离线关系网络根据真实三角形遮挡边生成共享先验，再用受约束的逐实例残差修复共享生成器对局部实例表达不足的问题；浏览器只读取融合后的 28 维生存系数。
2. **视点区域矩包络频谱查询**：用一次中心视线查询及视点区域的低秩矩描述近似区域内可见性并集，不在前端展开多个子视点。
3. **安全裕度工作区效用损失**：保留归一化 RVL 对高 weighted recall 的作用，在正例形成安全裕度后，再优化负例剔除和 GLB 资源效率。

实验只回答四个问题：完整组合能否稳定训练、能否保持重要可见实例召回、能否提高分类与剔除质量、能否降低实例和 GLB 资源成本。任何中间指标都不用于取消后续训练。

## 二、不可变执行原则

### 2.1 扫描一定产生配置

八组参数全部完成后必须冻结其中一组。达到安全标准的配置优先参与效率比较；如果八组均未达到安全标准，则按诊断排序选择相对最优者。两种情况下都继续正式长训，不能因为扫描结果不理想而停止、缩减 epoch 或取消消融。

### 2.2 正式矩阵一定训练完整

正式矩阵固定为五个变体、三个随机种子、每个成员 80 epoch。安全标准只决定结果能否成为论文主模型，不决定成员是否训练。失败成员必须在修复代码、数值或硬件问题后续跑到 80 epoch，不能从矩阵中删除。

### 2.3 评价失败不回滚训练

训练、validation 回放、统计汇总、图像评价、资源评价和运行时评价分别记录完成状态。后处理失败时保留全部 checkpoint，只补做失败的评价阶段，不重新训练已经完成的成员。

### 2.4 不临时改实验

扫描开始后不追加第九组参数，不根据中间曲线修改损失，不给单个消融单独调参，不降低 weighted recall 标准，不补入 GT，不改变后退相机候选集合，也不在 validation 或 test 上重新选择阈值。

## 三、固定输入与运行资源

| 输入 | 固定路径或口径 |
|---|---|
| PoseCSR | `neural_instance_culling/dataset/out/pose_csr_hkust_v3_bounded_relation_moment_fov66_v3` |
| 原生遮挡关系 | `neural_instance_culling/dataset/out/pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/bounded_relation_csr_v3` |
| 固定几何表 | `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_geo_features_fp16.bin` |
| 运行元信息 | `hkust-v3/assets/runtimeVisibilityMeta.json` |
| GLB 索引与文件根目录 | `hkust-v3/assets/glbIndex.json`、`hkust-v3/assets/` |
| 实例数 | 18,831 |
| split | train 2,772、calibration 168、validation 213 |
| 候选相机 | 后退相机，垂直 FOV 66 度 |
| 查询区域 | 同朝向水平圆盘 view-cell，半径 2 m |
| 真实渲染相机 | 垂直 FOV 60 度 |
| GPU | RTX A6000 × 4，动态任务队列，每卡一个训练进程 |

所有成员读取相同 pose 顺序、原生候选、实例 GT、`visible_weights`、实例到 GLB 映射和资源成本。运行前只检查文件存在性、schema、shape、split 数量、`visible_ids` 是否属于候选、数值有限性以及变体配置是否一致。

## 四、启动前代码收口

本阶段只确保程序能按计划完整运行，不评价模型效果。

### 4.1 runner 模式

将 `run_pvs_bounded_relation_survival_moment_v4.py` 收敛为三个模式：

| 模式 | 作用 | 成员数 |
|---|---|---:|
| `smoke` | 单成员、单 step 的 CUDA 和产物契约检查 | 1 |
| `scan8` | 八组完整模型快速参数扫描 | 8 |
| `formal80` | 完整组合和四个核心消融的三种子长训 | 15 |

直接删除 `pilot12` 和 `refine24` 分支；`formal80` 删除保持度数置乱关系成员。summarizer、route decision 和对应测试同步使用五变体矩阵，不能保留双重默认逻辑。

### 4.2 可恢复执行

runner 增加同一阶段内的明确续跑能力：

- 已完成且训练产物齐全的成员跳过；
- 未完成成员写入新的 attempt 子目录，不能覆盖已有日志或 checkpoint；
- 扫描只有在 8/8 成员完成后生成 `frozen_config.json`；
- 正式矩阵只有在 15/15 成员达到 80 epoch 后标记 `training_complete`；
- 单个进程异常不取消队列中其他 GPU 的任务，问题修复后只补跑失败成员。

### 4.3 一次性检查

修改完成后只执行以下检查：

1. v4 模型、损失、训练、导出、runner、汇总和路线判定相关 unittest；
2. Python 静态编译和三个入口的 self-test；
3. `scan8` dry-run，必须得到 8 个任务；
4. `formal80` dry-run，必须得到 15 个任务；
5. 一个完整模型 CUDA 单 step smoke，确认 loss/梯度有限、checkpoint 和 124 维运行表可导出。

smoke 的指标只证明链路可运行，不参与参数选择。只有代码错误、非有限值、CUDA OOM 或硬件故障需要先修复；修复完成后立即进入扫描，不再增加额外准入步骤。

### 4.4 后处理续跑前的代码修复

正式后处理真正启动前，先完成以下代码修复并通过对应测试。此阶段只修改代码和测试，不训练、不导出、不回放 validation，也不启动浏览器：

1. **按内容结构判断 replay 是否可复用。** 不能因为 `validation_evaluation.json` 存在就直接复用。脚本必须读取该文件，检查它是 validation、没有读取 test、pose 顺序完整、候选与 GT 字段长度和语义正确、预测值有限、变体和 seed 正确，并检查运行特征表与当前 checkpoint 的维度和来源一致。任何一项不满足都重新生成该成员的 replay。
2. **移除活动路径中的文件摘要依赖。** replay、checkpoint、导出和汇总阶段不得以文件 SHA、代码 SHA、候选摘要或 GT 摘要作为成功条件；复用只由上面的结构化检查决定。历史状态文件中的旧字段不作为新阶段的输入。
3. **收口 v4 runner。** 删除活动路径中的摘要计算和比较函数；preflight 改为检查 schema、shape、dtype、实例数、split 数量、pose 顺序、候选/GT 子集关系、变体、seed、checkpoint 与运行表来源及数值有限性。validator 命令、manifest、续跑状态、runtime bundle 和阈值 manifest 不再生成摘要字段。validation 命令必须持久化逐 pose 的候选 ID 和预测 ID。
4. **收口 v4 validator。** 保留 checkpoint schema、变体/seed/split、模型维度、生存系数融合有限性、阈值和 calibration 安全门校验；删除摘要字段必需性和摘要相等性校验。
5. **收口 v4 summarizer。** 验证 15 个成员、poseIndices、每个 pose 的候选 ID/预测 ID、TP/FP/FN/TN 和指标顺序；paired bootstrap 按相同 seed 和 pose 结构配对。输出成员级和顶层汇总时不写入 `*Sha256`、digest 或 hash 字段。
6. **收口路线判定和通用运行时加载器。** 删除 route 输出及 `model_runners.py` 中的 checkpoint、特征表和几何表摘要比较，保留 schema、shape、dtype、byteLength、来源语义和冻结阈值检查。
7. **补齐回归测试。** 覆盖旧 replay 缺少逐 pose ID 时不可复用、合法 replay 可复用、`visible_ids ⊆ candidate_ids`、TP/FP/FN/TN 一致性、pose 顺序一致性、无摘要字段仍能通过 schema 校验，以及资源汇总 15 成员覆盖。

上述修复完成后，才运行静态编译、v4 相关 unittest、汇总 self-test、schema self-test、`scan8 --dry-run` 和 `formal80 --dry-run`。dry-run 只检查任务数量和参数，不创建训练权重或评价结果；本轮计划确认阶段不运行训练、导出、validation、图像评价、资源评价或 bootstrap。

### 4.5 文件级实施清单

| 文件 | 计划修改 | 完成判据 |
|---|---|---|
| `run_pvs_bounded_relation_survival_moment_v4.py` | 删除活动摘要依赖；增加语义 preflight、逐 pose ID 持久化、合法 replay 判断和结构化续跑 | 15 个已有成员可被识别为“训练已完成、后处理待补”，不触发新训练 |
| `validate_pvs_bounded_relation_survival_moment_v4.py` | 只保留结构化 checkpoint/阈值/calibration 校验 | 无摘要字段的合法成员可以通过 validator |
| `summarize_pvs_bounded_relation_survival_moment_v4.py` | 按成员、seed、pose 顺序验证并汇总；保留 10,000 次配对 bootstrap | 缺 ID 或顺序错误会拒绝，合法 15 成员可汇总 |
| `decide_pvs_bounded_relation_survival_moment_v4.py` | 删除摘要输出，保留安全、分类、图像、资源和运行成本分层判定 | route 结果不依赖摘要字段 |
| `model_runners.py` | 删除运行时摘要比较，保留 schema/shape/dtype/来源/阈值检查 | 运行包结构正确即可加载 |
| `evaluate_pvs_bounded_relation_survival_moment_v4.py` | 保持 validation-only、完整 pose、逐 pose ID 和有限值输出 | 每个成员输出 213 个 validation pose 的完整结构化记录 |
| v4 相关 tests | 替换摘要断言，增加 replay、ID、集合语义和资源覆盖测试 | unittest、self-test、dry-run 全部通过 |

这一阶段的输出只有代码修改和测试日志，不产生新的模型权重，不覆盖已有结果，也不启动 GPU 训练或浏览器评价。

## 五、阶段一：八组快速参数扫描（已完成，禁止重跑）

### 5.1 固定预算

- 模型：只训练 `full`；
- 随机种子：`20260801`；
- 配置数：8；
- 每配置：8 epoch，每 epoch 50 optimizer step；
- checkpoint：epoch 4 和 epoch 8；
- calibration：完整 168 pose；
- validation：完整 213 pose，只回放 calibration 冻结阈值；
- 扫描统计：1,000 次重采样，用于快速诊断；
- 并行：GPU 0、1、2、3 动态队列，共两轮任务。

该预算用于区分明显错误的损失尺度和较有前景的参数区域，不作为创新点正式结论。按照当前最慢 CUDA 单步观测，训练加评价预计约 1.5 至 3 小时。

### 5.2 固定参数表

| 配置 | 学习率 | 生存监督 | 关系一致性 | 正例尾部 | 负例工作带 | GLB 资源 | 关系梯度上限 | 效率梯度上限 | 残差正则 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `s00` | 1e-4 | 0.10 | 0.05 | 0.15 | 0.01 | 0.005 | 0.10 | 0.10 | 0.005 |
| `s01` | 2e-4 | 0.25 | 0.10 | 0.30 | 0.03 | 0.015 | 0.25 | 0.25 | 0.020 |
| `s02` | 3e-4 | 0.40 | 0.10 | 0.50 | 0.06 | 0.030 | 0.25 | 0.25 | 0.080 |
| `s03` | 1e-4 | 0.25 | 0.05 | 0.50 | 0.06 | 0.005 | 0.10 | 0.25 | 0.005 |
| `s04` | 3e-4 | 0.10 | 0.10 | 0.15 | 0.01 | 0.030 | 0.25 | 0.10 | 0.020 |
| `s05` | 2e-4 | 0.40 | 0.05 | 0.30 | 0.03 | 0.030 | 0.10 | 0.25 | 0.080 |
| `s06` | 1e-4 | 0.40 | 0.10 | 0.50 | 0.01 | 0.015 | 0.25 | 0.10 | 0.020 |
| `s07` | 3e-4 | 0.25 | 0.05 | 0.15 | 0.06 | 0.015 | 0.10 | 0.25 | 0.005 |

其余参数固定：weight decay `1e-5`、视觉效用损失 `0.10`、下载排序损失 `0.10`、调度梯度上限 `0.25`、残差绝对上限 `4.0`、稀疏实例附加系数 `3.0`、残差 warmup/ramp `0.10/0.20`。

### 5.3 无条件冻结规则

每个配置使用自己的 calibration split 选择阈值，并在 validation 上只评价该阈值。配置选择采用确定词典序，不使用人工加权总分：

1. 先按是否同时满足 validation weighted recall `> 0.99` 及其单侧 95% 下界 `> 0.99` 排序；该字段只是优先标签，不会阻止长训；
2. 达标配置内部依次比较 balanced accuracy、useful cull、precision、预测 GLB 字节和平均预测实例数；
3. 如果没有配置达标，则全部配置依次比较 weighted recall 下界、weighted recall、balanced accuracy、useful cull、precision、预测 GLB 字节和平均预测实例数；
4. 最后用配置 ID 打破完全相同的结果。

`frozen_config.json` 必须记录选中配置、完整参数、排序依据和是否达到安全标准。该文件生成后不再人工修改，`formal80` 无条件读取它。

### 5.4 计划命令

以下命令仅作为历史复现记录，本次不执行；现有 `scan8` 结果保持不变：

```bash
LOG_ROOT="neural_instance_culling/benchmark/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_logs"
mkdir -p "$LOG_ROOT"

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_bounded_relation_survival_moment_v4.py \
  scan8 --gpu-ids 0 1 2 3 \
  > "$LOG_ROOT/scan8_stdout.log" \
  2> "$LOG_ROOT/scan8_stderr.log"
```

扫描输出固定为：

```text
neural_instance_culling/model/out/
  pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_scan8/
```

## 六、阶段二：三种子 80 epoch 完整长训与消融（已完成，禁止重跑）

### 6.1 正式矩阵

所有成员使用扫描冻结的同一组超参数、相同数据和相同训练预算。正式随机种子固定为 `20260801`、`20260802`、`20260803`。

| 变体 | 改动 | 对应结论 |
|---|---|---|
| `full` | 完整组合 | 论文候选模型 |
| `without_bounded_relation` | 用容量匹配的纯几何生成器替代分层遮挡关系先验 | 实例间真实遮挡关系是否有效 |
| `without_viewcell_moment_envelope` | 只查询 view-cell 中心视线 | 区域矩包络频谱查询是否有效 |
| `without_safety_reserve_utility` | 换回归一化 RVL | 新损失是否在保留高召回作用时改善分类和资源效率 |
| `without_instance_calibration_residual` | 保留共享关系先验，关闭逐实例校准残差 | 逐实例表达能力是否修复共享生成器欠拟合 |

总量为 `5 变体 × 3 seed × 80 epoch × 100 step = 15 个成员、120,000 个 optimizer step`。四张 GPU 使用动态队列；单个成员结束后，该 GPU 立即领取下一成员。

### 6.2 checkpoint 规则

每个成员都保存完整训练曲线和 epoch 快照：

- 若训练过程中存在满足 calibration 安全标准的 checkpoint，记录其中 useful cull 最优者为 `best_safe.pt`；
- 同时始终记录一个 `best_diagnostic.pt`，按 weighted recall、balanced accuracy、useful cull 和 precision 诊断排序；
- 正式评价优先使用 `best_safe.pt`，不存在时使用 `best_diagnostic.pt` 并明确标记未达到安全标准；
- 不使用 `last.pt` 代替最佳 checkpoint；
- 运行特征表必须从被评价 checkpoint 中保存的逐实例生存系数重新导出，不能复用末轮或其他成员的表。

没有安全 checkpoint 的成员仍完成 80 epoch、导出诊断运行包并进入全部 validation 指标计算。

### 6.3 登记命令（本次不执行新的任务）

以下命令是正式训练阶段的历史登记，仅用于说明矩阵来源，本次不执行。15 个成员已经达到 80 epoch；后续只续跑缺失的后处理阶段：

```bash
SCAN_ROOT="neural_instance_culling/model/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_scan8"
LOG_ROOT="neural_instance_culling/benchmark/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_logs"

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_bounded_relation_survival_moment_v4.py \
  formal80 --scan-root "$SCAN_ROOT" --gpu-ids 0 1 2 3 \
  > "$LOG_ROOT/formal80_stdout.log" \
  2> "$LOG_ROOT/formal80_stderr.log"
```

正式训练输出固定为：

```text
neural_instance_culling/model/out/
  pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal80/
```

训练阶段已经完成，不再估算或启动新的训练时长。任何成员的评价失败只补做对应后处理，不删除 checkpoint，不降低矩阵规模。

### 6.4 当前后处理续跑命令（只登记，不执行）

修复并通过后处理相关测试后，使用已有 `scan8` 冻结配置调用同一 runner。runner 必须识别 15 个成员已完成 80 epoch，并跳过训练，只执行缺失的导出和评价阶段：

```bash
SCAN_ROOT="neural_instance_culling/model/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_scan8"
LOG_ROOT="neural_instance_culling/benchmark/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_logs"

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_bounded_relation_survival_moment_v4.py \
  formal80 --scan-root "$SCAN_ROOT" --gpu-ids 0 1 2 3 \
  > "$LOG_ROOT/formal80_postprocess_stdout.log" \
  2> "$LOG_ROOT/formal80_postprocess_stderr.log"
```

执行前必须先确认：15 个成员目录、每个成员的 80 epoch checkpoint 和 `frozen_config.json` 均存在；执行命令本身不应创建新的训练 attempt。若 runner 仍尝试训练，必须停止并修复续跑判断后再执行。

## 七、阶段三：完整评价

正式训练已经完成后按固定顺序执行，任一评价子项失败都只补做该子项。当前尚未形成正式 validation 汇总，因此不能提前写路线结论。

### 7.1 calibration 与 validation

每个 checkpoint 只在自己的 calibration split 冻结阈值，validation 不重新扫描阈值。15 个成员全部回放 213 个 validation pose，并同时输出 pose 宏平均和 aggregate 指标。

画面安全指标：

- pose recall、aggregate recall；
- pose weighted recall、aggregate weighted recall及单侧 95% 下界；
- bad cull、平均 FN；
- mean/p95 miss-pixel、wrong-ID pixel、extra-pixel。

分类指标：

- pose 与 aggregate precision、F1、Jaccard；
- instance accuracy、balanced accuracy、specificity；
- 正负样本概率分位数、冻结阈值和阈值邻域稳定性。

剔除与资源指标：

- useful cull、平均预测数、GT 数、候选数、FP、FN、TN；
- 预测数/候选数、预测数/GT 数；
- 预测 GLB 数量和字节、GLB 数量削减和字节削减；
- download utility recall、同等视觉效用所需 GLB 字节。

运行成本：

- CUDA mean/p50/p95 前向延迟；
- 固定特征表、查询权重和频谱查表大小；
- 最终候选模型的硬件 WebGPU mean/p50/p95、主线程时间和峰值内存；
- 首次加载、首次预测和稳态预测时间。

### 7.2 配对统计

使用相同 seed 和相同 validation pose 进行 10,000 次 paired bootstrap：先按 seed 聚类，再在 seed 内重采样 pose。固定比较为：

- `full - without_bounded_relation`；
- `full - without_viewcell_moment_envelope`；
- `full - without_safety_reserve_utility`；
- `full - without_instance_calibration_residual`。

每组报告差值、双侧 95% 置信区间、方向和是否跨零。必须同时显示 weighted recall、precision、accuracy、balanced accuracy、useful cull、bad cull、平均预测数、GLB 字节和图像指标，不能用单个指标替代完整判断。

### 7.3 图像和资源评价

图像评价使用 validation view-cell 的实例级 Color-ID 渲染，Chrome 必须通过 NVIDIA Vulkan/ANGLE 硬件路径。资源评价使用同一 GLB 索引和实际字节；实例可见性保持实例粒度，下载排序才聚合到 GLB 粒度。

图像与资源评价不再作为训练启动条件。训练完成后如浏览器环境异常，先保留模型与离线 validation 结果，修复硬件评价入口后继续，直到全部 15 个成员的图像和资源结果齐全。

## 八、最终模型冻结与前端验证

### 8.1 论文资格只影响结论

完整模型满足以下条件时可以晋级论文主模型：

- validation aggregate weighted recall `> 0.99` 且单侧 95% 下界 `> 0.99`；
- 相对至少一个核心消融，在 balanced accuracy、precision、useful cull、GLB 字节或图像指标中出现稳定收益；
- 图像安全没有稳定恶化；
- 神经资产不超过 7 MiB，运行时仍为固定实例表加轻量查询，不引入在线关系图或多子视点查询。

若完整模型未满足这些条件，实验仍视为执行完成，报告明确写出失败原因，并从正式矩阵中选出最强诊断成员作为后续研究起点；当前前端默认模型保持不变。

### 8.2 冻结 test

只有 validation 完成并冻结最终候选后，才对“当前部署基线”和“最终候选”各执行一次完整 test。test 不参与阈值、checkpoint、超参数或模型选择。若没有任何新成员获得论文资格，则不读取 test，并在报告中说明原因。

### 8.3 前端范围

只为最终候选实现一次 v4 WebGPU 适配，不为 15 个训练成员维护十五套前端资产。前端继续使用融合后的 `96 + 28 = 124` 维固定实例表、轻量视角查询和实例级输出；不运行关系传播、邻居搜索、PointNet、AABB 八角点投影或在线子视点展开。硬件 WebGPU 通过数值一致性和性能测试后，才允许规划替换默认前端资产。

## 九、恢复与监控

- 所有长任务写入成员级 stdout/stderr，日志包含 epoch、step、损失分项、学习率、速度、ETA 和 GPU 绑定；
- 当前只监控后处理阶段，每 3 至 4 小时检查一次，不使用高频自动监控脚本；不再监控或重启已经完成的训练成员；
- 检查导出成员数、validation 回放进度、图像 GPU 证据、资源汇总、bootstrap 次数和日志增长；
- GPU 门失败、指标缺失、产物结构错误或进程退出时，记录失败原因，修复后只补齐对应后处理阶段；
- 不静默修改候选集合、阈值来源、checkpoint、损失权重、epoch 或矩阵成员；正式结论前不替换默认前端资产；
- 后处理耗时不作为启动新训练的理由，完成时间以实际阶段日志为准。

## 十、输出与完成定义

| 产物 | 固定位置 | 完成条件 |
|---|---|---|
| 快速扫描 | `..._v4_scan8/` | 8/8 成员各 8 epoch |
| 冻结配置 | `..._v4_scan8/frozen_config.json` | 无条件选出一组完整参数 |
| 正式训练 | `..._v4_formal80/` | 15/15 成员各 80 epoch |
| validation 汇总 | `..._v4_formal_validation/validation_summary.json` | 15 成员、213 pose、全部必报指标 |
| 配对统计 | formal validation 目录 | 10,000 次、四组核心比较 |
| 图像与资源 | formal validation 下的 `image/`、`resource/` | 15 成员结果完整 |
| 路线结论 | `route_decision.json`、`route_report.md` | 安全、分类、图像、资源、运行成本分层结论 |
| 正式报告 | `docs/evaluation/pvs_bounded_relation_survival_moment_v4_formal_validation_*.md` | 方法、训练曲线、消融、失败项和成本完整记录 |

本计划只有在以下事项全部完成后结束：8 组扫描、一个冻结配置、15 个 80 epoch 成员、calibration 阈值、完整 validation、配对统计、图像和资源评价、最终运行资产大小与运行时状态、正式报告。安全标准只决定“是否晋级主模型”，不允许缩短这条执行链。

## 十一、本轮确认后的实际执行顺序

本节是后续真正执行时的顺序清单，当前只登记，不启动：

1. **代码契约收口。** 完成 replay 复用修复、移除文件摘要门控，运行静态检查、单元测试和两个 dry-run。
2. **已有产物审计。** 读取 `scan8` 的八个成员和 `formal80` 的十五个成员，检查成员目录、epoch、checkpoint、运行表、校准摘要、日志和当前 pipeline 状态；不重训已完成成员。
3. **成员导出。** 为每个正式成员从被评价 checkpoint 重新确认并导出对应的固定运行表和模型包，确保逐实例生存系数、阈值和 checkpoint 属于同一成员。
4. **校准与 validation replay。** 每个成员只使用自己的 calibration 冻结阈值，完整回放 213 个 validation pose；不读取 test，不补入 GT，不改变原生后退相机候选集合。
5. **图像与资源评价。** 使用同一 validation pose 执行实例级 Color-ID 图像评价和 GLB 资源评价。正式图像评价必须有 NVIDIA Vulkan/ANGLE 硬件证据；软件后端只能记录为失败，不能混入正式均值。
6. **统计与汇总。** 对五个变体和三个 seed 生成完整指标，运行按 seed 聚类、seed 内按 pose 重采样的 10,000 次 paired bootstrap，输出四组核心消融差值和置信区间。
7. **路线判定与报告。** 同时报告 weighted recall 安全门、普通分类指标、useful/bad cull、图像损失、GLB 成本和运行延迟；安全门只决定是否晋级论文主模型，不跳过任何失败成员的报告。
8. **最终状态审计。** 检查报告、汇总、图像/资源证据和日志齐全，确认默认模型、默认阈值、前端资产和部署包未被修改；完成后才决定是否另行申请前端接入。

后续每个阶段都采用“完成则复用，缺失或结构不合法则只补该阶段”的续跑策略。任何指标不达标都不能取消矩阵、缩短训练或改写候选集合；只有代码错误、非有限数值、CUDA OOM 或硬件故障可以暂停，修复后从未完成阶段继续。
