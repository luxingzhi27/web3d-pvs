# 论文全创新组合扫参、长训与消融执行计划

- 日期：2026-08-13
- 状态：v2 代码修复、scan12 和 refine24 已完成，formal80 正在运行；本文保留为当前 v2 的预登记，不再作为后续代码改进入口
- 唯一实验前缀：`pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2`

> 2026-08-14 审计已确认当前积分公式、query center、关系层级、生存观察和资源损失存在实现问题。当前 formal80 按原预登记继续完成，不中断、不改参；其后的代码纠错与新实验以 [`pvs_bounded_relation_survival_moment_envelope_safety_reserve_plan_2026-08-14.md`](pvs_bounded_relation_survival_moment_envelope_safety_reserve_plan_2026-08-14.md) 为唯一行动计划。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-08-13
- Verification Status：V2_FORMAL80_RUNNING_AUDITED_AS_HISTORICAL_PREREGISTRATION
- Version Label：full_innovation_scan_longtrain_ablation_v2
- 研究对象：分层遮挡关系生存网络、视点区域积分频谱查询和安全工作点对齐效用损失的完整组合
- 数据访问边界：仅允许 train、calibration 和 validation；模型、阈值、资产和对比全部冻结前不得读取 test

## 一、目的与执行顺序

本计划是当前正在运行 v2 实验的原始执行依据。它取代此前关于 16/32 epoch 学习曲线、模块复验和旧 `formal80` 的重复计划，不改写已经完成的 pilot、32 epoch 学习曲线或历史正式矩阵。v2 完成后，本文只作为预登记和复现证据保留，后续实现不再沿用其已确认错误的代码语义。

后续严格按以下顺序执行：

1. 修复训练、校准、评价和导出的已确认代码契约错误；
2. 只对论文完整组合模型做快速超参数扫描，重点扫描损失权重和优化参数；
3. 用两个 seed、24 epoch 复验少量候选，并按预登记规则冻结一组最优可运行超参数；
4. 无论 pilot、复验或冻结配置是否达到安全门，都从头完成完整组合三个 seed、80 epoch 长训；
5. 使用同一冻结超参数，从头完成四个核心消融成员各三个 seed、80 epoch 长训；
6. 在 validation 上完成配对统计、图像、下载和运行成本评价；
7. 只有模型、阈值、导出 schema 和前端实现全部冻结后，才对最终模型读取一次 test。

本句记录 2026-08-13 制定计划时的执行决定：不直接续训 R2，并以修复后的 v2 独立启动 `formal80`。截至 2026-08-14，该 `formal80` 已经启动且仍在运行；当前默认模型、默认阈值、前端资产和部署包保持不变。

阶段零的代码、数据和 schema 契约是启动训练的唯一硬门。契约通过后，快速扫描负责选择配置，不负责取消后续实验；正式阶段固定完成 `5 variants x 3 seeds = 15` 个 80 epoch 成员。weighted recall 安全门、分类指标、图像指标、资源指标和运行成本只决定成员能否晋级论文主模型、默认资产或前端部署，不决定是否完成长训和消融。没有合格安全工作点的成员必须完成训练并如实标记，不能通过提前终止隐藏失败曲线。

## 二、研究假设与完整组合定义

### 2.1 三项候选创新

完整组合模型由三个可独立移除的模块构成。

**分层遮挡关系生存网络。** 输入为 96 维离线实例几何、train-only 真实有向遮挡关系和实例到局部/结构层级的映射；输出为每个实例 `4 x 7` 个遮挡生存系数。图传播、层级聚合和逐实例恢复只在离线训练与导出阶段运行，浏览器不加载关系图。

**视点区域积分频谱查询。** 输入为当前候选实例的九维 ray-space 中心量、由已登记 view-cell 几何解析得到的九维方差，以及固定遮挡生存系数；输出为一次查询所需的紧凑频谱基和八维生存语义。浏览器不展开 subpose，不运行邻居搜索或动态图传播。

**安全工作点对齐效用损失。** 保留 `rvl_strong_v2` 的高召回主干，在统一量纲的视觉权重上增加 view-cell 稀有正例尾部保护、低阈值带上的误请求抑制和真实 GLB 字节/排序监督。它必须在不读取 calibration 阈值参与反向传播的前提下，使训练目标覆盖实际可能出现的低阈值安全工作区。

### 2.2 完整运行数据流

离线阶段执行：

```text
实例几何 + train-only 真实遮挡边 + 分层映射
  -> 分层关系编码与逐实例恢复
  -> 每实例 28 维遮挡生存系数
  -> 与 96 维几何共同导出固定表
```

浏览器阶段执行：

```text
后退 66 度视锥候选实例
  -> 固定几何/生存系数 gather
  -> 一次视点区域积分频谱查询
  -> 共享轻量主干
  -> 实例可见性、实例视觉效用、GLB 下载优先级三个输出头
  -> 当前 60 度真实视锥过滤与实例级显示
```

前端禁止新增在线图传播、KNN、HZB、AABB 八角点全量投影、subpose 循环或主线程逐实例复杂后处理。下载可以聚合到 GLB，渲染过滤仍必须保持实例级。

### 2.3 需要验证的假设

- 真实遮挡来源和层级传播能够比容量匹配的几何单体编码更稳定地生成生存场，并在 weighted recall 安全门内提高 balanced accuracy、precision 或 useful cull；
- 区域积分相对同频率、同容量的中心点查询，能够改善 view-cell 边界稳定性或图像漏检，而不显著增加浏览器运行成本；
- 新损失能够保留原 RVL 的高召回能力，同时减少低阈值下的 false positive、平均预测实例数和无需求 GLB 字节；
- 三项组合后的收益能够跨三个 seed 重现，并且不是由更低阈值、补入 GT 或改变候选集合造成。

## 三、计划制定时的证据与代码审计

本节冻结记录 2026-08-13 启动 v2 前的证据，不表示 2026-08-14 的当前进度。当前运行状态和修正后的 validation 口径见 2026-08-14 综合诊断。

### 3.1 已有结果

当前最接近新关系结构的 R0/R2 学习曲线已经完成两个 seed、32 epoch。以下是相同 213 个 validation pose、各 checkpoint 自身 calibration 阈值下的 epoch 32 结果。

| 变体 | seed | weighted recall / LCB | precision | accuracy | balanced accuracy | useful cull | 平均预测数 | 预测 GLB 字节 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 自由生存场 R0 | 20260801 | 0.993755 / 0.990775 | 0.199721 | 0.827726 | 0.891587 | 0.785156 | 1102.9 | 67.22 MB |
| 自由生存场 R0 | 20260802 | 0.994810 / 0.992509 | 0.218454 | 0.847292 | 0.896438 | 0.805223 | 996.5 | 65.29 MB |
| 分层关系 R2 | 20260801 | 0.998073 / 0.996767 | 0.125719 | 0.694403 | 0.836058 | 0.650513 | 1806.4 | 121.71 MB |
| 分层关系 R2 | 20260802 | 0.993332 / 0.990348 | 0.213977 | 0.847169 | 0.880638 | 0.806560 | 982.0 | 68.92 MB |

R2 的两个 seed 均达到 weighted recall 安全门，但效率和分数分离没有稳定改善。seed 20260801 的过预测尤其严重。在本计划制定时，完整组合只有单 seed、8 epoch 的 L2/L3 pilot，validation weighted recall 约 `0.87`，不能作为完整模型训练结果；当时 `formal80` 尚未启动，随后已按本计划启动。

因此，新长训不能建立在“增加 epoch 自然会解决问题”的假设上。必须先修复损失与实际工作点的错位，再通过快速扫参选择当前最优可运行配置。扫参结果是否达到安全门只影响最终论文判定；完整长训仍用于获得完整学习曲线、跨 seed 稳定性和同预算消融证据。

### 3.2 正式训练前的阻塞问题

| 优先级 | 已确认问题 | 影响 | v2 处理 |
|---|---|---|---|
| 阻塞 | 导出器优先读取 `pose_weighted_recall`，未严格读取 `aggregateWeightedRecall` 及其下界 | 训练和导出可能对同一 checkpoint 给出不同安全判定 | 只接受 aggregate 字段；宏平均字段不能回退为安全主值；增加冲突字段回归测试 |
| 阻塞 | AABB 实际写出 FP32，但文件名和 meta 声明 `instance_aabb_fp16.bin` / FP16 | 可能造成前端字节宽度和空间谓词错误 | 统一改为 `instance_aabb_fp32.bin`、`float32`、每元素 4 bytes；不保留错误别名 |
| 阻塞 | `quality_resource` 用 `sigmoid(logits / temperature)`，作用中心仍为概率 0.5 | calibration 阈值常在 0.02--0.15，资源项没有约束真实请求工作区 | 删除旧含义的正式注册，改为预登记低阈值带上的平滑请求概率 |
| 阻塞 | 当前 HKUST `visible_weights` 是 Color-ID 屏幕覆盖 ppm，范围约 `7--877021`，进入 RVL 前没有归一化 | `1 + weight` 可使极少正例支配梯度，且损失尺度依赖分辨率/场景 | 所有 v2 成员先除以 pose 内正例最大权重，再做固定 `log1p(99r)/log1p(99)` 压缩；该修复不作为独立论文贡献 |
| 阻塞 | 模型训练了可见性、视觉效用和下载优先级三头，正式 evaluator 只推理可见性头 | 当前 GLB 指标只是可见集合映射，不能证明下载头有效 | 增加下载头独立排序与预算评价；可见实例集合和下载排序分别报告 |
| 阻塞 | `candidateGlbBytes` 没有完整进入 evaluator summary、standalone summary 和 bootstrap | GLB 字节削减缺少统一分母和置信区间 | 补齐逐 pose、aggregate、pose macro 和 paired bootstrap 字段 |
| 高 | R0 自由逐实例系数与 R2 共享关系编码器容量和归纳偏置不匹配 | `R2 - R0` 不能解释为关系模块纯贡献 | 新增容量匹配的几何单体系数生成器；R0 只作冻结工程参考 |
| 高 | view-cell 半径、形状、朝向扰动和 back offset 主要依赖代码默认值 | 数据、训练和前端可能在未来静默失配 | 写入 dataset/checkpoint/export provenance，并由 runner 强校验 |
| 高 | 当前简单来源平移置乱不保持几何分布和节点度数 | 不能严格证明模型使用真实来源身份 | 预计算方向、深度和度数分层内的确定性置乱控制 |
| 诊断 | calibration 网格覆盖到 `1e-8`，低阈值很密集 | 可发现安全尾部，但也容易用大量过预测换取 recall | 保留低端覆盖，增加两阶段局部加密和阈值健康报告；阈值接近 0.5 不设为硬门 |

### 3.3 当前正确且必须保留的实现边界

- train-only 关系 CSR 明确拒绝 test 和非 train split；
- 训练和评价不补入 GT，不改变原始后退相机候选集合；
- 每个 checkpoint 使用自己的 calibration 冻结阈值；
- 浏览器运行表只包含固定几何、固定生存系数和轻量查询网络；
- 三个任务头共享同一基础表示，但实例显示和 GLB 下载保持不同输出粒度；
- 输出目录拒绝覆盖已有成员。

## 四、v2 损失的明确实现

### 4.1 视觉权重统一量纲

对 pose `p` 中真实可见实例 `i` 的原始覆盖权重 `w_pi`，先除以该 pose 的正例最大权重，再使用固定压缩常数 `kappa=99`：

```text
r_pi = w_pi / max_j_in_Gp(w_pj)
q_pi = log1p(kappa * r_pi) / log1p(kappa)
```

不可见候选的 `q_pi` 为零。该变换保留 pose 内重要性排序，把输入限制在 `[0, 1]`，压缩大构件与细小构件之间的极端差距，并对同一 pose 全部 ppm 权重的整体倍数严格不变。`kappa` 在 v2 中固定，不参与首轮超参数搜索。训练使用 `q_pi`，正式 weighted recall 和报告仍使用原始 ppm 权重；两者必须以不同字段记录，不能把 ppm 误写成真实像素数。

### 4.2 保留 RVL 安全主干

`rvl_strong_v2` 的正样本加权 BCE、非对称 Tversky、集合数量、正负排序和 FP/FN 项保留为安全主干，第一轮不搜索其内部固定系数。这样先验证新增项是否在保留原 RVL 作用的前提下改善效率，避免同时改写所有内部项后无法解释结果。

RVL 的总系数固定为 `1.0`。只有当全部 v2 组合在两个 seed、24 epoch 下仍出现一致的正负分数重叠，才允许建立独立的第二轮内部系数扫描；不能在同一个实验名下临时改变定义。

### 4.3 view-cell 稀有正例尾部保护

使用 train-only dense-subpose sidecar 中的可见命中率区分“多数子位姿可见”和“少量合法子位姿才可见”的正例。正例风险为归一化视觉权重、漏检概率和稀有系数的乘积，并对每个 pose 的最高风险尾部取 CVaR：

```text
r_pi = q_pi * (1 - predicted_visibility_pi) * (1 + lambda_rare * (1 - sqrt(hit_rate_pi)))
L_tail = mean_pose(CVaR_alpha(r_pi for positive instances))
```

该项只使用 train split sidecar，不读取 calibration、validation 或 test 标签。`alpha` 通过 pilot 扫描，正式训练后冻结。

### 4.4 与安全工作区对齐的误请求抑制

训练不直接使用某个 checkpoint 的 calibration 阈值，以避免循环选择和 calibration 泄漏。改为预登记阈值带：

```text
T = {0.01, 0.02, 0.05, 0.10, 0.20}
soft_keep(logit, tau) = sigmoid((logit - logit(tau)) / temperature_logit)
```

对每个阈值 `tau`，实例级 soft keep 按 GLB 做 noisy-OR。只对当前 pose 没有 GT 可见实例的 GLB 计算误请求成本。训练成本使用真实字节的固定对数归一化值 `c_g=log1p(bytes_g)/max_h(log1p(bytes_h))`，评价仍报告未经变换的真实字节：

```text
L_false_request = mean_tau,pose(
  sum_no_demand_glb(soft_request_pg(tau) * c_g)
)
```

这使资源梯度覆盖实际 0.01--0.20 安全阈值区域，不再固定围绕 0.5。资源梯度继续与安全主干做冲突投影；若投影前后出现非有限值或连续抑制安全梯度，则该成员立即失败。

### 4.5 三任务联合目标

完整训练目标为：

```text
L_total = L_RVL
        + lambda_survival * L_survival_censoring
        + lambda_tail * L_tail
        + lambda_false_request * L_false_request
        + lambda_utility * L_visual_utility
        + lambda_download * L_glb_priority
        + lambda_reg * L_regularization
```

`L_survival_censoring` 使用真实遮挡事件和右删失事件训练生存场；`L_visual_utility` 学习实例视觉效用；`L_glb_priority` 用真实 GLB 字节和 GT 可见效用学习预算内排序。下载头不能再只训练不评价。

## 五、固定数据、候选与来源

所有新成员必须使用相同输入：

| 输入 | 固定值或路径 |
|---|---|
| 数据集 | `neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1` |
| 实例数 | 18,831 |
| train / calibration / validation pose | 2,772 / 168 / 213 |
| 候选口径 | train、calibration、validation 均使用各自 split 的原生 66 度后退相机候选，不补入 GT |
| 关系 CSR | `neural_instance_culling/dataset/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/relation_csr` |
| subpose 质量 sidecar | `neural_instance_culling/dataset/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/subpose_quality_sidecar` |
| 初始几何表 | 当前固定 96 维 HKUST 几何特征表及其 schema、shape 和来源说明 |
| 运行元数据 | `hkust-v3/assets/runtimeVisibilityMeta.json` |
| GLB 索引与真实字节 | `hkust-v3/assets/glbIndex.json` 与 `hkust-v3/assets/` |
| 相机口径 | 真实渲染 FOV 60 度；采样、候选和模型输入 FOV 66 度 |
| view-cell | 当前 HKUST 水平半径 2 m、固定高度、固定朝向、back offset 3.4641 m；必须写入 provenance 后才可训练 |

训练前 manifest 必须记录上述路径、schema、shape、pose 顺序、split、候选口径、权重语义、view-cell 形状、FOV、aspect、关系产物说明、代码 commit、随机种子和 GPU。任何一项语义不一致都必须停止，不能用兼容参数继续。

## 六、阶段零：代码修复与契约门

此阶段不启动正式训练。

### 6.1 必须修改的入口

- 训练损失：新增 v2 的归一化权重、尾部风险和阈值带误请求损失；移除旧 `quality_resource` 作为当前正式默认含义；
- 导出器：安全门只读取 aggregate weighted recall；AABB 文件、meta 和前端读取统一为 FP32；
- evaluator：一次推理同时收集可见性、视觉效用和下载头；实例显示指标与 GLB 排序指标分开；
- summarizer：补齐候选/预测 GLB 数量、候选/预测字节、字节削减、预算效用和 paired bootstrap；
- runner：新增 v2 的 `scan12`、`refine24`、统一 `formal80` 和 `evaluate` 模式，写不可覆盖 manifest；`formal80` 一次登记完整组合和四个消融的全部 15 个成员；
- provenance：显式记录 view-cell、FOV、候选口径、权重语义和 train-only sidecar。

### 6.2 必须通过的测试

1. 构造 pose-macro weighted recall 合格但 aggregate 不合格的 checkpoint，导出必须拒绝；反例必须接受；
2. AABB 导出文件必须为 `numInstances * 6 * 4` bytes，meta 为 FP32，PyTorch/前端读取一致；
3. 同一 pose 的全部 `visible_weights` 乘任意正常数后，归一化权重和仅依赖该权重的 v2 损失分项在数值容差内保持不变；
4. soft request 在每个预登记阈值附近连续、单调，且不会回退到固定 0.5 工作点；
5. 资源梯度投影不能降低已登记安全梯度的一阶内积，所有损失和梯度有限；
6. 修改 download logits、保持 visibility logits 不变时，下载 NDCG/预算效用必须变化，而实例可见性指标不变；
7. `candidateGlbBytes` 和 `predictedGlbBytes` 必须进入逐 pose、aggregate 和 bootstrap；
8. 几何单体控制、真实关系和分层置乱控制具有相同输出形状及可比较容量；
9. integrated 与 learned-point 控制只在 view-cell 方差是否生效上不同；
10. train、calibration、validation pose 数和候选集合口径全部匹配，manifest 明确 `testRead=false`；
11. 两 pose 前后向 smoke、FP16 导出 smoke、schema self-test、`py_compile` 和相关 unittest 全部通过。

只有阶段零全部通过，才能生成扫参 manifest。

## 七、阶段一：完整组合快速超参数扫描

### 7.1 固定架构

扫参期间只训练完整组合，不混入结构消融：

- 真实分层关系生成 `4 x 7` 生存系数；
- 16 个可学习联合频率的区域积分查询；
- 96 维几何 + 28 维生存系数固定运行表；
- hidden dim 64、relation hidden dim 64、query basis rank 4；
- 三任务头保持相同结构；
- 每个候选只执行一次查询。

架构宽度在本阶段固定，避免把容量变化与损失贡献混在一起。

### 7.2 扫描空间

| 参数 | 粗扫范围 | 说明 |
|---|---|---|
| 学习率 | `1e-4, 2e-4, 3e-4` | AdamW |
| weight decay | `1e-6, 1e-5, 5e-5` | 防止频谱和关系头过拟合 |
| `lambda_survival` | `0.10, 0.25, 0.40` | 生存事件/删失监督 |
| `lambda_tail` | `0.15, 0.35, 0.60` | 稀有正例质量尾部 |
| `lambda_false_request` | `0.005, 0.015, 0.030, 0.050` | 低阈值带无需求 GLB 成本 |
| `lambda_utility` | `0.08, 0.18, 0.30` | 实例视觉效用头 |
| `lambda_download` | `0.10, 0.20, 0.35` | GLB 下载排序头 |
| logit 温度 | `0.5, 0.75, 1.0` | 阈值带平滑宽度 |
| CVaR tail fraction | `0.005, 0.010, 0.020` | 每 pose 正例风险尾部 |
| rare-positive 系数 | `0.5, 1.0, 2.0` | 少量 subpose 可见实例保护 |

RVL 内部固定系数、网络宽度、关系层级、频率数量、阈值带和真实 GLB 成本来源不参与首轮搜索。

### 7.3 12 epoch 粗扫

- 使用固定 seed `20260801`；
- 生成 16 个确定性筛选配置：4 个手工锚点和 12 个固定 Sobol 组合；设计点在训练前一次性写入 manifest，运行中不得追加；
- 每个成员从头训练 12 epoch、每 epoch 100 step；
- 四张 GPU 同时运行四个独立单卡进程，共四轮；
- 每 4 epoch 使用完整 168-pose calibration 选择自身阈值，并在 213-pose validation replay；
- 粗扫可使用 2,000 次 bootstrap 作早期不确定性诊断，但不能称为正式安全 LCB；
- 保留最多 6 个成员进入复验。

粗扫先剔除存在非有限 loss/梯度、候选摘要变化、split/schema 失配或输出不完整的无效成员。其余数值有效成员全部进入指标排序；没有 calibration 安全候选、validation weighted recall 低于 `0.99`、预测数膨胀或下载字节只因漏检下降，均记为负面诊断，但不能使候选池为空或取消后续长训。按预登记排序保留最多 6 个成员进入复验；若数值有效成员少于 4 个，必须先修复共同代码或训练稳定性问题，再用同一 manifest 重跑，不得临时改变搜索空间。

### 7.4 24 epoch 双种子复验

- 对粗扫保留的 4--6 组配置使用 seed `20260801/20260802`；
- 每组从头训练 24 epoch、每 epoch 120 step；
- 使用完整 calibration 和 10,000 次 aggregate weighted recall 单侧 LCB；
- validation 使用 calibration 冻结阈值，并做 10,000 次 seed/pose 配对 bootstrap；
- 将两个 seed 均满足 validation `aggregate weighted recall > 0.99` 且 LCB `> 0.99` 的配置划入安全候选池，其余数值有效配置划入诊断候选池；
- 若安全候选池非空，先按 useful cull、balanced accuracy、precision、accuracy、平均预测数、GLB 字节和跨 seed 方差做 Pareto 筛选，再依次以平均 useful cull 更高、balanced accuracy 更高、precision 更高、预测 GLB 字节更低、跨 seed 方差更小作确定性决胜；
- 若安全候选池为空，从全部数值有效配置中依次按最差 seed 的 aggregate weighted recall、更高的单侧 LCB、balanced accuracy、useful cull、precision、更低的预测 GLB 字节和更小的跨 seed 方差选择诊断最优配置；不用任意加权总分。

最终必须冻结一组超参数。冻结记录同时写明它来自“安全候选池”还是“诊断候选池”，并保存完整排序依据。即使所有配置均未达到安全门，或均弱于 R0 的效率指标，也继续使用诊断最优配置完成正式 80 epoch 队列；这些结果不能晋级论文主模型，但必须保留为完整失败证据。

### 7.5 阈值健康只作诊断

每个 checkpoint 记录正例 Q05/中位数、负例 Q95/中位数、正负间隔、Brier、ECE、冻结阈值和不同 seed 的阈值波动。阈值接近 0.5 不是安全硬门，也不能通过 bias 或 temperature 后处理把阈值人为移到中间。

若收益仅在阈值低于 `1e-3` 时出现，并伴随预测实例或 GLB 字节显著增加，则标记为分数分布不健康。该标记降低论文晋级资格，但不阻止被选中的最优配置完成长训。

## 八、阶段二：完整组合三种子 80 epoch 长训

### 8.1 启动条件

- 阶段零全部测试通过；
- 24 epoch 双种子复验已按预登记规则从安全候选池或诊断候选池冻结一组数值有效配置；
- 超参数、代码 commit、输入路径与 schema、阈值选择规则、GPU 和输出路径已写入不可覆盖 manifest；
- test 仍未读取，默认前端资产仍未改变。

以上均为契约和可复现性条件，不包含最低 precision、weighted recall、阈值位置或效率要求。只要阶段零契约通过且存在至少一个数值有效扫描配置，就必须启动正式长训。

### 8.2 训练配置

- 变体：`full_hierarchical_integrated_threshold_aligned`；
- seeds：`20260801/20260802/20260803`；
- 每个成员从头训练 80 epoch、每 epoch 150 step；
- AdamW 和 cosine schedule，具体权重使用阶段一冻结值；
- 每 4 epoch 完整 calibration，每 8 epoch 保存不可覆盖 checkpoint，epoch 80 强制保存和评价；
- `best_safe.pt` 只能来自 calibration aggregate weighted recall 和 LCB 均大于 `0.99` 的 checkpoint；每个成员另按预登记诊断规则保存 `best_diagnostic.pt`，但后者不得改名为安全模型或进入默认前端；
- checkpoint 的 calibration 阈值固定后才能 replay validation；
- 正式阶段由四张 GPU 组成统一持久任务队列：第一批在 GPU 0、1、2 分别运行 full 的三个 seed，GPU 3 同时运行第一个消融成员；任一 GPU 完成后立即领取剩余正式成员，直到 15 个任务全部完成；
- 每个 worker 固定一个 `CUDA_VISIBLE_DEVICES`，同一 GPU 同时只运行一个训练进程。不得按任务创建时预分配后无锁复用 GPU，也不得因 full 只有三个 seed 长期闲置第四张卡；
- validation、bootstrap、汇总和导出优先使用 CPU，或在某张卡完成成员后的切换间隙执行，不能长期占用第四张训练卡；
- stdout/stderr、epoch/step、损失分项、梯度范数、学习率、吞吐、ETA、GPU 绑定和 checkpoint 角色全部落盘。

长训不能因中间 epoch 指标较差而静默早停。基础设施中断使用同一 checkpoint、代码 commit 和 manifest 恢复；若出现非有限值、候选/GT 语义错误或 manifest 失配，则保留现场、暂停受影响队列并修复共同契约，随后按同一预登记配置从头重跑受影响成员。不得静默修改损失权重或只重跑表现较差的 seed。

### 8.3 完整组合论文晋级门

三个 seed 必须全部满足：

- validation aggregate weighted recall `> 0.99`；
- 同指标单侧 95% LCB `> 0.99`；
- 不补 GT、候选集合完全一致；
- 相比冻结 R0 安全参考，平均 balanced accuracy、useful cull 或 precision 至少有一项改善，且平均预测实例数/GLB 字节没有反向大幅恶化；
- 图像平均和 p95 miss-pixel 没有稳定恶化；
- 固定神经资产不超过 7 MiB；
- 同候选 CUDA p95 不超过当前强参考的 1.10 倍。

该门只决定完整组合能否成为论文主模型、导出默认资产或进入前端部署。未通过时必须标记为“没有合格安全工作点”或“通过安全门但未通过效率/运行成本门”，并继续完成全部正式消融；不得用晋级失败取消消融训练。

## 九、阶段三：核心创新消融

所有消融使用完整组合冻结的训练超参数、80 epoch、三个相同 seed 和相同 checkpoint/阈值规则。消融不单独重新调参，避免每个成员获得不同的优化预算。它们与 full 一起登记到统一 `formal80` 队列；逻辑上先报告完整组合，再报告消融，但执行时允许并行重叠以充分使用四张 GPU。

| 变体 | 关系表示 | 视点查询 | 损失 | 目的 |
|---|---|---|---|---|
| `full` | 真实分层关系 | 区域积分频谱 | v2 对齐效用损失 | 完整组合 |
| `without_hierarchical_relation` | 参数量误差不超过 1% 的几何单体系数生成器 | 区域积分频谱 | v2 对齐效用损失 | 分层遮挡关系的独立贡献 |
| `shuffled_relation_source` | 方向/深度/度数分层内置乱来源 | 区域积分频谱 | v2 对齐效用损失 | 真实遮挡来源身份是否被使用 |
| `without_viewcell_integration` | 真实分层关系 | 同频率、同容量中心点查询，方差置零 | v2 对齐效用损失 | view-cell 区域积分的独立贡献 |
| `without_threshold_aligned_utility` | 真实分层关系 | 区域积分频谱 | 归一化权重的 RVL 主干，尾部/误请求项关闭 | 新损失相对 RVL 主干的增量贡献 |

几何单体控制使用相同 96 维几何输入和相同 `4 x 7` 输出，通过只含逐实例残差块的离线网络匹配分层关系编码器的可训练参数量，误差不超过 1%；导出表、在线查询和三任务头与 full 完全相同。历史 Fourier117、R0 自由生存场和当前部署模型只作冻结参考，不重新解释为容量匹配消融。真实关系贡献至少需要同时查看 `full - without_hierarchical_relation` 和 `full - shuffled_relation_source`，不能只比较 R2 与 R0。

四张 GPU 以一张卡一个持久 worker 的方式领取独立成员。每个成员输出到独立目录，不覆盖完整组合或历史结果。每个消融无论指标高低都完成三个 seed 的 80 epoch；找不到合格 calibration 安全阈值时，保存 calibration 冻结的非安全诊断工作点并明确标记“没有合格安全工作点”。

## 十、阈值、评价和统计

### 10.1 阈值协议

每个 checkpoint 使用自己的 calibration split 冻结阈值。两阶段阈值扫描先使用固定全局网格，再只在 calibration 上对安全边界相邻区间做 logit-space 局部加密。validation 和 test 不得重新选择阈值。

正式安全工作点必须同时满足：

- calibration aggregate weighted recall `> 0.99`；
- calibration aggregate weighted recall 单侧 95% LCB `> 0.99`。

validation 使用该冻结阈值报告同口径结果。pose recall 和 pose-macro weighted recall只作诊断，不替代 aggregate weighted recall 安全门。

若某成员不存在合格安全工作点，仍在 calibration 上冻结一个非安全诊断工作点：优先选择 aggregate weighted recall 单侧 LCB 最高的阈值，再依次比较 aggregate weighted recall、balanced accuracy、useful cull 和 precision。该工作点只用于绘制完整曲线和分析分类失效，所有 validation 表格必须显式标注 `diagnostic_non_safe`，不得与正式安全工作点混排或用于默认模型选择。

### 10.2 实例分类和剔除

必须同时报告 pose macro 与 aggregate：

- recall、weighted recall、precision、F1、Jaccard；
- instance accuracy、balanced accuracy、specificity；
- useful cull、bad cull、TP/FP/FN/TN；
- 平均预测数、候选数、GT 数、预测/候选、预测/GT。

大量候选可能压低 precision，因此 balanced accuracy 是核心辅助指标；普通 accuracy 也必须报告，但不能由大量 TN 单独证明模型有效。useful cull 必须与 weighted recall、recall 和 bad cull 同表出现。

### 10.3 下载头和资源效率

下载头按与训练相同的 GLB 聚合规则独立评价，不能由 visibility predicted IDs 代替。至少报告：

- GLB priority 的 NDCG@20、NDCG@50、NDCG@100；
- 候选 GLB 数量/字节和预测/预算内 GLB 数量/字节；
- 候选字节削减、达到相同视觉效用所需字节；
- 在候选字节 `5%/10%/20%/40%` 预算下的可见效用召回；
- 固定 `10/25/50 MiB` 首屏预算下的效用召回；
- required-GLB recall、无需求 GLB 字节和首屏队列顺序；
- 下载头、仅按可见性聚合和按效用/字节启发式三者对照。

### 10.4 图像与运行成本

- 使用相同 validation pose 的硬件 Color-ID 评价，报告平均/p95 miss-pixel、wrong-ID 和 extra-pixel；
- CUDA 报告单 pose 和固定候选规模的 p50/p95；
- 导出报告固定表、查询权重、全部神经资产和推理输入字节；
- WebGPU 必须读取 adapter 并通过 NVIDIA Vulkan 硬件门，记录 WebGL renderer、adapter info、Chrome 参数和同窗口 `nvidia-smi/pmon`；
- SwiftShader 只能做数值 smoke，不能写入硬件性能结论；
- 真实移动设备可用后执行 10k 候选 p95 `<50 ms` 协议；设备暂不可用时明确标记未验证。

### 10.5 配对统计

正式 validation 使用 10,000 次 paired bootstrap：外层按 seed 聚类，内层在每个 seed 内对相同 pose 重采样。每个消融相对 full 报告差值、95% 区间、方向和是否跨零。至少覆盖 weighted recall、precision、accuracy、balanced accuracy、F1、useful cull、bad cull、平均预测数、预测 GLB 字节、预算效用和图像指标。

一项创新只有在安全门通过、没有稳定图像恶化，并且 precision 提升至少 0.02、balanced accuracy/useful cull 提升至少 0.01、GLB 字节下降至少 5% 或运行成本明确下降中的至少一项获得不跨零证据，才作为论文正贡献。否则按失败消融如实报告。

## 十一、输出目录与实现后命令

### 11.1 独立输出

```text
neural_instance_culling/model/out/pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2_scan12_20260813/
neural_instance_culling/model/out/pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2_refine24_20260813/
neural_instance_culling/model/out/pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2_formal80_20260813/
neural_instance_culling/benchmark/out/<同名实验>/
```

每个根目录包含不可覆盖 manifest、成员日志、checkpoint、calibration rows、validation replay、schema 校验和汇总。smoke 只放在对应实验根下的 `smoke/`，完成后删除无保留价值的临时产物。

### 11.2 计划实现的唯一 runner

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_hierarchical_survival_threshold_aligned_v2.py \
  scan12 --gpu-ids 0 1 2 3 --dry-run

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_hierarchical_survival_threshold_aligned_v2.py \
  refine24 --gpu-ids 0 1 2 3 --dry-run

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_hierarchical_survival_threshold_aligned_v2.py \
  formal80 --variants full without_hierarchical_relation shuffled_relation_source without_viewcell_integration without_threshold_aligned_utility \
  --seeds 20260801 20260802 20260803 --gpu-ids 0 1 2 3 --dry-run
```

这些命令在阶段零完成前只是固定接口登记，当前不存在可直接执行的 v2 runner。实现后必须先 `--dry-run` 检查正式队列恰为 5 个变体、3 个 seed、15 个成员、4 个独占 GPU worker，并核对输入摘要和 `testRead=false`，再启动训练。

## 十二、完成、停止与清理条件

### 12.1 完成条件

- v2 契约测试全部通过；
- 12 epoch 粗扫与 24 epoch 双种子复验有完整 manifest 和结果报告；
- full 三种子 80 epoch 全部完成，无论其是否通过安全门；
- 四个消融成员各三种子 80 epoch 全部完成，共获得 15 个正式成员的完整训练曲线；
- validation 统计、下载头、图像、CUDA、导出和硬件 WebGPU 评价完成；
- 若存在合格论文主模型，最终变体和阈值冻结后 test 只读取一次；若不存在合格成员，明确记录 test 未读取；
- 结论写入 `docs/evaluation/` 并更新 `docs/README.md`。

### 12.2 唯一允许阻断训练的条件

- split 泄漏、GT 补入、候选摘要变化、AABB/权重 schema 失配或其他使不同成员不可比较的代码/数据契约错误；
- 持续非有限 loss/梯度、损坏 checkpoint 或无法恢复的训练数值错误；
- CUDA、存储或调度基础设施不可用，且无法在相同代码、manifest 和超参数下恢复。

上述条件只暂停受影响成员或整个可比队列，以便修复并按相同预登记配置恢复或重跑；它们不是根据模型效果设置的提前停止规则。weighted recall、LCB、precision、accuracy、balanced accuracy、阈值过低、跨 seed 指标差、资产超预算或运行成本偏高均不能阻止 15 个成员完成，只决定最终失败标签和论文晋级。失败成员不能进入默认 runner、README 主线或前端资产。

## 十三、保护边界

- 不修改当前默认 checkpoint、阈值、前端资产或 Route 文档结论；
- 不覆盖或重命名 2026-08-11、2026-08-12 正式矩阵、2026-08-13 pilot 和 32 epoch 学习曲线；
- 不读取 test 选择模型、epoch、损失权重或阈值；
- 不把低阈值位置本身包装成模型能力，也不把 useful cull 单独作为优劣结论；
- 不把服务器 CUDA 延迟写成浏览器或移动端性能；
- 新代码语义确认后直接替换错误 v2 主线实现，不增加旧字段回退、别名文件或双重默认路径。

## 十四、关联证据

- 架构与 pilot 实施记录：[`pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md`](pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md)
- 32 epoch 学习曲线结果：[`../evaluation/pvs_hierarchical_relation_survival_learning_curve_e32_2026-08-13.md`](../evaluation/pvs_hierarchical_relation_survival_learning_curve_e32_2026-08-13.md)
- 当前优化阶段：[`../current/optimization_restart_2026-08-10.md`](../current/optimization_restart_2026-08-10.md)
- 统一指标：[`../evaluation/unified_pvs_metrics_evaluation.md`](../evaluation/unified_pvs_metrics_evaluation.md)
- 硬件 GPU 规范：[`../current/hardware_gpu_execution_policy.md`](../current/hardware_gpu_execution_policy.md)
