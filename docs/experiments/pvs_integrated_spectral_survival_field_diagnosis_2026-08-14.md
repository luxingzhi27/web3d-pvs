# 区域积分频谱、分层遮挡生存场与安全效用损失综合诊断

- 日期：2026-08-14
- 状态：已完成代码、数据语义和现有结果的只读审计；审计后 v2 `formal80` 已自然完成 15/15，但本文表格仍保留当时审计快照，不构成最终 paired-bootstrap 结论；纠错实施状态见 2026-08-15 v3 实施记录
- 适用实验：`pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2`

## 一、诊断目的与边界

本文汇总当前论文完整组合训练效果不稳定的原因，统一回答以下问题：

- 当前结果是否只是训练轮数不足；
- 分层遮挡关系是否真正进入了逐实例生存场；
- 区域积分频谱是否正确表达了 view-cell 内的相机分布；
- 生存监督是否覆盖了声称使用的四百余万条观察；
- 安全工作点对齐损失是否真正约束了最终低阈值工作区；
- 当前消融能否独立证明三项候选创新的贡献。

审计只读取 train、calibration 和 validation 相关代码与产物，不读取 test，不改变候选集合，不补入 GT，不修改当前默认模型或前端资产。审计期间所有正在运行的训练进程保持运行，没有发送停止、重启或重复后处理命令。

本文把证据分为三类：

- **已确认实现错误**：代码或数据语义可以直接证明实现与数学定义、实验定义不一致；
- **已确认实验设计缺陷**：不会使程序崩溃，但会使消融无法回答原研究问题；
- **待统计验证的性能解释**：现有结果支持该解释，但必须等完整矩阵和配对统计后才能形成正式结论。

### 1.1 代码缺陷与纠错入口总表

下表把后文证据归并到具体实现位置。它用于确定修改顺序，不代表已经修复；新实现、测试和输出必须使用独立的 v3 schema，不能原地改变正在运行的 v2 语义。

| 优先级 | 当前实现位置 | 已确认问题 | 直接影响 | v3 纠错入口 |
|---|---|---|---|---|
| P0 | `dataset/build_viewcell_pose_csr_for_fixed_geo.py`、`model/pose_csr_dataset.py` | 派生 CSR 只把后退候选相机写入 `camera_world`，没有把原始 view-cell 中心作为独立查询字段保留下来 | 候选覆盖位置和可见性查询分布中心相差 3.4641 m | 分开保存 `candidate_camera_world` 与 `query_center_world`，候选只使用前者，ray 与区域矩只使用后者 |
| P0 | `model/common/viewcell_integrated_spectral_query.py` | Fourier 相位使用 cycles 口径，衰减却漏掉 `(2*pi)^2`；同时用启发式九维对角方差替代真实水平圆盘 | 高频衰减、方向相关性和 view-cell 边界响应均与登记分布不一致 | 统一频率单位，由水平圆盘经过 ray-space Jacobian 得到两个低秩轴，并计算圆盘频谱的一、二阶矩包络 |
| P0 | `dataset/build_triangle_depth_layer_evidence.py` | 同一实例部分可见、部分受遮挡时，后层事件会覆盖首层可见证据 | 生存监督把仍应保留的实例整体推向遮挡 | 分别输出右删失可见证据和遮挡事件证据，保留各自像素权重 |
| P0 | `dataset/build_observed_relation_csr.py` | 无容量约束的 connected-components 发生 single-link 渗流 | 局部层几乎全是单例，结构层出现覆盖 15,346 个局部组的巨型分量 | 改为强关系优先、带组容量和空间直径上限的确定性层级 |
| P0 | `model/train_hierarchical_relation_survival_integrated.py` | 每一步用同一组 `linspace` 索引读取 4,096/4,039,231 条观察 | 只直接监督约 11.3% 的实例，四百万条观察名义存在但没有被遍历 | 实现按事件/删失、方向、深度和实例分层的跨 step/epoch 轮换采样器 |
| P0 | `model/train_hierarchical_relation_survival_integrated.py` | 当前 shuffled control 只平移来源 ID，仍保留原相对特征和真实层级 | 负对照没有真正破坏来源-目标遮挡关系 | 使用保持度数和分层分布的双边交换，重新计算相对几何并重建层级 |
| P1 | `model/common/train_observed_relation_csr.py`、`model/common/hierarchical_relation_survival.py` | 每个方向/深度格对大量弱边做普通平均，且生存系数可绕过关系消息 | 少量强遮挡来源被稀释，非零关系贡献不可辨识 | 每格选择真实证据 top-k，加入置信度注意力、关系增量门和 train-only 关系一致性监督 |
| P1 | `dataset/build_observed_relation_csr.py`、`model/common/survival_loss.py` | `d/(d+r)` 使大部分深度压缩到 `[0.94,1]` | 两个 Logistic 分量难以分辨远距离细小实例的遮挡边界 | 改为由 train observation 冻结分位数的半径相对对数深度 |
| P1 | `model/common/viewcell_quality_rvl_loss.py`、训练器梯度组合 | 资源门从首步常开，资源梯度只占安全梯度约 0.2%--0.5%，阈值带未覆盖 0.001--0.005，noisy-OR 随 GLB 实例数饱和 | 新损失主要依赖正例尾部项，难以减少实际工作点的 FP 和无需求字节 | 拆分安全、关系、调度和效率梯度；增加 warmup/安全裕度门、低阈值 hard-negative CVaR 与固定 top-k GLB smooth-max |
| P1 | 当前 formal runner 与 benchmark 后处理 | 训练完成后不会自动执行独立 replay、候选口径核验和 paired bootstrap | 训练退出码为零可能被误当成正式评价完成 | runner 串接 checkpoint/schema 校验、calibration 冻结 validation replay、10,000 次 paired bootstrap 和报告状态机 |

完整修改文件、测试、数据重建和晋级门见后续纠错计划。当前 v2 的结果只用于暴露这些问题，不能作为修复后架构的性能代理。

## 二、当前运行状态与可比性

当前正式输出目录为：

```text
neural_instance_culling/model/out/
  pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2_formal80_20260814_rerun/
```

本次代码和结果审计取证时已经完成 11/15 个成员：

- `full`：3/3；
- `shuffled_relation_source`：3/3；
- `without_hierarchical_relation`：3/3；
- `without_viewcell_integration`：2/3；
- `without_threshold_aligned_utility`：0/3。

在审计取证后，一个 `without_viewcell_integration` seed 和两个 `without_threshold_aligned_utility` seed 已自然结束。文档校验时已完成 14/15，只剩 `without_threshold_aligned_utility_seed20260803` 仍在运行。下表保持 11/15 时的审计快照，不把新结束成员未经独立 replay 的训练器内摘要混入既有均值；损失消融尚未全部完成，因此本文不提前判定新损失相对 RVL 的最终贡献。

已完成成员满足以下可比性条件：

| 项目 | 固定值 |
|---|---|
| train pose | 2,772 |
| calibration pose | 168 |
| validation pose | 213 |
| 实例数 | 18,831 |
| 候选集合 | train、calibration、validation 使用各自 split 的原生 66 度后退相机候选 |
| 候选语义 | 原生 66 度后退相机 AABB 候选，不补入 GT |
| 阈值来源 | 每个 checkpoint 自身 calibration split |
| test | 未读取 |

不同变体的候选、GT、pose 顺序和固定几何表一致；分层关系模型和容量匹配几何控制的总可训练参数分别为 182,073 和 182,112，参数量差异不是当前结果差异的主要解释。

## 三、现有结果的正确口径

### 3.1 原诊断表格的口径错误

旧版本文第 36 行把表中数据称为“validation 汇总”，但表中数值实际是三个 seed 的 calibration 安全工作点平均。例如 Full 的 aggregate precision 为：

```text
(0.09760 + 0.17915 + 0.16237) / 3 = 0.1464
```

该数值不能写成正式 validation 结果。以下全部改用各 checkpoint 在 calibration 冻结阈值后，对 213 个 validation pose 的 replay 结果。

### 3.2 Full 的 validation 结果

| seed | 冻结阈值 | aggregate precision | accuracy | balanced accuracy | weighted recall | weighted recall LCB | useful cull | bad cull | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 0.020 | 0.1633 | 0.7782 | 0.8709 | 0.99227 | 0.98808 | 0.7351 | 0.00121 | 1364.5 |
| 20260802 | 0.005 | 0.2654 | 0.8930 | 0.8456 | 0.99106 | 0.98824 | 0.8624 | 0.00817 | 657.2 |
| 20260803 | 0.005 | 0.2721 | 0.8995 | 0.8367 | 0.99047 | 0.98751 | 0.8610 | 0.01125 | 673.3 |
| 三 seed 平均 | - | 0.2336 | 0.8569 | 0.8511 | 0.99127 | 0.98794 | 0.8195 | 0.00688 | 898.3 |

三个 Full seed 的 validation weighted recall LCB 均低于 0.99，所以当前 Full 没有通过正式 validation 安全门，不能进入默认模型或论文主结果。

### 3.3 已完成结构对照

| 变体 | seed 数 | aggregate precision | accuracy | balanced accuracy | weighted recall | weighted recall LCB | useful cull | bad cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 3 | 0.2336 | 0.8569 | 0.8511 | 0.99127 | 0.98794 | 0.8195 | 0.00688 | 898.3 |
| Shuffled relation source | 3 | 0.2342 | 0.8603 | 0.8486 | 0.99107 | 0.98764 | 0.8233 | 0.00727 | 876.4 |
| Without hierarchical relation | 3 | 0.2510 | 0.8765 | 0.8378 | 0.99301 | 0.99030 | 0.8413 | 0.00906 | 774.2 |
| Without view-cell integration | 2 | 0.1749 | 0.8012 | 0.8340 | 0.99723 | 0.99608 | 0.7627 | 0.00576 | 1198.1 |

这里的 LCB 是 seed 结果的描述性平均，安全资格仍须逐 seed 判断。无分层关系只有 seed 20260802 的 validation LCB 高于 0.99；不能因为三 seed LCB 平均略高于 0.99 就声称该变体稳定通过安全门。

区域积分在已完成的两个同 seed 对照中提高 precision 和 useful cull，但降低 weighted recall 及其 LCB。当前证据更接近“区域积分用安全尾部换取剔除量”，不能描述为全面改善。

## 四、已确认的积分频谱实现错误

### 4.1 Fourier 单位不一致

当前实现为：

```python
phase = 2 * pi * (center @ frequencies.T)
quadratic = einsum(frequencies, variance, frequencies)
attenuation = exp(-0.5 * quadratic)
```

若 `frequency` 使用 cycles 单位，严格高斯 Fourier 矩应为：

\[
E[e^{i2\pi f^T x}]
=
e^{i2\pi f^T\mu}
e^{-\frac12(2\pi)^2f^T\Sigma f}.
\]

因此当前衰减缺少 `(2*pi)^2`，相差约 39.48 倍。若把频率解释为 radians，当前相位又不应乘 `2*pi`。这是确定的数学单位错误。

已训练 Full checkpoint 的高频范数仍约为 4.18、5.00、5.77 和 6.85，没有整体缩小到可同时补偿相位与衰减的尺度。可学习频率能够部分适配错误公式，但不能同时保持正确相位和正确不确定性衰减。

### 4.2 积分中心不是 GT view-cell 中心

原始 view-cell GT 是在 view-cell 中心周围半径 2 m 的水平圆盘内，对同朝向 subpose 的可见实例取并集。CSR 构建时保存的 `camera_world` 却是：

```text
viewcell_center - forward * 3.4641016 m
```

模型直接把这个后退相机位置作为九维 ray-space 均值，再叠加 2 m 方差。实际积分均值与 GT subpose 分布中心沿相机前向相差 3.4641 m。候选相机与 view-cell 查询中心在实现中被混成了同一个位置。

### 4.3 方差形状与真实采样区域不一致

真实采样区域是世界水平面上的圆盘，竖直位移为零。当前实现使用基于 `radius / distance` 的近似角度方差，并只保留九维对角项：

- 没有表达世界水平圆盘经过 ray-space Jacobian 后的非对角相关性；
- 对俯视、仰视和倾斜相机引入了与真实水平位移不一致的方向扰动；
- 没有使用 dataset 中实际登记的 view-cell 形状和轴向；
- 没有验证解析矩与真实 subpose Monte Carlo 矩的一致性。

因此当前实现不能称为对真实 view-cell 分布的正确解析积分，只能称为围绕错误中心的启发式对角不确定性编码。

### 4.4 平均矩与保守并集目标不等价

即使修正均值和协方差，当前网络计算的仍是：

\[
g(E[\phi(x)]),
\]

而 view-cell PVS 目标更接近：

\[
\max_{p\in\text{viewcell}}V_i(p).
\]

频谱均值会弱化只在少数边界位置可见的实例。当前虽额外输出一个与衰减相关的 unresolved energy，但它没有区分正弦和余弦的相位相关方差，也没有直接学习并集上包络。区域积分成员降低安全尾部、迫使阈值下降，与这一结构性错位一致。

## 五、已确认的分层遮挡关系问题

### 5.1 connected-components 层级发生 single-link 渗流

当前局部组和结构组都由超过置信度阈值的边执行无容量约束 union-find 得到。实际关系图统计为：

| 层级 | 数量与分布 |
|---|---|
| 实例 | 18,831 |
| 局部组 | 17,954 |
| 单例局部组 | 17,952 |
| 非单例局部组 | 2 个，大小为 877 和 2 |
| 结构组 | 2,592 |
| 单例结构组 | 2,575 |
| 最大结构组 | 含 15,346 个局部组 |

局部层几乎退化为逐实例映射，结构层则通过少量桥接边形成巨型全局分量。编码器在巨型分量内做 mean/max 后，大量实例收到近似相同的全局状态，方向和深度相关的局部遮挡关系被冲淡。

### 5.2 全量弱边的普通平均稀释强遮挡证据

关系 CSR 包含 1,211,555 条边，置信度统计为：

| 统计量 | 值 |
|---|---:|
| 平均置信度 | 0.00428 |
| 中位数 | 0.00117 |
| p90 | 0.01137 |
| p99 | 0.04123 |
| `confidence >= 0.01` | 137,799，约 11.37% |
| `confidence >= 0.08` | 2,951，约 0.24% |

18.36% 的目标/方向/深度格有至少一条边；在这些活跃格内，边数中位数为 2、p95 为 41、p99 为 128，最大为 2,838。当前 `segment_mean` 对全部来源等数量平均，没有 top-k、显式置信度归一化或来源注意力。即使 MLP 学会把弱边输出压低，分母仍包含全部弱边，强遮挡来源会被数量稀释。

### 5.3 Full 可以绕过真实关系边

Full 的系数路径包含：

```text
96 维几何
  -> node projection
  -> 实例/局部/结构 pooling
  -> coefficient head
  -> 28 维生存系数
```

关系消息不是生成系数的必要条件。模型可以把方向关系消息学成近零，继续依靠几何和 pooling 生成完整生存场。容量匹配几何控制拥有近乎相同参数量且部分 seed 更好，说明当前 Full 没有形成必须利用真实遮挡边的可辨识路径。

### 5.4 当前 shuffled control 没有打乱完整关系

当前置乱只把 `source_ids` 整体平移，仍保留：

- 原 target；
- 原 direction 和 depth shell；
- 原 relative center、depth gap、pixel support 和 confidence；
- 由真实原图生成的 local/structural group IDs。

因此该对照只测试“来源实例的几何身份”，没有破坏完整来源-目标关系和图层级。它与 Full 接近，不能证明全部遮挡关系未被使用，也不能作为严格的关系负对照。

## 六、已确认的生存监督问题

### 6.1 四百万条观察每步只使用同一批 4,096 条

关系产物包含 4,039,231 条生存观察，覆盖 18,197 个实例。训练端每一步使用固定 `linspace` 索引选择 4,096 条，索引不随 epoch、step 或 seed 轮换。

固定样本的实际覆盖为：

| 项目 | 值 |
|---|---:|
| 样本数 | 4,096，约占全部观察 0.10% |
| 覆盖实例 | 2,133，约占全部实例 11.3% |
| 事件率 | 91.80% |
| 四个水平锚点 | 3,885 条 |
| 八个上/下倾锚点 | 211 条 |

同一批观察在 12,000 个正式训练 step 中反复出现。共享编码器仍可从最终可见性损失获得梯度，但不能据此声称全部 4.04M 生存观察参与了训练。

### 6.2 部分可见和部分遮挡被压成单一事件标签

对于同一实例，如果它在第一层部分像素可见，同时在其他像素作为后层实例被遮挡，构建器把整条观察标为 `event=1`。这把“仍有可见表面”压成纯遮挡事件。

view-cell GT 使用可见并集：只要存在合法 subpose 或像素可见，实例就必须保留。91.6% 的总体事件率会把生存场推向高遮挡概率，与最终可见并集损失产生系统性梯度冲突。正确语义应保留可见像素对应的右删失证据和遮挡像素对应的事件证据，而不是二选一覆盖。

### 6.3 深度坐标严重挤压

当前使用：

\[
\rho=\frac{d}{d+r_i},
\]

其中 `d` 是相机到实例中心距离，`r_i` 是实例半径。全部 observation 的分布为：

```text
min=0.0066, p05=0.6265, median=0.9395,
p95=0.9976, p99=0.9990, max=1.0
```

绝大多数远距离细小构件集中在 `[0.94, 1.0]`。两个 Logistic 分量需要在极窄区间内表达大量不同深度遮挡，数值分辨率和跨实例可比性都不足。

## 七、安全工作点对齐损失的实现问题

### 7.1 资源门实际上从第一步常开

正式命令没有设置额外 warmup 或正的安全门阈值，因此：

```text
resource_warmup_steps = 0
resource_gate_quality = 0
```

所有已完成 threshold-aligned 成员的 `resourceGateRate` 均为 1.0。当前资源项不是“获得安全裕度后再优化效率”，而是从第一步就与安全目标竞争。

### 7.2 资源梯度过弱，限幅不是主因

Full 三 seed 的训练平均量级为：

| 分项 | 典型范围 |
|---|---:|
| RVL | 1.57--1.66 |
| 加权正例尾部损失 | 0.21--0.23 |
| 加权生存损失 | 0.048--0.054 |
| 加权资源损失 | 0.00499--0.00534 |
| 资源梯度投影率 | 43.4%--47.9% |
| 资源梯度限幅率 | 约 0% |

资源梯度投影后范数约为 0.040--0.048，当前所谓 safety gradient 范数约为 11.5--18.0，只有约 0.2%--0.5%。资源分支几乎没有足够梯度改变大量负样本；当前结果差不能归因于资源梯度被 0.25 比例上限频繁截断。

### 7.3 被保护的“安全梯度”混入了非安全任务

当前投影中的 `safety_objective` 不仅包含 RVL 和正例尾部，还包含：

- 生存似然；
- 视觉效用头；
- 下载优先级头；
- 正则项。

资源梯度被投影到这个复合目标的正交补，而不是只对“保持重要可见实例”的安全梯度做保护。复合梯度很大，进一步掩盖了资源项是否真正与可见性安全冲突。

### 7.4 训练阈值带没有覆盖最终低阈值尾部

资源项固定使用：

```text
{0.01, 0.02, 0.05, 0.10, 0.20}
```

而当前正式成员的冻结阈值多次落在 0.001、0.002 或 0.005。Full 的两个 seed 使用 0.005。资源损失没有约束这些真正决定安全工作的低分尾部，因此“threshold aligned”只对部分成员成立。

### 7.5 noisy-OR 与大 GLB 的实例数量耦合

当前 GLB 请求概率使用所有实例 soft keep 的 noisy-OR。HKUST 的 GLB 实例数中位数为 1，但最大 GLB 含 5,610 个实例。大量单实例概率即使都低于阈值，noisy-OR 也会随实例数量快速饱和到 1，并使单个实例梯度接近零。这与前端真实的“任一实例分数超过阈值即请求”规则近似不佳，也使资源损失对超大实例化 GLB 尤其无效。

### 7.6 当前损失消融混合了两个新增作用

`without_threshold_aligned_utility` 同时关闭：

- 约 0.21--0.23 量级的正例尾部保护；
- 约 0.005 量级的 GLB 误请求抑制。

正式消融可以评价“整个新损失包相对 RVL”的总贡献，但不能单独说明收益来自稀有正例保护还是资源抑制。内部快速诊断至少需要分别记录两项梯度与工作区行为。

## 八、训练与正式评价链路问题

### 8.1 refine 阶段没有跨两个 seed 的合格安全配置

`frozen_config.json` 明确记录：

```text
safeConfigurationCount = 0
```

当前 frozen 配置是按照预登记规则从非安全诊断候选中选出的最优配置。按既定要求继续完成 80 epoch 是正确的，因为安全门只决定论文晋级，不决定是否长训；但结果不稳定并不是进入 formal80 后才出现。

### 8.2 最佳 checkpoint 高度依赖早期偶然工作点

Full 三个 seed 的最佳安全 checkpoint 分别来自 epoch 68、8 和 8，阈值为 0.02、0.005 和 0.005。两个“80 epoch 模型”实际导出的最佳权重来自第 8 epoch。长训后期没有稳定超过早期 calibration useful-cull 工作点，说明当前优化和空间泛化具有明显 seed 方差。

### 8.3 formal pipeline 没有自动接上正式评价

当前 shell pipeline 在训练完成后只验证 15 个成员和 loss lineage，然后结束。独立 evaluator、完整 validation replay、10,000 次按 seed/pose 配对 bootstrap 和路线判定脚本已经存在，但没有被 pipeline 自动调用。

因此训练器内的 `validationAtCalibration` 可以用于当前诊断，却不能代替全矩阵候选口径复核、逐 pose 配对比较和正式报告。15 个成员完成后必须显式执行评价阶段，不能因为训练 pipeline 退出码为零就宣称正式实验完成。

## 九、原因优先级

当前性能问题按优先级归纳如下。

### P0：必须先修复，否则继续训练不能回答研究问题

1. Fourier 相位与衰减的 `2*pi` 单位错误；
2. 后退候选相机被错误用作 view-cell 积分中心；
3. 水平圆盘被启发式对角方差替代；
4. connected-components 形成“几乎全单例 + 一个巨型结构组”的退化层级；
5. 生存观察固定为同一 4,096 条；
6. 部分可见/部分遮挡被合并成纯遮挡事件；
7. shuffled control 保留真实层级和大部分真实边属性。

### P1：会降低三项创新的实际收益

1. 全量弱边普通平均，缺少强来源选择和置信度归一化；
2. `rho` 在远距离实例上严重饱和；
3. 频谱均值缺少针对 PVS 并集的相位相关边界不确定性；
4. 资源梯度量级过小且投影参照混入非安全任务；
5. 固定阈值带未覆盖实际 0.001--0.005 工作点；
6. noisy-OR 对 GLB 实例数量敏感并在大组中饱和。

### P2：正式结论和工程交付仍缺少的证据

1. 15 个成员全部完成后的独立 validation replay；
2. 10,000 次按 seed 聚类、seed 内 pose 重采样的 paired bootstrap；
3. 三个下载头与可见性聚合基线的独立 GLB 排序比较；
4. 硬件 Color-ID 图像指标；
5. 新区域积分实现的 NVIDIA WebGPU adapter 硬件延迟和移动端预算。

## 十、当前可以和不可以得出的结论

当前可以确认：

- Full 没有通过 validation weighted recall LCB 安全门；
- 当前关系层级和生存采样存在足以掩盖真实关系贡献的实现问题；
- 当前区域积分公式和分布中心均与登记定义不一致；
- 当前资源损失实际梯度很弱，正例尾部项才是新增损失的主要训练信号；
- 当前结果不能只用输入维度从 117 降到 57 解释；
- 简单增加训练 epoch 不会自动修复数学、图结构和监督语义错误。

当前不能声称：

- 分层遮挡关系本身没有价值；
- 正确的 view-cell 区域频谱一定无效；
- 新损失整体弱于或强于 RVL，因为 RVL 对照尚未全部完成独立 replay 和正式配对汇总；
- 无分层关系稳定优于 Full，因为逐 seed 方向不一致且正式配对统计尚未执行；
- 新模型具有浏览器性能优势，因为新 schema 尚未接入默认前端，也没有通过硬件 WebGPU 性能门。

## 十一、对三项论文创新的影响

三项创新方向仍然成立，但当前实现不能继续原样长训：

1. **分层遮挡关系生存网络**应保留“真实遮挡边离线生成逐实例生存场”的核心叙事，同时改成有界层级、强证据注意力和覆盖充分的生存监督；
2. **视点区域积分频谱查询**应保留“前端一次解析查询、不展开 subpose”的轻量目标，同时修复 query center、真实水平区域矩和 Fourier 单位，并增加面向并集边界的矩包络；
3. **安全工作区资源效用损失**应保留 RVL 的高召回作用，同时把安全裕度、阈值局部负样本、GLB 成本和梯度保护分开建模，避免当前资源项常开却几乎无梯度。

具体代码修改、数据重建、快速验证、正式训练和前端验收方案见 [`pvs_bounded_relation_survival_moment_envelope_safety_reserve_plan_2026-08-14.md`](pvs_bounded_relation_survival_moment_envelope_safety_reserve_plan_2026-08-14.md)。在该计划完成并通过 validation 安全门前，当前默认 checkpoint、阈值和前端资产保持不变。
