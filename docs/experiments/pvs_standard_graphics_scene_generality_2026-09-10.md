# 标准图形学场景 PVS 正式实验协议

日期：2026-09-10；2026-09-15 按 Connected-SAH 主线重写

状态：执行中。本文档只描述当前正式协议与当前正式产物。

## 1. 实验目的

Sponza、Viking Village 和 Big City 是论文正式测试场景，不只是“证明不依赖 BIM”的附加
例子。三个场景用于回答：

1. Full V4 在普通图形场景上能否满足画面安全门并有效剔除遮挡单位；
2. 相同单位、候选、GT 和 split 下，Full 是否优于 Keep-All 与 AABB + Ray MLP；
3. 相比 Geometry-shell HZB，Full 能否以更小的启动资产和更低的区域查询成本得到有用 PVS；
4. 模型延迟随候选单位数的变化是否适合桌面和移动 WebGPU/WASM。

标准场景不要求 GLB 原型复用。每个输出 renderable unit 对应一个 component ID 和一个
resource ID：

```text
renderable unit -> componentGlobalId -> resource ID -> 96D fixed geometry row
```

模型是场景专属 visibility codec，不宣称跨场景 zero-shot。HKUST/IFCBench 仍承担真实 GLB
复用与 progressive streaming 主实验；标准场景主要报告剔除、图像、启动资产和运行成本。

## 2. 场景与来源

| 场景 | 类型 | 固定来源 | 许可边界 |
|---|---|---|---|
| Sponza | 室内多层遮挡 | Khronos glTF Sample Assets 的 Sponza glTF | 仅本地实验，不分发源纹理和派生资产 |
| Viking Village | 大型室外村落 | `standard_graphics_sources/viking_village/VikingVillage.glb` | 论文报告统计和结果，不公开分发受限资产 |
| Big City | 大型室外城市 | NeuralPVS 官方资产中的 `BigCity/scene.gltf` | 本地实验；不据此声明派生资产可再分发 |

每个场景必须保留 `scene_source_audit.json`，记录实际源版本、节点、primitive、三角形、材质、
坐标单位、场景范围和许可。NeuralPVS 论文中的 primitive 概数不能替代本项目源审计。

## 3. Renderable-unit 协议

正式转换名称为 `standard_graphics_connected_sah_128k_v1`，partition schema 为
`connected-sah-pack-v2`。单位化只读取 scene graph、几何、材质和压缩字节，不读取标签、
模型分数或 HZB 结果。

1. 每个源 node、primitive、材质和 alpha mode 独立处理；不跨 primitive 或材质打包。
2. 先按共享顶点求三角形连通分量。
3. 小连通分量作为原子项，用确定性 16-bin SAH 按空间紧凑性打包到接近 `128 KiB`。
4. 超大连通分量按三角形质心做确定性 SAH 递归拆分。
5. 输出经固定 meshopt 参数编码；实际编码几何超过 `128 KiB` 时继续拆分。
6. 单三角形是唯一不可再分的超限例外，必须在 manifest 中登记。
7. 每个单位内部重索引但不减面；按固定源顺序分配连续 ID。
8. manifest 记录 `componentCount`、`sourceComponentCount`、`partitionReason`、
   `oversizedComponentSplit`、AABB、三角形数和编码字节。

正式统计必须同时报告单位数、源连通分量数、每单位分量数、超大分量拆分数、三角形分布、
编码字节 p50/p95/max 和 AABB 尺度。主结果按 unit 统计，并补 triangle/byte cull，避免单位
粒度本身抬高剔除率。

材质政策固定为：OPAQUE 可写关系和 HZB 深度；MASK 在 reference Color-ID 中保留 alpha
test，但不作为无纹理外壳的确定不透明遮挡物；BLEND 不进入静态遮挡剔除；未知材质按非
遮挡物处理。HZB 关闭背面剔除以保持保守。

## 4. 相机、View-cell 与 Split

三个场景使用同一空间契约：

- 显示相机 FOV 为 `60 degrees`；模型/候选相机 FOV 为 `66 degrees`；aspect 使用实际值；
- view-cell 是世界 XZ 水平圆盘，半径 `r=0.75 m`；
- 每 cell 使用 32 个确定性面积均匀 subpose，方向固定，只改变合法位置；
- 表面 clearance 为 `0.80 m`；
- 后退距离为 `r/tan(30 degrees)=1.299038 m`；
- GT 为全部成功 subpose 可见单位的并集，`visible_weight` 为单位跨 subpose 的最大屏幕覆盖；
- 浏览器运行时只做一次后退相机查询，不展开 32 个 subpose。

高空或不同楼层通过不同 Y 的 view-cell 中心表达，不在单个圆盘中加入垂直扰动。候选必须
覆盖整个 cell，正式数据要求 `visible_ids` 是 `candidate_ids` 的子集。

Split 按物理中心分组，使用固定 seed 做确定性随机划分。同一中心的全部方向和 subpose
不得跨 split。顺序统一为 train/calibration/validation/test；calibration 只选阈值，
validation 只选成员，test 在全部方法冻结后读取一次。

## 5. 当前正式数据

| 场景 | 源连通分量 | 单位 | 96D 表 | 成功 subpose | Train / Cal / Val / Test | 平均候选 | 平均 GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sponza | 8,091 | 129 | `[129,96] FP16` | 213,504 | 4800 / 528 / 672 / 672 | 44.1667 | 7.9519 |
| Viking Village | 141,456 | 1,763 | `[1763,96] FP16` | 86,784 | 1944 / 216 / 276 / 276 | 544.4108 | 125.9956 |
| Big City | 652,925 | 2,861 | `[2861,96] FP16` | 514,560 | 11580 / 1284 / 1608 / 1608 | 748.5629 | 155.0919 |

三场景均无编码超限单位，`candidateMissVisible=0`。Color-ID 使用 NVIDIA Vulkan/ANGLE
硬件路径并保存 GPU evidence。固定几何特征由同一个冻结几何编码器从每单位 1024 个归一化
表面点导出；几何编码器不在目标场景上更新。

训练关系只使用 train split。每个稠密深度 shard 完成后立即压缩成稀疏遮挡边、生存观察和
深度矩，再删除 ID/depth payload；不构建全场景稠密缓存。当前 K=8 关系结果为：

| 场景 | 深度 shard | Train render rows | 关系边 | 生存观察 | 未覆盖 train pose |
|---|---:|---:|---:|---:|---:|
| Sponza | 32 | 43,200 | 23,280 | 1,739,324 | 0 |
| Viking Village | 16 | 17,496 | 128,997 | 4,922,117 | 0 |
| Big City | 64 | 104,220 | 438,877 | 62,247,879 | 0 |

## 6. Full 训练协议

三个场景使用同一 V4 架构和配置，从随机初始化训练：

| 项目 | 固定值 |
|---|---:|
| Epoch x step | `40 x 900` |
| Learning rate | `5e-5` |
| Poses per batch | `8` |
| Pose sampling | `ambiguity_balanced` |
| Hard pose fraction / quantile | `0.35 / 0.65` |
| Survival rank / relation K | `4 / 8` |
| Weighted-recall target | `0.99` |
| Recall guard final weight | `0.30` |
| Tail separation final weight | `0.30` |
| Positive tail / negative frontier | `0.01 / 0.02` |
| Positive importance floor / power | `0.25 / 0.5` |
| Calibration bootstrap | `10,000` |

Epoch 1-4 关闭尾部分离和召回保护；epoch 5-12 逐步加入尾部分离；epoch 13-40 再逐步加入
召回保护。固定 `p=0.5` 不作为安全门，也不使用 bias/temperature 移动阈值伪装效果。

每个 checkpoint 使用 calibration 的实际 float32 score change-points，预测规则为
`score >= threshold`。选择满足 weighted recall 及单侧 95% bootstrap LCB 都严格大于
`0.99` 的最高安全阈值；validation 只重放该阈值。安全成员依次比较 Useful Cull、CNOR、
balanced accuracy、Occlusion Recall、precision 和平均保留数。

## 7. Scene-first 调度

Full 固定按 seed round 执行：

1. Sponza、Viking Village、Big City 的 seed20260801 同时训练；
2. 首轮稳定后，如果单卡显存和利用率仍有余量，同时启动三个场景 seed20260802；
3. 再按同一资源门决定是否同时启动三个场景 seed20260803；
4. 三种子 validation 完成后，冻结每场景唯一 Full 成员与阈值。

不能只给一个场景提前启动额外种子；增加并发时必须整轮覆盖三个场景。GPU 参数表示并发
slot，允许重复 GPU ID，但启动前必须按实际显存、利用率和吞吐判断。同卡并发不能与正式
Color-ID、HZB、WebGPU timing 同时运行。GPU0 上的外部任务不得终止或抢占。

正式 runner：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/run_standard_graphics_connected_sah_full.py \
  train --seeds 20260802 --gpu-slots 1 2 3
```

## 8. 基线与执行顺序

所有方法使用相同 Connected-SAH units、`66 degrees` candidate CSR、区域 GT 和 split。

| 方法 | 作用 |
|---|---|
| Keep-All | 绝对安全、零剔除参照 |
| AABB + Ray MLP | 检验包围盒、投影和相机射线特征是否足够 |
| Geometry-shell HZB | 几何外壳已经预下载后的现代 GPU 遮挡基线 |
| Full V4 | 本文紧凑场景专属 PVS 表示 |

执行阶段门固定为：

```text
Full 三场景三种子并冻结
  -> AABB MLP 学习率扫描和三种子
  -> lossless/equal-asset HZB calibration、test、timing
  -> Full/AABB/HZB 的一次 frozen test、图像和端侧 runtime
```

AABB 使用与 Full 相同的训练步数、三种子、安全校准和候选，不扩充到 96D 输入；HKUST 的
核心消融已隔离 Full 各模块贡献，而 AABB 基线的定义就是只使用几何包围盒与视角信息。

HZB 唯一算法为：确定不透明外壳深度光栅化、max-depth pyramid、候选 projected AABB 的
保守测试。仅当 `HZBMax + bias < candidateNear` 时剔除；近裁剪面、非有限投影、相机位于
AABB 内和其他不确定情况全部保留。

- Point60：只测试当前真实相机点，是单点在线遮挡能力参照；
- Region66：对登记 subpose 做 HZB 并取可见并集；多次查询的总耗时全部计入；
- Lossless shell：保留全部确定不透明 Connected-SAH 外壳单位；
- Equal-asset shell：以冻结神经运行资产字节为上限，从同一输出单位按 train-only 投影收益/
  压缩字节选择完整遮挡单位。它是预算敏感性基线，不宣称保守或最优简化。

论文只声称实现现代批量 Geometry-shell HZB，不冒称完整复现 HROC。相关工作包括
Hierarchical Z-Buffer Visibility、HROC 的 GPU 批组织，以及基于平面截面的遮挡物简化。

## 9. 指标

| 类别 | 必须报告 |
|---|---|
| 排序 | pose-macro PR-AUC、pose prevalence、AP lift；aggregate AP 作为辅助 |
| 画面安全 | Visible Recall、False Occlusion Rate、weighted recall、WR 单侧 95% LCB |
| 遮挡能力 | Occlusion Recall、CNOR、Useful Cull、Bad Cull |
| 分类诊断 | Precision、instance accuracy、balanced accuracy、平均候选/GT/保留数 |
| 图像 | PER、miss-pixel、wrong-ID、extra-pixel 的 aggregate/mean/p95 |
| 系统成本 | 启动传输、解码内存、纯查询 p50/p95、候选规模拟合、冷启动 |

其中：

```text
Occlusion Recall = TN / (TN + FP) = specificity
False Occlusion Rate = FN / (TP + FN) = 1 - Visible Recall
Useful Cull = TN / candidate
Bad Cull = FN / candidate
CNOR = sum_p(TN_p / C_p) / sum_p((TN_p + FP_p) / C_p)
```

CNOR 先按每个 pose 的候选数归一化剔除机会，避免宏平均被极少负样本 pose 放大，也避免
aggregate 被超大候选 pose 主导。它不替代 weighted-recall 安全门或 Useful Cull。

## 10. 论文表图

`Table G1` 报告场景范围、源三角形、源连通分量、单位数、单位字节分布、split、候选和 GT。

`Table G2` 报告每场景 Keep-All/AABB/HZB/Full 的 pose AP、prevalence、Visible Recall、
WR/LCB、Occlusion Recall、CNOR、Useful/Bad Cull 和平均保留数。

`Table G3` 报告神经资产、equal-asset shell 和 lossless shell 的传输字节、解码内存、
Point/Region 查询次数与 p50/p95、冻结安全点 Useful Cull 和图像 miss p95。

核心图包括：

1. 启动资产大小（对数轴）与安全门下 Useful Cull 的 Pareto；
2. 候选单位数与桌面/移动端延迟；
3. 每场景 median、p95、worst miss 的 Reference/Prediction/Difference；
4. 若时间允许，在一个代表场景补 `64/128/256 KiB` 单位粒度敏感性。

标准场景的一单位一资源 streaming 若补做，只单独报告 score ranking 的 units/bytes-to-
visible-weight coverage，不与 HKUST/IFCBench 的资源复用主表混合。

## 11. 当前状态与下一步

截至 2026-09-15：

- 三场景转换、固定几何表、硬件 Color-ID、Pose CSR、深度分片和 K=8 关系均完成；
- 三场景 preflight 与 seed20260801 smoke 均通过，且没有读取 test；
- seed20260801、seed20260802 和 seed20260803 均已按 Sponza/Viking/Big City 完整三场景
  round 启动；每张训练卡并发三个种子；
- 三任务同卡后 GPU 1/2/3 显存约 `1.6/3.3/7.8 GiB`，利用率约 `99-100%`，功耗约
  `123/131/130 W`；后续不再增加训练任务，只监控总吞吐、数值和首次 validation；
- AABB、HZB、标准场景 test、图像和 runtime 尚未启动，等待 Full 冻结；
- HKUST/IFCBench checkpoint 不重训，已完成精确 calibration/validation 重校准；新阈值下
  的 test、图像、streaming 和前端资产尚未重放。

训练监控必须检查进程存活、epoch/step、loss 数值有限、step/s、ETA、显存和 GPU 利用率。
中间 validation 只用于健康诊断，不提前取消登记的完整矩阵。

## 12. Big City 剔除能力专项优化

日期：2026-09-15。

Big City 原 `128 KiB / K=8` 三种子长训在 epoch 20-22 主动停止，只保留为训练诊断，
不进入论文正式成员池。停止时三个种子均未通过 validation 的 `weighted recall LCB > 0.99`
安全门，CNOR 为 `0.2101/0.2232/0.2958`。该结果不能继续消耗约 14 小时/种子的剩余训练时间，
因为数据和训练入口存在以下结构性瓶颈：

1. Big City 的 652,925 个源连通片被打包为 2,861 个单位，每单位连通片数均值 228.2、
   p50 227、p95 476；一个 1024 点、96 维几何表征需要概括过多离散碎片；
2. `K=8` 关系单元有 74.0% 发生截断，关系质量 q01 只有 0.5196；
3. 训练集约 22.6% view-cell 没有 GT 可见单位，其中候选非空的纯负例 pose 未进入旧
   `ambiguity_balanced` 采样，模型没有直接学习这批重要剔除机会；
4. seed02 epoch20 的正样本 q01 为 0.464973、负样本 q99 为 0.476780，安全阈值落在重叠
   尾部内，说明低 CNOR 来自分数不可分，而不是候选负例不足。Big City 平均约 736 个候选、
   160 个 GT，可剔除机会约占 78%。

### 12.1 第一阶段：固定 128 KiB 单元上的训练专项扫描

实验名固定为 `pvs_v4_bigcity_connected_sah_occlusion_opportunity_v1`。只读取
train/calibration/validation，不读取 test。每个 pilot 从随机初始化训练 `8 x 450`，使用
相同 V4 架构和精确 float32 calibration：

| 配置 | 关系 K | 学习率 | 纯负例 pose | 困难边界 pose | 普通可见 pose |
|---|---:|---:|---:|---:|---:|
| P1 | 16 | `5e-5` | 25.0% | 37.5% | 37.5% |
| P2 | 16 | `2e-5` | 25.0% | 37.5% | 37.5% |
| P3 | 16 | `5e-5` | 37.5% | 25.0% | 37.5% |
| P4 | 24 | `5e-5` | 25.0% | 37.5% | 37.5% |

这里的“纯负例增强”不是修改相机或标签，而是从 train split 中显式重采样
`candidate_count > 0 && visible_count == 0` 的合法 view-cell。每个 8-pose batch 按表中固定
配额组成；困难边界池仍由 train-only 实例可见频率和 subpose 边界命中率构造。困难负例尾部
比例由 `0.02` 提到 `0.04`，召回保护从总步数 10% 后开始、30% 时离开慢速段，避免旧协议
到 epoch 13 后才有效约束安全尾部。

完整四项 pilot 不由中间指标提前取消。选择时先取通过 calibration 和 validation 安全门的
成员，再比较 Useful Cull、CNOR、balanced accuracy、Occlusion Recall 和平均保留数；若无
安全成员，按 validation LCB、weighted recall、CNOR、Useful Cull 的顺序选相对最优配置，
如实标记为诊断配置。选定配置必须从头完成三种子 `40 x 900`，不能从已停止 checkpoint
续训，也不能用 test 选参数。

关系重建与训练 smoke 已完成：`K=16` 的截断 cell 为 30,986、保留质量 q01 为 0.7327；
`K=24` 为 21,266 和 0.8456。两者均来自同一 train-only 深度缓存。修正后的两步 smoke
每批实际进入损失的 pose 数为 8，纯负例配额没有在训练张量构造阶段被丢弃，loss 数值有限。

短训只缩短实际运行步数，不能压缩训练课程。pilot 的 3,600 个实际 optimizer step 必须按
完整 36,000 步课程计算学习率、实例校准、尾部分离和召回保护进度，即只观察完整训练的前
10%。2026-09-15 首次启动错误地把课程也压缩为 3,600 步，导致末步学习率变成 0、召回
保护接近满权重，ROC-AUC 退化到随机水平；该批结果不具参数选择含义，删除后按本段口径
原地重跑四项矩阵。

### 12.2 第二阶段：单元粒度重建触发条件

若第一阶段相对最优配置仍未通过 `weighted recall LCB > 0.99`，或安全工作点 CNOR 仍低于
`0.50`，则必须执行场景无关的细粒度规则：目标压缩大小 `64 KiB`，且每个单位最多包含
128 个源连通片。两项限制都只使用源几何和编码字节，不读取标签或模型结果。单位 ID 改变后
必须重建 GLB、1024 点/96D 几何表、硬件 Color-ID、Pose CSR、train-only depth/relation，
并用同一 split 中心划分协议重新训练。正式论文同时报告单位数、候选数、模型资产、运行时
和 byte/triangle cull，避免只靠更细单位获得好看的 unit-level 指标。

第一阶段完成前不启动 AABB MLP、HZB 或 test；这些基线必须使用最终冻结的 Big City 单位、
候选、GT 和 split。

## 13. 引用边界

- Wang et al., NeuralPVS: Learned Estimation of Potentially Visible Sets, SIGGRAPH Asia 2025。
- Greene et al., Hierarchical Z-Buffer Visibility。
- NeuralPVS official rendering/training repositories 仅用于场景来源与方法关系说明。

Sponza 版本、输出空间、view-cell、GT 采样和训练目标未与 NeuralPVS 官方协议完全对齐，
因此不能直接摘录其论文数值放入同协议结果表。若以后加入 NeuralPVS 数值行，必须先固定
同场景、同相机、同 GT、froxel 到 renderable unit 的映射和统一指标重算协议。
