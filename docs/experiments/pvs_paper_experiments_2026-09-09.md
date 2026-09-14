# PVS 论文核心实验与一周执行计划

日期：2026-09-09；2026-09-10、2026-09-11、2026-09-13、2026-09-15 更新正式执行状态

状态：执行中。截至 2026-09-15，HKUST/IFCBench 的冻结 V4 checkpoint 保持不变，只用
统一精确 calibration 阈值重放现有评价。三个标准图形学场景使用 Connected-SAH 128 KiB
单位，先完成并冻结 Full，再执行同单位 AABB、HZB、图像和 runtime。
本文是论文实验、图表、结果目录和执行顺序的唯一计划。

## 论文需要证明的内容

论文围绕五个问题组织：实例 PVS 是否安全且有效；分层关系、生存场和区域频谱查询为何有效；神经启动资产相比几何外壳是否更小；模型能否在桌面和手机实时运行；连续可见性分数能否让正确 GLB 更早到达。

论文重点突出三个系统优势：昂贵遮挡关系在离线完成，前端只查询固定实例表；一次查询覆盖整个 view-cell；相对 geometry-shell HZB，启动传输和运行内存更小。所有优势必须分别由 test 指标、真实设备时间和冷缓存下载实验支撑。

## 当前证据与场景

| 项目 | HKUST | IFCBench/Metropolis |
|---|---:|---:|
| 实例 / GLB | 18,831 / 3,273 | 41,298 / 3,669 |
| Train / calibration / validation / test | 5926 / 659 / 730 / 684 | 19647 / 2183 / 2712 / 2710 |
| 完整 V4 | 三种子 `40 x 900` 已完成 | 三种子 `40 x 900` 已完成 |
| 核心消融 | 六项三种子已完成 | 不重训；只重做精确阈值复评 |
| 生存场 rank | 2/4/8/12 已完成 | rank 4 固定 |
| Validation 图像 | seed 2 已完成 | seed 2 已完成 |

论文正式评价使用 HKUST、IFCBench/Metropolis、Sponza、Big City 和 Viking Village 五个场景。三个标准图形学场景采用一 renderable unit 对应一 resource 的协议；一个单位可以包含同一源 node/primitive/material 内经 SAH 紧凑打包的多个小连通分量，但不跨 primitive 或材质。它们正式报告可见性、图像、HZB、资产和运行成本。Streaming 资源复用与下载排序实验集中在 HKUST/IFCBench。完整转换、训练和表图要求见[标准图形学场景方案](pvs_standard_graphics_scene_generality_2026-09-10.md)。

## 统一评价协议

- 实例任务使用 `66` 度区域候选和区域可见并集；图像与即时显示使用真实 `60` 度相机及每个 pose 的实际 aspect。
- 每个 checkpoint 只在 calibration 冻结阈值。安全条件为 aggregate weighted recall 及其单侧 95% bootstrap 下界均严格大于 `0.99`。
- Validation 选择配置；模型、阈值和评价结构冻结后，每个 test 样本只读取一次。零 GT pose 保留并单独统计。
- 同时报告 pose-macro 和 aggregate precision、recall、F1、Jaccard、accuracy、balanced accuracy、specificity、PR-AUC、正样本比例、AP lift，以及候选归一化遮挡召回率 CNOR、useful cull、bad cull、预测数量和 GLB 字节。论文主表的 PR-AUC 固定为 pose-macro AP，并紧邻 pose 正样本比例与 AP lift；aggregate AP 只作为完整汇总的辅助口径。CNOR 是跨 pose 的主要遮挡效率指标，但不替代 weighted-recall 安全门或 Useful Cull。
- AP 使用合并同分值的 Average Precision。Pose-macro AP 只对有 GT 的 pose 计算并报告有效 pose 数；零 GT pose 的 FP 仍进入集合和资源指标。
- 三种子报告 mean 和 sample standard deviation。核心差值使用按 seed 聚类、seed 内配对 pose 重采样的 10,000 次 bootstrap。
- 图像统计以 view-cell 为重采样单位，报告 PER、miss 和 wrong-ID 的 aggregate、mean、median、p95；extra 使用全图像素分母。

## 正文方法与基线

| 方法 | 输入和用途 |
|---|---|
| Keep-All | 保留相同候选集合，作为绝对安全和零剔除参照 |
| AABB + Ray MLP | 18 维 AABB、投影和视角特征，小型 MLP，不使用固定几何表或遮挡关系 |
| Geometry-only | HKUST 已有无生存场消融，保留现有查询和损失，用于隔离结构化遮挡表示 |
| Geometry-shell HZB | 预下载不透明纯几何外壳，GPU depth/HZB/AABB 测试 |
| Full V4 | 分层关系校准生存场、区域矩包络频谱查询和综合损失 |

AABB MLP 正式版使用与 Full 相同的逐 pose 平衡、weighted-recall 保护和困难边界损失。先扫描学习率 `2e-4/1e-3`，每项 `6 x 300` updates，再以选定配置完成三种子 `40 x 900`。所有方法独立校准阈值；标准场景 AABB 扫描和长训必须等三个场景 Full 全部冻结后开始。

现有核心消融继续按逐项移除解释，不能改写成未经训练的累加模型。Generic-28 是与结构化生存场同容量的可训练实例记忆对照。

## Geometry-shell HZB

正式主基线 `geometry_shell_hzb_lossless` 保留 LOD0 中确定不透明的 POSITION、INDEX 和实例变换，删除法线、UV、颜色、纹理和无关材质，采用 meshoptimizer 传输压缩。关闭背面剔除；透明、玻璃、alpha-cutout 和材质不确定表面不写遮挡深度。

附加 `geometry_shell_hzb_equal_asset`：资产字节限制为同场景冻结神经资产大小，从与 Full 相同的输出 renderable units 中按固定 128 个 train 中心视点的投影面积累计值/压缩字节选择完整 occluder 单位，不移动顶点。该项用于资产敏感性，不宣称保守或最优简化，也不能从旧源 primitive 外壳复用结果。

当前唯一 HZB 算法为 `opaque depth -> max pyramid -> conservative projected AABB test`：只将确定不透明外壳写入深度，构建 max pyramid，再对同一 candidate CSR 中的每个候选执行保守 projected AABB test。四套外壳的 `shell_meta.json` 使用修正后的 `geometry-shell-hzb-v2` schema。投影矩形向外取整并检查覆盖 mip 的全部 texel；仅当 `HZBMax + bias < candidateNear` 时剔除。近裁剪面、非有限投影、相机位于 AABB 内和其他不确定情况全部保留。HZB 论文口径只保留这条算法路径。

HKUST/IFCBench 已导出外壳的交接资产统计如下；这些是启动资产和解码内存数字，不是 HZB 可见性或性能结果。标准场景外壳须在其 Full 资产冻结后按 Connected-SAH 单位重新导出：

| 场景 | 变体 | 传输 B | 展开几何 MiB | 展开运行时 MiB | prototype | prototype 三角形 | 展开三角形 | occluder 实例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HKUST | lossless | 399,312,852 | 1,697.30 | 1,697.80 | 3,042 | 54,771,324 | 55,537,431 | 18,566 |
| HKUST | equal-asset | 5,147,576 | 17.78 | 18.29 | 81 | 682,664 | 780,713 | 1,257 |
| IFCBench | lossless | 59,924,084 | 153.30 | 154.40 | 3,669 | 8,648,833 | 25,556,160 | 41,298 |
| IFCBench | equal-asset | 11,819,713 | 25.43 | 26.54 | 1,405 | 1,342,146 | 6,290,848 | 31,909 |

正式 HZB 编排固定为 `36 tasks`：`2 preflight + 16 calibration + 2 select + 12 frozen test + 4 timing`。其中 calibration 为 `512x288 / 1024x576` × depth bias `0.01/1/10/100 m`，每场景 8 个 lossless 配置、每配置一次 invocation；按安全条件下 useful cull 最高选取，平局选择耗时更低配置。投影矩阵逐 pose 使用真实 aspect，HZB 纹理分辨率作为基线自身预算固定，不冒充前端图像分辨率。

完整 calibration 和 frozen test 每个 pose 只执行一轮可见性查询，避免把精度矩阵重复五遍；独立的 120-pose 性能任务才对每个 pose 执行五轮并统计 p50/p95。两类任务都必须通过同一硬件 GPU 和无并发计算证据门。

八个 lossless calibration task 由 `select_geometry_shell_hzb.py` 冻结配置：安全池要求 weighted recall 及其单侧 95% 下界均严格大于 `0.99`，安全池内按 useful cull、balanced accuracy、specificity、precision 和延迟排序；没有安全成员时仍输出诊断配置，但不得标成安全 HZB。

公平比较分为：

- Point60：canonical center 的真实相机和同点 GT。
- Region66：相同区域候选，对多个真实 subpose 做 HZB 后取并集。HKUST 比较 `1/5/9/all`；IFCBench 比较 `1/all`。子集由中心最近点开始做最远点空间覆盖，不读取可见标签。
- 全 subpose HZB 是区域质量参照，多次查询的总耗时全部计入；有限点结果不宣称连续区域保证。

主要相关工作包括 Greene 等人的 Hierarchical Z-Buffer Visibility、HROC 的 GPU 批量组织方式，以及 Occluder Simplification using Planar Sections。论文只数值实现一个现代 GPU HZB 管线，不冒称复现这些完整系统。

## 论文实验与图表

| 实验 | 动作 | 论文产物 |
|---|---|---|
| 场景统计 | 统计范围、原型/展开三角形、GLB 字节分布、split、候选和 GT 分布；标准场景另报划分 schema、源连通分量、每单位分量数、超大分量拆分数和编码字节分布 | Table 1 |
| Test 可见性 | Keep-All、AABB MLP、HZB、Full V4 全量 test | Table 2 |
| Test 图像 | Full、AABB MLP、选定 HZB 的真实60度 Color-ID | 图像表、Fig. 7 |
| 核心消融 | 复用六项三种子和 Generic-28，统一重新汇总 | Table 3 |
| 容量与资产 | 复用 rank 2/4/8/12，补同候选延迟 | Pareto 图、Table 4 |
| 端侧性能 | 两场景 WebGPU/WASM，A6000、M2、vivo | Table 4、Fig. 5 |
| HZB 系统成本 | 两种外壳、Point/Region、冷启动和分阶段 GPU 时间 | Table 4 |
| Progressive streaming | 全 test 冷缓存离线模拟和固定子集真实调度 | Table 5、Fig. 6 |
| 安全效率曲线 | Validation 的 WR-useful cull 和 WR-GLB bytes | Fig. 4 |
| GT 收敛 | 每场景100个分层 validation cell，嵌套采样到128点 | 附录图 |
| 离线成本 | 采样、关系、编码、训练、校准、导出时间与峰值资源 | 附录表 |
| 标准图形学场景主实验 | Sponza/Big City/Viking Village，固定 128 KiB renderable units，一单位一资源；同单位比较 Keep-All、AABB MLP、Hi-Z 与 Full | 正式场景结果表、资产-剔除 Pareto、定性图 |

正文保留五张表：场景统计、test 可见性、消融、资产与运行时间、streaming。核心图为系统总览、模型架构、生存场可视化、安全效率曲线、候选数量延迟曲线、下载覆盖曲线和定性对比。研究图输出 PDF、SVG、PNG 和源 CSV；架构图保留可编辑源文件。

定性图每场景选择 PER 中位数附近、p95 附近、最大错误和一个预登记细构件/遮挡边界视角，报告 pose ID。不能只挑低误差画面。

## 端侧性能

扩展现有 HTTPS 测试站，使场景配置提供模型、实例数、pose 数和候选文件，并允许 WebGPU/WASM 选择。

- 每个场景和后端使用 50 次 warmup、5 个 session，纯模型前向遍历该场景完整 test 候选负载。
- WebGPU 报告 kernel 和 submit-completion；WASM 使用一次批量模型调用。二者排除资产下载、初始化和候选构建。
- 候选分桶为 `0-1k/1-2k/2-5k/5-10k/10-15k/15k+`，必要时为 IFCBench增加高候选桶，报告 count、mean、p50、p95和区间。
- 拟合 `T(N)=a+bN` 并报告拟合误差；不强行宣称完全线性。
- HZB 和完整 PVS 管线在每场景120个按候选规模分层的固定 test pose 上做5次重复，记录压缩和回读总成本。
- 冷启动另外计入资产 fetch、解码、GPU upload/init、首次查询和首个结果。

## Progressive streaming

每个 test pose 都从空缓存开始。结果明确分成三类且不混合：`threshold filtering` 使用冻结阈值形成预测资源集合；`threshold-free ranking` 在完整 candidate GLB 集合上排序，不读取阈值；`scheduler replay` 单独重放 urgent/warm/speculative 调度状态机并计入资源处理过程。

过滤实验使用 calibration 冻结的阈值，报告候选到预测 GLB 的数量/字节、最终 visible-weight coverage 上限以及漏掉的必需资源。

排序实验对所有方法使用相同完整 candidate GLB set，不应用阈值。神经 GLB 分数先按 `p_g=max_i(p_i)` 聚合；成本感知分数为 `p_g/Bytes_g^alpha`，`alpha` 只能在 validation 从 `{0,0.5,1}` 选择并写入冻结 manifest，test 只读取该 manifest。比较 Full、AABB MLP、原始顺序、距离、投影面积、投影面积/字节、20 个固定随机种子、HZB visible-first 和 GT-informed utility/byte oracle。Distance 与 projected AABB area 必须先逐候选实例计算，再按 GLB 聚合为 `distance=min`、`area=max`；不使用合并 GLB AABB 的投影结果。

每个 GLB 完整到达后才累计 coverage。主要指标为 Bytes@95/99/99.9/100、99%前无用字节、必需资源平均排名；时间按 `10/25/50/100 Mbps` 换算，正文展示25和50 Mbps。Oracle 按 GT utility/byte 贪心排序，并另给分数背包字节下界，不能把贪心值称为所有目标的全局最优。

全 test 的主覆盖字段明确为 visible-weight coverage：使用数据集提供的 `visible_weights`，在实例映射到 GLB 后累计可见权重；Color-ID 屏幕覆盖权重按 view-cell subpose 最大值解释，不解释为隐藏表面总覆盖。若补充 reference-frontmost screen utility，则单独用于四个下载阶段的真实重渲染核验，核验前景移除后露出的 wrong-ID，不替代 visible-weight coverage。

真实调度在每场景12个固定 test pose、25/50 Mbps、每项3次，调用现有 urgent/warm/speculative 状态机，取消与固定首页位置相关的 startup-100 前缀。计入方法资产、初始化、GLB 下载、解析、挂载和首个正确画面。无法达到目标时报告未达到和覆盖上限。

## HKUST/IFCBench 精确重校准

HKUST 与 IFCBench 不重新训练。对每个现有 checkpoint，在 calibration 保存 float32 分数、
标签、可见权重和 pose offsets；阈值候选使用实际分数变化点，预测规则固定为
`score >= threshold`。复用固定的 10,000 组 pose bootstrap 索引，找到最高安全阈值，
再只在 validation 比较成员。精确重校准不会改变 AP，收益必须单独记录。模型和阈值冻结后
才允许重放一次 test、图像、streaming 和前端资产。

## 必要接口

- 评价入口增加冻结 test 模式，并将 test 结果明确标为 `testRead=true`。训练与 calibration 继续拒绝 test。
- 大型分数使用按pose offsets对齐的二进制sidecar；JSON只保存小型指标和路径。图像manifest每个view-cell只保存一次预测集合。
- 测试站移除HKUST固定684/18831限制，改为场景清单驱动，并增加WASM测量。

## 并行执行和结果目录

Full 冻结后的独立工作可从同一提交创建不同 worktree：

| 执行者 | 责任 |
|---|---|
| AABB 工作树 | 三场景 AABB 扫描、三种子训练与 validation 冻结 |
| HZB 工作树 | 同单位外壳导出、WebGPU calibration/test/timing |
| 评价工作树 | 冻结 test、图像评价、端侧 runtime 与统一指标表 |
| Streaming 工作树 | HKUST/IFCBench 全 test 模拟、真实 scheduler replay 和论文图 |
| 主工作树 | 公共接口、阶段门、结果集成与最终审阅 |

GPU 1-3用于当前训练和评分；GPU 0 的外部任务不干预。正式浏览器采样和计时另等独占窗口，
所有长任务写 stdout/stderr 日志。

统一结果目录为 `neural_instance_culling/benchmark/out/paper_results/`，包含 `scene_statistics.csv`，以及 `test_metrics/`、`image_metrics/`、`ablation/`、`rank_sweep/`、`mobile_runtime/`、`hzb/`、`streaming/`、`threshold_curves/`、`gt_convergence/`、`preprocessing/` 和 `figures/`。每组结果记录实际样本数、冻结配置、来源和复现命令。

执行顺序不按日历压缩：先完成三个标准场景 Full 三种子并冻结成员和阈值；随后并行执行
AABB 与 HZB；最后统一打开 frozen test，完成图像、端侧 runtime、HKUST/IFCBench
streaming、GT 收敛和论文数据包。任何中间指标都不能提前绕过阶段门。

验收覆盖同分AP、零GT统计、阈值边界、checkpoint和固定表一致性、HZB深度/近裁剪面、区域并集、GLB原子到达、覆盖不可达、像素直方图与重渲染一致性，以及硬件计时边界。禁止新增兼容层、前端历史模型路径或手工哈希步骤。

## CNOR 评价口径

新增候选归一化遮挡召回率：

\[
\mathrm{CNOR}=\frac{\sum_p TN_p/C_p}{\sum_p(TN_p+FP_p)/C_p}.
\]

该指标从正式 `perPose` 混淆计数或 pose-aligned score sidecar 直接重算，不由 aggregate
specificity 近似。Full 与 AABB 的三种子结果报告 mean +/- sample std；只有一份逐 pose
资产时明确标成单成员结果。

HKUST 核心消融已从 730 个 validation pose 的逐 pose 混淆计数重新汇总：Full 的三种子
CNOR 为 `0.903764`；去除分层关系、去除生存场、通用 28 维、去除矩包络、去除召回保护、
去除困难边界的 CNOR 依次为 `0.875568/0.838316/0.894385/0.895451/0.882723/0.897642`。
生存场 rank `2/4/8/12` 的 CNOR 分别为
`0.893121/0.906264/0.890525/0.905261`。这些结果仍是 validation 消融，不与 frozen test
主表混用。

核心消融和容量汇总命令为：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/summarize_core_ablation.py \
  --root neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1 \
  --output neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1/paper_core_ablation_summary.json \
  --bootstrap-replicates 10000 --seed 20260823

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_survival_rank_capacity.py summarize
```

## 执行状态（2026-09-15）

- 不修改当前 V4 的 96D 几何表、结构化生存场和运行时查询头；HKUST/IFCBench 不重训。
- 标准场景统一使用 `connected-sah-pack-v2`：先求共享顶点连通分量；小分量只在同一源
  node/primitive/material 内用确定性 16-bin SAH 打包；超大分量按三角形 SAH 递归拆分；
  meshopt 实际编码字节执行 `128 KiB` 最终上限。
- Full 固定使用 `lr=5e-5`、`poses_per_batch=8`、`hard_pose_fraction=0.35`，尾部分离权重
  升至 `0.30`，正/负尾部比例为 `0.01/0.02`，正样本重要性下限为 `0.25`；采用先分类与
  生存场、再尾部分离、最后召回保护的课程。
- 三场景 GLB、1024 点/96D 几何表、硬件 Color-ID、Pose CSR 和 train-only K=8 关系均已
  完成。Sponza/Viking/Big City 单位数为 `129/1763/2861`，split 分别为
  `4800/528/672/672`、`1944/216/276/276`、`11580/1284/1608/1608`。
- seed20260801、seed20260802 和 seed20260803 已分别按完整三场景 round 在 GPU 1/2/3
  启动。每张训练卡并发三个种子，后续只监控总吞吐、数值稳定性和首次 validation。
- calibration 阈值统一改为实际 float32 分数变化点；HKUST/IFCBench 对现有 checkpoint
  重做 calibration/validation 阈值冻结和派生结果，不用 test 选阈值、不重新训练。
- 精确重校准已完成。HKUST 三种子均通过 validation 安全门，选择 seed02；其
  WR/LCB/Useful Cull/CNOR 为 `0.996847/0.994558/0.904606/0.915014`。IFCBench 只有
  seed03 通过 validation 安全门，对应 `0.991350/0.990509/0.559296/0.648690`；seed01/02
  的 validation LCB 为 `0.989853/0.989395`，不进入精确阈值安全池。当前尚未用新阈值
  重放 test、图像、streaming 或前端资产。
- 标准场景执行顺序固定为 `Full -> AABB MLP / HZB -> frozen test/image/runtime`，后两类
  基线不得提前占用 Full 的采样、训练和浏览器 GPU 资源。
- 完整单位契约和重建步骤以
  [标准图形学场景方案](pvs_standard_graphics_scene_generality_2026-09-10.md)第 3-11 节为准。
