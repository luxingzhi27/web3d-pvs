# PVS 论文核心实验与一周执行计划

日期：2026-09-09；2026-09-10、2026-09-11 更新正式执行状态

状态：执行中。截至 2026-09-11，HKUST/IFCBench 的神经、AABB、正式 HZB、test 图像、A6000 runtime 和全 test streaming 模拟均已完成；真实 scheduler replay、GT 收敛和标准图形场景扩展仍在执行。本文是论文实验、图表、结果目录和执行顺序的唯一计划。

## 论文需要证明的内容

论文围绕五个问题组织：实例 PVS 是否安全且有效；分层关系、生存场和区域频谱查询为何有效；神经启动资产相比几何外壳是否更小；模型能否在桌面和手机实时运行；连续可见性分数能否让正确 GLB 更早到达。

论文重点突出三个系统优势：昂贵遮挡关系在离线完成，前端只查询固定实例表；一次查询覆盖整个 view-cell；相对 geometry-shell HZB，启动传输和运行内存更小。所有优势必须分别由 test 指标、真实设备时间和冷缓存下载实验支撑。

## 当前证据与场景

| 项目 | HKUST | IFCBench/Metropolis |
|---|---:|---:|
| 实例 / GLB | 18,831 / 3,273 | 41,298 / 3,669 |
| Train / calibration / validation / test | 5926 / 659 / 730 / 684 | 19647 / 2183 / 2712 / 2710 |
| 完整 V4 | 三种子 `40 x 900` 已完成 | 三种子 `40 x 900` 已完成 |
| 核心消融 | 六项三种子已完成 | 本轮只做完整模型微调 |
| 生存场 rank | 2/4/8/12 已完成 | rank 4 固定 |
| Validation 图像 | seed 2 已完成 | seed 2 已完成 |

本周冻结结果仍以两个现有场景为主。标准图形学场景作为独立的 non-BIM generality 扩展：采用一 renderable unit 对应一 resource 的协议，不要求 GLB 原型复用。Big City 官方 NeuralPVS 资产已完成来源审计，并按固定 `128 KiB` 目标转换为 `2734` 个单位；其采样、训练、test 和性能结果仍未形成，不阻塞 HKUST/IFCBench 的 HZB、图像和 streaming 主结果。完整转换、训练和表图要求见[标准图形学场景方案](pvs_standard_graphics_scene_generality_2026-09-10.md)。

## 统一评价协议

- 实例任务使用 `66` 度区域候选和区域可见并集；图像与即时显示使用真实 `60` 度相机及每个 pose 的实际 aspect。
- 每个 checkpoint 只在 calibration 冻结阈值。安全条件为 aggregate weighted recall 及其单侧 95% bootstrap 下界均严格大于 `0.99`。
- Validation 选择配置；模型、阈值和评价结构冻结后，每个 test 样本只读取一次。零 GT pose 保留并单独统计。
- 同时报告 pose-macro 和 aggregate precision、recall、F1、Jaccard、accuracy、balanced accuracy、specificity、PR-AUC、正样本比例、AP lift、useful cull、bad cull、预测数量和 GLB 字节。论文主表的 PR-AUC 固定为 pose-macro AP，并紧邻 pose 正样本比例与 AP lift；aggregate AP 只作为完整汇总的辅助口径。
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

AABB MLP 正式版改用与 Full 相同的逐 pose 平衡、weighted-recall 保护和困难边界损失。先扫描学习率 `2e-4/1e-3`，每项 `6 x 300` updates，再以选定配置完成三种子 `40 x 900`。所有方法独立校准阈值。

现有核心消融继续按逐项移除解释，不能改写成未经训练的累加模型。Generic-28 是与结构化生存场同容量的可训练实例记忆对照。

## Geometry-shell HZB

正式主基线 `geometry_shell_hzb_lossless` 保留 LOD0 中确定不透明的 POSITION、INDEX 和实例变换，删除法线、UV、颜色、纹理和无关材质，采用 meshoptimizer 传输压缩。关闭背面剔除；透明、玻璃、alpha-cutout 和材质不确定表面不写遮挡深度。

附加 `geometry_shell_hzb_equal_asset`：资产字节限制为同场景神经资产大小，从原始不透明 primitive 中按固定 128 个 train 中心视点的投影面积累计值/压缩字节选择 occluder，只删除完整 primitive，不移动顶点。该项用于资产敏感性，不宣称保守或最优简化。

当前唯一 HZB 算法为 `opaque depth -> max pyramid -> conservative projected AABB test`：只将确定不透明外壳写入深度，构建 max pyramid，再对同一 candidate CSR 中的每个候选执行保守 projected AABB test。四套外壳的 `shell_meta.json` 使用修正后的 `geometry-shell-hzb-v2` schema。投影矩形向外取整并检查覆盖 mip 的全部 texel；仅当 `HZBMax + bias < candidateNear` 时剔除。近裁剪面、非有限投影、相机位于 AABB 内和其他不确定情况全部保留。HZB 论文口径只保留这条算法路径。

四套已导出外壳的交接资产统计如下；这些是启动资产和解码内存数字，不是 HZB 可见性或性能结果：

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
| 场景统计 | 统计范围、原型/展开三角形、GLB 字节分布、复用率、split、候选和 GT 分布 | Table 1 |
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
| 标准图形场景泛化 | Sponza/Big City 等标准场景，固定 128 KiB renderable units，一单位一资源；同单位比较 Keep-All、AABB MLP、Hi-Z 与 Full | Generality 表、资产-剔除 Pareto、定性图 |

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

## IFCBench 校准与微调

现有 IFCBench 三种子使用相同 `40 x 900` from-scratch 配置。Aggregate AP 为 `0.2536/0.4051/0.3190`，正样本比例 `0.09536`。当前高分阈值网格步长为约0.02，而困难分数集中在0.774附近，因此先区分网格量化和排序重叠。

先在 calibration 保存 float32 分数、标签、可见权重和 pose offsets。阈值候选使用实际分数变化点，预测规则固定为 `score >= threshold`。复用固定的10,000组 pose bootstrap索引，找到最高安全阈值，再冻结到 validation。精确重校准不会改变AP，收益必须单独记录。

随后从 seed 20260802 原 `best_safe.pt` 端到端微调，每组 `4 x 900` updates：

| 配置 | RVL | 边界权重 | Margin / temperature | LR |
|---|---:|---:|---:|---:|
| 原损失继续训练 | 0.30 | 0.20 | 0.50 / 0.25 | 2e-5 |
| 边界减半 | 0.30 | 0.10 | 0.50 / 0.25 | 2e-5 |
| 去除边界诊断 | 0.30 | 0 | 0.50 / 0.25 | 2e-5 |
| 软边界 | 0.30 | 0.20 | 0.25 / 0.50 | 2e-5 |
| 较高学习率 | 0.30 | 0.20 | 0.50 / 0.25 | 5e-5 |

固定96维几何输入和124维运行schema，保留生存/关系/实例校准正则权重 `0.25/0.10/0.02`、四 pose batch、8192 observation batch和0.25关系梯度上限。加载 checkpoint 模型状态但使用新的 AdamW；继承实例校准 blend并从第一步启用完整尾部项。

扫描先检查 calibration 安全和冻结阈值下 validation 安全，再比较 useful cull、balanced accuracy、specificity、普通 precision/recall、pose/aggregate AP及分数健康。若没有安全成员，仍选择召回下界最高的相对最优成员完成三种子确认。

选中配置从三个各自原 checkpoint 出发，各额外完成 `8 x 900` updates，每1800步评价。Seed 20260802不能复用扫描后的额外更新；三种子确认都从原 checkpoint开始。结果决定是否替换IFCBench最终模型，不修改HKUST模型。

## 必要接口

- 训练入口增加 `--init-checkpoint`，记录初始 seed/epoch、额外更新数和新优化器状态；不改变前端模型schema。
- 评价入口增加冻结 test 模式，并将 test 结果明确标为 `testRead=true`。训练与 calibration 继续拒绝 test。
- 大型分数使用按pose offsets对齐的二进制sidecar；JSON只保存小型指标和路径。图像manifest每个view-cell只保存一次预测集合。
- 测试站移除HKUST固定684/18831限制，改为场景清单驱动，并增加WASM测量。

## 并行执行和结果目录

计划文档提交后，从同一提交创建独立 worktree：

| 执行者 | 责任 |
|---|---|
| Subagent A | IFCBench精确校准、warm-start、五组扫描和三种子确认 |
| Subagent B | 外壳导出、HZB WebGPU和测量 |
| Subagent C | AABB MLP、场景统计、统一test评价和分数sidecar |
| Subagent D | Streaming模拟、真实调度和论文图 |
| 主agent | 公共接口、集成、设备测试、图像评价和最终审阅 |

GPU 1-3用于训练和评分，GPU 0用于浏览器开发验证；正式计时使用独占窗口。所有长任务写 stdout/stderr 日志。

统一结果目录为 `neural_instance_culling/benchmark/out/paper_results/`，包含 `scene_statistics.csv`，以及 `test_metrics/`、`image_metrics/`、`ablation/`、`rank_sweep/`、`mobile_runtime/`、`hzb/`、`streaming/`、`threshold_curves/`、`gt_convergence/`、`preprocessing/` 和 `figures/`。每组结果记录实际样本数、冻结配置、来源和复现命令。

执行顺序：第1天完成共享评价、校准、微调/基线扫描和外壳导出；第2-3天完成三种子确认、基线训练、HZB和streaming；第4天冻结并执行test与图像评价；第5-6天完成设备、HZB、streaming和GT收敛；第7天统一表图和论文数据包。

验收覆盖同分AP、零GT统计、阈值边界、checkpoint和固定表一致性、HZB深度/近裁剪面、区域并集、GLB原子到达、覆盖不可达、像素直方图与重渲染一致性，以及硬件计时边界。禁止新增兼容层、前端历史模型路径或手工哈希步骤。

## 执行进度（2026-09-10）

| 项目 | 状态 | 已有产物 / 下一动作 |
|---|---|---|
| 场景统计 | 完成 | 两场景规模、三角形、GLB 字节、split、候选/GT 分布已进入 Table 1 |
| HKUST Full test | 完成 | 三种子均通过安全门；WR `0.997082 +/- 0.001634`，LCB `0.994334 +/- 0.003271`，useful cull `0.901758 +/- 0.007830` |
| IFCBench 微调与 test | 完成 | v2 边界减半三种子 confirmation 的 validation LCB 为 `0.991106/0.990764/0.990215`，均通过；正式 `2710` test 的 pose PR-AUC `0.482290 +/- 0.021681`、WR `0.991179 +/- 0.000668`、LCB `0.990523 +/- 0.000552`、useful cull `0.602498 +/- 0.015504`，三份 test 各读取一次。运行资产使用 validation useful cull 最高的 seed01 |
| AABB + Ray MLP | 完成 | 两场景三种子 `40 x 900` 与 frozen test 已完成。IFCBench pose PR-AUC `0.28504 +/- 0.00047`、prevalence `0.12232`、useful cull `0.03942`；HKUST 与 Full 的正式对照产物也已登记 |
| HZB 外壳与运行时 | 完成 | `36/36` 正式任务完成；两场景都冻结 `512x288 / 100 m`。Lossless Region66 test 的 WR/LCB/useful cull 为 HKUST `0.997633/0.996552/0.328166`、IFCBench `0.995325/0.994716/0.237997`；总查询 p50 分别 `989.55/120.15 ms` |
| 资产与容量 | 完成 | 神经资产为 lossless shell 的 `1.37%`（HKUST）和 `19.78%`（IFCBench）；rank Pareto、Table 4 及 PDF/SVG/PNG 已生成 |
| 端侧模型前向 | 部分完成 | A6000 两场景五 session 已完成：HKUST/IFCBench WebGPU kernel p50 `1.568/1.385 ms`、p95 `3.555/3.590 ms`；WASM p50 `3.7/46.8 ms`、p95 `100.5/143.2 ms`。HKUST M2 与 vivo 已完成，IFCBench 移动端仍缺失 |
| Safety-efficiency | 完成 | 六个核心变体的 calibration-safe validation 曲线及论文图已生成 |
| Streaming | 全 test 模拟完成；真实 replay 执行中 | Visible-weight Bytes@99：HKUST Full/成本感知/面积字节/HZB 为 `18.031/10.819/12.881/17.498 MiB`；IFCBench 为 `8.078/7.119/13.426/13.404 MiB`。冻结 alpha 为 `1.0/0.5`；HKUST 真实 scheduler replay 正在运行，随后执行 IFCBench |
| Test 图像 | 完成 | 六组正式硬件 Vulkan/ANGLE 图像评价完成，均无失败 subpose 或缺失 GLB。HKUST Full/AABB/HZB aggregate PER 为 `0.3662/0.8613/0.3128%`；IFCBench 为 `0.5011/0.0583/0.5312%` |
| GT 收敛 | 部分完成 | 两场景各 100 cell x 128 点水平圆盘计划已生成，IFCBench 半径为 `2.5 m`；新 evaluator 已按计划/raw 三元组对齐，正式 Vulkan Color-ID 采样待执行 |
| 离线成本 / 结果包 | 完成当前可得项 | 六阶段成本、artifact registry、三种子 Table 2 汇总已生成；未记录时间保持 unavailable，不作推算 |
| 标准图形场景 | Sponza 训练中；Big City 采样中 | Sponza `38,784` 个硬件 Color-ID subpose、`2,424` Pose CSR 和 train-only 稀疏关系已完成，Full 三种子与 AABB 扫描已启动。Big City `129,408` 个 subpose 正在正式硬件采样 |

截至 2026-09-11，现有两个主场景只剩真实 scheduler replay、GT 收敛和缺失的移动端组合。标准场景按同一显式四分割、同一 V4 和同一 AABB/HZB 评价口径执行；Sponza 正式训练期间继续采样 Big City，任何中间指标都不替代三种子 validation 选择和一次 frozen test。

本轮实现回归已通过 benchmark `167` 项、model `48` 项、完整前端 `npm test` 和 sampler `7` 项测试；测试过程禁用 CUDA，不作为任何正式性能结果。
