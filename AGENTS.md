# AGENTS.md

本文件定义本仓库中 Codex / 其他 agent 的工作准则。目标是避免无约束实验、重复命名、错误复用资源和文档散乱，保证神经剔除实验能够被复现、比较和交接。

## 1. 总体原则

- 所有工作必须优先保持当前可运行版本稳定；新实验不能破坏已有前端、训练、采样和 benchmark 链路。
- 不允许随意创建大量临时版本名、临时输出目录或一次性脚本；新方案必须先定义清楚目标、输入、输出、评估指标和保留/删除条件。
- 不允许用旧实验名称复用新含义。例如同一个目录名不能先表示 PointNet++ 原始版，后又表示高召回版。
- 不允许把失败实验伪装成当前主线；失败实验可以记录，但不能污染默认 runner、README 或前端默认资产。
- 不允许用 sample-level 二分类指标替代 pose-level 集合指标作为主结论。
- 当前主要验收目标是：在画面安全约束下最大化有效剔除和资源节省。普通集合 recall 应尽量接近或超过 `0.95`，weighted recall 应尽量接近或超过 `0.99`；在满足安全约束后，优先比较 `useful cull = TN / candidate`、`bad cull = FN / candidate`、平均预测数量、GLB 字节削减和前端延迟。普通 precision、F1 和逐实例 accuracy 必须报告，但不能单独作为主结论。
- 不允许把 `recall_high` 当作额外变体后缀来逃避主目标；每个正式 PVS 实验默认就必须按高召回目标训练和选 checkpoint。若一个模型需要降低召回才能得到好看的 F1，它不能作为合格主线版本。历史路径或旧 benchmark 中出现的 PointNet / Triplane / dynamic-pool 命名只作为已删除旧实验理解，不能作为当前默认路径。
- 向用户解释模型、训练流程或前端推理时，必须优先使用可读的中文概念和数据流描述；内部类名、函数名、变量名只能作为定位参考放在括号中，不能用一串代码英文名代替解释。每个新名词都要说明“它输入什么、输出什么、为什么需要它”。

## 1.1 当前运行环境

- 当前项目已从 Windows / AMD ROCm 机器迁移到 Linux，训练优先使用 NVIDIA CUDA。
- 新训练环境必须使用 conda 创建和记录，例如 `conda create -n slm_pvs python=3.11`；不要再新增旧 ROCm 环境或 Windows PowerShell 反引号命令作为当前推荐命令。
- 文档和新脚本中的可复现命令默认使用 Linux shell 口径，例如 `conda run -n slm_pvs python ... --device cuda`。
- 如果必须保留历史 Windows/ROCm 命令，只能放在历史说明中并明确标注“历史迁移前命令，不作为当前推荐运行方式”。
- 长时间训练、导出和 benchmark 仍必须写入明确 stdout/stderr 日志；Linux 下推荐使用 `> train_stdout.log 2> train_stderr.log` 或等价的日志方案。
- 正式浏览器采样、实例级 Color-ID 图像评价和三角形 HZB 光栅化必须使用 Chrome 的硬件 GPU 路径，默认开启 `--use-angle=vulkan` 与 `--disable-software-rasterizer`，并保存页面 `gpuBackend`、`gpuGate` 以及同一执行窗口的 `nvidia-smi`/`pmon` 证据。检测到 SwiftShader、llvmpipe、softpipe、swrast 或无法确认后端时必须失败；只有显式 `--allow-software-gpu` 的小规模语义调试可以使用软件后端，不能进入正式数据集、GPU 性能或移动端性能结论。详细政策见 `docs/current/hardware_gpu_execution_policy.md`。
- WebGL/ANGLE 与 WebGPU adapter 是两条独立的硬件证据链：WebGL 回报 NVIDIA 不能替代 WebGPU adapter 核验，`nvidia-smi` 中出现 Chrome 进程也不能替代被测 API 的后端字段。凡是 WebGPU/WGSL 性能实验都必须单独读取 adapter；若 adapter 回报 SwiftShader 或其他软件后端，必须停止硬件性能汇总。
- Color-ID 正式采样还必须保留分片旁的 `*.jsonl.gpu_evidence.json` 和 view-cell 总目录的 `gpu_execution_summary.json`；其中要有页面 `gpuBackend`/`gpuGate`、Chrome 参数及同一窗口的 `nvidia-smi`/`pmon`。没有这些证据的历史 JSONL 不能追认为硬件采样。

## 1.2 当前核心研究目标

- 当前长期目标不是单纯堆高某个离线 benchmark 分数，而是用一个优美、统一、轻量、可部署的方式，同时解决“视点区域实例可见性预测”和“GLB 下载优先级排序”两个问题。
- 目标模型必须具有清晰论文叙事：从视角无关的实例几何/上下文表征出发，用当前视角进行轻量查询，并在同一输出体系中自然得到实例级可见性分数和 GLB 级下载优先级；不能把二者做成互不相关的后处理拼接。
- 这是后续 agent 的长期任务准则：新的 PVS 方案必须优先回答“如何在手机端预算下统一决定哪些实例显示、哪些 GLB 先下载”，而不是只把现有 visibility MLP、GLB 排序器、阈值扫描和前端规则机械叠加。
- 目标模型必须具有实际前端价值：运行时应尽量依赖固定离线特征表、轻量视角特征、少量预计算关系或压缩结构；必须面向手机端流畅运行，避免在线 PointNet++、Triplane、大规模动态图传播、全量 AABB 8 角点逐帧投影或其他主线程重计算。
- 必须正视当前任务的结构性困难：在候选集合很大时，pose-level 直接对每个实例做共享 MLP 二分类，很难对大量细小构件同时达到极高 precision 和 recall。后续方案应优先研究更合适的层次化、集合化、图形学启发、遮挡关系、重要性排序、预算感知或区域/资产联合建模，而不是只继续调低阈值或加大普通二分类 loss。
- GLB 下载优先级的监督和输出应与实例可见性共享基础表征，并显式考虑可见实例集合、重要性权重、GLB 成本、预下载价值和前端加载预算。视锥内立即需要与视锥外预下载可以在调度层分队列，但底层分数体系应尽量统一。
- GLB priority 监督不能简单等同于“这个 GLB 是否含有当前阈值下的可见实例”。更合理的监督应优先来自视觉效用、缺失画面损失、可见权重、资源字节/解码成本、首屏预算和预下载收益；实例可见性和 GLB 下载可以共享表征，但不能把“某个实例不可见”反向解释成“其所属 GLB 不该下载”。
- 模型设计必须区分两个粒度：实例级输出用于 `componentModelList` 和渲染过滤，GLB 级输出用于下载/解码/缓存排序。二者应在同一个轻量模型中通过共享基础特征和不同任务头统一，而不是让 GLB 粗粒度信号污染实例级遮挡判别。
- 判断一个方案是否“优美优雅”，至少要满足：输入输出清楚、训练监督和论文叙事一致、前端运行路径短、移动端算子可实现、失败原因可分析、指标能解释真实画面和加载体验。只靠相机哈希记忆、阈值压低、前端重计算或复杂后处理得到的结果不应作为主线。
- 新方案应同时追求论文叙事感与工程实用性：可以主动查阅并吸收相关深度学习、图形学、可见性、PVS、遮挡建模、集合预测、排序学习、压缩推理和移动端推理研究，但引入任何复杂机制前必须说明它输入什么、输出什么、为什么比当前方案更优雅、更轻量、更可部署。
- 相关研究可以作为方案来源，包括但不限于 NeuralPVS、Neural Visibility of Point Sets、Hardly-Visible Sets、可见性预取、遮挡图/深度排序、集合预测、learning-to-rank、budgeted inference、感知图像质量指标和移动端神经网络压缩。引用这些研究时必须落到本项目的数据结构、监督信号、前端成本和可复现实验，不允许只堆论文名。
- 判断一个方案是否值得继续时，除 pose-level 指标外，还必须看图像损失、候选削减、GLB 下载数量/字节削减、运行时延迟、前端实现复杂度和论文叙事一致性。不能因为一个方案离线 precision 略高，就忽略其移动端代价或叙事割裂。
- 如果普通 precision 长期难以达到目标，可以提出更合理的主指标或辅助主指标，但必须有严谨解释和实验证据：大量细小构件的误判与大构件、近景构件、高像素贡献构件、关键 GLB 资源的误判严重程度不同，不能机械地把每个 instance false positive/false negative 等权看待。
- 替代或补充 precision 的指标应优先围绕最终效果定义，例如图像 PER、miss pixel rate、wrong ID pixel rate、可见权重召回、重要 GLB 下载召回、字节级节省、首屏/交互延迟、预算内可见效用、感知可忽略错误比例等。使用这些指标时必须同时报告普通集合 precision/recall，说明“为什么普通 precision 不足但最终体验仍可接受”，不能用新指标掩盖大量有视觉影响的错误。
- 如果一个方案无法把普通 precision 提升到目标值，但能证明“大部分误判来自低视觉效用、低像素贡献、低成本或可延迟下载的细小构件”，并且图像 PER、miss utility、预算内 GLB utility、字节节省和移动端耗时均优于或不弱于主线基线，则可以作为论文主线候选；反之，如果新指标只是在统计上淡化错误，而最终画面或下载体验没有改善，必须降级为失败实验。
- 论文叙事允许把任务从“每个实例等权二分类”提升为“预算约束下的可见效用最大化”：目标是在不影响最终画面和加载体验的前提下，用轻量模型选择最有显示价值和下载价值的实例/GLB。这个叙事必须和训练 loss、阈值选择、GLB priority、图像指标和前端调度保持一致。

## 2. 实验版本管理

- 每个正式实验必须有清晰稳定名称，推荐格式为：
  - `pvs_<核心方法>_<关键特性>`
  - `pvs_directional_occlusion_proxy_encoder_rvl_w042_full40`
  - `baseline_aabb_hzb`
- 禁止使用含糊名称，如 `new_model`、`test2`、`final_final`、`v6_recall_tmp`。
- 新实验输出目录必须独立，不能覆盖已有保留版本：
  - `neural_instance_culling/model/out/<experiment_name>`
  - `neural_instance_culling/benchmark/out/<experiment_name>`
  - `neural_instance_culling/dataset/out/<dataset_name>`
- 如果实验被废弃，必须清理默认 runner 引用、README 引用和文档中的主线描述。
- 如果只调阈值，不得命名成新架构；必须标注为同一 checkpoint 的 threshold calibration。

## 3. 资源依赖检查

每次训练、导出或评估前，必须检查依赖资源是否存在且语义正确。

### 训练前必须检查

- `dataset_meta.json` 是否存在，数据集格式是否符合当前脚本要求。
- `visible_ids` 与 `candidate_ids` 是否满足 `visible_ids ⊆ candidate_ids`。
- `visible_pixels.bin` 如果来自 rvcServer `component_weights`，必须按 `visible_weights` 解释，不能当作真实像素覆盖率。
- `runtimeVisibilityMeta.json` 的实例数量、AABB、`instance_to_glb` 是否与数据集一致。
- 如果训练 fixed-geo 模型，必须确认 `instance_geo_features_fp16.bin` 的 `numInstances`、`geoDim` 与模型配置一致。
- 如果训练 screen-grid/MVP 模型，必须确认 `mvp.bin` 存在且 pose 数量一致。
- 如果训练上下文化实例点云 / fixed-context 模型，必须确认上下文化特征表的 `numInstances`、`featureDim`、特征来源、是否使用 camera hash 先验、是否使用完整 AABB 投影特征都写入 meta。
- 长时间训练、采样和 benchmark 必须把 stdout/stderr 写入明确日志文件，例如 `train_stdout.log` 和 `train_stderr.log`；不能只依赖终端进度条。日志中必须定期输出 epoch、step、loss、主要指标、速度和 ETA，防止后台报错或卡死时 agent 只能傻等。
- 如果使用 viewcell / 后退扩大视锥，训练前必须检查扩大视锥是否覆盖位置扰动、yaw 抖动、pitch 抖动、采样 aspect 和安全余量；不能只用纵向 FOV 推导横向 FOV。改变相机扰动范围后必须重新计算包络并重建数据集。

### 导出前必须检查

- checkpoint 的 `config` 与导出脚本 schema 匹配。
- 前端运行时需要的资产全部存在；前端不需要的训练端资产不能误导出。
- 如果模型承诺前端不运行 Triplane/PointNet/implicit field，导出资产中不能包含运行时依赖这些模块的布局。
- 如果模型承诺前端只做 fixed contextual feature 查询，导出资产中不能包含运行时图传播、PointNet++、Triplane、动态边传播或完整 AABB 投影依赖，除非文档明确新增 schema 并完成前端 smoke。
- 导出目录必须写入清晰的 `model_meta.json` 或同等 meta 文件，说明 schema、阈值、输入维度、特征来源和适用数据集。

### 评估前必须检查

- 对比模型必须使用同一候选集合和同一 GT 口径。
- 正式 CSR test 评估必须默认遍历 test split 中全部唯一可见 pose/viewcell，测试数量必须写实际 split 大小，例如当前 `pose_csr_viewcell_back_camera_pvs_v1` 是 `684 unique test viewcells`。
- 不允许把正式 test 固定写成某个固定样本数。只有复现历史有放回抽样时，才可以写 `sampled-with-replacement`，并且必须明确它不是严格 test split 结论。
- `50 live` 必须注明是否为重新请求 rvcServer 得到的新 pose，不能混成数据集内 pose。
- 如果报告 weighted 指标，必须说明权重来源，并同时报告普通 precision / recall / F1 / Jaccard。

## 4. 指标准则

- PVS 不是普通均衡二分类任务，不能把裸 `precision`、`F1` 或裸 `candidate reduction = 1 - avg_pred / avg_candidate` 作为唯一主结论。裸 candidate reduction 会把正确剔除不可见实例 `TN` 和错误剔除可见实例 `FN` 混在一起；`FN` 是画面风险，不是效率。
- 当前主指标优先级：
  - 画面安全约束：pose-level / live-level `recall`、`weighted recall`、`image PER`、`miss pixel rate`。
  - 有效剔除：`useful cull = TN / candidate`，只在满足安全约束的阈值或工作点之间比较。
  - 错误剔除：`bad cull = FN / candidate` 和 `FN / GT`，必须同时报告，防止用少预测掩盖漏预测。
  - 资源效率：`avg pred`、`avg pred / avg GT`、`avg pred / avg candidate`、`GLB count reduction`、`GLB byte reduction`、预算内 GLB utility recall。
  - 运行成本：模型 forward latency、总调度 latency、runtime feature size、前端 Worker/WebGPU smoke latency。
- 辅助准确性指标必须保留：
  - `instance accuracy = (TP + TN) / candidate`，用于说明逐实例分类正确率。
  - `balanced accuracy = (recall + specificity) / 2`，用于避免普通 accuracy 被大量不可见候选抬高。
  - `precision`、`F1`、`Jaccard`，用于诊断误报/漏报权衡，但不再单独决定方案是否可作为主线。
- `weighted recall` 只能说明重要 GT 是否被找回，不惩罚 false positive；不能单独作为模型有效性的结论。
- 如果模型通过降低阈值提高 recall，必须同时报告 precision、accuracy、balanced accuracy、avg pred、avg GT、useful cull、bad cull 和 GLB 字节指标，避免用过量预测或错误剔除掩盖问题。
- 每个正式模型或方案训练、导出或评估完成后，必须在 `docs/current/` 或 `docs/evaluation/` 下新增或更新对应指标报告。报告必须说明指标口径、阈值选择、普通集合指标、weighted 指标、平均预测数量、有效剔除指标、错误剔除指标、前端性能指标和图像指标；如果某项指标未实现，必须明确写出“未实现”和原因。
- 图像 PER、miss pixel rate、wrong ID pixel rate 主要衡量漏预测造成的画面损失，不能单独证明模型剔除能力。剔除能力必须同时看 `useful cull`、`bad cull`、avg pred / avg candidate、GLB 下载数量或字节级 reduction。
- 报告中必须区分：
  - `pose precision/recall/accuracy`：每个 pose 单独算，再平均。
  - `agg precision/recall/accuracy`：跨所有 pose 合并 TP/FP/FN/TN 后再算。
  - `weighted recall`：按 `visible_weights` 统计 GT 被找回比例。
  - `specificity / negative recall`：不可见候选被正确剔除的比例，即 `TN / (TN + FP)`。

## 5. 文件与目录卫生

- 不要在根目录堆积临时 Markdown、临时脚本、日志或一次性结果。
- 根目录只允许保留全局项目说明类文件，例如 `AGENTS.md`。
- 当前 PVS 文档按类别放在：
  - `docs/current/`：当前主线、保留版本、正式指标报告。
  - `docs/evaluation/`：指标解释、图像 PER、weighted 指标和评价口径。
  - `docs/frontend/`：前端调度、Worker、WebGPU、部署和浏览器 smoke。
  - `docs/experiments/`：新模型计划、实验过程、失败分析和消融尝试。
- `docs/` 根目录只保留 `README.md` 索引；不要把新的 Markdown 直接放在根目录。
- 临时 benchmark 结果如果只是 smoke，完成后必须删除。
- 长期保留 benchmark 结果必须有清楚命名，并在文档里说明数据来源和运行命令。
- 不要把训练 raw、点云缓存、大模型权重误放入前端部署包。
- 不要把废弃版本继续留在默认模型列表或 README 中。

## 6. 文档记录要求

每次发生重要文件更改、架构变更或实验尝试，必须写入文档。

必须记录的情况：

- 新增或删除模型架构。
- 修改训练损失、阈值策略、候选生成方式或数据语义。
- 新采样数据集或重建 CSR 数据集。
- 导出前端资产。
- 与 rvcServer 进行正式对比。
- 删除旧版本、重命名版本或改变默认 runner。
- 完成一个正式模型的训练、阈值校准、benchmark、图像 PER 或前端部署 smoke。

记录内容必须包含：

- 日期。
- 变更目的。
- 修改文件。
- 依赖资源。
- 运行命令。
- 主要结果。
- 指标表和每个指标的中文含义，至少区分“画面损失指标”和“剔除效率指标”。
- 是否保留为主线。
- 后续风险或待验证事项。

文档应该写清楚“为什么这么改”，不能只写“做了什么”。

## 7. 代码修改准则

- 修改代码前先阅读相关实现，不凭记忆重写关键链路。
- 代码改动应优先小步、可验证、可回滚。
- 不要大面积重构与当前任务无关的文件。
- 不要删除用户或其他 agent 的未确认改动。
- 手工编辑文件时使用 patch 方式，避免不可审查的大段覆盖。
- 删除目录或文件前必须确认：
  - 是否被 runner 引用。
  - 是否被训练脚本默认参数引用。
  - 是否被前端资产或 README 引用。
  - 是否是唯一一份可复现实验结果。
- 如果发现依赖资源缺失、命名冲突或语义不确定，必须先记录并澄清，不要继续堆新实验。

## 8. 前端接入准则

- 前端默认路径只能指向已经通过 smoke 和同位姿对比的资产。
- 前端推理必须保持实例级输出 `componentModelList`，下载可以聚合到 GLB，但渲染过滤不能退化成 GLB 级。
- WebGPU / WebGL 初始化、shader 编译和权重解码的改动必须单独记录，并说明对启动时间和兼容性的影响。
- 不要在前端保留大量 debug log、临时开关或未使用 backend。
- 如果新增 runtime schema，必须保证旧 schema 要么明确兼容，要么明确移除并更新文档。
- 上下文化实例点云路线的前端首版必须保持“固定上下文化实例特征表 + 轻量视角查询头”；不能把 camera hash 作为唯一或主要可见性记忆来源。
- 完整 AABB 8 角点 MVP 投影默认不进入前端首版运行时。如果后续确实上线，必须只在 Worker/WebGPU 中对候选实例低频计算，禁止主线程逐帧或全量实例计算，并必须报告额外耗时。

## 9. 采样与数据集准则

- 新采样必须先写清楚 pose 分布目标，不能盲目全场均匀采样。
- 采样点必须避免建筑内部穿模，除非实验目标明确需要内部点。
- rvcServer 采样完成或失败后都必须清理服务进程。
- 数据集构建必须保留 raw 来源、采样脚本参数、split 规则和候选生成逻辑。
- 如果使用 live pose 评估，必须说明它不是 CSR replay。

## 10. 保留与删除准则

一个实验可以保留为主线，至少需要满足：

- 有清晰命名。
- 有可复现命令。
- 有同口径 benchmark。
- 有文档说明。
- 不破坏当前前端或默认 runner。

一个实验应该删除或降级为历史记录，如果：

- 指标明显弱于已有基线且没有新的解释价值。
- 依赖链路混乱，无法复现。
- 只靠降低阈值造成大量 false positive。
- 与当前研究方向不一致。
- 文件命名误导后续工作。

删除实验时必须同步清理：

- 训练脚本默认参数。
- benchmark 默认模型名。
- README 和 docs 引用。
- 前端资产引用。
- 旧 benchmark 输出。

## 11. 当前项目特定约束

- 当前保留模型名：
  - `baseline_aabb_hzb`
  - `pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best`
- 当前训练与导出输出保留在 `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40` 和 `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best_eval`，包括 `best.pt`、`last.pt`、epoch 快照、训练日志、导出特征和评测摘要；清理旧实验时不能删除这些结果。
- `pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best` 运行时读取离线固定实例特征表，不在前端运行 PointNet++、Graph U-Net、Triplane、dynamic-pool 或任何动态图传播。
- 当前前端资产默认路径为 `slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best`，并同步保留 `slm2viewer/assets/` 与已构建部署目录中的同名资产。
- 历史字段 `visible_pixels.bin` 当前按 `visible_weights` 处理，不能宣称是真实 pixel coverage。
- 后退扩大视锥候选上的 no-hash 主线相机输入必须参考 `Neural Visibility of Point Sets` 的视角条件化方式：以“当前相机到实例中心的单位视线方向 / ray direction”及轻量 ray-space 标量查询固定实例特征，不能把 raw camera xyz 或 raw world-space delta xyz 直接作为 visibility MLP 的主要输入。camera hash 只能作为消融或辅助，不得替代这种 view-ray 查询叙事。

## 12. 每次任务结束前检查清单

- 是否改动了默认模型列表。
- 是否改动了训练或导出默认路径。
- 是否新增了未记录的输出目录。
- 是否留下了 smoke 输出、临时日志或 pycache。
- 是否需要更新 `docs/`。
- 是否运行了最小静态检查或 smoke。
- 是否在最终回复中说明了未验证项。
