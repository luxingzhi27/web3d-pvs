# AGENTS.md

本文件定义本仓库中 Codex / 其他 agent 的工作准则。目标是避免无约束实验、重复命名、错误复用资源和文档散乱，保证神经剔除实验能够被复现、比较和交接。

## 1. 总体原则

- 所有工作必须优先保持当前可运行版本稳定；新实验不能破坏已有前端、训练、采样和 benchmark 链路。
- 不允许随意创建大量临时版本名、临时输出目录或一次性脚本；新方案必须先定义清楚目标、输入、输出、评估指标和保留/删除条件。
- 不允许用旧实验名称复用新含义。例如同一个目录名不能先表示 PointNet++ 原始版，后又表示高召回版。
- 不允许把失败实验伪装成当前主线；失败实验可以记录，但不能污染默认 runner、README 或前端默认资产。
- 不允许用 sample-level 二分类指标替代 pose-level 集合指标作为主结论。
- 当前主要验收目标是：以 `weighted recall` 及其置信下界作为画面安全主门，在安全约束下最大化有效剔除和资源节省。普通 pose recall 必须报告，用于诊断均匀实例覆盖，但不能单独否决或证明画面安全；除非实验协议另有登记，不得把 pose recall `0.95` 当作 weighted recall 的替代门。满足 weighted recall 安全约束后，优先比较 `useful cull = TN / candidate`、`bad cull = FN / candidate`、平均预测数量、GLB 字节削减和前端延迟。普通 precision、F1、逐实例 accuracy 和 balanced accuracy 必须报告，但不能单独作为主结论。
- `precision` 受候选集合中的正负样本比例影响：在真正率和假正率相同的情况下，加入更多不可见候选仍会增加 FP 并降低 precision。因此跨候选规模、跨场景或跨候选生成策略比较时，必须同时报告平均候选数、GT/候选比例、specificity、instance accuracy 和 balanced accuracy。普通 accuracy 可能被大量 TN 抬高；`balanced accuracy = (recall + specificity) / 2` 对正负样本比例更稳健，是安全门之后的重要分类参考，不能只在附录中出现。
- 新优化模型必须区分“安全工作点”和“分布健康诊断”。安全工作点只由 checkpoint 自己的 calibration 阈值是否满足 weighted recall 及其置信下界决定；固定概率边界 `0.5` 或 `[0.4, 0.6]` 不能作为额外硬门。低阈值必须结合正负尾部、阈值扰动和校准误差解释；禁止用 bias、temperature 或其他后处理把阈值移动到中间后伪装成模型效果提升。
- 不允许把 `recall_high` 当作额外变体后缀来逃避主目标；每个正式 PVS 实验默认就必须按高召回目标训练和选 checkpoint。若一个模型需要降低召回才能得到好看的 F1，它不能作为合格主线版本。历史路径或旧 benchmark 中出现的 PointNet / Triplane / dynamic-pool 命名只作为已删除旧实验理解，不能作为当前默认路径。
- 向用户解释模型、训练流程或前端推理时，必须优先使用可读的中文概念和数据流描述；内部类名、函数名、变量名只能作为定位参考放在括号中，不能用一串代码英文名代替解释。每个新名词都要说明“它输入什么、输出什么、为什么需要它”。
- 发现当前安全口径、代码实现或文档结论错误时，必须直接修正主线并删除错误的旧代码、旧文档和旧结果；不得增加兼容开关、双重默认路径、别名脚本或“旧逻辑仍可用”的过渡层。只有明确标注且仍有复现实验价值的原始数据或模型权重可以保留。
- 禁止 agent 为日常代码验证、临时 smoke、checkpoint、运行包或数据文件手工计算、对比和汇报哈希，也不得把哈希确认写成计划步骤、验收条件或结果结论。程序内部为防止数据、checkpoint 和运行资产误配而自动维护的一致性字段可以保留，但执行与汇报必须聚焦 schema、shape、split、候选/GT 语义、数值有限性、测试结果和实际指标。
- 已登记的完整训练计划不能被 pilot、参数扫描或中间指标门控提前取消。快速扫描必须始终选出一个相对最优配置；即使没有配置达到论文安全标准，也必须按计划完成完整组合与核心消融长训，并把不合格结果如实标记。只有代码无法运行、数值非有限、CUDA OOM 或外部硬件故障可以暂时中断执行；修复后应从未完成成员继续，而不是缩减矩阵。

## 1.1 当前运行环境

- 当前项目已从 Windows / AMD ROCm 机器迁移到 Linux，训练优先使用 NVIDIA CUDA。
- 新训练环境必须使用 conda 创建和记录，例如 `conda create -n slm_pvs python=3.11`；不要再新增旧 ROCm 环境或 Windows PowerShell 反引号命令作为当前推荐命令。
- 文档和新脚本中的可复现命令默认使用 Linux shell 口径，例如 `conda run -n slm_pvs python ... --device cuda`。
- 如果必须保留历史 Windows/ROCm 命令，只能放在历史说明中并明确标注“历史迁移前命令，不作为当前推荐运行方式”。
- 长时间训练、导出和 benchmark 仍必须写入明确 stdout/stderr 日志；Linux 下推荐使用 `> train_stdout.log 2> train_stderr.log` 或等价的日志方案。
- 正式浏览器采样、实例级 Color-ID 图像评价和三角形 HZB 光栅化必须使用 Chrome 的硬件 GPU 路径，默认开启 `--use-angle=vulkan` 与 `--disable-software-rasterizer`，并保存页面 `gpuBackend`、`gpuGate` 以及同一执行窗口的 `nvidia-smi`/`pmon` 证据。检测到 SwiftShader、llvmpipe、softpipe、swrast 或无法确认后端时必须失败；只有显式 `--allow-software-gpu` 的小规模语义调试可以使用软件后端，不能进入正式数据集、GPU 性能或移动端性能结论。详细政策见 `docs/current/hardware_gpu_execution_policy.md`。
- WebGL/ANGLE 与 WebGPU adapter 是两条独立的硬件证据链：WebGL 回报 NVIDIA 不能替代 WebGPU adapter 核验，`nvidia-smi` 中出现 Chrome 进程也不能替代被测 API 的后端字段。凡是 WebGPU/WGSL 性能实验都必须单独读取 adapter；若 adapter 回报 SwiftShader 或其他软件后端，必须停止硬件性能汇总。
- Color-ID 正式采样还必须保留分片旁的 `*.jsonl.gpu_evidence.json` 和 view-cell 总目录的 `gpu_execution_summary.json`；其中要有页面 `gpuBackend`/`gpuGate`、Chrome 参数及同一窗口的 `nvidia-smi`/`pmon`。没有这些证据的历史 JSONL 不能追认为硬件采样。

### 浏览器 Vulkan 硬件路径（采样与 WebGPU）

以下参数来自当前实际入口代码，后续新增脚本必须沿用这套口径，不能凭记忆删减或改成软件后端：

- Three.js Color-ID 采样入口 `neural_instance_culling/sampler/run_sampler.mjs` 使用系统 Chrome/Chromium，并传入：
  `--disable-dev-shm-usage`、`--ignore-gpu-blocklist`、`--enable-gpu`、`--enable-webgl`、`--use-angle=vulkan`、`--enable-accelerated-2d-canvas`、`--enable-zero-copy`。正式采样必须再传 `--require-hardware-gpu`，脚本据此追加 `--disable-software-rasterizer`。
- view-cell 采样入口 `run_scene_viewcell_colorid_sampling.mjs` 不自行启动渲染器，而是为每个分片调用上述 `run_sampler.mjs`，强制传递 `--require-hardware-gpu`；所有分片都必须生成 GPU evidence，最后由 `gpu_execution_summary.json` 汇总检查。
- 当前 V4 WebGPU/WGSL 页面采集入口 `slm2viewer/scripts/capture_v4_frontend_parity.mjs` 使用：
  `--headless=new`、`--ozone-platform=headless`、`--ozone-override-screen-size=1280,720`、`--no-sandbox`、`--no-first-run`、`--disable-dev-shm-usage`、`--disable-background-networking`、`--disable-extensions`、`--enable-gpu`、`--enable-unsafe-webgpu`、`--enable-webgpu`、`--enable-webgl`、`--enable-features=Vulkan`、`--use-vulkan`、`--use-angle=vulkan`、`--enable-accelerated-2d-canvas`、`--enable-zero-copy`、`--ignore-gpu-blocklist` 和 `--disable-gpu-sandbox`；正式模式再追加 `--disable-software-rasterizer`。脚本还记录 `VK_ICD_FILENAMES` 和 Chrome DevTools `SystemInfo.getInfo`，用于区分 WebGL ANGLE 后端与 WebGPU adapter 后端。`--enable-unsafe-webgpu`、`--enable-webgpu`、`--enable-features=Vulkan` 和 `--use-vulkan` 是 WebGPU 路径的额外参数，不能误加到只做 WebGL 采样的脚本中作为替代证据。
- WebGPU 页面必须调用 `navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })`，保存 `adapter.info.vendor`、`architecture`、`device`、`description`，并同时保存 WebGL renderer 作为辅助诊断。适配器或 renderer 文本含 `SwiftShader`、`llvmpipe`、`softpipe`、`swrast`、`software` 或为空时，硬件门失败。
- 硬件门不是由命令退出码决定：采样和 WebGPU parity 都必须保存 API 后端字段、Chrome 启动参数，以及浏览器执行窗口的 `nvidia-smi` 和 `nvidia-smi pmon` before/during/after 证据。WebGL 的 NVIDIA/ANGLE 证据只能证明 WebGL 光栅化硬件路径，不能证明 WebGPU adapter 使用 NVIDIA 硬件。

正式入口示例：

```bash
node neural_instance_culling/sampler/run_sampler.mjs \
  --assets-dir <scene-assets> \
  --pose-plan <pose-plan.jsonl> \
  --output <formal-output.jsonl> \
  --require-hardware-gpu

node slm2viewer/scripts/capture_v4_frontend_parity.mjs \
  --viewer-dir slm2viewer/public \
  --out <v4-parity-capture.json> \
  --require-hardware-gpu

conda run -n slm_pvs python slm2viewer/scripts/verify_v4_frontend_parity.py \
  --checkpoint <best_safe.pt> \
  --asset-dir slm2viewer/assets/neural_instance_culling/pvs_mainline_v4 \
  --capture <v4-parity-capture.json>
```

若 WebGPU 当前只能返回 SwiftShader，必须记录为“WebGPU 软件数值 parity 通过、硬件门失败”，不能写成硬件 WebGPU 延迟或移动端性能结果；不得通过删除 `--disable-software-rasterizer`、加入 SwiftShader 参数或复用 WebGL 证据绕过该门。

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

### 当前优化阶段覆盖（2026-08-23）

- 当前论文模型的训练协议是 `docs/experiments/pvs_mainline_training_2026-08-21.md`，统一实验前缀为 `pvs_v4_integrated_visibility_mainline_v1`。固定架构为分层遮挡关系先验与逐实例校准生存场、视点区域矩包络频谱查询，以及逐 pose 平衡分类、单侧 RVL 加权召回保护和共享困难边界对比组成的综合可见性损失。
- 本轮只优化实例可见性。视觉效用、下载优先级、GLB 字节预算和资源调度损失必须为零，不参与参数扫描排名；训练期对比投影头不得进入运行时导出。旧 108 维尾部分离器、冻结主干 refinement 和完整旧 RVL 叠加不再是执行入口。
- 主实验固定使用 `5926 train / 659 calibration / 730 validation / 684 test`；旧 684 test 不变。每个 checkpoint 只用自己的 calibration 冻结阈值，validation 比较配置，test 在模型和阈值全部冻结后读取一次。
- 已完成八组单种子快速扫描、配置复核、完整模型与四个核心消融的三种子 `40 epoch × 900 step` 从头长训。正式评价入口是 `neural_instance_culling/benchmark/reaudit_pvs.py`，结论见 `docs/evaluation/pvs_mainline_validation_2026-08-23.md`。
- 本阶段暂不把 `bad cull` 置信区间上界作为路线否决条件，但仍必须报告 `bad cull`、漏检数量和图像级漏检指标；不得用减少预测数量掩盖画面风险。
- 画面安全主门仍是每个 checkpoint 在 calibration 上冻结的 `weighted recall > 0.99` 及其单侧 95% 置信下界大于 `0.99`。普通 pose recall 只作诊断。
- 安全阈值的位置只作分布健康诊断，不作固定 `p=0.5` 硬门。必须同时记录安全阈值区间、阈值扰动稳定性、正样本加权 q01/q005、负样本 q99/q99.5、logit 间隔、Brier/ECE 和可靠性图；低阈值本身不能否决模型，因为 bias 或温度缩放可以移动概率阈值而不改变排序。不得用这种后处理伪装模型改进，主线资格仍由 calibration 的 weighted recall 安全门，以及同一安全门下的 precision、accuracy、balanced accuracy、specificity、useful cull、图像和资源指标共同决定。
- 如果启用按最差 pose 加权的锚点风险项，训练批次必须包含至少两个 pose；单 pose 批次只能报告普通锚点项，不能宣称完成尾部风险约束。关系系数反向传播的显存峰值必须单独记录，不能为了尾部统计扩大到超出单卡预算。
- 2026-08-09 以前的 M4/M5/M6 路线结论属于历史记录，不回写、不改名；后续新实验不得继续引用旧路线名称作为默认主线。

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
- 口径修正不建立兼容版本：先确认新口径的输入和输出，再原地替换主线入口；旧口径产生的错误汇总、路线判定、阈值清单和报告必须删除，不能继续被 runner 或 README 引用。

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
- 当前 V4 神经运行时后端顺序固定为 `WebGPURenderer 共享 GPUDevice 的页面异步 WebGPU 查询 -> WebGL2 backend + Worker WASM SIMD`。两条路径都必须完整执行后退 `66°` 视锥 AABB 候选、V4 模型、冻结阈值、真实 `60°` 视锥过滤、实例编号压缩和按 GLB 的最高分聚合；AABB 不能直接决定神经可见集合。WebGPU 普通运行只回读最终实例编号和压缩后的 GLB 队列；WASM 的特征表、权重和关系结果必须初始化一次后常驻线性内存，每个 pose 只允许一次批量模型调用并复制最终编号与队列。不得恢复纯 JavaScript CPU 神经计算、主线程全实例扫描或 WebGPU 失败后的隐式 AABB fallback；无神经权重场景使用独立的显式 AABB 视锥模式。逐候选编号和概率只允许在显式 parity 调试模式输出。
- Three.js 主渲染器统一使用 `WebGPURenderer`。有硬件 WebGPU adapter 时，渲染和 PVS 查询共享同一个 `GPUDevice`；没有硬件 adapter、adapter 为 SwiftShader/llvmpipe 等软件实现或初始化失败时，同一 Viewer 使用 Three.js WebGL2 backend，并把神经查询放入 Worker WASM SIMD。背景、GTAO 和 SMAA 使用 TSL/NodeMaterial 定义，由 WebGPU 和 WebGL2 backend 分别编译；不得恢复 WebGL 专用 `EffectComposer`、N8AO 或 RawShaderMaterial 背景。当前仍需把最终实例 ID 回读给资源调度和稠密实例槽位，不能宣称已经实现渲染着色器零回读可见性位图。
- GLB 资源调度只能由 `GlbResourceScheduler` 的单一状态机负责。Worker 只输出后退 66° 模型可见集合、真实 60° 最终集合和低分预取集合；真实 60° 集合必须全部进入 `urgent`，不得因队列长度上限降级为预取。其余模型可见项进入 `warm`，仅高于预取阈值的低分项进入 `speculative`。每个 GLB 只能处于 queued/fetching/parsing/mounting/resident/failed 之一；进入和离开真实视锥必须对称升降级，下载、解析、挂载和重试完成事件必须直接推进管线，不得依赖连续 RAF 或重新维护并行的立即/预取数组。

## 9. 采样与数据集准则

- 新采样必须先写清楚 pose 分布目标，不能盲目全场均匀采样。
- View-cell PVS 的运行契约必须保持唯一：多个同朝向 `subpose` 只用于离线采样、保守 GT/候选并集构造和图像评价；浏览器不得展开这些 `subpose` 或对一个 view-cell 多次运行网络。前端以当前相机建立预测锚点，构造一次后退扩展候选并执行一次批量查询，输出该 view-cell 内任一合法位置可能可见的实例并集；结果只在相机仍处于已登记的空间和方向范围内复用。
- 只在 view-cell 边缘某个 `subpose` 可见的实例属于合法区域正例，不能标记为标签歧义、采样噪声或 false positive。正式负例必须在该 view-cell 的全部成功 `subpose` 中均不可见。
- View-cell 半径、形状、朝向范围、前端位置/角度复用门限、后退距离、模型 FOV 和真实渲染 FOV 是同一保守契约。修改其中任一项都必须先验证新候选仍覆盖整个 cell，并重建不再匹配的数据集；不能通过增加在线 `subpose` 查询或降低阈值掩盖口径错误。
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

- 当前 HKUST 前端只允许加载 `pvs_mainline_v4`，运行 schema 为 `pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4`。旧方向代理模型和 `baseline_aabb_hzb` 可以作为历史 benchmark 保留，但不能再出现在前端兼容分支、默认资产映射或部署包中。
- 当前 HKUST 前端 checkpoint 为 `pvs_v4_integrated_visibility_mainline_v1_20260821/formal40_s02_ablation_without_contrastive_separation_s02_guard030_sep020_mix025_seed20260802_e40/best_safe.pt`，导出选中 epoch 36，calibration 阈值为 `0.6800000071525574`。
- 当前默认模型运行时读取 `96` 维几何和 `28` 维融合生存场组成的 `124` 维离线固定实例特征表，不在前端运行 PointNet++、Graph U-Net、分层关系网络、Triplane、dynamic-pool 或任何动态图传播。
- 当前 HKUST 前端资产路径为 `slm2viewer/assets/neural_instance_culling/pvs_mainline_v4`，生产构建同步到 `slm2viewer/public/assets/neural_instance_culling/pvs_mainline_v4`。其他场景没有匹配 V4 权重时必须使用实例 AABB 视锥模式，不得复用 HKUST 权重或回退旧神经模型。
- 必须保留 2026-08-11 修正正式矩阵、2026-08-12 Fourier 补充矩阵的 model/benchmark 输出，以及 `neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_v1_subpose5_20260811_directchrome` 三角形深度层硬件缓存；后者是新分层关系网络构建 train-only 遮挡关系 CSR 的数据依赖。
- 当前论文模型唯一训练主线由共享分层遮挡关系先验与逐实例校准生存场、视点区域矩包络频谱查询和综合可见性损失组成。计划、runner、评价和前缀分别为 `docs/experiments/pvs_mainline_training_2026-08-21.md`、`neural_instance_culling/benchmark/run_pvs.py`、`neural_instance_culling/benchmark/reaudit_pvs.py` 与 `pvs_v4_integrated_visibility_mainline_v1`；HKUST 当前前端已使用该主线的冻结 V4 导出。
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
