# 视线关系场与遮挡生存场正式实验协议

日期：2026-08-11
状态：正式 validation 已完成并冻结；三角形深度证据、8×3×40 训练结果、10,000 次 paired bootstrap 和路线判定均已登记。Color-ID 图像评价通过 NVIDIA 硬件 WebGL；WebGPU 数值 parity 通过软件适配器，但当前 WebGPU 适配器未通过 NVIDIA 硬件门，因此不报告硬件 WebGPU 延迟。

## 1. 研究对象

本实验在现有实例级可见性预测和 GLB 调度链路上增加三个可插拔部分：离线球面深度关系上下文、低秩单调遮挡生存场，以及安全约束效用损失。后端离线处理三角形级遮挡证据；浏览器只读取固定实例特征表，计算九维视线查询和小型共享网络，不执行点云编码、图传播、邻居查询、HZB 或 AABB 八角点投影。

正式实验名称为 `pvs_direction_depth_relation_survival_constrained_v1`。本实验不得覆盖 M4-v2、历史方向深度关系结果、默认 checkpoint 或默认前端资产。

## 2. 相机和数据口径

| 项目 | 固定值 | 说明 |
|---|---:|---|
| 模型查询/后退相机垂直视场角 | 66° | 采样、候选集合和模型输入统一使用 |
| 前端真实渲染垂直视场角 | 60° | 只用于图像显示和 Color-ID 图像评价 |
| 深度证据分辨率 | 320×180 | 固定，不随显示器分辨率变化 |
| 深度层数 | 6 | 首层为普通未剥离 Color-ID 层 |
| view-cell 子姿态 | 每个 view-cell 5 个 | 中心和空间覆盖代表位置，方向保持该 view-cell 方向 |

候选集合始终来自数据集保存的后退相机候选 CSR。真实可见集合来自对应数据集 GT，禁止将 GT 可见实例补入候选集合，禁止使用前端白名单修复，禁止在评价时重新构造候选集合。每个正式成员必须保存并校验：

- canonical PoseCSR 候选数量、顺序和字段语义；
- 重复 subpose 渲染顺序的候选数量、顺序和字段语义；
- validation pose 顺序；
- 实例 GT、可见权重和 `visible_ids ⊆ candidate_ids` 检查结果。

代表 subpose 的渲染记录可以重复引用同一个 canonical PoseCSR 行，但不能把这些 subpose 的 ID 或 GT 做并集后写回候选文件。关系统计把每个实际渲染 subpose 当作独立观察。

## 3. 离线三角形遮挡证据

Chrome 通过 Three.js WebGL Color-ID 渲染所有实例和 GLB。每个子姿态依次执行深度剥离，输出实例 ID 和线性深度。首层先以 `peelEnabled=false` 渲染，再重复一次相同的未剥离 Color-ID pass；两次 RGBA 缓冲必须逐字节一致。所有相邻有效层的深度必须严格递增，背景像素不参与该检查。

正式采样必须满足以下硬件门：

- Chrome 使用 `--enable-gpu --enable-webgl --use-angle=vulkan`；
- 正式硬件模式使用 `--disable-software-rasterizer`；
- 页面后端必须报告 NVIDIA/ANGLE 硬件 renderer；
- 同一执行窗口必须保存 `nvidia-smi` 和 `nvidia-smi pmon` 的 before/during 证据；
- 检测到 SwiftShader、llvmpipe、softpipe、swrast、软件后端或证据缺失时，缓存只能作为 smoke，不能进入正式训练。

相邻不同实例的层形成有向关系“前方实例遮挡后方实例”。关系权重由像素支持和线性深度间隔得到。对于深度剥离未覆盖的候选目标，使用真实 GLB 表面点、实例变换和当前相机进行离线深度比较补全；AABB 重叠不作为正式遮挡监督。补全不改变候选集合和 GT。fallback 关系必须按实际渲染子姿态保存 `renderPoseId/source/target/pixelCount/gap` 记录，后续关系构建器再按每个子姿态合并，不能把已聚合的 top-k 表误当成独立 pose 观测。

深度单位校验：表面点 fallback 的 `gap` 是线性深度缓冲中的归一化差值，关系构建时先将目标实例中心距离按同一相机远裁剪尺度归一化，再计算相对深度差；禁止将归一化差值直接除以米制世界距离。该检查用于避免 fallback 关系被错误压入最浅深度层。

## 4. 固定实例表和模型输入

每个实例离线保存：

1. 96 维 PointNet++ 几何表征；
2. 4×8 的球面方向上下文系数，共 32 维；
3. 4×7 的遮挡生存场系数，共 28 维。

运行时固定表总维度为 156。相机查询只使用三维单位视线方向、归一化距离、相机前向夹角、屏幕水平/垂直位置和水平/垂直角尺度等九个直接标量。方向基函数为共享的 `3→16→4` 小网络；生存场输出保持距离方向上的单调非增生存概率，并派生无阻挡概率、预计遮挡深度、离散度和局部变化率等语义量。

上下文编码器在离线阶段依次聚合来源实例、深度层和十二个球面方向。前端不保留关系表、不查询邻居、不运行该编码器。可见性头、视觉效用头和 GLB 下载头共享轻量基础表征；实例输出用于渲染过滤，GLB 输出用于下载排序。

## 5. 损失和训练协议

正式比较的可见性损失为：

- `rvl_strong_v2`：冻结的当前控制组；
- 安全约束效用损失：平衡 BCE、软误报项和超额 GLB 字节项，并由 pose miss、weighted miss 和 weighted miss 的 95% CVaR 三个非负对偶变量施加约束。

安全约束目标为：

```text
E[1 - pose_recall] <= 0.05
E[1 - weighted_recall] <= 0.01
CVaR_0.95(1 - weighted_recall_per_pose) <= 0.03
```

对偶变量初值为零，更新学习率为 0.05，范围为 `[0, 20]`。资源混合系数使用 pilot 扫描 `gamma ∈ {0.25, 0.50, 0.75}`，只能根据 validation 的安全工作点选择。遮挡生存场使用深度观察的事件/右删失似然；未观察到遮挡物的射线作为右删失样本。

pilot 使用单 seed、12 epoch，目的只有数值稳定性检查和 gamma 粗选。正式成员从头训练 40 epoch，固定 seed 为 `20260801`、`20260802`、`20260803`，四张 GPU 并行。训练日志必须分别保存 stdout/stderr，并记录 epoch、loss、主要安全统计和速度。

## 6. 阈值和数据分割

每个 checkpoint 使用自己的 calibration split 扫描并冻结阈值。安全工作点必须同时满足：

- `weighted recall > 0.99`；
- weighted recall 的单侧 95% 置信下界 `> 0.99`；
- 普通 pose recall 作为诊断指标报告，不替代 weighted recall 安全门。

没有合格安全工作点的模型只能保存诊断阈值，必须标记为 `no_qualified_safety_workpoint`，不能进入安全路线排名。validation 只用于比较冻结工作点；test 在模型、checkpoint、阈值和资产全部冻结后才允许读取一次，不能参与任何阈值或 gamma 选择。

## 7. 评价和统计

对每个 pose 使用候选集合 `C`、GT 可见集合 `G` 和预测集合 `P` 计算：

```text
TP = P ∩ G
FP = P - G
FN = G - P
TN = C - (P ∪ G)
```

每个工作点必须同时报告 pose-level 宏平均和全部 pose 合并的 aggregate 结果：recall、weighted recall、precision、F1、Jaccard、accuracy、balanced accuracy、specificity、useful cull、bad cull、平均预测数、FP/FN/TN、预测/候选、预测/GT、GLB 数量和字节削减、特征表大小及前向延迟。Color-ID 图像评价另外报告 miss-pixel、wrong-ID pixel、extra-pixel 和 p95 miss-pixel；尚未生成时明确标记为未实现。

Validation 使用至少 10,000 次 paired bootstrap。每次重采样先按 seed 聚类，再在抽中的 seed 内对 pose 重采样；比较成员使用同一 pose、同一候选集合和同一 GT。对每个指标输出差值、95% 置信区间、方向和是否跨零。test 不参与 bootstrap 的模型选择。

核心 2×2×2 变体为：上下文关闭/开启、生存场关闭/开启、RVL/安全约束损失。补充对照包括 AABB 投影证据、三角形深度剥离证据、192 维旧代理、同容量无单调 28 维代理、32/64 维上下文以及 Fourier/九维 ray 输入。

## 8. 路线保留条件

一项创新只有在安全工作点成立、weighted recall 没有稳定恶化，并且 precision、balanced accuracy、F1、useful cull 或资源成本至少一项的 paired 置信区间稳定改善时，才可写为预测贡献。只提高 useful cull、同时降低 recall 或提高 bad cull 的变体不能判定为更好。

最终神经资产目标不超过 7 MiB；同候选 CUDA p95 不高于当前约 4.52 ms；前端不得新增在线邻居查询。真实移动设备可用后，执行既有 10k 候选 p95 小于 50 ms 的测试协议。未满足条件的模块降级为失败消融或辅助表征，不修改默认模型。

## 9. 可复现入口

以下命令是协议入口，正式运行前必须将输出路径替换为本实验独立目录：

```bash
node neural_instance_culling/benchmark/build_triangle_depth_layer_evidence_browser.mjs \
  --manifest <shard-manifest.json> \
  --output <shard-dir>/triangle_depth.bin \
  --chrome-exe /usr/bin/google-chrome \
  --width 320 --height 180 --max-layers 6 \
  --require-hardware-gpu

conda run -n slm_pvs python \
  neural_instance_culling/dataset/merge_triangle_depth_layer_caches.py \
  --shard-root <formal-shard-root> \
  --manifest <full-render-manifest.json> \
  --output-dir <merged-cache>

conda run -n slm_pvs python \
  neural_instance_culling/dataset/build_triangle_depth_layer_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --layer-cache-dir <merged-cache> \
  --output-dir <triangle-evidence> \
  --splits train --surface-point-fallback \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin

conda run -n slm_pvs python \
  neural_instance_culling/dataset/build_ray_context_relation_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --layer-cache-dir <merged-cache> \
  --survival-evidence-dir <triangle-evidence-with-surface-fallback> \
  --output-dir <relation-evidence-v3> \
  --splits train --require-surface-fallback
```

正式输出目录、manifest、候选摘要、GPU 证据和运行日志必须和模型/benchmark 结果一同保存；不把 smoke 或失败 partial 当作正式结果。

## 10. 本次正式冻结记录

- 正式输出目录：`neural_instance_culling/benchmark/out/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811/`。该目录替代了上下文关闭误设为全零输入的旧 formal 输出；旧目录不再作为可复现实验结果保留。
- 矩阵规模：8 个完整三因素变体、3 个 seed（`20260801/02/03`）、213 个 validation pose；所有成员的候选集合、实例 GT 和 pose 顺序一致。
- 统计：按 seed 聚类、seed 内按 pose 重采样的 paired bootstrap，共 10,000 次；test split 未读取，阈值和 gamma 未由 test 选择。
- 图像硬件证据：Color-ID 图像评价为正式 NVIDIA 硬件 WebGL 结果。
- WebGPU parity：8 个 parity case 的软件数值校验通过，最大绝对误差 `3.814697265625e-06`，最大相对误差 `3.7417979910969734e-05`；capture 报告的适配器为 SwiftShader，故 `formalReady=false`，不能把该结果写成 NVIDIA WebGPU 硬件性能。
- 2026-08-12 使用 Playwright 无头 Chrome 进行了独立 Vulkan 参数复核。入口现在显式加入 `--ozone-platform=headless` 和 `--ozone-override-screen-size=1280,720`，并记录启动参数、`VK_ICD_FILENAMES` 和 Chrome DevTools `SystemInfo.getInfo`。在当前 Chrome `146.0.7680.177`、NVIDIA RTX A6000 驱动 `535.183.01` 下，当前 Vulkan、headless ozone、DefaultANGLEVulkan 和 Skia/Vulkan 组合均仍由 `navigator.gpu.requestAdapter({powerPreference: "high-performance"})` 返回 `vendor=google, architecture=swiftshader`；`--disable-vulkan-surface` 则无法创建适配器。CDP 同时显示 WebGL renderer 为 `ANGLE (NVIDIA, Vulkan ...)`，但 `hardwareSupportsVulkan=false`、`webgpu_on_vk_via_gl_interop=enabled`。因此 Playwright 无头模式可以作为正式 WebGL 采样和软件 WebGPU 数值 parity 的统一入口，但当前环境没有通过 WebGPU NVIDIA 硬件门；`nvidia-smi/pmon` 中的 Chrome 进程只作为旁路证据，不能改变该结论。独立软件 parity 重新校验通过，最大绝对误差 `2.86102294921875e-06`、最大相对误差 `6.78728520142613e-06`，未写入正式硬件结果目录。
- 路线判定：`degrade_to_auxiliary_or_failed_ablation`。在 weighted recall 安全门、分类/剔除收益和资源收益的联合规则下，没有因素获得稳定的正式路线保留资格；方向上下文、生存场和安全约束损失均只作为分析结果保留，不替换默认 checkpoint、默认阈值或前端资产。
- validation 冻结选择：`context_off_survival_off_safety`、seed `20260802`、阈值 `0.20000000298023224`；该选择只用于本实验的一次性 test 授权，不改变当前默认模型。

## 11. 2026-08-12 补充机制矩阵

为区分上下文宽度、ray 编码、单调参数化和遮挡证据来源的独立影响，另行执行了补充矩阵。补充矩阵使用独立名称和输出目录，不覆盖本协议第 10 节的核心 8×3 formal 结果：

| 变体 | 固定表/查询变化 | 对照目的 |
|---|---|---|
| `triangle_context32_direct9_monotone` | 三角形关系、32 维上下文、9 维 ray、单调生存 | 补充参考 |
| `triangle_context64_direct9_monotone` | 仅上下文宽度增至 64 维 | 上下文容量对照 |
| `triangle_context32_fourier117_monotone` | 仅 ray 查询恢复为历史 117 维 Fourier 输入 | 输入编码上界/成本对照 |
| `triangle_context32_direct9_unconstrained28` | 仅取消 28 维生存参数的单调构造 | 单调约束对照 |
| `aabb_context32_direct9_monotone` | 仅将关系证据替换为 AABB 投影关系 | 证据来源对照 |

每个变体使用 `20260801/02/03` 三个 seed、40 epoch 和同一 213 个 validation pose。每个 checkpoint 从自己的 calibration split 冻结阈值；安全门仍只看 weighted recall 及其 calibration 单侧置信下界，pose recall 作为诊断。所有成员的候选集合、逐 pose 候选顺序、GT 和 Color-ID 图像评价口径一致，test split 未读取。

正式结果文件：

- manifest：`neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_matrix_manifest.json`；
- summary：`neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_summary.json`；
- validator：`neural_instance_culling/benchmark/validate_ray_context_survival_owrb_supplement.py`；
- schema validation：`neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812/formal_supplement_schema_validation.json`；
- 明细报告：[`pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md`](../evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md)；
- 结论报告：[`pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md`](../evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md)。

补充矩阵不生成核心 2×2×2 因子效应；只对预注册的四组机制差值执行 10,000 次 paired bootstrap。Playwright 无头 Chrome 的硬件 GPU 规则保持不变：Color-ID 图像评价必须通过 NVIDIA Vulkan/ANGLE WebGL 门；WebGPU 数值 parity 单独记录，当前软件适配器结果不能宣称硬件 WebGPU 性能。
