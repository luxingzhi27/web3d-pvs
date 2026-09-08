# IFCBench 稀疏三角形遮挡关系流水线改造

日期：2026-09-02

## 目的

当前正式深度剥离协议把每个训练 subpose 的六层实例 ID 和线性深度完整写入磁盘。IFCBench 的 78,588 个渲染 pose 因此产生约 202.36 GiB 稠密缓存，随后关系构建器还要单线程重复扫描这些数据并用 Python 字典、集合和列表进行全局聚合。该实现语义正确，但存储和构建复杂度不适合作为当前主线。

本次改造保持以下训练语义不变：

- 模型输入 FOV 为 66 度；
- 每个 view-cell 的代表 subpose 独立渲染；
- 第一层必须和独立的普通 Color-ID 渲染逐像素一致；
- 遮挡边只来自相邻三角形深度层中不同实例的严格正深度差；
- 所有关系端点必须属于该 source pose 已存储的原生后退视锥候选；
- 生存场监督同时保留首层可见右删失记录和后层遮挡事件；
- 关系与生存监督只读取 train split，不补入 GT 可见实例。

## 新流水线

### 分片渲染与即时稀疏化

Chrome 仍使用 NVIDIA Vulkan/ANGLE 硬件路径生成六层深度剥离结果。一个分片完成后，立即在该分片内完成：

1. 线性轴向深度到真实相机射线距离的转换；
2. 相邻层实例对、像素支持、深度间隔均值和方差的提取；
3. 目标实例方向 bin、深度 shell 和相对深度的计算；
4. 首层可见右删失记录与后层遮挡事件的提取；
5. 每个 subpose 内生存监督权重归一化。

稀疏分片完成 schema、数值、候选边界、pose 覆盖和硬件证据检查后，删除该分片的稠密 ID/depth 文件。失败时保留稠密文件用于重试。正式缓存只长期保留稀疏关系行、生存观察、pose 身份、元数据、日志和 GPU evidence。

### 全局向量化归并

最终关系构建不再逐 pose 读取像素，也不再构造全局 Python 对象图。它按固定结构化数组执行：

- 按目标实例、方向、深度层和来源实例排序；
- 用分段归约计算像素支持、pose 支持和深度间隔矩；
- 在每个目标方向深度单元内选择 top-k 遮挡来源；
- 直接写出关系 CSR、生存观察和层级分组。

每个 subpose 的生存权重在分片阶段已经归一化，禁止再次用“逐 subpose 扫描完整观察数组”的二次复杂度实现。

## 输入与输出

输入：

- 已审计的 `triangle-depth-layer-render-manifest-v2`；
- PoseCSR 数据集；
- `runtimeVisibilityMeta.json`；
- 每个分片的硬件 WebGL 深度剥离结果及 GPU evidence。

分片输出 schema：`triangle-depth-relation-sparse-shard-v1`。

最终输出保持训练端需要的 `pvs-viewcell-train-observed-relation-csr-v3`，不改变 V4 模型运行时 schema、checkpoint 或前端资产。

## 性能目标

- 全场长期缓存从 202.36 GiB 降到数 GiB以内；`shard_00` 正式数据实测由 4.7 GiB 降为 77 MiB，按 44 个分片线性估算约 3.3 GiB。
- 正式运行峰值稠密空间约为“并发分片数 × 单分片约 4.7 GiB”，每个分片稀疏化后立即释放。
- 现有四并发硬件渲染实测约 3.18 小时；`shard_00` 的 1,787 pose 稀疏提取、关系矩聚合和 zstd 压缩实测 113.13 秒。四并发全场稀疏化与最终归并预计 0.5 至 1.5 小时，总时间预计 3.5 至 5 小时。
- 避免当前约 8 小时以上的单进程全局扫描和不可预测的单线程后处理。

## 验证与删除条件

必须通过：

- 小型固定数组上稠密路径与稀疏路径的遮挡边、深度矩和生存记录等价；
- 第一层一致性检查数等于渲染 pose 数，mismatch 为零；
- 每个关系端点属于原生候选集合；
- 每个非空 subpose 的生存权重和为一；
- 分片 renderPoseId 无重复，完整归并后覆盖 78,588 个 pose；
- 只读取 train split；
- 正式硬件 GPU evidence 完整；
- 最终关系 schema 能通过现有 V4 训练 preflight 和单步 smoke。

只有新稀疏关系 CSR 完成训练 smoke，并与当前稠密构建结果在同一小型输入上等价后，才能删除 IFCBench 的 203 GiB 合并缓存和分片稠密 payload。GPU evidence、渲染日志、manifest 和稀疏正式结果必须保留。

## 预计运行命令

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_triangle_depth_layer_shards.py \
  --manifest-dir <manifest-dir> \
  --output-root <sparse-cache-root> \
  --jobs 4 \
  --require-hardware-gpu

conda run -n slm_pvs python \
  neural_instance_culling/dataset/build_observed_relation_csr.py \
  --dataset-dir <pose-csr> \
  --runtime-meta <runtimeVisibilityMeta.json> \
  --sparse-cache-root <sparse-cache-root> \
  --output-dir <relation-csr> \
  --splits train \
  --source-k 12
```

## 主线状态

这是数据预处理实现优化，不改变论文三项模型创新、训练损失、阈值协议或前端运行资产。新关系结果通过完整验证前，当前默认模型和 HKUST 前端不变。

## 实施记录

2026-09-02 已完成以下 smoke：

- 1,787 pose 的正式 `shard_00` 从 4.7 GiB 压缩为 77 MiB；原始 8,687,468 条逐 pose 关系行无损归约为 941,184 个分片关系矩；
- 一分片最终关系输出通过 `pvs-viewcell-train-observed-relation-csr-v3` 校验，保留 611,475 条 top-12 关系边和 4,020,775 条生存观察；
- 生存观察覆盖 1,787 个 subpose，每个 subpose 权重守恒；
- V4 训练完成 `1 epoch × 2 step` CUDA smoke，观察实例覆盖率达到 1.0，未出现非有限 loss；该短 smoke 只验证链路，不作为指标结论。

训练 smoke 同时发现并修复 IFCBench split 物化时的相机字段错误：旧实现把 `candidate_camera_world` 复制为 query center。当前 2.5 m view-cell 已按 `radius / tan(30°)` 写入 4.330127 m 后退候选相机，并保持 `poses.camera_world == candidate_camera_world`；候选 CSR、GT、可见权重和 split 未改动。

2026-09-03 完成全场执行：

- 44/44 稀疏分片完成，`formalReady=true`，共覆盖 78,588 个渲染 pose；
- 368,943,836 条原始逐 pose 关系观察归约为 39,083,069 个分片关系矩，压缩缓存总大小为 3,373,033,354 bytes，磁盘显示约 3.2 GiB；
- 最终全局归并得到 12,222,965 条唯一关系边，按每个目标/方向/深度单元 top-12 后保留 3,373,965 条；
- 保留 172,516,349 条生存观察，其中 132,103,256 条为遮挡事件。77,614 个具有非空证据的 subpose 通过权重守恒检查；其余 974 个渲染 subpose 没有可见或遮挡证据，不产生伪造记录；
- 最终训练关系目录约 4.3 GiB，完整 IFCBench runner CUDA smoke 耗时 101.67 秒并通过；
- 删除 217,280,102,400 bytes 分片稠密 payload、203 GiB 合并副本、42 GiB 失败分片以及临时 smoke 输出。正式稀疏缓存、GPU evidence、manifest、日志和最终关系 CSR 保留。

主要修改文件：

- `neural_instance_culling/dataset/compact_triangle_depth_relation_shard.py`
- `neural_instance_culling/dataset/build_observed_relation_csr.py`
- `neural_instance_culling/dataset/build_triangle_depth_layer_evidence.py`
- `neural_instance_culling/dataset/build_ray_context_relation_evidence.py`
- `neural_instance_culling/dataset/apply_pose_split_manifest.py`
- `neural_instance_culling/benchmark/run_triangle_depth_layer_shards.py`
- `neural_instance_culling/benchmark/run_ifcbench_mainline.py`
- `neural_instance_culling/model/common/train_observed_relation_csr.py`

正式检查包括 32 个相关 unittest、44 分片 schema/pose/GPU evidence 检查、最终关系 schema 校验、IFCBench preflight 和完整关系 CUDA smoke。当前默认 HKUST checkpoint、阈值和前端资产未修改。
