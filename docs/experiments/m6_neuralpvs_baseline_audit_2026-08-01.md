# M6 NeuralPVS 对照审计

日期：2026-08-01  
状态：未实现，M6 总门保持 No-Go。本文记录审计结论，不把已有的 AABB、froxel 诊断或三角形 HZB 结果改名为 NeuralPVS。

## 审计目的

投稿计划要求将当前实例级可见性模型与 NeuralPVS 风格的视锥体体素方法在相同的 66° 模型相机、相同 view-cell、相同候选集合和相同校准规则下比较。审计检查仓库是否已有可以公平运行的实现，以及缺少哪些输入、监督和运行时资产。

## 已核验的事实

- `build_neuralpvs_viewcell_pose_plan.mjs` 能生成固定朝向、view-cell 内随机位置的 subpose；这只定义采样组织，不生成 NeuralPVS 的体素输入。
- Color-ID 采样器目前输出实例 ID 和覆盖权重，未输出供视锥体体素重建使用的完整深度片段。
- `build_rvc_viewcell_pose_csr.py` 能生成 view-cell 级实例可见集合并集和严格候选并集，但实例集合标签不能替代每个 froxel 的深度/实例占据监督。
- `froxel_metrics` 只是将 AABB 投影到诊断网格，不能称为三角形 froxel 几何输入。
- 当前 `model_runners.py` 注册了 keep-all、训练频率、相机距离、投影 AABB、AABB+ray、AABB depth proxy、train view-cell bitset 和三角形 HZB；没有 NeuralPVS runner。
- 当前前端模型读取固定实例特征并进行轻量射线查询，运行时不执行稀疏三维卷积，也没有 NeuralPVS 的稀疏体积编解码器。

## 与论文方法的边界

NeuralPVS 论文描述的是 froxelized 场景表示、稀疏卷积/体积交错压缩以及 Dice 与 repulsive visibility loss。论文公开资料没有提供足够的逐层网络、全部输入通道和复现实验超参数，因此本项目不能用自行猜测的网络冒充原始实现。后续若实施，只能命名为 `baseline_neuralpvs_froxel_viewcell_<scene>_fov66`，并明确标注为 NeuralPVS-style adaptation。

## 公平实现所需的新链路

1. 离线对每个 view-cell 的同方向 subpose 渲染深度和实例 ID。
2. 将可见深度片段反投影到固定的相机对齐视锥体网格，形成几何占据和 froxel-to-instance 的保守 CSR 映射。
3. 在 train view-cell 上训练轻量三维体积网络，用 from-region 的 froxel 标签监督；实例分数由关联 froxel 概率的最大值或预注册聚合得到。
4. 将体素、映射、索引和模型权重作为 L1 冷启动资源计入字节和启动时间，运行时不得请求目标 GLB 三角形。
5. 只用 calibration 选择阈值；test 只执行一次，不补 GT 候选、不扫描 test 阈值。

当前仓库缺少深度片段采样、froxel 数据集、froxel-to-instance 资产、网络、训练器和 runner，因此本日不启动一个信息层级不公平的伪实现。

## 已有相关基线的正确命名

| 资产/方法 | 允许的名称 | 不允许的解释 |
|---|---|---|
| `baseline_aabb_hzb` | `baseline_aabb_depth_proxy` | 真实 HZB、NeuralPVS |
| `froxel_metrics` | AABB 投影诊断网格 | 三角形 froxel 输入 |
| `baseline_triangle_hzb` | 完整几何 warm-cache HZB | Cold-0 NeuralPVS 对照 |
| train view-cell bitset | 训练视点查表基线 | NeuralPVS 的连续视锥编码 |

## 结论与后续

M6 的 L0 元数据基线和 L2 真实三角形 HZB warm-cache 子门已有证据；NeuralPVS 子门未实现，M6 总门不能通过。除非先完成上述深度/体素数据链路并将其输入资源计入预算，否则论文中只能把 NeuralPVS 作为相关工作和未实现的公平基线缺口，不能报告“NeuralPVS 对比结果”。

