# M6 NeuralPVS 对照审计

日期：2026-08-01  
状态：未实现，M6 总门保持 No-Go。本文记录审计结论，不把已有的 AABB、froxel 诊断或三角形 HZB 结果改名为 NeuralPVS。

## 审计目的

投稿计划要求将当前实例级可见性模型与 NeuralPVS 风格的视锥体体素方法在相同的 66° 模型相机、相同 view-cell、相同候选集合和相同校准规则下比较。审计检查仓库是否已有可以公平运行的实现，以及缺少哪些输入、监督和运行时资产。

## 已核验的事实

本次补充审计于 2026-08-02 复核了 NeuralPVS 官方代码仓库。官方训练与推理实现是可以获得的，不能再表述为
“公开资料不足以复现”。本项目保留一份临时、只读的审计副本，commit 为
`946088616cad18de81cde12fecd6ab204e52eac9`；该副本不属于本项目运行时依赖，也不写入仓库。

- `build_neuralpvs_viewcell_pose_plan.mjs` 能生成固定朝向、view-cell 内随机位置的 subpose；这只定义采样组织，不生成 NeuralPVS 的体素输入。
- Color-ID 采样器目前输出实例 ID 和覆盖权重，未输出供视锥体体素重建使用的完整深度片段。
- `build_rvc_viewcell_pose_csr.py` 能生成 view-cell 级实例可见集合并集和严格候选并集，但实例集合标签不能替代每个 froxel 的深度/实例占据监督。
- `froxel_metrics` 只是将 AABB 投影到诊断网格，不能称为三角形 froxel 几何输入。
- 当前 `model_runners.py` 注册了 keep-all、训练频率、相机距离、投影 AABB、AABB+ray、AABB depth proxy、train view-cell bitset 和三角形 HZB；没有 NeuralPVS runner。
- 当前前端模型读取固定实例特征并进行轻量射线查询，运行时不执行稀疏三维卷积，也没有 NeuralPVS 的稀疏体积编解码器。
- 官方仓库包含 `models/oacnn.py`、`models/vnet.py`、`modules/interleaver.py`、`modules/dataset.py`、
  `train.py` 和 `infer.py`，并提供 `torchnn`/`spconv` 后端；其数据目录约定为 `gv/*.bin.gz` 和
  `pvv/*.bin.gz`，每个文件是 bit-packed 三维体素网格。
- 官方实现的依赖链包含 `spconv`，并可选用 `cupy`；当前 `slm_pvs` 环境没有
  `spconv`、`torch_geometric` 和 `cupy`。这只是当前环境限制，不是对官方代码可获得性的判断。

## 与论文方法的边界

NeuralPVS 论文和官方代码描述的是 froxelized 场景表示、稀疏卷积/体积交错压缩以及 Dice、focal 和
repulsive/no-guess 类损失。官方代码可用于复现其网络和训练入口，但它没有直接读取本项目的实例级 Color-ID
CSR，也没有提供本项目所需的“froxel 到实例”和严格后退视锥候选接口。因此，当前不能把已有的 AABB、HZB
或实例 MLP 结果称为原始 NeuralPVS，也不能在缺少适配链路时声称完成公平对比。后续若实施，只能命名为
`baseline_neuralpvs_froxel_viewcell_<scene>_fov66`，并在报告中明确区分“官方实现复现”和“本项目的实例级适配”。

## 公平实现所需的新链路

1. 离线对每个 view-cell 的同方向 subpose 渲染深度和实例 ID。
2. 将可见深度片段反投影到固定的相机对齐视锥体网格，形成几何占据和 froxel-to-instance 的保守 CSR 映射。
3. 在 train view-cell 上训练轻量三维体积网络，用 from-region 的 froxel 标签监督；实例分数由关联 froxel 概率的最大值或预注册聚合得到。
4. 将体素、映射、索引和模型权重作为 L1 冷启动资源计入字节和启动时间，运行时不得请求目标 GLB 三角形。
5. 只用 calibration 选择阈值；test 只执行一次，不补 GT 候选、不扫描 test 阈值。

当前仓库缺少与官方格式对齐的深度片段采样、froxel 数据集、froxel-to-instance 资产、实例聚合规则和
严格候选 runner；当前环境也缺少官方 `spconv`/`cupy` 依赖。因此本日不启动一个信息层级不公平的伪实现。

## 已有相关基线的正确命名

| 资产/方法 | 允许的名称 | 不允许的解释 |
|---|---|---|
| `baseline_aabb_hzb` | `baseline_aabb_depth_proxy` | 真实 HZB、NeuralPVS |
| `froxel_metrics` | AABB 投影诊断网格 | 三角形 froxel 输入 |
| `baseline_triangle_hzb` | 完整几何 warm-cache HZB | Cold-0 NeuralPVS 对照 |
| train view-cell bitset | 训练视点查表基线 | NeuralPVS 的连续视锥编码 |

## 结论与后续

M6 的 L0 元数据基线和 L2 真实三角形 HZB warm-cache 子门已有证据；NeuralPVS 官方代码审计已完成，
但本项目的 NeuralPVS 实例级适配子门仍未实现，M6 总门不能通过。除非先完成上述深度/体素数据链路、
依赖环境和资源预算核算，否则论文中只能把 NeuralPVS 作为已公开的相关方法，并诚实记录本项目尚未完成
公平适配，不能报告“NeuralPVS 对比结果”。

## 2026-08-02 复核记录

- 复核对象：官方代码临时副本 `9460886`，不是本项目提交内容。
- 已确认：官方网络、交错模块、数据集读取器、训练器、推理器和损失实现均存在；官方输入是压缩三维体素，
  不是本项目已有的实例并集标签。
- 未完成：将三个场景的真实深度/实例 ID 采样转换为官方 `gv/pvv` 语义；建立保守的体素到实例映射；在同一
  `66°`、同一 view-cell、同一 candidate CSR 和同一 calibration/test 协议下运行实例级评测；计入冷启动
  体素资产、权重、解码和前端查询成本。
- 决策：不修改现有主线模型，不把 AABB depth proxy 或三角形 HZB warm-cache 的结果填入 NeuralPVS
  对比表。M6 保持 `No-Go / adaptation not implemented`，直到上述链路有可复现产物。
