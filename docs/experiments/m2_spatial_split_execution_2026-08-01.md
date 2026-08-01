# M2：空间隔离四路 split 执行记录

日期：2026-08-01  
状态：空间隔离门通过，类别覆盖门保留为风险；正式模型尚未训练。

## 目的

原始随机 view-cell 划分不能排除相邻相机位置泄漏。M2 将同一物理 view-cell、其所有
subpose 以及相邻支持位置视为一个空间单元，在训练、validation、calibration 和 test
之间加入 guard band。这样后续模型结果可以解释为未见空间区域泛化，而不是记忆相邻相机。

## 实现

修改/新增：

- `neural_instance_culling/dataset/build_spatial_viewcell_split.py`
  - 使用 canonical view-cell center 和 dense subpose XZ 支撑位置构造空间单元；
  - 保存 group、block、类别计数和跨 split 最近距离；
  - 对 guard 触发的跨 split 支撑点对进行单独计数。
- `neural_instance_culling/dataset/apply_spatial_split_manifest.py`
  - 将不可变 manifest 以 hard link 方式应用到正式 Pose CSR，不复制大型候选/可见数组；
  - 把 `splitIds` 改为 `train/validation/calibration/test/guard` 五类语义。
- `neural_instance_culling/dataset/materialize_viewcell_geometry.py`
  - 从 subpose 原始采样物化 canonical center 和 dense 支撑位置，避免用随机 subpose 均值
    代替 view-cell 几何。

## 正式 manifest

### HKUST

路径：`neural_instance_culling/dataset/out/spatial_split_hkust_fov66_20260801`  
应用数据集：`neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1`

| 集合 | pose 数 | group 数 |
|---|---:|---:|
| train | 2,772 | 2,772 |
| validation | 213 | 213 |
| calibration | 168 | 168 |
| test | 238 | 238 |
| guard | 4,608 | 4,608 |

跨 split 最近支持位置距离为 `16.03 m` 至 `16.22 m`，超过 `16 m` guard 要求。六类采样
类别在四个正式集合中均有记录，HKUST 的空间 split 质量门通过。

### Metropolis

路径：`neural_instance_culling/dataset/out/spatial_split_metropolis_dense_subpose_union_fov66_20260801`  
应用数据集：`neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v1`

| 集合 | pose 数 | group 数 |
|---|---:|---:|
| train | 19,296 | 536 |
| validation | 2,088 | 58 |
| calibration | 2,376 | 66 |
| test | 2,340 | 65 |
| guard | 1,152 | 32 |

Metropolis 以 `40 m` block 和 `10 m` guard 运行，dense subpose 支撑位置共 109,008 个；
guard 排除了 33,620 对跨 split 支撑位置，正式集合最近距离为 `10.94 m` 至 `12.92 m`，
空间隔离门通过。

## 类别稀疏风险

Metropolis 当前正式采样只有 `street_gap`、`near_building`、`sky` 和 `far` 四类。场景中
只有约三个独立的 sky 物理位置，因此无法在四个空间集合都放入 sky，同时又保持物理位置
不跨 split。实际结果是 validation/calibration 没有 sky，test 只有三个 sky group。

这不是通过伪造类别或随机复制 pose 解决的问题。当前处理是：

1. 空间隔离门按几何泄漏标准通过；
2. 类别覆盖作为 Metropolis 泛化实验的限制单独报告；
3. 后续 M11 不宣称每类均衡泛化，并补充按类别分层的置信区间；
4. 如果需要完整类别覆盖，必须新增真实 sky/perimeter/plaza 物理位置后重建采样和 split，
   不能把已有位置重复分配到不同集合。

## M2 准入判断

M2 的核心科学门是“空间相邻视点不跨正式集合”，当前已通过。Metropolis 的稀有类别
限制不会被隐藏，作为后续 M11 的预注册风险。训练前仍必须完成：

- 正式空间数据的方向遮挡证据重建；
- `K=4` subpose 对投稿级稳定性的验证，必要时追加 K16/K32 对照；
- 使用固定 validation/calibration/test 协议训练，而不是沿用旧随机 split checkpoint。

## 复现信息

- model/sampling FOV：66°；真实前端渲染 FOV：60°；
- split seed：`20260801`；
- Metropolis block/guard：`40 m / 10 m`；
- HKUST block/guard：`64 m / 16 m`；
- 详细 manifest、源数据 SHA-256 和 split 标签 SHA-256 位于各 manifest 的 `manifest.json`
  和正式 CSR `dataset_meta.json`。
