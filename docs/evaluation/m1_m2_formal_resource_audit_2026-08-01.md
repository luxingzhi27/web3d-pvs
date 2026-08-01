# M1-M2 正式资源与空间划分审计

日期：2026-08-01  
状态：通过资源准入；正式模型训练正在等待新 HKUST 方向遮挡证据完成。

## 目的

投稿计划要求先证明数据语义可靠，再进行 checkpoint、阈值和 test 结论。本文记录当前两个正式场景的候选集合、可见标签、实例映射、点云缓存和空间四路划分。旧的候选正样本补入数据不会进入正式训练。

## 固定协议

- 模型查询视场：66°；前端真实渲染视场：60°。
- 一个 pose 是一个 view-cell 的 canonical 相机状态；候选集合是该 view-cell 内所有成功密集子姿态的完整 AABB 后退视锥候选并集。
- GT 可见实例是同方向密集子姿态的 Color-ID 可见实例并集。
- `visible_weights` 是 view-cell 内最大池化的 Three.js Color-ID 屏幕覆盖率，单位为 ppm；不是严格真实像素覆盖率。
- 正式候选不允许把 GT 可见实例补回候选。候选遗漏正样本直接使数据不合格。
- 点云缓存按可下载 GLB 去重；实例通过 `instance_to_glb` 访问对应原型点云。

## HKUST

正式原始 CSR：

```text
neural_instance_culling/dataset/out/pose_csr_hkust_v3_raw_subpose_aabb_fov66_v1
```

正式空间 CSR：

```text
neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1
```

| 项目 | 数值 |
|---|---:|
| 实例数 | 18,831 |
| 可下载 GLB 数 | 3,273 |
| view-cell / pose 数 | 7,999 |
| 成功密集子姿态数 | 279,008 |
| 平均候选实例数 | 5,014.44 |
| 平均 GT 可见实例数 | 107.53 |
| 候选漏正样本 | 0 |
| GT 补入候选数 | 0 |
| 点云缓存失败 / fallback | 0 / 0 |

空间划分采用 64 m block 和 16 m dense-subpose guard：

```text
train / validation / calibration / test / guard
5563 / 664 / 690 / 722 / 360
```

所有 split 共享同一 canonical view-cell，不跨组拆分。四个可评测集合之间的密集子姿态最近距离最小约 17.53 m，高于 16 m guard。类别覆盖已记录在 manifest；test 中 sky 类没有样本，不能对 HKUST sky 类作泛化结论。

## Metropolis

正式空间 CSR（修复顶层 split metadata 后的版本）：

```text
neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2
```

| 项目 | 数值 |
|---|---:|
| 实例数 | 41,298 |
| 可下载 GLB 数 | 3,669 |
| view-cell / pose 数 | 27,252 |
| 成功密集子姿态数 | 109,008 |
| 平均候选实例数 | 9,991.01 |
| 平均 GT 可见实例数 | 953.76 |
| 候选漏正样本 | 0 |
| GT 补入候选数 | 0 |
| 空 GT pose | 299 |
| 空候选 pose | 203 |
| 点云缓存失败 / fallback | 0 / 0 |

Metropolis 使用 40 m block 和 10 m guard：

```text
train / validation / calibration / test / guard
19296 / 2088 / 2376 / 2340 / 1152
```

Metropolis 可评测集合中存在空 GT pose。评测器现在保留这些 pose：空候选且空 GT 的 pose 按无误报、无漏报计入普通 pose 指标，但 useful cull 和字节节省不为其制造虚假分母收益。空 GT 但仍有候选的 pose 会正常经过模型预测。

## 审计命令与结果

两个正式 CSR 均执行：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/audit_pvs_resources.py \
  --dataset-dir <formal-csr> \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --glb-points <formal-glb-points>.bin \
  --glb-index <scene>/assets/glbIndex.json \
  --output <formal-csr>/resource_audit.json
```

两份审计的 `status` 均为 `pass`，并同时满足：

- `candidateRawIndependentlyAvailable = true`；
- `candidateIncludesForcedVisiblePositives = false`；
- `candidateMissVisible = 0`；
- 运行时实例数、AABB、GLB 映射和点云行数一致；
- 3,273 / 3,669 个 GLB 点云行全部成功解码且 hash 一致。

数据加载 smoke 还逐场景构造了 test 子集的 pose-set batch，验证存储候选顺序和 pose offsets 一致；空 pose 现在保留为零长度 offset 行，不会静默从正式评测分母消失。

## 历史目录边界

以下目录只作为历史审计或旧 split 证据保留，不能进入正式训练或主表：

- 旧 HKUST 的 GT candidate-union CSR；
- `*_stale_wrong_split_*` 目录；
- Metropolis 原 `..._spatial_dense_subpose_union_fov66_v1` 的旧顶层 split metadata。

Metropolis 的方向遮挡证据由同一候选完成的空间数据生成，使用 GT 可见来源与投影几何证据，不使用 dynamic-pool teacher。历史证据目录的二进制内容与 v2 CSR 逐文件 SHA-256 一致；已建立 `directional_occlusion_evidence_metropolis_spatial_dense_subpose_union_fov66_v2` 硬链接别名并修正 `datasetDir` 元数据，后续实验必须使用该明确路径。HKUST 新空间 CSR 的对应证据已经生成，正式训练正在使用它。

## 代码修复

- `build_spatial_viewcell_split.py` 修复 `viewcell` 分组模式引用不存在全局变量的问题。
- `PoseCSRDataset` 增加评测专用的 `include_empty` 路径，训练仍保持正样本批次效率。
- 统一评测器和训练校准统计空 GT / 空候选 pose，并对零正样本 pose 使用明确的无误报约定。
- 统一评测器和训练器默认拒绝缺失 GLB 文件的中位数成本填充；只有显式 exploratory 选项才能使用估算成本。
- GLB 排序使用“分数降序、GLB 编号升序”的固定 tie-break。

## 下一步

1. 完成新 HKUST 方向证据并运行证据元数据审计。
2. 以 `rvl_strong_v2`、CUDA、FP32、40 epoch、无 exploratory flags 从头训练两个场景。
3. 用 calibration 选择冻结阈值，validation 只用于 checkpoint 选择，test 只运行一次。
4. 在正式 checkpoint 上进入 M3 代理干预和 M4 消融，不把旧 test-calibrated 阈值写入主表。
