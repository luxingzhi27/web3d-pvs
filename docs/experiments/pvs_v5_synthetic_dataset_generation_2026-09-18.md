# V5 Synthetic Dataset Generation

日期：2026-09-18  
状态：生成器与小场景数据契约已实现，120 场景正式生成尚未在本次 smoke 中全部执行。

## 目的

把 V5 已登记的 120 个 seed 级场景转成可训练的程序化数据。每个 renderable unit 是一个闭合
box、cylinder、beam 或 panel primitive 的真实网格；AABB 只作为候选和纯几何关系的输入，不能
替代几何本身。

## 修改文件

- `neural_instance_culling/dataset/v5/generate_synthetic_datasets.py`
- `neural_instance_culling/dataset/v5/tests/test_generate_synthetic_datasets.py`

同目录已有 V5 preprocessing 模块未修改。生成器复用它们的 surface、relation 和 catalog API。

## 输出契约

默认正式参数写入 `synthetic_dataset_manifest.json`：64 个 view-cell/scene、4 个离线同方向
subpose/view-cell、160x90 Color-ID 近似渲染、模型 FOV Y 66 度、显示渲染 FOV Y 60 度，以及
16 个 surface starts、36 个 probe directions 和 13 个距离比。

每个场景的编译目录固定为：

```text
<scene>/compiled/surface/
<scene>/compiled/relation/
<scene>/compiled/probes/
<scene>/pose_csr/
```

`surface/` 是 `[N,256,6]` xyz+unit-normal dense asset；`relation/` 明确写入
`usesVisibilityLabels=false`；训练 seed 的 `probes/` 以 `R=N*16*36` 行列式保存
`unitIds/directions/hitDistances/maxDistances/startIds/directionIds`，manifest 使用与真实
probe 生成器相同的 `rowCount`、column dtype/shape、little-endian 和 source-train-only 权限。
每行只保存最近外部命中，未命中用 NaN 右删失；训练通过固定 13 个 distance ratios 展开
event/right-censored observation，并跳过当前整个 primitive。validation/diagnostic seed 不生成
可读 probe 监督，避免把非 source-train 场监督混入训练。Pose CSR 保留现有 `PoseCSRDataset`
所需的 poses、offsets、IDs、weights、view-cell 和 subpose 文件。

Color-ID 近似使用解析 box/cylinder 最近交点；区域 GT 是四个 subpose 的 unit ID 并集，pixel
weight 是子位姿间 max-pool 的 pixel parts-per-million。candidate 只由四个 subpose 的 66 度
AABB 几何视锥并集生成，不读取 GT 或可见标签构图；生成后强制检查 `visible_ids ⊆ candidate_ids`。

## 运行命令

```bash
conda run -n slm_pvs python neural_instance_culling/dataset/v5/generate_synthetic_datasets.py \
  --scene-id synthetic_v5_f00_s00 \
  --device cuda \
  --output-root neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/synthetic \
  > synthetic_stdout.log 2> synthetic_stderr.log
```

生成全目录时使用 `--all`；可以用 `--views-per-scene`、`--render-width` 和 `--render-height`
登记一套明确的重生成参数。`--device cpu` 与 `--device cuda` 使用相同的解析交点和确定性
ID tie-break；CUDA parity 已用随机射线检查。

## 验证结果

- 生成器单测：5 passed。
- `neural_instance_culling/dataset/v5/tests`：17 passed。
- CPU/CUDA 随机 ray-primitive ID 与距离 parity：通过。
- 小场景 Pose CSR：现有 `PoseCSRDataset` 可读，visible 是 candidate 子集；probe 单 unit
  不命中自身，13 距离格点展开形状正确，列式 manifest 通过 V5 probe schema 校验。

## 保留与风险

该生成器作为 V5 synthetic 主线入口保留。当前 smoke 没有执行 120 场景全量生产；正式运行前
仍需按仓库日志要求保存 stdout/stderr，并检查全量场景的耗时、磁盘占用、候选覆盖和训练器读取。
