# V5 Synthetic Dataset Generation

日期：2026-09-18  
状态：生成器与小场景数据契约已完成回归；120 场景正式生成正以 4 个互斥 scene shard 执行。

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
weight 是子位姿间 max-pool 的 pixel parts-per-million。candidate 只由四个 subpose 的后退
候选相机的 66 度 AABB 几何视锥并集生成，不读取 GT 或可见标签构图；生成后强制检查
`visible_ids ⊆ candidate_ids`，发现缺失直接失败，不再静默扩成全量候选。

## 2026-09-18 契约修正与根因

首次生成 `synthetic_v5_f00_s05` 时出现两个问题：

1. 生成器写出 `compiled/probes/probe_manifest.json`，而 V5 保护读取器只接受
   `external_hit_probe_manifest.json`。
2. 64 个 pose 的候选数都为 487。检查 `dataset_meta.json` 后确认
   `candidateFallbackAllUnitsViewcells=0`，问题不是 fallback，而是旧相机轨道半径为场景水平范围
   `0.9×`；从这么远的位置看，66 度视锥确实覆盖了整个合成场景。

当前修正为：

- 合成 view-cell 轨道半径固定为水平范围的 `0.20×`（下限 8 m），使候选测试覆盖场景内部的
  几何裁剪情况；这不改变模型 FOV 或四 subpose 协议。
- 对每个 subpose 使用
  `candidate_camera = subpose_camera_pos - subpose_camera_forward * back_offset`，再对四次
  66 度 AABB 测试取并集。
- 删除候选缺失时的全量 fallback；候选契约错误现在抛出异常，避免生成看似可训练但实际没有
  frustum 剔除的 CSR。
- probe manifest 改为 `external_hit_probe_manifest.json`，并统一使用正式列文件名
  `probe_unit_ids_uint32.bin`、`probe_directions_fp32.bin`、`probe_hit_distances_fp32.bin`、
  `probe_max_distances_fp32.bin`、`probe_start_ids_uint8.bin` 和 `probe_direction_ids_uint8.bin`。
- disk region manifest 直接使用实际水平轴写出支持点，保证 `scene_manifest.json` 与 subpose
  几何一致。

## 回归与小场景结果

集成测试独立重算四个后退 subpose 的 AABB 并集，并逐 pose 对比生成的 CSR；同时通过
`V5SceneTrainingData` 加载并采样 pose/probe：

```bash
conda run -n slm_pvs python -m unittest discover \
  -s neural_instance_culling/dataset/v5/tests -p 'test_*.py' -v
```

结果：`17 passed`。其中合成集成测试验证了：

- 4 个 pose 的 candidate CSR 与独立 frustum 并集完全相等；
- 每个候选集合都小于 64 个单位且候选数量不被固定为全量；
- `visible_ids` 始终是 candidate 子集；
- `V5SceneTrainingData` 能加载 16 维区域查询特征，并采样 32 条 probe observation。

按正式小场景参数重新生成的 `synthetic_v5_f00_s05`：

| 项目 | 结果 |
|---|---:|
| view-cell / subpose | 64 / 4 |
| renderable units | 487 |
| candidate 总数 | 20,251 |
| 平均 candidate 数 | 316.42 |
| candidate 范围 | 270–357 |
| 全量 candidate pose 数 | 0 / 64 |
| visible 子集违规 | 0 |
| probe 行数 | 280,512 = 487×16×36 |
| loader 采样 pose | 4 |
| loader 采样 probe | 8,192 |

小场景修正阶段只重新生成了 `synthetic_v5_f00_s05`；随后已启动 120 场景正式生产。正式输出路径为：

```text
neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/synthetic/synthetic_v5_f00_s05
```

## 运行命令

```bash
conda run -n slm_pvs python neural_instance_culling/dataset/v5/generate_synthetic_datasets.py \
  --scene-id synthetic_v5_f00_s00 \
  --device cuda \
  --output-root neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/synthetic \
  > synthetic_stdout.log 2> synthetic_stderr.log
```

单进程生成全目录时使用 `--all`。当前机器为提高 GPU 利用率使用四个互斥 shard：

```bash
for shard in 0 1 2 3; do
  conda run --no-capture-output -n slm_pvs \
    python -m neural_instance_culling.dataset.v5.generate_synthetic_datasets \
    --all --scene-shard-index "$shard" --scene-shard-count 4 --device cuda \
    > "all_shard_${shard}_stdout.log" 2> "all_shard_${shard}_stderr.log" &
done
wait

conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.dataset.v5.generate_synthetic_datasets \
  --finalize-existing --device cuda
```

四个 shard 按排序后 scene ID 的步长切分，测试保证互不重叠且并集恰为 120。每个 shard 只写
自己的分片 manifest；`--finalize-existing` 只有在 120 份 scene manifest 全部存在时才写唯一
`synthetic_dataset_manifest.json`。`--device cpu` 与 `--device cuda` 使用相同的解析交点和确定性
ID tie-break；CUDA parity 已用随机射线检查。

## 验证结果

- 生成器专项单测：6 passed。
- `neural_instance_culling/dataset/v5/tests`：18 passed。
- CPU/CUDA 随机 ray-primitive ID 与距离 parity：通过。
- 小场景 Pose CSR：现有 `PoseCSRDataset` 可读，visible 是 candidate 子集；probe 单 unit
  不命中自身，13 距离格点展开形状正确，列式 manifest 通过 V5 probe schema 校验。

## 保留与风险

该生成器作为 V5 synthetic 主线入口保留。全量任务日志位于正式 synthetic 输出根目录的
`logs/all_shard_*_{stdout,stderr}.log`；完成后仍需运行 finalize、全场景 loader preflight，并
汇总耗时、磁盘占用与候选覆盖。
