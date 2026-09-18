# IFCBench V5 Surface 映射修复记录

日期：2026-09-18

## 目的

修复 IFCBench V5 surface extractor 将可渲染 component 错误标记为退化 unit 的问题，并确认逐 GLB 的 component-to-geometry 映射与 IFCBench 实际 runtime 元数据一致。该修复不使用 epsilon 几何或零行掩盖缺失几何。

## 根因

IFCBench runtime 的一个 prototype GLB 可以对应多个空间分离的 component。GLB 使用 `EXT_mesh_gpu_instancing` 保存一份 prototype 网格和多个实例的平移/旋转矩阵；`runtimeVisibilityMeta.globalGlbRecords[*].componentGlobalIds` 保存与实例数组相同顺序的 component ID。

旧实现已经展开实例矩阵，但仍优先以变换后三角形的 component AABB 做匹配。由于旋转、浮点误差和源 AABB 松紧程度，少量三角形可能不与 AABB 相交；旧逻辑随后把这些三角形标记为 fallback，并把包含任何 fallback 三角形的整个 component 写成零特征退化 unit。于是实际可渲染几何被错误地计入 `degenerateUnitIds`，IFCBench 出现了 8,879 个假退化 unit。

## 修复

- 对 `EXT_mesh_gpu_instancing` 展开的每个三角形，直接使用实例数组顺序对应的 component ID；显式实例映射是该 GLB 的权威归属信息。
- 普通非实例化多网格 GLB 仍使用几何 AABB/质心匹配。
- 对实例化 GLB 严格检查 `componentGlobalIds.length === object.count`，映射不一致时直接失败，避免静默生成错误数据。
- 每个 GLB 提取完成后释放几何、材质和纹理，并清空 GLTFParser 的依赖缓存、关联表和 primitive/source/texture cache。
- 对一个 GLB 对应一个 component 的普通 Mesh，使用累计面积和源三角形索引的 typed-array 两遍采样，不把全网格展开成 JavaScript triangle 对象；这一步覆盖 HKUST 中的百万级三角形 GLB。
- 读取 surface 特征前移除材质/纹理依赖；surface 只需要世界位置和由位置计算的三角形法线，不需要解码图片或创建 Blob URL。
- 通过 `node --expose-gc` 启动编译入口；默认每 10 个 GLB 调用一次 `global.gc()`（`--gc-every 0` 可关闭）。GC 不参与几何分配、采样或标签计算，不改变输出数值。

## 修改文件

- `neural_instance_culling/dataset/generate_v5_instance_surface_samples.mjs`
- `neural_instance_culling/dataset/test_generate_v5_instance_surface_samples.mjs`
- `neural_instance_culling/dataset/v5/compile_registered_scene.py`

## 验证

Node 单元测试：

```bash
node neural_instance_culling/dataset/test_generate_v5_instance_surface_samples.mjs
node --expose-gc neural_instance_culling/dataset/test_generate_v5_instance_surface_samples.mjs
```

测试覆盖普通多 component GLB、空 GLB、带平移和旋转的 `EXT_mesh_gpu_instancing` GLB、GC 调用路径，以及 GC 前后输出字节一致性。两条命令均通过。

真实 IFCBench 小规模验证使用实际 `runtimeVisibilityMeta.json`、`glbIndex.json` 和前 101 个 GLB：

- 1,471 个 component
- `degenerateUnitCount = 0`
- 无 stderr

另外抽查实际 prototype GLB 0、1、100、1000、2000、3668。实例化 GLB 均为 `instance-index-explicit`，所有 source triangle 均被分配，`unmatchedTriangleCount = 0`；末尾单 component GLB 使用几何匹配且同样无 unmatched triangle。

全量 IFCBench 重建结果：

- 3,669 个 GLB
- 41,298 个 unit
- `degenerateUnitCount = 0`
- `surface_points_fp32.bin`：253,734,928 bytes
- stderr：0 bytes
- 使用固定 GC 后 RSS 约 1.1--1.3 GiB；未提高 Node heap 上限，未再出现 4.7 GiB 后 SIGABRT。

HKUST 内存回归（实际资产，前 1,700/3,273 个 GLB）：

- 完成写出，stderr 为 0
- `/usr/bin/time -v` 最大 RSS：706,596 KiB
- 旧故障点附近不再触发 V8 OOM/SIGABRT
- 退化 unit 仅为 runtime 中没有面积的真实占位：component 4238 的 AABB 只有 Y 线段范围，component 4528 的 AABB 为零体积；未使用 epsilon 填充。component 15082 同样是零体积，但位于未处理的后续 GLB 中。

## 正式重建命令

以下命令将修复后的结果写入 V5 IFCBench canonical surface 目录；它只重建 surface stage，不触碰模型或训练 runner：

```bash
node --expose-gc neural_instance_culling/dataset/generate_v5_instance_surface_samples.mjs \
  --assets-dir ifcbench_fantasy_metropolis_instanced_v2/assets \
  --runtime-meta ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json \
  --glb-index ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json \
  --output-dir neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/ifcbench_metropolis/surface \
  --progress-every 50 \
  --gc-every 10 \
  > neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/ifcbench_metropolis/logs/surface_fixed_stdout.log \
  2> neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/ifcbench_metropolis/logs/surface_fixed_stderr.log
```

## 主线状态与风险

修复后的 IFCBench surface 资产保留为 V5 主线输入；旧的 8,879 退化统计不再作为训练或论文结果。该修复只改变错误的 component-to-geometry 归属，不改变 256 点/unit、世界坐标三角形、AABB、采样种子或 V5 模型接口。正式训练前仍需用 canonical loader 检查 manifest、unit ID、AABB 和 relation/probe 资产的一致性。
