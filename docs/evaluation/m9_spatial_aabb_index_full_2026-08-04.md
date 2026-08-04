# M9 全量 Pose 空间 AABB 候选审计

日期：2026-08-04  
状态：候选正确性通过；空间索引加速门未通过，默认前端路径不宣称使用索引获得加速

## 评价口径

本次使用 HKUST 的完整 `7,999` 个模型查询 pose，实例数为 `18,831`，模型查询视场角为 `66°`，远裁剪面为
`2,000 m`，候选集合由相同的实例 AABB 和 Three.js 视锥测试产生。比较对象是当前 `InstancePVS` 的 64 米
空间 AABB 索引和逐实例全量 AABB 扫描。两条路径都只返回后退相机候选，不读取 GT，也不补入可见实例。

空间索引为保持集合保守性，将跨 64 米 cell 边界的构件放入 overflow；本次共有 `2,778` 个 overflow 实例，
真正放入 cell 桶的实例为 `16,053` 个。候选集合通过逐 pose 排序后的实例 ID 逐项比较，不能只比较数量。

## 结果

| 路径 | 实际来源 | pose 数 | 候选集合不一致 | 平均耗时 | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|
| 默认上限 `maxQueryCells=100000` | `full_aabb_scan` | 7,999 | 0 | 0.889 ms | 0.356 ms | 2.852 ms | 3.401 ms |
| 强制允许索引 `maxQueryCells=1e9` | `spatial_aabb_index` | 7,999 | 0 | 10.316 ms | 9.770 ms | 13.928 ms | 14.766 ms |

默认上限下的路径统计为 `full_aabb_scan: 7,999`；索引强制路径统计为
`spatial_aabb_index: 7,999`。两组候选集合均为 `0` 个 mismatch，候选数量分布也完全相同：均值
`4,961.168`、p50 `673`、p95 `18,414.1`、最大值 `18,831`。

独立的 512 pose 参数扫描显示，查询 cell 上限为 `1`、`100`、`1,000`、`10,000` 和 `100,000` 时均安全回退
到全量扫描；只有 `1e9` 才在这批 pose 上使用空间索引。该结果说明当前默认配置没有把较慢的字符串桶索引强制用于大视锥查询，但也不能把空间索引写成已经带来前端加速。

## 结论与边界

1. M9 候选正确性子门通过：默认和索引路径在完整 pose 集上没有候选集合差异。
2. 当前 64 米字符串桶加 overflow 的实现没有通过独立的 CPU 加速子门；在强制索引条件下比默认全量扫描慢约一个数量级。
3. 本次结果不改变默认模型、候选语义、前端显示逻辑或空间索引参数。前端仍以实际运行时的
   `candidateSelection.source` 记录为准，不能从离线索引结构推断每次浏览器查询都走索引。
4. 后续若要让索引成为性能优化，必须使用扁平化桶/紧凑候选缓冲或经过校准的自适应查询策略，并在同一 pose、同一浏览器和硬件门下重新测量；不能仅提高 `maxQueryCells`。

复现命令：

```bash
node slm2viewer/scripts/benchmark_m9_spatial_index.mjs \
  --samples 7999 \
  --index-max-query-cells 100000 \
  --out neural_instance_culling/benchmark/out/m9_spatial_aabb_index_audit_full_default_20260804.json

node slm2viewer/scripts/benchmark_m9_spatial_index.mjs \
  --samples 7999 \
  --index-max-query-cells 1000000000 \
  --out neural_instance_culling/benchmark/out/m9_spatial_aabb_index_audit_full_20260804.json
```

