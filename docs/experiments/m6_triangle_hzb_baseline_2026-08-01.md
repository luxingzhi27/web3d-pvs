# M6 三角形 HZB 基线实现记录

日期：2026-08-01

状态：生成器和查询 runner 已实现，合成缓存与浏览器 smoke 通过；正式完整场景 validation/calibration 尚未运行。

是否保留为主线：保留为 L2 warm-cache 图形学基线，不作为当前冷启动模型或前端默认路径。

## 变更目的

此前的 `baseline_aabb_hzb` 实际只使用实例 AABB 深度近似，显示名已固定为
`baseline_aabb_depth_proxy`，不能代表标准层次深度缓冲（Hierarchical Z-Buffer，HZB）。本次新增
真实三角形 HZB 基线，用完整本地 GLB 的浏览器光栅化结果生成最底层深度，再从该深度图逐级做
最小池化。这样可以把“已有完整几何时的遮挡质量上界”与“没有目标 GLB 几何时的神经/元数据方法”
分开比较。

## 实现内容

- `neural_instance_culling/benchmark/build_triangle_hzb_cache_browser.mjs`
  - 从 Pose CSR 的 64 字节方向相机记录读取世界坐标、前向和 FOV；世界坐标位于记录偏移 12，不能误读为归一化相机坐标。
  - 固定使用模型查询 FOV `66°`，真实前端渲染 FOV 仍为 `60°`。
  - 用 Three.js/Chrome 加载指定 GLB 集合，使用深度测试光栅化三角形，并把线性视深编码为 RGBA8。
  - 浏览器端对 level-0 深度逐级执行 `2x2 min-pooling`，Node 按 pose 固定偏移写入 Float32 二进制。
  - 输出同名 JSON 元数据，记录层级尺寸、pose 索引、相机、深度语义、GLB 完整性和浏览器构建耗时。
- `neural_instance_culling/benchmark/triangle_hzb.py`
  - 校验缓存 schema、二进制大小和层级描述；按候选 AABB 投影矩形选择层级并读取三角形深度。
  - 返回连续的潜在可见覆盖分数，不把它宣称为精确三角形可见性 mask。
- `neural_instance_culling/benchmark/model_runners.py`
  - 注册 `baseline_triangle_hzb`，运行时要求显式缓存路径和保存的 `pose_indices`。
  - runner 只给当前保存候选打分，不读取 GT，也不把缓存 pose 的可见集合并入当前候选。
- `test_visibility_baseline_runners.py`
  - 增加缓存格式、最小池化、前后深度查询和缺少精确 pose 索引时拒绝运行的测试。

## 运行命令

先用小规模 smoke 验证浏览器和缓存格式：

```bash
node neural_instance_culling/benchmark/build_triangle_hzb_cache_browser.mjs \
  --assets-dir slm2viewer/assets/scenes/ifcbench_fantasy_metropolis_instanced_v2 \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2 \
  --output /tmp/triangle_hzb_smoke/values.bin \
  --split test --max-poses 1 --glb-id-list 1109 \
  --width 64 --height 64 --chrome-exe /opt/google/chrome/google-chrome \
  --timeout-ms 120000 --force
```

正式 baseline 不能使用 `--max-poses` 或 `--glb-id-list` 子集。正式运行还必须把完整 GLB 清单、
完整 validation/calibration pose split 和独立输出目录写入日志，并在 runner 评测时使用同一份
缓存元数据。缓存 `formalReady=false` 或 `geometryScope=explicit_glb_subset_non_formal` 时不得进入
正式论文表格。

统一评测器通过显式参数传入缓存路径，避免把不带缓存路径的默认 key 当作可执行模型：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models baseline_triangle_hzb \
  --triangle-hzb-cache /absolute/path/to/values.bin \
  --dataset-dir <dataset> --runtime-meta <runtimeVisibilityMeta.json> \
  --glb-index <glbIndex.json> --glb-root <glb-root> \
  --split validation --output-dir <independent-output-dir> --device cpu
```

`load_runner` 仍接受显式 `{"kind":"triangle_hzb","cache":"..."}` 规格，供 frozen manifest
等严格入口复用；`baseline_triangle_hzb` 没有缓存路径时会直接失败，且没有加入任何默认 benchmark
命令。

## Smoke 结果

2026-08-01 使用 Metropolis 的一个 test pose、一个 GLB、`64x36` 深度图生成缓存：

| 项目 | 结果 |
|---|---:|
| 缓存 level-0 | `64x36` |
| 层级数 | 7 |
| 每 pose Float32 值 | 3,081 |
| level-0 深度范围 | `0.029750–1.000000` |
| 浏览器加载 GLB | 1 |
| 浏览器渲染 pose | 1 |
| Python runner 候选打分 | 7,405 行，全部有限 |
| 缓存正式状态 | `formalReady=false` |

Chrome 的 Vulkan 初始化在无显示环境下报告 `vkCreateInstance() failed: -7`，但 headless WebGL
仍完成了光栅化，smoke 元数据记录的 renderer 为 `WebKit WebGL`。因此本次结果只证明生成链路和
数据格式正确，不证明 NVIDIA 硬件 GPU 路径或正式运行性能；正式性能实验必须单独记录 Chrome
GPU 信息、实际后端、完整 GLB 加载耗时和逐 pose HZB 构建耗时。

## 指标与限制

该基线的输入信息等级是 `L2_warm_triangle_depth`：目标 GLB 已经在本地、已加载并可光栅化。
它不能与 L0/L1 冷启动方法直接进行“同资源预算”胜负比较，只能报告：

- 三角形 HZB 构建时间和峰值内存；
- 每 pose 查询耗时；
- 与实例 AABB 候选集合相同条件下的 recall、useful cull、bad cull 和图像指标；
- HZB 额外缓存字节及其相对于直接动态重建的收益。

当前仍未完成：完整 HKUST/Metropolis 缓存、validation/calibration 工作点、与标准真实深度渲染的
同位姿校验、以及正式 test。任何这些结果产生前，M6 总质量门继续保持未通过。

## 质量验证

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest discover \
  -s neural_instance_culling/benchmark/tests -p 'test*.py' -v
```

2026-08-01 结果：32 项全部通过。生成器另通过 `node --check`、`git diff --check` 和上述浏览器
smoke。该记录对应提交 `139b092`（runner）及后续生成器提交，旧提交历史保留不重写。
