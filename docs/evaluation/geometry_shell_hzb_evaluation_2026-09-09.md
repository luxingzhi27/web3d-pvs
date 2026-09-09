# Geometry-shell HZB 评价报告

日期：2026-09-09
状态：MVP 已实现；完整 CPU 外壳导出已完成；浏览器结果目前只属于并发硬件 smoke，尚未形成正式 HZB 硬件计时或 test visibility 结论。

## 目的与边界

本基线用预下载的 LOD0 不透明几何外壳构造层次深度缓冲区，再对同一候选集合做批量实例 AABB 遮挡测试。它用于和神经 PVS 比较启动资产大小、GPU 深度/HZB 查询路径、实例输出和 GLB 聚合成本，不修改训练、PVS 评价或 streaming 主线。

输入根目录必须显式传入：

- HKUST：`/mnt/sda/rhyang/slm/hkust-v3`；
- IFCBench：`/mnt/sda/rhyang/slm/ifcbench_fantasy_metropolis_instanced_v2`；
- 也可传入直接包含 `sceneWeb.json`、`glbIndex.json` 和 `runtimeVisibilityMeta.json` 的 assets/data 目录。

所有新生成资产都写在 worktree 的 `neural_instance_culling/benchmark/out/paper_results/hzb/`，未创建软链接。

## 真实 GLB/schema 核查

| 场景 | GLB | 实例 | 源 GLB 字节 | 材质/结构核查 |
|---|---:|---:|---:|---|
| HKUST | 3,273 | 18,831 | 563,269,492 | 3,042 个 GLB 含确定 OPAQUE primitive；229 个 BLEND primitive；384 个 GLB 使用 `EXT_mesh_gpu_instancing` |
| IFCBench Metropolis | 3,669 | 41,298 | 183,335,200 | 全部 3,669 个 primitive 为确定 OPAQUE；2,765 个 GLB 使用 `EXT_mesh_gpu_instancing` |

HKUST 中发现 2 个真实空占位 GLB：runtime 元数据登记了实例，但 GLB 的默认 scene 无 node、无 mesh 且 buffer 长度为零。exporter 将它们标为 `empty-source-placeholder`，不作为深度 occluder；对其他非空 GLB，runtime 实例数量与源 instancing 数量不一致会直接报错。

## Exporter schema

输出 schema 为 `geometry-shell-hzb-v1`，由 `shell_meta.json` 描述：

- 每个确定 OPAQUE primitive 保留原始 `POSITION` float32、索引三角形顺序和最终 node/`EXT_mesh_gpu_instancing` float32 变换；
- 用独立 meshopt `ATTRIBUTES`/`TRIANGLES` segment 压缩 POSITION、INDEX 和 transforms；不写入纹理、UV、法线、切线、颜色和材质；
- 透明、alpha-cutout、材质 alpha 小于 1、透明材质扩展、未知材质扩展和空占位均不作 occluder；
- runtime 仍保留所有实例 AABB、instance-to-GLB 映射和 instance occluder mask；mask 为 `1` 的已选外壳实例是深度贡献者，查询时直接保留且不计 uncertainty，mask 为 `0` 的所有其他候选统一对已渲染外壳执行保守 AABB/HZB 测试；
- lossless 逐 primitive 输出完整几何，不做顶点移动、三角形简化或 primitive 内切分；
- `equal-asset` 只按固定 train 中心视点的投影面积/压缩字节排序，并删除完整 primitive，作为资产敏感性分析。
- `shell_meta.json` 只保留已选 prototype、segment、实例表和聚合统计；逐 primitive 审计写入 output 目录之外的
  `<output-dir>.offline.json`，该文件明确标记 `runtimeAsset=false`，不属于启动下载资产。
- equal-asset 的预算是实际需要下载的六个二进制文件加 `shell_meta.json` 的文件字节；日志、目录块和 offline sidecar
  不计入传输集合。exporter 写完文件后再次逐文件求和，并断言 `selection.totalAssetBytes <= selection.budgetBytes`。
- `queryContract.occluderSet` 固定记录上述二值语义：`selectedValue=1` 只作 selected-self 保留，`nonSelectedValue=0` 必须进入 AABB/HZB；透明和不确定材质不会写入外壳深度，但其候选仍可被外壳保守测试。

## 完整 CPU 导出结果

字节均为实际文件字节数；“传输几何”是三个 meshopt stream，“运行时展开几何”是 POSITION/INDEX/transforms 解码到 float32/uint32 后的大小。

| 场景 | 输出 prototype | prototype 三角形 | 实例展开三角形 | POSITION / INDEX / transforms 压缩字节 | 传输几何 | 固定 runtime payload | binary payload | 运行时展开几何 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HKUST lossless | 3,042 | 54,771,324 | 55,537,431 | 338,880,643 / 56,327,956 / 606,477 | 395,815,076 | 602,592 | 396,417,668 | 1,779,745,256 |
| IFCBench lossless | 3,669 | 8,648,833 | 25,556,160 | 25,951,283 / 27,735,937 / 1,503,387 | 55,190,607 | 1,321,536 | 56,512,143 | 160,744,560 |

lossless HKUST 的 229 个 BLEND primitive 和 2 个空占位没有进入外壳 occluder；最终 `occluderInstanceCount=18,566`。IFCBench `occluderInstanceCount=41,298`。
lossless runtime 下载总量还包括 `shell_meta.json`：HKUST 为 `399,388,459 B`（metadata `2,970,791 B`），IFCBench 为
`60,089,560 B`（metadata `3,577,417 B`）；这两个 metadata 也不含逐 primitive audit。

对应完整输出目录：

- `geometry_shell_hzb_lossless_hkust/`
- `geometry_shell_hzb_lossless_ifcbench/`

逐 segment meshopt 解码检查通过：HKUST 3,042 个 prototype、IFCBench 3,669 个 prototype 的 positions/indices/transforms 均无尾部字节，解码值有限。

## Equal-asset 敏感性

预算取现有同场景神经 runtime asset 目录的实际文件总字节，重要性使用固定 128 个 train 中心视点；这不是正式准确性结论。

| 场景 | 神经资产预算 | shell binary payload | runtime 总量余量 | 选中完整 primitive | 候选 primitive | occluder 实例 |
|---|---:|---:|---:|---:|---:|---:|
| HKUST | 5,479,213 | 5,384,833 | 2,907 | 91 | 3,042 | 1,257 |
| IFCBench | 11,883,919 | 10,470,682 | 48,904 | 1,412 | 3,669 | 31,909 |

两项 meta 都明确写入 `selection.mode=complete-primitive-deletion-only`，没有 primitive 内拆分。对应目录为 `geometry_shell_hzb_equal_asset_hkust/` 和 `geometry_shell_hzb_equal_asset_ifcbench/`。
最终 runtime 下载集合还包括 metadata：HKUST `shell_meta.json=91,473 B`、总计 `5,476,306 B`；IFCBench
`shell_meta.json=1,364,333 B`、总计 `11,835,015 B`。两者分别低于预算 `2,907 B` 和 `48,904 B`；这两个值由六个二进制文件和
`shell_meta.json` 的实际文件字节再次求和得到。
两个目录之外的 offline audit 分别为 `1,205,941 B` 和 `1,347,295 B`，不计入上述总量。

## HZB runtime MVP

浏览器 runtime `GeometryShellHZB` 提供：

1. depth-only pass：正线性眼空间深度，背景为 far，普通 depth attachment 取最近表面；
2. max mip：每个 2x2 block 取最大正深度，完整处理奇数宽高到 1x1；
3. 保守 AABB：投影 8 个角点，向外取整覆盖矩形；相机在 AABB 内、近裁剪面穿越、非有限投影和空/越界 footprint 都保留；selected occluder 自身直接保留，其他候选（包括 non-occluder）执行 HZB 测试；
4. 只有 `hzbMax + depthBiasM < candidateNear` 才允许剔除；
5. `queryPoint60` 对单个真实 60°相机查询，`queryRegion66` 对离线 subpose 分别查询后取实例并集；
6. GPU 只回读计数、最终可见实例编号和 GLB flag，实例输出保持 component 粒度，GLB 队列由可见实例聚合得到。

WebGPU MVP 明确固定 color/HZB texture、depth texture 和 render pipeline 为 `sampleCount=1`，不做 MSAA 或 resolve；
所有 mip 的源/目标宽高由 uniform 传入，WGSL 不依赖 `textureDimensions`，以兼容当前 Chrome WebGPU storage texture
路径。Region66 每个子姿态单独完成 depth/HZB/query/readback，最后对 component ID 取并集，`passCount` 和分阶段 timing
随结果保存。

## 浏览器 smoke 与 GPU 证据

已执行 2 pose、128x72 的小型浏览器 smoke，使用正式 Chrome Vulkan 参数、`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`；当前结果为修正后的 query contract 实现。
Point60 结果文件为：

`neural_instance_culling/benchmark/out/paper_results/hzb/smoke_point60_v4/point60.json`

Region66 结果文件为：

`neural_instance_culling/benchmark/out/paper_results/hzb/smoke_region66_v4/region66.json`

实际结果：

- WebGPU adapter：`vendor=nvidia`、`architecture=ampere`；
- browser result 和旁路 evidence 均保存 `gpuBackend`（API=`webgpu`、adapter 字段）以及独立 `gpuGate`；
- WebGL 辅助 renderer：NVIDIA RTX A6000 的 ANGLE Vulkan；
- Point60：2 个 pose、91 个候选引用，平均 45.5 个候选和 44 个保留实例；timing total p50=`34.00 ms`、p95=`58.57 ms`。aggregate precision=`0.068182`、recall=`1.0`、specificity=`0.035294`、balanced accuracy=`0.517647`、useful cull=`0.032967`、bad cull=`0`、weighted recall=`1.0`、weighted recall lower95=`1.0`。每个 pose 保留数为 `32/33`、`56/58`；该结果使用重新生成的 contract asset，仍为并发 smoke，不是正式 timing 或完整 test 结论；
- Region66：2 个 pose、4 个子姿态/pose，平均 6,433 个候选和 5,705 个并集保留实例；timing total p50=`36.10 ms`、p95=`44.74 ms`。aggregate precision=`0.140316`、recall=`0.995647`、specificity=`0.128708`、balanced accuracy=`0.562178`、useful cull=`0.112622`、bad cull=`0.000544`、weighted recall=`0.999021`、weighted recall lower95=`0.998910`。两个 pose 的保留数为 `5,611/6,013`、`5,799/6,853`，每个 pose 的 4 个 subpose 查询后取并集；该结果使用重新生成的 contract asset，仍为并发 smoke，不是正式 test 结论；
- 两个 smoke 均报告 WebGPU adapter `vendor=nvidia`、`architecture=ampere`，WebGL 辅助 renderer 为 NVIDIA RTX A6000 ANGLE Vulkan，`gpuValidationErrors=[]` 且无 page error；同一窗口的 `nvidia-smi pmon` 检出 GPU 0-3 上的 IFCBench `python` 计算进程，因此写入 `formalReady=false`、`executionClass=hardware-smoke-concurrent`。
- GLB 资源统计按 pose 内去重后再做 pose macro 平均，不再把跨 pose union 作为主指标。Point60 的主 `glb.poseMacro` 为 `44` 个 predicted、`3` 个 truth、`3` 个 intersection，源 GLB 字节分别为 `3,580,428 / 106,684 / 106,684 B`；两个 pose predicted 数/字节为 `32/2,163,372 B`、`56/4,997,484 B`。Region66 的主 `glb.poseMacro` 为 `932`、`277`、`277`，源 GLB 字节为 `27,644,000 / 6,357,804 / 6,357,804 B`；两个 pose predicted 数/字节为 `842/21,871,312 B`、`1,022/33,416,688 B`。`glb.unionDiagnostic` 仅作跨 pose 重复度诊断；bytes 来自显式 `glbIndex.json` 与 `--glb-root` 的 `stat().st_size`。

每个 smoke 目录旁都有 `geometry_shell_hzb_gpu_evidence.json`，保存 `gpuBackend`、adapter/WebGL gate、Chrome 参数、同窗口
`nvidia-smi` 和 `pmon` before/during/after。并发 smoke 的 timing 字段只用于确认路径可运行，不进入正式性能汇总。

页面仍有 Chrome 的 `A valid external Instance reference no longer exists.` warning；它未形成 WebGPU validation error，需在正式计时前继续定位。favicon 请求已由 benchmark server 处理为 204。

## 画面损失与剔除效率口径

本轮没有正式全量 HZB visibility test，因此尚未产生可用于论文结论的 pose/aggregate precision、recall、weighted recall 及其置信下界、specificity、balanced accuracy、useful cull、bad cull、image PER、miss pixel、wrong-ID pixel、GLB byte reduction 和正式冷启动/端侧延迟。2 pose smoke 的上述诊断值已单独标注，不能替代完整 test。

正式评价需使用与 PVS 相同的 candidate CSR、GT visible IDs/weights 和 split：

- 画面损失指标：weighted recall、miss pixel、image PER、wrong-ID pixel；
- 剔除效率指标：useful cull、specificity、预测/候选比例、bad cull、GLB 数量/字节削减；
- 资源/运行指标：shell 传输字节、解码后 GPU 内存、depth/mip/query/readback 分阶段时间和总调度时间。

`evaluate_geometry_shell_hzb.py` 已实现上述集合口径、pose macro/aggregate 计数、visible weights 加权召回和按 pose 的 GLB 计数；传入 `--glb-index` 与 `--glb-root` 时还会输出每 pose/pose macro/union diagnostic 的源 GLB 字节。主资源指标是 `glb.poseMacro`，跨 pose union 只在 `glb.unionDiagnostic`。

## 变更记录

日期：2026-09-09。目的：完成 Geometry-shell HZB 的 lossless 外壳导出、严格 equal-asset 传输预算、WebGPU depth/HZB MVP、Point60/Region66 接口及可审计的硬件 smoke 路径；不改训练、PVS evaluation 或 streaming。

修改文件：`neural_instance_culling/benchmark/geometry_shell_hzb_exporter.mjs`、`neural_instance_culling/benchmark/evaluate_geometry_shell_hzb.py`、对应 exporter/evaluator 测试，以及 `slm2viewer/src/GeometryShellHZB*.js`、benchmark page/runner 和 core 测试。追加修正把 selected occluder 自保留与非 selected AABB/HZB 测试写入 WGSL、CPU contract 和 `queryContract`，并把 GLB 主指标改为 per-pose macro。依赖为现有 `meshoptimizer`、Three.js/Parcel、Chrome WebGPU 和显式 scene/data root；源场景只读取主工作区的 HKUST/IFCBench 目录，输出写入本 worktree。

CPU 导出命令使用 `--scene-root`、`--output-dir`、`--variant`；equal-asset 另传 `--budget-from-dir`、`--importance-pose-plan` 和 `--importance-pose-count 128`。代表性命令如下：

```bash
node neural_instance_culling/benchmark/geometry_shell_hzb_exporter.mjs \
  --scene-root /mnt/sda/rhyang/slm/ifcbench_fantasy_metropolis_instanced_v2 \
  --output-dir neural_instance_culling/benchmark/out/paper_results/hzb/geometry_shell_hzb_equal_asset_ifcbench \
  --variant equal-asset --budget-from-dir <neural-runtime-dir> \
  --importance-pose-plan <ifcbench-pose-plan.jsonl> --importance-pose-count 128 --overwrite
```

浏览器结果评价在可得源 GLB 字节时显式传入 index/root，例如：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/evaluate_geometry_shell_hzb.py \
  --result neural_instance_culling/benchmark/out/paper_results/hzb/smoke_region66_v4/region66.json \
  --dataset-dir <ifcbench-csr-dataset> \
  --runtime-meta /mnt/sda/rhyang/slm/ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json \
  --output neural_instance_culling/benchmark/out/paper_results/hzb/smoke_region66_v4/metrics.json \
  --glb-index /mnt/sda/rhyang/slm/ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json \
  --glb-root /mnt/sda/rhyang/slm/ifcbench_fantasy_metropolis_instanced_v2/assets
```

该版本保留为论文 Geometry-shell HZB 基线/资产敏感性对照；当前风险是外壳仍未完成独占 GPU 的全 test visibility、图像损失和端侧正式计时，Chrome 还有 external Instance warning。

## 验证与剩余工作

已通过：

- `node neural_instance_culling/benchmark/test_geometry_shell_hzb_exporter.mjs`；
- `node slm2viewer/scripts/test_geometry_shell_hzb_core.mjs`；
- `conda run -n slm_pvs python -m unittest neural_instance_culling.benchmark.tests.test_evaluate_geometry_shell_hzb`；
- 修正后的 Point60/Region66 browser smoke，均有 WebGPU adapter 和 before/during/after GPU evidence；
- 两场景完整 lossless/equal-asset CPU exporter，runtime 文件总量和 equal-asset 硬预算均已复核；
- 四个 runtime 目录无 `primitiveAudits`，四个对应的 `.offline.json` sidecar 含离线审计且不在 runtime 目录；
- 两场景完整 meshopt stream 解码与长度/有限性检查；
- Parcel benchmark bundle 构建；
- `npm run probe:webgpu-hardware`（独立硬件门）。

`npm test` 未能完成：第一项 `test_current.mjs` 找不到 worktree 内的 `assets/scenes/hkust-v3/glbIndex.json`；单独运行的其余现有前端测试中 11 项通过，`test_instance_pvs_wasm_runtime.mjs` 因 `/model/model_meta.json` 缺失失败。这些是缺少前端测试资产造成的环境阻碍，不是 HZB 专项失败。

剩余正式运行：

1. 等待 GPU 0-3 的 IFCBench 扫描结束，在独占 GPU 窗口重做 Point60 和 Region66 小规模验证；当前 v4 两次 smoke 均因并发明确标为 `hardware-smoke-concurrent`；
2. 按计划校准 resolution/depth bias，并在完整 test split 上执行 HZB visibility 与图像评价；
3. 在 A6000、M2 和真实移动设备分别记录冷启动、depth-only、mip、AABB/query/readback p50/p95；
4. 定位 Chrome external Instance warning 后再冻结正式 HZB 性能表。

本报告不把并发 smoke 或 CPU 导出耗时当作正式硬件性能结论。
