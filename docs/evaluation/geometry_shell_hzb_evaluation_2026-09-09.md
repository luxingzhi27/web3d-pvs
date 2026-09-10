# WebGPU Batched Geometry-shell Hi-Z 评价报告

日期：2026-09-09；2026-09-10 更新

状态：四套 v2 外壳已完成。修正后的 calibration、frozen test 与 120-pose timing 正在独占 A6000 上执行，尚未形成完整 test 结论。

## 目的

该基线在正式场景 GLB 到达前预下载不透明几何外壳，通过 WebGPU 深度光栅、max-depth pyramid 和候选 AABB 测试生成实例级 PVS。它与神经方法比较启动资产、画面安全、剔除效率和运行压力。

正式名称固定为 **WebGPU Batched Geometry-shell Hi-Z**。实现借鉴经典 Hierarchical Z-Buffer 和现代 GPU 批量查询方法，但不称为 HROC 复现。它不包含对象 BVH、连续帧可见集、fragment-ray traversal 或 indirect multidraw。

## 输入与外壳语义

| 场景 | GLB | 实例 | 源 GLB 字节 | 可作外壳的几何 |
|---|---:|---:|---:|---|
| HKUST | 3,273 | 18,831 | 563,269,492 | 3,042 个确定 OPAQUE primitive；18,566 个 occluder 实例 |
| IFCBench Metropolis | 3,669 | 41,298 | 183,335,200 | 3,669 个确定 OPAQUE primitive；41,298 个 occluder 实例 |

HKUST 的 229 个 BLEND primitive 和 2 个空占位 GLB 不写遮挡深度。透明、alpha-cutout、材质 alpha 小于 1、透明材质扩展和未知材质扩展均按非遮挡物处理；对应候选仍能作为 occludee 被外壳测试。

`geometry-shell-hzb-v2` 运行资产包含：

- Meshopt 压缩的 POSITION、INDEX 和实例变换；
- 全部实例 AABB 与 instance-to-GLB 映射；
- 外壳实例对应的真实 `componentGlobalId`；
- 每实例是否进入外壳的二值表；
- `shell_meta.json`。

法线、切线、UV、颜色、纹理和 PBR 参数不进入运行资产。逐 primitive 审计只存在于目录外的 `.offline.json`，不计入启动下载。

## 资产结果

以下字节来自 v2 运行目录内实际下载文件。解码内存是 POSITION/INDEX/transforms 展开后加固定 runtime 表，不包括 HZB 纹理和浏览器实现。

| 场景 | 变体 | 传输字节 | 传输 MiB | 解码运行内存 MiB | prototype | 展开三角形 | occluder 实例 |
|---|---|---:|---:|---:|---:|---:|---:|
| HKUST | lossless | 399,388,459 | 380.89 | 1,697.87 | 3,042 | 55,537,431 | 18,566 |
| HKUST | equal-asset | 5,476,306 | 5.22 | 19.34 | 91 | 822,121 | 1,257 |
| IFCBench | lossless | 60,089,560 | 57.31 | 154.56 | 3,669 | 25,556,160 | 41,298 |
| IFCBench | equal-asset | 11,835,015 | 11.29 | 26.20 | 1,412 | 6,267,776 | 31,909 |

神经运行资产为 HKUST `5,479,213 B`、IFCBench `11,883,919 B`。因此 equal-asset 外壳分别保持在对应神经预算以内。Equal-asset 只删除完整 primitive，不移动顶点；它按 128 个固定 train 中心视点的投影面积/压缩字节选择外壳，是资产敏感性对照，不是保守简化上界。

## GPU 查询

每个 subpose 执行：

```text
RenderBundle depth + nearest component ID
  -> nearest-ID attachment 压缩成实例 bitset
  -> max-depth mip chain
  -> 候选实例 AABB/bitset 查询
  -> 可见实例和 GLB flag 压缩
  -> 只回读最终编号和阶段计时
```

已经进入外壳的实例只有在贡献最近深度像素时才保留。没有进入外壳的候选使用保守 AABB/Hi-Z 测试。只有 `hzbMax + depthBiasM < candidateNear` 才剔除；相机位于 AABB 内、穿越近裁剪面、投影非有限或 footprint 越界时保留。

固定绘制命令预录为 WebGPU `RenderBundle`，避免每个 pose 在 JavaScript 中重新提交数千个 draw。当前候选已经由同一 `66°` candidate CSR 完成视锥筛选，Hi-Z 查询对候选一线程一个 AABB。对象 BVH 只可能减少这一阶段，不会减少外壳下载、解码、深度光栅和 mip 构建成本。

## Point60 与 Region66

- `Point60`：真实 `60°` canonical 相机只运行一次，GT 来自该 view-cell 的 `subpose_id=0` 硬件 Color-ID 重渲染。
- `Region66`：使用和学习方法相同的区域候选，对登记 subpose 分别执行 Hi-Z 后取可见实例并集。

Point60 不读取 Region66 的可见并集。Region66 的 `1/5/9/all` 子集按空间覆盖选择，不读取可见标签；所有 subpose 查询时间都计入基线成本。

## 正式矩阵

唯一正式入口：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_geometry_shell_hzb_paper.py all
```

执行顺序为 `preflight -> calibration -> selection -> frozen test -> timing`：

| 阶段 | HKUST | IFCBench |
|---|---:|---:|
| Calibration | `512x288/1024x576` × `0.1/1/10 mm`，lossless | 相同 |
| Frozen test | Region `1/5/9/all` × lossless/equal-asset | Region `1/all` × lossless/equal-asset |
| Timing | 120 个固定 test pose × 5 轮 × 两种外壳 | 相同 |

Calibration 只决定分辨率和 depth bias。选择器要求 weighted recall 与其单侧 95% LCB 均严格大于 `0.99`，安全成员中先比较 useful cull，再比较 balanced accuracy、specificity、precision 和延迟。Equal-asset 复用 lossless 冻结配置，不重新校准。

正式浏览器结果必须具有 NVIDIA WebGPU adapter、NVIDIA ANGLE/Vulkan 辅助 renderer、`formalReady=true`、`executionClass=formal-hardware-gpu`、无 WebGPU validation error，以及同一窗口无其他 GPU 计算进程的 `nvidia-smi/pmon` 证据。

## 指标

所有方法使用相同 candidate CSR、GT、split 和 evaluator。正文报告：

- pose-macro 与 aggregate precision、Visible Recall、Occlusion Recall、False Occlusion Rate、balanced accuracy；
- weighted recall 及其单侧 95% LCB；
- Useful Cull Ratio、Bad Cull Ratio、平均保留实例和 GLB 数/字节；
- image PER、miss-pixel、wrong-ID 和 extra-pixel；
- 外壳传输字节、解码内存、depth/ID/mip/query/compaction/readback 与总时间 p50/p95。

本项目以可见为正类：`Occlusion Recall = specificity = TN/(TN+FP)`，`False Occlusion Rate = 1-recall = FN/GT`。`usefulCull=TN/candidate`、`badCull=FN/candidate` 使用不同分母，不能互相替代。完整定义见[统一评价协议](unified_pvs_metrics_evaluation.md)。

## 验证与当前状态

2026-09-10 已通过：

- v2 exporter、Meshopt 解码、visible-ID compaction 和 Region 选择专项测试；
- 48 个 HZB、图像和 streaming Python `unittest`；
- 完整 `slm2viewer npm test`；
- 32 项正式 dry-run；
- NVIDIA A6000 无头 WebGPU adapter 门。

当前没有可报告的正式 HZB test 或图像数字。早期两 pose smoke、旧 selected-self 逻辑和使用 Region union 作为 Point60 GT 的结果均已删除；后续只从当前正式根目录生成论文表格和 streaming 输入。详细任务目录、恢复规则和失败诊断见[HZB 正式执行文档](pvs_hzb_formal_execution_2026-09-09.md)。

## 变更记录

- 修改代码：`geometry_shell_hzb_exporter.mjs`、`evaluate_geometry_shell_hzb.py`、`run_geometry_shell_hzb_paper.py`、`GeometryShellHZB*.js`、benchmark runner/page 及对应测试。
- 依赖：现有场景 GLB/runtime meta、Pose CSR、Region source、Meshopt、Chrome WebGPU 和硬件证据入口。
- 主线决定：保留 lossless 为完整几何基线，equal-asset 为同启动预算敏感性；两者都不替代 Full V4。
- 待完成：正式 Region66/Point60、test 图像、A6000/移动端计时和 visible-weight streaming 接入。
