# WebGPU Batched Geometry-shell Hi-Z 评价报告

日期：2026-09-09；2026-09-11 更新

状态：四套 v2 外壳已完成。修正后的 calibration、frozen test 与 120-pose timing 正在独占 A6000 上执行，尚未形成完整 test 结论。

## 目的

该基线在正式场景 GLB 到达前预下载不透明几何外壳，通过 WebGPU 深度光栅、max-depth pyramid 和候选 AABB 测试生成实例级 PVS。它与神经方法比较启动资产、画面安全、剔除效率和运行压力。

运行契约固定为 shell schema `geometry-shell-hzb-v2` 和浏览器结果 schema `geometry-shell-hzb-browser-result-v2`。

正式名称固定为 **WebGPU Batched Geometry-shell Hi-Z**。实现借鉴经典 Hierarchical Z-Buffer 和现代 GPU 批量查询方法，但不称为 HROC 复现。它不包含对象 BVH、连续帧可见集、fragment-ray traversal 或 indirect multidraw。

## 输入与外壳语义

| 场景 | GLB | 实例 | 源 GLB 字节 | 外壳几何规模 |
|---|---:|---:|---:|---|
| HKUST | 3,273 | 18,831 | 563,269,492 | 3,042 个确定 OPAQUE primitive；18,566 个外壳实例 |
| IFCBench Metropolis | 3,669 | 41,298 | 183,335,200 | 3,669 个确定 OPAQUE primitive；41,298 个外壳实例 |

HKUST 的 229 个 BLEND primitive 和 2 个空占位 GLB 不写入外壳深度。透明、alpha-cutout、材质 alpha 小于 1、透明材质扩展和未知材质扩展均不进入外壳几何；对应实例仍在候选集合中接受统一查询。

`geometry-shell-hzb-v2` 运行资产包含：

- Meshopt 压缩的 POSITION、INDEX 和实例变换；
- 全部实例 AABB 与 instance-to-GLB 映射；
- `shell_meta.json`。

法线、切线、UV、颜色、纹理和 PBR 参数不进入运行资产。逐 primitive 审计只存在于目录外的 `.offline.json`，不计入启动下载。

## 资产结果

以下字节来自 v2 运行目录内实际下载文件。解码内存是 POSITION/INDEX/transforms 展开后加固定 runtime 表，不包括 HZB 纹理和浏览器实现。

| 场景 | 变体 | 传输字节 | 传输 MiB | 解码运行内存 MiB | shell prototype | 展开三角形 | shell 实例 |
|---|---|---:|---:|---:|---:|---:|---:|
| HKUST | lossless | 399,312,852 | 380.81 | 1,697.80 | 3,042 | 55,537,431 | 18,566 |
| HKUST | equal-asset | 5,147,576 | 4.91 | 18.29 | 81 | 780,713 | 1,245 |
| IFCBench | lossless | 59,924,084 | 57.15 | 154.40 | 3,669 | 25,556,160 | 41,298 |
| IFCBench | equal-asset | 11,819,713 | 11.27 | 26.54 | 1,405 | 6,290,848 | 31,796 |

神经运行资产为 HKUST `5,479,213 B`、IFCBench `11,883,919 B`。因此 equal-asset 外壳分别保持在对应神经预算以内。Equal-asset 只删除完整 primitive，不移动顶点；它按 128 个固定 train 中心视点的投影面积/压缩字节选择外壳，是资产敏感性对照，不是保守简化上界。

## GPU 查询

每个 subpose 执行：

```text
opaque shell raster -> positive linear view depth
  -> untouched pixels retain camera far depth
  -> max-depth mip chain
  -> every candidate conservative projected-AABB/HZB test
  -> candidate component IDs and GLB flags compacted after classification
  -> 只回读最终编号和阶段计时
```

外壳 pass 的唯一遮挡输入是正线性视空间深度；每个 candidate 都执行同一 conservative projected-AABB/HZB test。投影后的 NDC-y 转换为纹理坐标时固定使用 `texture_y = (1 - ndc_y) / 2`。只有 `hzbMax + depthBiasM < candidateNear` 才剔除；非有限投影、近裁剪面穿越、相机位于 AABB 内和屏幕外投影等不确定情况全部保留。

固定绘制命令预录为 WebGPU `RenderBundle`，避免每个 pose 在 JavaScript 中重新提交数千个 draw。当前候选已经由同一 `66°` candidate CSR 完成视锥筛选，Hi-Z 查询对每个 candidate 执行一个 AABB 查询线程；分类结束后才回读 component ID 和 GLB flag 压缩结果。

## Point60 与 Region66

- `Point60`：真实 `60°` canonical 相机只运行一次，GT 来自该 view-cell 的 `subpose_id=0` 硬件 Color-ID 重渲染。
- `Region66`：使用和学习方法相同的区域候选，对登记 subpose 分别执行统一的 Hi-Z 查询后取可见实例并集。

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
| Calibration | `512x288/1024x576` × `0.01/1/10/100 m`，lossless | 相同 |
| Frozen test | Region `1/5/9/all` × lossless/equal-asset | Region `1/all` × lossless/equal-asset |
| Timing | 120 个固定 test pose × 5 轮 × 两种外壳 | 相同 |

Calibration 只决定分辨率和 depth bias。选择器要求 weighted recall 与其单侧 95% LCB 均严格大于 `0.99`，安全成员中先比较 useful cull，再比较 balanced accuracy、specificity、precision 和延迟。Equal-asset 复用 lossless 冻结配置，不重新校准。

深度余量采用 `0.01/1/10/100 m` 对数跨度。该范围覆盖厘米级数值误差到大型场景中大 AABB 的保守深度余量；它只在 calibration 上选择，不读取 test。较大余量会降低错误剔除，同时也会增加保留数量，因此不会无代价地增强 HZB。

正式浏览器结果必须具有 schema `geometry-shell-hzb-browser-result-v2`、NVIDIA WebGPU adapter、NVIDIA ANGLE/Vulkan 辅助 renderer、`formalReady=true`、`executionClass=formal-hardware-gpu`、无 WebGPU validation error，以及同一窗口无其他 GPU 计算进程的 `nvidia-smi/pmon` 证据。

## 指标

所有方法使用相同 candidate CSR、GT、split 和 evaluator。正文报告：

- pose-macro 与 aggregate precision、Visible Recall、Occlusion Recall、False Occlusion Rate、balanced accuracy；
- weighted recall 及其单侧 95% LCB；
- Useful Cull Ratio、Bad Cull Ratio、平均保留实例和 GLB 数/字节；
- image PER、miss-pixel、wrong-ID 和 extra-pixel；
- 外壳传输字节、解码内存、depth/mip/query/compaction/readback 与总时间 p50/p95。

本项目以可见为正类：`Occlusion Recall = specificity = TN/(TN+FP)`，`False Occlusion Rate = 1-recall = FN/GT`。`usefulCull=TN/candidate`、`badCull=FN/candidate` 使用不同分母，不能互相替代。完整定义见[统一评价协议](unified_pvs_metrics_evaluation.md)。

## 验证与当前状态

2026-09-11 已通过：

- v2 exporter、Meshopt 解码、candidate component-ID compaction 和 Region 选择专项测试；
- HZB、图像和 streaming Python 专项 `unittest`；
- 完整 `slm2viewer npm test`；
- 36 项正式 dry-run；
- NVIDIA A6000 无头 WebGPU adapter 门。

当前没有可报告的正式 HZB test 或图像数字。后续只从当前正式根目录生成论文表格和 streaming 输入。详细任务目录、恢复规则和失败诊断见[HZB 正式执行文档](pvs_hzb_formal_execution_2026-09-09.md)。

## 变更记录

- 修改代码：`geometry_shell_hzb_exporter.mjs`、`evaluate_geometry_shell_hzb.py`、`run_geometry_shell_hzb_paper.py`、`GeometryShellHZB*.js`、benchmark runner/page 及对应测试。
- 依赖：现有场景 GLB/runtime meta、Pose CSR、Region source、Meshopt、Chrome WebGPU 和硬件证据入口。
- 主线决定：保留 lossless 为完整几何基线，equal-asset 为同启动预算敏感性；两者都不替代 Full V4。
- 待完成：正式 Region66/Point60、test 图像、A6000/移动端计时和 visible-weight streaming 接入。
