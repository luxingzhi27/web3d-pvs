# M5 Dense Subpose 评价协议（2026-08-04）

## 目的

修正第一轮 M5 图像评价只从每个 view-cell 选取一个 subpose 的执行缺陷。HKUST view-cell 数据集包含 `7,999` 个 view-cell 和 `279,008` 个 dense subpose，每个 view-cell 内的位置扰动具有相同的后退相机朝向。模型仍只对 view-cell 的冻结后退相机查询一次，图像安全评价则在该 view-cell 的全部 dense subpose 相机上检查同一预测集合的画面结果。

本次是评价协议修正，不改变模型、候选集合、真实可见集合、FOV、阈值或前端规则。旧的 sampled-subpose 结果保留在 `neural_instance_culling/benchmark/out/m5_visual_safety_repair_image_chunked_20260803/`，不与本次结果合并。

## 固定语义

- `--subposes-per-viewcell 0`：选择当前 view-cell 的全部 dense subpose；
- 正整数：按照已有的确定性等距规则选择指定数量；
- 负数：立即报错；
- 模型查询使用冻结的 66° 后退相机输入；图像参考和预测使用真实 60°相机；
- reference 是完整本地 GLB inventory 的实例级 Color-ID 渲染，prediction 只改变实例可见性掩码；
- 不向候选集合补入 GT，不使用前端白名单，不读取 test split；
- calibration 阈值来自 checkpoint 自己的 calibration 摘要，评价过程不重新扫描阈值。

manifest 和汇总结果记录请求值、实际 subpose 总数、每个 view-cell 的最小/最大/平均数量以及采样模式。浏览器批处理保持完整 GLB inventory 约束，分块只限制浏览器进程寿命，不改变样本、候选或预测实例列表。

## 执行入口

选择函数契约：

```bash
conda run --no-capture-output -n slm_pvs \
  python -u neural_instance_culling/benchmark/evaluate_viewcell_image_per.py --self-test
```

全量 manifest 入口使用独立输出名和独立 schema 根目录：

```bash
conda run --no-capture-output -n slm_pvs \
  python -u neural_instance_culling/benchmark/run_m5_visual_safety_image_evaluation.py \
  --subposes-per-viewcell 0 \
  --schema-only \
  --output-name m5_visual_safety_repair_image_dense_20260804 \
  > neural_instance_culling/benchmark/out/m5_visual_safety_repair_image_dense_20260804_schema_stdout.log \
  2> neural_instance_culling/benchmark/out/m5_visual_safety_repair_image_dense_20260804_schema_stderr.log
```

正式浏览器阶段使用同一 schema 根目录，采用有界页面和同相机 reference 复用；默认要求浏览器使用硬件 WebGL/Vulkan 后端，若后端识别为 SwiftShader 或其他软件渲染器则直接失败；该阶段仍仅运行 validation/calibration：

```bash
conda run --no-capture-output -n slm_pvs \
  python -u neural_instance_culling/benchmark/run_m5_visual_safety_image_evaluation.py \
  --subposes-per-viewcell 0 \
  --chunk-samples 512 \
  --max-chunk-manifest-bytes 400000000 \
  --output-name m5_visual_safety_repair_image_dense_hw_20260804 \
  > neural_instance_culling/benchmark/out/m5_visual_safety_repair_image_dense_hw_20260804_stdout.log \
  2> neural_instance_culling/benchmark/out/m5_visual_safety_repair_image_dense_hw_20260804_stderr.log
```

## Pilot 结果

硬件路径核验和 Chrome 参数记录见 `docs/evaluation/m5_hardware_gpu_renderer_2026-08-04.md`。当前最佳第一轮视觉修复 checkpoint `m5_visual_mass_soft / seed20260803` 已完成一个 validation view-cell 的 dense pilot：

| 项目 | 结果 |
|---|---:|
| dense subpose | 32 |
| GLB 加载失败 | 0 |
| 渲染失败 | 0 |
| self-consistency PER | 0 |
| miss-pixel rate | 0.09833% |
| wrong-ID pixel rate | 0.70099% |
| 浏览器耗时 | 134.3 s（历史 pilot，使用软件 SwiftShader） |

pilot 输出为 `neural_instance_culling/benchmark/out/m5_dense_subpose_pilot_20260804/`。它只验证 manifest、实例绑定、FOV 和浏览器渲染链路，不替代 validation/calibration 质量门。由于单个 view-cell 需要加载约 3,181 个 GLB，正式阶段必须使用同页面参考复用和 128 样本分块。旧 pilot 的 SwiftShader/ReadPixels 延迟不能解释为硬件 GPU 性能；硬件 GPU 复核必须使用独立输出目录并保留 `gpuBackend`、`gpuGate` 和 `nvidia-smi` 证据。

独立硬件 GPU pilot 输出为 `neural_instance_culling/benchmark/out/m5_dense_subpose_pilot_hw_20260804/`：32 个 subpose、3,181 个 GLB、加载/渲染失败均为 0，self-consistency PER 为 0，miss-pixel rate 为 `0.09900%`，wrong-ID pixel rate 为 `0.94841%`，页面总耗时 `22.59 s`。这组数字只验证真实 GLB 在硬件 GPU 路径上的渲染语义，不作为完整 M5 质量门结果。

## 质量门

正式 dense 结果仍使用预登记门槛：validation/calibration 的 mean miss-pixel rate `<0.5%`、view-cell miss-pixel p95 `<1%`，并检查系统性 pop-in、GLB/实例绑定错误和渲染失败。若 dense 评价未通过，记录 M5 `No-Go`，不能通过降低阈值、改变候选集合或补入 GT 修复。

## 全量硬件 GPU 结果（2026-08-04）

正式输出为 `neural_instance_culling/benchmark/out/m5_visual_safety_repair_image_dense_hw_20260804/`。48 个浏览器分块共完成 `536,256` 个 validation/calibration dense subpose，24 个 batch、3,273 个完整 GLB inventory；GLB 加载失败和渲染失败均为 `0`，self-consistency PER 为 `0`。Chrome 48 个分块均回报 `ANGLE (NVIDIA, Vulkan ... RTX A6000 ...)`，`gpuGate.hardware=true`，并保存了 `nvidia-smi` 快照。

全量 aggregate PER 为 `1.5286%`，miss-pixel rate 为 `0.7488%`，wrong-ID pixel rate 为 `0.7798%`。validation miss-pixel rate 为 `0.8418%`，calibration 为 `0.6599%`；各 batch 的 dense subpose p95 均高于 `1%`。因此硬件渲染链路通过，但视觉安全质量门为 `No-Go`。详细表格和历史软件结果边界见 `docs/evaluation/m5_visual_safety_repair_image_dense_hw_2026-08-04.md`。

执行中第 40 个 manifest 曾达到约 `840 MB` 并触发 Node 单字符串上限。分块器现默认限制 manifest 为 `400,000,000` 字节，并在恢复时按样本 ID 集合验证后复用已完成分块；该工程修复不改变样本、候选、GT、预测、FOV 或阈值。
