# M5 硬件 GPU Dense Subpose 图像评价报告

日期：2026-08-04  
场景：HKUST v3  
状态：完成，视觉安全门 `No-Go`

## 评价范围

本次评价修正了上一轮只抽取少量 subpose 的执行问题。每个 view-cell 使用全部 dense subpose，模型只查询一次冻结的 66° 后退相机输入，随后在这些同方向位置上使用真实 60° 相机进行实例级 Color-ID 图像检查。评价只读取 validation 和 calibration，未读取 test，也没有重新扫描阈值、补入真实可见实例、改变候选集合或使用前端白名单。

固定协议如下：

- `7,999` 个 view-cell，全部 dense subpose；总计 `536,256` 个图像样本；
- 4 个视觉修复变体、3 个随机种子、validation/calibration 两个 split，共 24 个 batch；
- 每个 batch 分别为 validation `21,888`、calibration `22,800` 个样本；
- 完整本地 GLB inventory 为 `3,273` 个，渲染分辨率为 `320 x 180`；
- reference 使用完整场景的实例级 `componentGlobalId` Color-ID，prediction 只应用冻结的实例预测集合；
- 前端真实渲染 FOV 为 `60°`，模型输入/后退相机 FOV 为 `66°`。

## 硬件路径

正式浏览器阶段使用 Playwright 启动 headless Chrome，参数包含 `--enable-gpu`、`--enable-webgl`、`--use-angle=vulkan`、`--enable-accelerated-2d-canvas`、`--enable-zero-copy` 和 `--disable-software-rasterizer`，并强制启用 `--require-hardware-gpu`。48 个有界浏览器分块的后端完全一致：

```text
vendor   = Google Inc. (NVIDIA)
renderer = ANGLE (NVIDIA, Vulkan 1.3.242 ... NVIDIA RTX A6000 ...)
gpuGate  = required=true, hardware=true
```

本机有 4 张 NVIDIA RTX A6000，驱动版本为 `535.183.01`。运行中的 `nvidia-smi pmon` 记录到 Chrome GPU 进程为 `C+G`；快照保存在评价输出目录的 `nvidia_smi_snapshot_20260804.csv` 和 `nvidia_smi_pmon_snapshot_20260804.txt`。GLB 加载失败和渲染失败均为 `0`，实例绑定自洽性 PER 为 `0`。

采样器也已改为默认硬件门，`run_sampler.mjs --smoke` 实测 4 个位置、8 个 GLB 加载成功，回报同一 RTX A6000 Vulkan 后端。`--allow-software-gpu` 仅用于明确标记的语义调试，不能用于正式数据集或硬件性能结论。

## 图像结果

全量合并结果为：

| 指标 | 结果 | 解释 |
|---|---:|---|
| 样本数 | 536,256 | validation/calibration 全部 dense subpose |
| PER | 1.5286% | 有效 reference 像素中的错误像素比例 |
| miss-pixel rate | 0.7488% | 预测漏掉实例造成的像素比例 |
| wrong-ID pixel rate | 0.7798% | 像素被错误实例覆盖的比例 |
| extra-pixel rate | 0% | 本次 reference/prediction 统计中的额外像素比例 |
| self-consistency PER | 0 | 实例 ID 绑定和重复渲染自洽性检查 |
| GLB 加载/渲染失败 | 0 / 0 | 浏览器资源和渲染完整性 |

按 split 合并：

| Split | 样本数 | miss-pixel rate | wrong-ID pixel rate | PER |
|---|---:|---:|---:|---:|
| validation | 262,656 | 0.8418% | 0.6743% | 1.5161% |
| calibration | 273,600 | 0.6599% | 0.8806% | 1.5406% |

各模型和种子的 aggregate miss-pixel rate 如下。表中的 `p95` 是对应 batch 的 dense subpose miss-pixel rate 第 95 百分位：

| 变体 | Seed | validation miss / p95 | calibration miss / p95 |
|---|---:|---:|---:|
| visual_mass_linear | 20260801 | 0.9083% / 6.2599% | 0.8037% / 6.0436% |
| visual_mass_tail | 20260801 | 1.0867% / 7.9976% | 0.5813% / 3.6279% |
| visual_mass_soft | 20260801 | 0.9606% / 6.2876% | 0.7223% / 5.5156% |
| control_log1p | 20260801 | 0.9415% / 6.5571% | 0.7923% / 5.4824% |
| visual_mass_linear | 20260802 | 0.9775% / 7.5010% | 0.8151% / 5.8845% |
| visual_mass_tail | 20260802 | 0.6873% / 4.0518% | 0.4924% / 2.7344% |
| visual_mass_soft | 20260802 | 0.9017% / 6.9698% | 0.6942% / 5.4997% |
| control_log1p | 20260802 | 1.1648% / 7.9665% | 0.5965% / 3.6842% |
| visual_mass_linear | 20260803 | 0.5943% / 3.6169% | 0.6125% / 4.1383% |
| visual_mass_tail | 20260803 | 0.6227% / 3.5303% | 0.6374% / 4.7654% |
| visual_mass_soft | 20260803 | 0.5766% / 3.4307% | 0.4904% / 2.8236% |
| control_log1p | 20260803 | 0.6800% / 4.1488% | 0.6814% / 5.0396% |

## 质量门结论

预登记门槛为 aggregate mean miss-pixel rate `<0.5%`，以及 dense subpose 的 p95 `<1%`。全量 miss-pixel rate 为 `0.7488%`，validation 为 `0.8418%`，calibration 为 `0.6599%`；各 batch 的 p95 均高于 `1%`。因此 M5 dense 图像安全门不通过，结论为 `No-Go`。

这个结论只说明当前冻结预测集合在 view-cell 内的位置扰动下仍有明显漏像素，不能归因于软件渲染、GLB 加载失败或实例绑定错误。硬件渲染链路已经通过，模型安全性仍未通过。不能通过降低阈值、改变候选集合或补入 GT 来修复该结论。

## 运行成本与分块修复

评价使用 48 个浏览器页面分块；每个分块最多保留每个源 batch 的 512 个样本，并保留完整 GLB inventory 语义。实际 renderer 累计耗时约 `7,755 s`，GLB 加载调用 `156,786` 次；这些数字是离线图像评价成本，不是移动端前端性能结论。相同相机的 reference 在每个有界页面内复用，页面结束后释放 GLB。

执行中发现第 40 个旧分块 manifest 约 `840 MB`，超过 Node 单字符串限制。分块器随后增加 `400,000,000` 字节上限和按样本 ID 校验的恢复机制，已完成分块直接复用，尾部重新渲染为 48 个可读分块。该修复只改变浏览器进程边界，不改变样本、候选、GT、预测、FOV 或阈值。

## 可复现产物

- 结果目录：`neural_instance_culling/benchmark/out/m5_visual_safety_repair_image_dense_hw_20260804/`；
- 汇总：`summary.json`、`summary.md`、`true_glb_render/render_summary.json`；
- GPU 证据：`nvidia_smi_snapshot_20260804.csv`、`nvidia_smi_pmon_snapshot_20260804.txt`；
- 协议：`docs/experiments/m5_dense_subpose_protocol_2026-08-04.md`；
- 硬件路径说明：`docs/evaluation/m5_hardware_gpu_renderer_2026-08-04.md`。

本次没有修改默认模型、默认前端资产、阈值、候选集合或历史软件渲染结果。历史 SwiftShader 输出仍可用于旧的语义调试，但不能与本报告的硬件路径耗时混合，也不能用来填写移动设备 GPU 性能结论。
