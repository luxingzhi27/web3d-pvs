# V4 主线 HKUST 与 IFCBench 图像级评价

日期：2026-09-09

## 目的与口径

本次评价检查当前 V4 实例可见性模型在真实前端相机下造成的画面损失。评价使用 validation split、seed `20260802` 的 `best_safe.pt` 和各 checkpoint 在 calibration 上冻结的阈值；评价过程不扫描阈值、不读取 test、不补入 GT，也不改变候选集合。

- HKUST 阈值为 `0.68`，IFCBench 阈值为 `0.76`。
- 模型仍以 `66°` 后退视锥候选执行一次 view-cell 查询；图像 reference 与 prediction 均使用真实 `60°` 显示相机。
- 每个 validation view-cell 的全部真实 subpose 都参与评价，分辨率为 `480×270`。
- reference 使用完整本地原始 GLB 三角形和实例变换；prediction 只改变实例级可见性掩码。
- Playwright 无头 Chrome 使用 NVIDIA RTX A6000 的 Vulkan/ANGLE WebGL 2 硬件路径。页面后端字段以及同一窗口的 `nvidia-smi`、`pmon` before/during/after 证据均已保存。

图像指标中的 aggregate 先合并全部像素再计算；mean、median 和 p95 则先按每个真实 subpose 计算，再汇总其分布。Image PER、miss 和 wrong-ID 的分母为 reference 非背景像素；extra 的分母为全部图像像素。

## 完整性检查

| 场景 | Validation view-cell | 真实 subpose | 本地 GLB | 可渲染实例 | 失败 subpose | 缺失 GLB | 硬件门 |
|---|---:|---:|---:|---:|---:|---:|---|
| HKUST | 730 | 25,568 | 3,273 | 18,829 | 0 | 0 | NVIDIA Vulkan/ANGLE 通过 |
| IFCBench | 2,712 | 10,848 | 3,669 | 41,298 | 0 | 0 | NVIDIA Vulkan/ANGLE 通过 |

HKUST 有两个不含 mesh node 的空 GLB，预检将其明确登记为不可渲染资源；它们没有造成缺失文件或失败帧。IFCBench 的全部实例均可建立 `componentGlobalId -> GLB instance slot` 绑定。两场景的 reference 自一致 PER 均为零。

## 图像结果

| 场景 | Aggregate PER | Mean PER | Median PER | P95 PER | Aggregate miss | P95 miss | Aggregate wrong-ID | P95 wrong-ID | Extra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HKUST | 0.3247% | 0.3196% | 0 | 1.1827% | 0.2232% | 0.6913% | 0.1016% | 0.3468% | 0 |
| IFCBench | 0.0545% | 0.0566% | 0 | 0.0317% | 0.000714% | 0 | 0.0538% | 0.0304% | 0 |

HKUST 的像素误差同时包含真正背景空洞和前景实例漏掉后露出后方实例造成的 wrong-ID。其 aggregate PER 为 `0.3247%`，但 p95 PER 达到 `1.1827%`，说明总体画面损失较低，困难视角尾部仍应单独报告。

IFCBench 的背景漏失仅为 `0.000714%`，主要误差是 wrong-ID。其 p95 小于 mean，是因为至少 95% 的 subpose 几乎无误差，而少量较大离群误差抬高了均值；不能把 p95 解释为最大误差。

## 同工作点的集合与剔除指标

| 场景 | Pose P | Pose R | Pose WR | WR 单侧 95% LCB | Agg P | Agg R | Agg WR | Agg accuracy | Agg balanced accuracy | Useful cull | Bad cull | Avg candidate / GT / pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HKUST | 0.4833 | 0.9725 | 0.99837 | 0.99642 | 0.2272 | 0.9703 | 0.99781 | 0.9235 | 0.9464 | 0.9013 | 0.000681 | 4747.9 / 109.0 / 465.6 |
| IFCBench | 0.2585 | 0.9964 | 0.99915 | 0.99882 | 0.1893 | 0.9956 | 0.99929 | 0.5929 | 0.7730 | 0.4980 | 0.000419 | 9884.2 / 942.5 / 4957.9 |

`Pose WR` 是逐 view-cell weighted recall 后取平均，`Agg WR` 是合并全部可见权重后计算。`Useful cull = TN/candidate`，`bad cull = FN/candidate`。低图像 PER 不能替代 precision、accuracy 或 useful cull：尤其 IFCBench 虽几乎不丢像素，但仍预测了约一半候选实例，说明大量 false positive 的视觉代价低，却仍会增加显示和资源负担。

| 场景 | Avg predicted GLB | Avg required GLB | Predicted GLB bytes / pose | Candidate-to-predicted byte reduction |
|---|---:|---:|---:|---:|
| HKUST | 177.61 | 53.89 | 40.94 MB | 77.26% |
| IFCBench | 675.05 | 239.82 | 16.32 MB | 65.94% |

GLB 指标来自同一 checkpoint、同一 validation pose 和同一冻结阈值的实例分数聚合，不是图像 renderer 的额外阈值或下载头结果。

HKUST 导出的神经资产为 `4,748,630` bytes（`4.53 MiB`）。IFCBench 有更多实例，其固定实例表使神经资产增至 `10,320,446` bytes（`9.84 MiB`）；本次导出将预算明确设为 `max(7 MiB, 256 bytes/instance + 128 KiB shared)`，而不是用只适合 HKUST 实例数的固定上限掩盖场景规模成本。

## 预览检查与结论

每个场景保存并检查了 8 张 `reference | prediction | diff` 预览。两侧相机、构件轮廓和实例 ID 对齐，没有空白帧、整体错位或后退 `66°` 相机误入图像评价。差异主要位于局部遮挡边界和少数漏失构件。

本次结果支持以下有限结论：当前 V4 主线在两个场景的代表性安全 checkpoint 上保持很低的真实画面像素损失；HKUST 的困难视角尾部明显高于整体均值，IFCBench 则主要存在低视觉影响的过量预测。由于本次没有对每个消融成员重复完整三角形渲染，它证明的是完整模型的绝对图像质量，不能单独证明某一创新模块改善了图像指标。

## 代码修正与复现

本次修正了三项正式评价边界：IFCBench split 应用时保留实际候选后退距离；导出资产预算按实例数扩展；空候选 view-cell 直接产生空预测。大型场景 manifest 改用紧凑 JSON，避免缩进数组超过 Node 字符串上限；renderer 也补齐逐 subpose PER、miss、wrong-ID 和 extra 的 mean、median、p95 字段。

主要修改文件：

- `neural_instance_culling/dataset/apply_pose_split_manifest.py`
- `neural_instance_culling/model/export_pvs.py`
- `neural_instance_culling/benchmark/model_runners.py`
- `neural_instance_culling/benchmark/evaluate_viewcell_image_per.py`
- `neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs`

正式命令使用以下统一入口，场景参数分别指向 HKUST 与 IFCBench 的 runtime meta、view-cell 数据、Pose CSR、原始 Color-ID 和本地 GLB：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --split validation --max-viewcells 0 --subposes-per-viewcell 0 \
  --width 480 --height 270 --device cuda --image-renderer true_glb \
  --formal-image-evaluation --require-hardware-gpu \
  --threshold <calibration-frozen-threshold> --preview-samples 8
```

正式结果保存在：

```text
neural_instance_culling/benchmark/out/pvs_v4_image_validation_hkust_seed20260802_v1/
neural_instance_culling/benchmark/out/pvs_v4_image_validation_ifcbench_seed20260802_v1/
```

当前结果保留为论文主线的 validation 图像证据。Test split 尚未读取；最终 test 只能在论文模型、阈值和协议全部冻结后执行一次。
