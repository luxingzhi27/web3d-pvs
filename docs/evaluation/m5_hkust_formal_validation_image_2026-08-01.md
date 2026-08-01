# M5 HKUST 正式模型 validation 图像评价

日期：2026-08-01
状态：validation 图像评价完成，图像安全门 No-Go；未读取 test split。

## 评价口径

本次使用 HKUST 空间隔离 validation split 的全部 `213` 个 view-cell，每个 view-cell 取一个真实 subpose。模型输入和后退相机使用 66°，真实图像使用固定 60°垂直视场角。浏览器加载全部 `3,273` 个本地 GLB，在实例槽位上绑定 `componentGlobalId`，reference 是完整场景的实例级 Color-ID 深度结果，prediction 是正式模型阈值下的实例集合。reference 与 prediction 都在同一个 Chrome 页面和同一相机下渲染。

运行命令：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --model-spec 'pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2|neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/best.pt|neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/instance_runtime_features_fp16.bin|neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/calibration_ready_summary.json' \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --viewcell-dataset neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source \
  --pose-csr neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1 \
  --glb-root hkust-v3/assets --glb-index hkust-v3/assets/glbIndex.json \
  --split validation --split-source pose_csr --max-viewcells 0 \
  --subposes-per-viewcell 1 --width 320 --height 180 \
  --image-renderer true_glb --chrome-exe /usr/bin/google-chrome \
  --device cuda --skip-raw-subpose-gt \
  --output-dir neural_instance_culling/benchmark/out/m5_hkust_strong_v2_validation_image_20260801_full
```

阈值由 checkpoint 的 calibration-ready 摘要读取，本次没有重新扫描 validation 或 test。渲染器明确记录 `formalImageEvaluationReady=false`，因为这是冻结 test 之前的 validation 质量检查，不是最终 one-shot test。

## 结果

| 指标 | 聚合结果 | 中文含义 |
|---|---:|---|
| view-cell / subpose | 213 / 213 | 全部 validation view-cell 均完成真实渲染 |
| 本地 GLB / 实例 | 3,273 / 18,831 | 完整场景清单和实例绑定规模 |
| 有效 reference 像素 | 6,821,854 | 参考图中属于实例的像素数 |
| miss pixels | 33,749 | 真实实例未被预测保留造成的漏像素 |
| wrong-instance pixels | 37,098 | 预测实例 ID 与 reference 实例 ID 不一致的像素 |
| aggregate miss-pixel rate | 0.4947% | `missPixels / validReferencePixels` |
| aggregate wrong-ID rate | 0.5438% | `wrongInstancePixels / validReferencePixels` |
| aggregate PER | 1.0385% | 漏实例和错误实例 ID 的有效像素比例 |
| extra-pixel rate | 0 | 预测额外像素占整幅图像的比例 |
| self-consistency PER | 0 | reference 与自身比较的渲染一致性检查 |

按单个 view-cell 统计的长尾结果如下：

| 指标 | 数值 |
|---|---:|
| miss-pixel rate 均值 | 0.5409% |
| miss-pixel rate 中位数 | 0% |
| miss-pixel rate p95 | 3.1973% |
| miss-pixel rate 最大值 | 13.1422% |
| 单 view-cell PER 均值 | 1.0661% |
| 单 view-cell wrong-ID rate 均值 | 0.5252% |

计划中的图像安全门是 mean miss-pixel rate `<0.5%` 且 p95 `<1%`。聚合值虽然略低于 0.5%，但单 view-cell 均值已经超过 0.5%，p95 明显超过 1%，所以 M5 图像门不通过。该结果不能通过改写统计口径或直接调整 test 阈值来宣称通过。

## 长尾诊断

漏像素主要集中在少数大型构件，例如 `componentGlobalId=15226` 的 `task-4/glb/LOD0/sub_5471.glb` 单项贡献 `25,303` 个漏像素，随后是同一任务组的若干构件。该现象说明当前模型在部分 view-cell 对高视觉贡献实例的安全保护不足，不能解释为仅由不可见细小构件造成的普通 precision 损失。后续改进应在 validation/calibration 上诊断候选生成、view-cell union 监督、构件权重和阈值工作点，再重新冻结模型；不能直接读取 test 寻找更低阈值。

## 运行成本

- 完整 GLB 加载：约 `22.61 s`。
- 213 个 subpose 的 reference/prediction 渲染：约 `1,250.27 s`，约 20.84 分钟。
- 页面总耗时：约 `1,273.08 s`。
- 浏览器使用 headless Chrome 软件 WebGL/SwiftShader 路径；该耗时不是硬件 WebGPU 或移动设备性能结论。

M5 的质量门和 M10 的设备性能门分别判断，不能用离线图像渲染耗时替代前端推理耗时。

## 保留判断

本结果作为正式模型 validation 图像失败证据保留，不能进入通过质量门的论文主表。现有实例 ID schema、完整场景 reference、同页 GLB 复用和 self-consistency 检查有效；当前需要解决的是模型/阈值在大型高贡献构件上的漏检长尾，以及真实图像评价的高成本。test 图像评价仍保持封存。
