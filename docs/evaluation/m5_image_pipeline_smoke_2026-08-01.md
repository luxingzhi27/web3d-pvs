# M5 实例级 Color-ID 图像评价链路 smoke

日期：2026-08-01  
状态：链路 smoke 通过；M5 正式图像质量门未通过。  
目的：确认正式 Pose CSR 空间 split、真实 60°相机、实例级 ID 缓冲和本地 GLB 浏览器渲染可以端到端执行。

## 运行范围

本次使用 HKUST 正式空间 Pose CSR 的 validation split，抽取 2 个 view-cell，每个 view-cell 取 1 个 subpose。渲染器加载本地 HKUST GLB 资产，并按 `componentGlobalId` 进行实例绑定和 Color-ID 输出。模型使用已有的探索性 `pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best`，不是尚未完成的 2026-08-01 正式训练 checkpoint。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -u neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --model-name pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --viewcell-dataset neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source \
  --pose-csr neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1 \
  --glb-root hkust-v3/assets --glb-index hkust-v3/assets/glbIndex.json \
  --glb-points-meta neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66_meta.json \
  --split validation --split-source pose_csr --max-viewcells 2 \
  --subposes-per-viewcell 1 --image-renderer true_glb --device cpu \
  --save-id-buffers --skip-raw-subpose-gt --render-timeout-sec 900 \
  --output-dir /tmp/neuralstream_m5_hkust_trueglb_smoke_20260801
```

输出目录为 `/tmp/neuralstream_m5_hkust_trueglb_smoke_20260801`，不作为长期 benchmark 资产。

## 结果

| 项目 | 结果 | 含义 |
|---|---:|---|
| evaluated subposes | 2 | 两个 subpose 均进入浏览器渲染 |
| render failed subposes | 0 | 浏览器/GLB 渲染没有失败 |
| missing GLB subposes | 0 | 本地 GLB 索引与实例映射完整 |
| valid reference pixels | 126,617 | 有参考实例 ID 的像素数 |
| miss pixels | 17,929 | GT 实例未被预测保留造成的画面损失 |
| wrong-instance pixels | 1,951 | 预测实例 ID 与参考实例 ID 不一致的像素 |
| miss-pixel rate | 14.1600% | `missPixels / validReferencePixels` |
| wrong-instance pixel rate | 1.5409% | `wrongInstancePixels / validReferencePixels` |
| PER | 15.7009% | 当前图像 ID 错误的聚合比例 |

本次输出证明的是评价链路和实例绑定可运行，不证明模型质量。miss-pixel rate 明显高于计划中的 mean `<0.5%` 门槛，因此该 checkpoint 不能进入 M5 主表，也不能通过调整阈值把 smoke 改写为正式结果。正式评价必须等待空间正式训练完成，使用 calibration 冻结阈值，在完整 validation/calibration 上运行后再封存 test one-shot。

## 其他协议核验

HKUST 的 schema-only 完整预检同时遍历了 validation 的 213 个 view-cell 和 calibration 的 168 个 view-cell，均成功生成 3,273 个 GLB 的实例绑定清单；两次输出均明确标记 `formalImageEvaluationReady=false`，因为没有启动真实浏览器渲染。这两次预检只通过了 split 和资源语义门，不计入图像质量门。

## 后续

1. 正式 checkpoint 完成后，用同一命令替换为正式模型和 calibration 冻结阈值。
2. 先完整运行 validation/calibration，检查所有 subpose 的渲染失败、缺失 GLB、实例 ID 冲突和 miss-pixel 长尾。
3. 只有在阈值、模型和图像协议冻结后，才运行 test 一次；test 不重新扫描阈值。
