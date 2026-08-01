# HKUST 正式模型导出与前端主线切换

**日期：** 2026-08-01  
**状态：** HKUST 导出完成；Metropolis 仍在正式训练；M5 图像质量门未完成。

## 目的

将已完成空间协议训练和独立校准的 HKUST `rvl_strong_v2` checkpoint 接入当前前端，替换仍被默认路径引用的历史 `rvl_w042` 资产。导出过程必须复用训练产出的冻结校准工作点，不能在测试集重新扫描阈值，也不能用旧的 `0.64` 兼容阈值覆盖新模型。

## 修改内容

- `export_directional_occlusion_proxy_frontend.py` 新增正式 `calibration_ready_pre_test` 摘要解析。
- 导出器验证 `testEvaluationCount=0`、冻结阈值与校准 selected 行一致、weighted recall 点估计、校准点下限和 bootstrap 下限均满足注册规则。
- HKUST 前端默认模型、M9 静态审计、当前完整性 smoke 和部署打包脚本统一指向：
  `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best`。
- 旧 `rvl_w042` 资产仍保留用于历史复现，但不再是当前默认前端模型。
- 同步更新当前版本和架构文档，明确 `0.64` 只属于旧探索性结果。

## 依赖资源

```text
checkpoint:
  neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/best.pt
runtime features:
  .../instance_runtime_features_fp16.bin
calibration:
  .../calibration_ready_summary.json
runtime metadata:
  hkust-v3/assets/runtimeVisibilityMeta.json
```

## 运行命令

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py \
  --checkpoint neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/best.pt \
  --runtime-features neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/instance_runtime_features_fp16.bin \
  --feature-meta neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/instance_features_meta.json \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --eval-summary neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/calibration_ready_summary.json \
  --eval-model-name pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best \
  --output-dir slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best

cd slm2viewer
npm run build
npm test
npm run test:m9
```

## 结果

| 项目 | 结果 |
|---|---:|
| 导出模型 | `rvl_strong_v2_full40_best` |
| 冻结校准阈值 | `0.02` |
| calibration weighted recall | `0.9930808` |
| bootstrap 单侧 95% 下界 | `0.9909417` |
| 前端运行资产 | `13,844,856` bytes |
| 场景实例 / GLB | `18,831 / 3,273` |
| 模型 FOV | `66°` |
| 真实渲染 FOV | `60°` |

M9 空间 AABB 集合审计仍为 `pass`，128 个采样 pose 没有集合差异；但索引查询的 p50/p95 仍高于当前全扫描实现，因此空间索引只证明正确性，尚未证明性能收益。完整 M5 图像评价和正式 frozen test 图像门仍在后台任务中，不能因本次导出完成而提前宣称通过。

## 主线判断

HKUST 的正式前端资产已切换到当前 `rvl_strong_v2` 模型。Metropolis 仍必须等待训练、校准和 frozen test 完成后单独导出；当前部署不会把旧模型替换为未验证的中间 checkpoint。

