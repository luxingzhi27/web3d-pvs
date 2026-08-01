# NeuralStreamWeb3D 导出、打包与部署资产

更新时间：2026-08-01

本文说明训练 checkpoint 如何变成浏览器运行资产，以及场景元数据、神经模型和主体 GLB 如何分别发布。

## 1. 模型运行资产

导出入口为 `neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py`。每个场景的模型目录至少包含：

```text
instance_pvs_assets.bin
instance_model_meta.json
```

`instance_pvs_assets.bin` 是没有文件头的连续 FP16 数据块，不能独立解析；`instance_model_meta.json` 描述 schema、每个数据块的 byte offset/shape、实例数、模型维度、阈值、FOV、场景边界和实例到 GLB 映射。当前 schema 是：

```text
directional-occlusion-proxy-scheduler-v1
```

默认运行维度为：

```text
实例固定特征：N × 352（96 几何 + 64 上下文 + 192 遮挡代理）
实例 AABB：N × 6
实例到 GLB：N
```

之后是方向代理门控、射线门控、可见性 MLP、遮挡抑制头、视觉效用头和 GLB 下载头。前端不需要 PyTorch checkpoint、点云缓存、遮挡证据源实例或训练日志。

当前导出前必须检查：

- checkpoint 的 `config` 与导出脚本 schema 一致；
- runtime feature 表行数等于 runtime metadata 的实例数；
- AABB 和实例到 GLB 映射数量一致；
- `workpoint` 与前端的阈值意图一致；
- `frontendRenderFovYDeg=60`、`modelInputFovYDeg=66` 和后退相机模式与训练数据一致。

阈值选择统一使用 `--workpoint primaryWeightedPrecision`：严格筛选 `weighted recall > 0.99`，再选择 `pose precision` 最高者。当前默认 HKUST 强模型
`pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` 的阈值为约 `0.02`，校准点 weighted recall 为 `0.9930808`，bootstrap 单侧
95% 下界为 `0.9909417`。旧的 `w042` 资产仍可能记录约 `0.64` 的诊断工作点，但它不是当前默认模型，不能把两个 checkpoint 的阈值混写。

## 2. 前端部署包和主体 GLB

`npm run package:deploy:direct` 先构建 Parcel 前端，再生成 `slm2viewer/public_deploy/`。部署包包含：

```text
index.html
混淆后的 app.*.js 和 LightweightPVSWorker.*.js
CSS/WASM/图标
assets/config.json
assets/scenes/<scene>/sceneWeb.json
assets/scenes/<scene>/glbIndex.json
assets/scenes/<scene>/runtimeVisibilityMeta.json
assets/scenes/<scene>/proxy/proxy.glb
assets/neural_instance_culling/<model>/instance_pvs_assets.bin
assets/neural_instance_culling/<model>/instance_model_meta.json
```

主体 GLB 不进入部署包。`npm run package:scene-glb -- --scene <scene>` 单独收集 `task-*/glb/LOD0/*.glb`，生成：

```text
dist_scene_glb/<scene>_glb.tar.gz
```

这里的 `tar.gz` 只是上传传输封装，解压后 GLB 文件本身保持原样。默认源目录是仓库根目录下对应场景的 `assets/`，可以用 `--source` 覆盖。

## 3. 按场景打包

```bash
cd slm2viewer
npm install
npm run package:deploy:direct
npm run package:deploy:direct -- --scene hkust-v3
npm run package:deploy:direct -- --scene ifcbench_fantasy_metropolis_instanced_v2
npm run package:scene-glb -- --scene hkust-v3
npm run package:scene-glb -- --scene ifcbench_fantasy_metropolis_instanced_v2
```

`--scene` 不是只由 `config.json` 自动发现。新增场景至少要同步：

1. `assets/config.json` 的场景配置；
2. `scripts/package_deploy.mjs` 的场景到模型目录映射；
3. `src/neuralCullingBackendMode.js` 的场景到模型映射；
4. 部署资产验证脚本中的必要路径；
5. 场景元数据和主体 GLB 目录。

单场景包会过滤掉另一个场景的元数据和神经模型。模型二进制在当前包中约为 HKUST 13.8MB、Metropolis 29.9MB，不能把这类大小写成一个固定的“系统模型大小”。

## 3.1 空间特征分页实验资产

导出器也支持显式的实验分页格式：

```bash
conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py \
  --checkpoint <checkpoint>/best.pt \
  --runtime-features <checkpoint>/instance_runtime_features_fp16.bin \
  --feature-meta <checkpoint>/instance_features_meta.json \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --eval-summary <checkpoint>/calibration_ready_summary.json \
  --output-dir slm2viewer/assets/neural_instance_culling/<paged-model> \
  --prediction-camera-mode viewcell-back-camera \
  --pvs-back-offset-m 3.464101552963257 \
  --spatial-pages
```

该模式的 `instance_pvs_model_weights.bin` 只保存模型权重；`spatial_pages/page_directory.json` 保存每页的联合 AABB、实例数和文件路径，
页二进制保存实例编号、FP32 AABB 和 FP16 固定特征。分页页选择必须与模型元数据中的 `modelInputFovYDeg=66`、
`predictionCameraMode=viewcell-back-camera` 和 `pvsBackOffsetM=3.464101552963257` 一致。当前 HKUST M9 实验资产为 45 页，完整页资源约 13.79MB，
但它的动态候选特征缓冲桌面 smoke 比完整特征表更慢，未接入默认场景，也不应写入正式部署包。

## 4. 混淆和预压缩

正式打包默认开启 JavaScript 混淆：

```bash
npm run package:deploy:direct -- --obfuscate-js=true
```

可用 `--obfuscate-js=false` 做调试包，但正式发布不应关闭。混淆只提高逆向成本，不是 DRM。

部署脚本会为可压缩的文本资源生成 `.br` 和 `.gz` 旁路文件。实际范围包括 JSON、HTML、JS、CSS、SVG、WASM 等；运行时模型 `.bin` 和大体积主体 GLB 默认跳过预压缩。主体 GLB 的 `tar.gz` 上传包仍会压缩传输封装，但不改变解压后的 GLB。

## 5. nginx 目录

建议将部署包和主体 GLB 分开：

```text
/var/www/slm2viewer/public_deploy/
/var/www/slm2viewer/scene_glbs/<scene>/task-*/glb/LOD0/sub_*.glb
```

nginx 必须同时提供部署根目录和主体 GLB alias：

```nginx
root /var/www/slm2viewer/public_deploy;

location ^~ /scene_glbs/ {
    alias /var/www/slm2viewer/scene_glbs/;
    add_header Cache-Control "public, max-age=31536000, immutable";
}

location ^~ /assets/neural_instance_culling/ {
    add_header Cache-Control "no-cache";
    try_files $uri =404;
}
```

`package_deploy.mjs` 自动生成的 nginx 模板主要覆盖部署根目录和神经资源，不应假定它已经包含 `/scene_glbs/` alias；使用主体 GLB 时必须人工合并 alias 或使用 `README_DEPLOY.md` 中的完整配置。

## 6. 发布与验证

推荐顺序：

1. 生成模型运行资产并检查 `instance_model_meta.json`；
2. 构建并按场景生成 `public_deploy`；
3. 生成主体 GLB tar 包并上传到独立目录；
4. 先部署元数据和前端，再部署主体 GLB；
5. `nginx -t && systemctl reload nginx`；
6. 使用 `verify:deploy-assets` 和浏览器 smoke 检查每个场景。

最低 HTTP 检查包括：

```bash
curl -f https://<host>/assets/config.json
curl -f https://<host>/assets/scenes/<scene>/runtimeVisibilityMeta.json
curl -f https://<host>/assets/neural_instance_culling/<model>/instance_model_meta.json
curl -f https://<host>/scene_glbs/<scene>/task-0/glb/LOD0/sub_0.glb
```

浏览器 smoke 还必须确认：WebGPU binding 没有 binding 3；模型结果经过真实视锥过滤；实例化 GLB 可以只更新部分实例；GLB 请求支持跨域和 Range（如果配置使用远端 GLB）；调试面板显示的是模型预测数、真实视锥后实例数和实际绘制实例数的不同字段。

## 7. HKUST 独立包

`npm run package:deploy:hkust-liteweb3d` 基于已有部署资产生成 HKUST 独立包，不重新构建场景 GLB。该包把主体 GLB 基址写成：

```text
https://www.liteweb3d.com/data/hkust-v3/
```

远端必须允许浏览器跨域请求，并正确响应 GLB 的路径和 Range 请求。独立包只携带 HKUST 的元数据和模型，不应把它误解为主体 GLB 的备份。
