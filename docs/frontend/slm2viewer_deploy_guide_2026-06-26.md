# SLM2Viewer 部署流程

日期: 2026-07-31

本页是通用流程记录，当前场景和资源数量以 `docs/current/repository_layout_2026-07-31.md` 和 `slm2viewer/assets/config.json` 为准。

## 当前范围

当前发布只保留两个场景：

| 场景 | `?scene=` 参数 | GLB 数 |
|---|---|---:|
| HKUST v3 | `hkust-v3`（默认） | 3273 |
| IFCBench Fantasy Metropolis 实例化 v2 | `ifcbench_fantasy_metropolis_instanced_v2` | 3669 |

配置唯一来源是 `slm2viewer/assets/config.json`。当前模型、场景元数据和前端代码通过
`npm run package:deploy:direct` 一起生成混淆部署包；场景 GLB 本体单独上传，避免每次
发布重复传输大文件。

## 打包

```bash
cd slm2viewer
npm run package:deploy:direct
npm run package:scene-glb -- --scene hkust-v3
npm run package:scene-glb -- --scene ifcbench_fantasy_metropolis_instanced_v2
```

如果只需要某一个场景，可以在部署包阶段筛选场景；这会同时筛选场景元数据、模型和
`config.json`，不会重新打包 GLB 本体：

```bash
npm run package:deploy:direct -- --scene hkust-v3
npm run package:deploy:direct -- --scene ifcbench_fantasy_metropolis_instanced_v2
```

需要并行保留两个单场景包时，可以指定输出目录：

```bash
npm run package:deploy:direct -- --scene hkust-v3 --output-dir public_deploy_hkust
npm run package:deploy:direct -- --scene ifcbench_fantasy_metropolis_instanced_v2 --output-dir public_deploy_metropolis
```

`public_deploy` 包含：

- 混淆后的 `app.*.js`、`LightweightPVSWorker.*.js` 和 CSS/HTML。
- HKUST 与 Metropolis 两个实例级模型的 `instance_pvs_assets.bin` 和 `instance_model_meta.json`。
- 两个场景的 `sceneWeb.json`、`glbIndex.json`、`runtimeVisibilityMeta.json` 和 proxy；离线转换清单
  保留在源场景中，不进入部署包。
- JSON 元数据的 `.br`/`.gz` 旁路文件；模型二进制和 GLB 不生成额外压缩副本。

IFCBench 原始 `sub_*.glb` 源保存在 `ifcbench_fantasy_metropolis_source/assets`，只用于
离线实例化、采样和重建，不进入前端部署包。

## 请求路径

```text
/assets/config.json
/assets/scenes/<scene>/sceneWeb.json
/assets/scenes/<scene>/glbIndex.json
/assets/scenes/<scene>/runtimeVisibilityMeta.json
/scene_glbs/<scene>/task-0/glb/LOD0/sub_*.glb
```

运行包不再携带根目录的 `assets/glbIndex.json` 或
`assets/runtimeVisibilityMeta.json` 历史副本。加载器只使用当前场景目录中的元数据，避免
某个场景请求失败时误用另一个场景的实例映射；场景元数据请求失败应被报告为部署问题。

部署根目录约定为 `/var/www/slm2viewer/public_deploy`，GLB 目录约定为
`/var/www/slm2viewer/scene_glbs`，nginx 使用 `location ^~ /scene_glbs/` 的 alias 提供
大文件，并保留 Range 请求支持。

## 远端维护

远端服务器为 `139.196.34.161`，登录使用：

```bash
ssh -i ~/.ssh/id_ed25519_slm_deploy -o IdentitiesOnly=yes root@139.196.34.161
```

当前服务器只应保留 `hkust-v3` 和 `ifcbench_fantasy_metropolis_instanced_v2` 的运行资源。
删除场景前必须先检查 nginx 配置和 `assets/config.json`，不要删除配置实际引用的文件。

验证命令：

```bash
bash slm2viewer/scripts/verify_deploy_assets.sh https://web3d.ysnb.asia
ssh -i ~/.ssh/id_ed25519_slm_deploy -o IdentitiesOnly=yes root@139.196.34.161 'nginx -t'
```

前端模型仍按实例输出和实例过滤，GLB 只作为下载、缓存和解码聚合粒度；不能因为一个
GLB 内存在可见实例就显示同一 GLB 的全部实例。

## 2026-07-31 HKUST 新模型增量部署

本次将 HKUST FOV66 方向遮挡代理模型部署到 `139.196.34.161`，远端目录为：

```text
/var/www/slm2viewer/public_deploy/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best
```

为了避免重复传输场景资源，只增量上传了当前混淆后的入口脚本、PVS Worker、入口 HTML 的压缩旁路文件，以及 HKUST 模型的 `instance_pvs_assets.bin` 和运行元数据；`/var/www/slm2viewer/scene_glbs` 未改动。远端模型元数据确认：阈值为 `0.6399999857`（前端显示 `0.64`），选择规则为 `primaryWeightedPrecision`，`poseWeightedRecall=0.990071`，严格满足 `weighted recall > 0.99`。

部署后执行了 `nginx -t` 和 `systemctl reload nginx`。在服务器本机通过 `https://web3d.ysnb.asia` 的 nginx HTTPS 虚拟主机验证了配置、HKUST 模型元数据、模型二进制和一个场景 GLB 均返回 `200`；从当前开发机访问公网域名时 TLS 连接被上游重置，属于外部域名入口问题，需在目标网络继续确认。

## 2026-07-31 复核记录

本次复核以 `docs/current/` 和当前 `assets/config.json` 为准，目标是让打包结果与两场景
运行边界一致。修改了 `scripts/package_deploy.mjs`、`slm2/SLM2Loader.js` 和当前完整性
测试，删除了前端源目录中重复的根级 HKUST `glbIndex.json` 与
`runtimeVisibilityMeta.json`。加载器现在只读取当前场景目录的索引和运行元数据，避免
Metropolis 请求失败时错误回退到 HKUST 映射。

验证命令为 `npm test`、`npm run build`、`node scripts/package_deploy.mjs --asset-mode=direct`
和 `node scripts/package_hkust_liteweb3d_deploy.mjs`。当前多场景混淆包包含 HKUST 与
Metropolis 两个场景，原始大小 `125,979,663` bytes，Brotli 传输估算 `72,921,441` bytes；
HKUST 独立包为 `14,439,732` bytes。该修改保留当前主线，不改变模型权重、阈值或场景
GLB，本地浏览器 WebGPU 交互仍需在目标服务器上复测。
