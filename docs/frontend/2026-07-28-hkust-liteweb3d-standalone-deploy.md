# HKUST 独立部署包

日期：2026-07-28

## 目的

为需要单独部署 HKUST 场景的服务器生成一个独立前端包。包内保留当前混淆后的前端、HKUST 场景元数据、代理几何、固定实例特征和轻量可见性查询模型；场景本体 GLB 不重复打包，由浏览器从远端地址按需请求。

## 运行配置

- 场景：`hkust-v3`。
- 场景元数据和代理模型：部署包本地的 `assets/scenes/hkust-v3/`。
- 神经运行资产：部署包本地的 `pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best`。
- GLB 请求基址：`https://www.liteweb3d.com/data/hkust-v3/`。
- 配置同时保留 `default_config` 和 `hkust-v3`，两者都指向 HKUST，避免不带查询参数或使用错误场景参数时落到空场景。
- 加载器对基址和相对路径进行规范化连接，尾斜杠不会产生双斜杠请求。

## 修改内容

- `slm2viewer/scripts/package_hkust_liteweb3d_deploy.mjs`：从当前 `public_deploy` 生成仅含 HKUST 的目录、清单、nginx/Caddy 模板和压缩包。
- `slm2viewer/package.json`：新增 `npm run package:deploy:hkust-liteweb3d`。
- `slm2viewer/slm2/SLM2Loader.js`：统一 GLB URL 拼接，兼容带尾斜杠的远端基址。
- `slm2viewer/hkust_v3_public_deploy_liteweb3d.tar.gz`：独立部署压缩包。
- `slm2viewer/public_deploy_hkust_liteweb3d/`：压缩包对应的可直接部署目录。

## 复现命令

```bash
cd /mnt/sda/rhyang/slm/slm2viewer
npm run package:deploy:direct
npm run package:deploy:hkust-liteweb3d
```

第一条命令重新构建并混淆当前多场景部署资产；第二条命令只从该部署目录提取 HKUST 内容，不把其他场景或 GLB 本体复制进独立包。

## 验证结果

2026-07-31 重新构建时同步锁定 FOV 协议：前端真实渲染为 `60°`，模型查询为 `66°`；部署包中的模型元数据不再携带旧的训练/推理 FOV 别名。

| 项目 | 结果 | 含义 |
|---|---:|---|
| 压缩包大小 | 14,448,638 bytes | 实际需要上传的独立部署包大小 |
| 解包原始内容 | 37,208,555 bytes | 服务器解包后需要提供的前端、元数据和模型资产 |
| 非压缩文件数 | 25 | 当前独立包中的原始文件数；另有 15 个 `.br`/`.gz` 预压缩旁路文件，总文件数为 40 |
| 场景数量 | 1 | 只保留 HKUST，避免错误加载其他场景 |
| 代理 GLB 数量 | 5 | 启动阶段用于场景代理显示的轻量几何 |
| 场景本体 GLB 数量 | 0 | 所有 `task-*/glb/LOD0/sub_*.glb` 均由远端地址提供 |
| 本地静态 HTTP smoke | 6/6 返回 200 | 配置、场景结构、运行元数据、模型元数据、模型二进制和主脚本可读取 |
| 远端 GLB smoke | 200 | 远端样例 GLB 支持浏览器请求，并返回 CORS 与 Range 相关响应头 |

这些大小和数量是部署检查指标，不是可见性模型的 precision、recall 或 F1；本次任务没有改变模型权重或阈值。

## 部署

将 `hkust_v3_public_deploy_liteweb3d.tar.gz` 解压到 nginx 根目录，并使用包内的 `nginx-neural-culling.conf`。远端服务需要允许浏览器的跨域 GET/Range 请求；服务器不需要配置本地 `scene_glbs` 目录。

## 保留状态与风险

该包作为 HKUST 独立部署产物保留，当前多场景 `public_deploy` 不被替换为单场景包。2026-07-31 已用当前紧凑运行元数据重新生成独立目录和归档，并完成静态资源检查；本次又移除了根目录历史元数据副本后重新归档。尚未在目标服务器上执行 nginx reload，也未在目标浏览器上做完整 WebGPU 交互 smoke；远端 GLB 服务的 CORS、Range 和具体 CDN 缓存策略仍需在实际域名页面上确认。
