# SLM2Viewer 部署包

当前状态和场景边界以 `../docs/README.md`、`../docs/current/repository_layout_2026-07-31.md` 和 `assets/config.json` 为准。本文只描述当前两场景发布流程。

完整的导出、压缩、场景模型映射和 nginx 资产边界见
`../docs/frontend/neuralstreamweb3d_deployment_assets.md`；本文保留可直接执行的部署步骤。

## 包里有什么

```
public_deploy/
├── index.html                         # 入口页面
├── app.a6a4d504.js                    # 混淆后的前端代码
├── LightweightPVSWorker.*.js          # PVS 推理 Worker(Web Worker)
├── style.*.css                        # 样式
├── favicon.df682a99.ico
├── assets/
│   ├── config.json                    # 场景配置(控制 ?scene= 切换哪些场景)
│   ├── neural_instance_culling/       # PVS 运行时模型(训练结果,每次会变)
│   │   ├── <model_1>/                 
│   │   │   ├── instance_pvs_assets.bin    # 模型权重
│   │   │   └── instance_model_meta.json   # 模型元信息
│   │   └── <model_2>/...
│   └── scenes/                        # 场景元数据(不含 GLB 本体)
│       └── <scene>/
│           ├── sceneWeb.json          # 场景结构(构件树/分组/包围盒)
│           ├── glbIndex.json          # 构件ID→GLB文件路径映射
│           ├── runtimeVisibilityMeta.json  # 每个构件的AABB/GLO映射
│           └── task-*/proxy/proxy.glb     # AABB代理模型(小文件,几个MB)
```

`package:deploy:direct` 默认生成目录 `slm2viewer/public_deploy`，不会自动生成 tar 归档；需要上传单文件时再执行 `tar -czf public_deploy.tar.gz -C slm2viewer public_deploy`。

离线转换记录 `conversionManifest.json` 保留在源场景和实例化产物中，但不进入前端部署包，
因为浏览器运行时不会读取它。

## 另一个包:场景 GLB 本体

每个场景的 GLB 文件单独打包,因为体量大且场景不变就不重传:

```
<scene>_glb.tar.gz
└── <scene>/
    └── task-N/glb/LOD0/sub_*.glb      # 场景构件几何(每个几KB,总数数千个)
```

解到 `/var/www/slm2viewer/scene_glbs/<scene>/`

## 预压缩规则

部署脚本会为 JSON、HTML、JS、CSS、SVG、WASM 等文本资源生成同名的 `.br` 和 `.gz` 旁路文件，供 nginx 的 `brotli_static`/`gzip_static` 直接发送。运行时模型二进制和主体 GLB 不生成预压缩副本；`tar.gz` 仅作为主体 GLB 或部署目录的上传封装，不代表运行目录中存在模型压缩副本。

## 按场景生成部署包

默认命令生成包含当前两个场景的多场景包：

```bash
cd slm2viewer
npm run package:deploy:direct
```

场景 GLB 本体单独生成：

```bash
npm run package:scene-glb -- --scene hkust-v3
npm run package:scene-glb -- --scene ifcbench_fantasy_metropolis_instanced_v2
```

也可以只生成一个场景。场景专用包只包含该场景的元数据、对应实例级模型和配置，
不会把另一个场景的模型带进去：

```bash
npm run package:deploy:direct -- --scene hkust-v3
npm run package:deploy:direct -- --scene ifcbench_fantasy_metropolis_instanced_v2
```

需要同时生成两个独立包时，可指定不同输出目录：

```bash
npm run package:deploy:direct -- --scene hkust-v3 --output-dir public_deploy_hkust
npm run package:deploy:direct -- --scene ifcbench_fantasy_metropolis_instanced_v2 --output-dir public_deploy_metropolis
```

`--scene` 的值必须是 `assets/config.json` 中注册的场景名，并且必须有对应的模型映射。当前模型映射还需要同步维护在
`scripts/package_deploy.mjs` 和 `src/neuralCullingBackendMode.js` 中。混淆默认开启；如需调试可显式追加
`--obfuscate-js=false`，正式发布不要关闭。

## 部署步骤

### 1) 上传部署目录和场景 GLB

```bash
# public_deploy 本体(每次发布都传); rsync 会移除远端已不再使用的旧文件
rsync -av --delete public_deploy/ root@<服务器>:/var/www/slm2viewer/public_deploy/

# 场景 GLB(只在场景几何变动时传); 文件名由脚本按场景名生成
scp dist_scene_glb/hkust-v3_glb.tar.gz root@<服务器>:/tmp/
ssh root@<服务器> "
  mkdir -p /var/www/slm2viewer/scene_glbs
  tar -xzf /tmp/hkust-v3_glb.tar.gz -C /var/www/slm2viewer/scene_glbs
"
```

若环境不能使用 `rsync`，可手动归档部署目录：

```bash
tar -czf public_deploy.tar.gz -C . public_deploy
scp public_deploy.tar.gz root@<服务器>:/tmp/
ssh root@<服务器> "tar -xzf /tmp/public_deploy.tar.gz --strip-components=1 -C /var/www/slm2viewer/public_deploy"
```

### 2) nginx 配置

```nginx
server {
    listen 443 ssl http2;
    server_name <你的域名>;

    root /var/www/slm2viewer/public_deploy;
    index index.html;

    brotli_static on;
    gzip_static on;
    gzip_vary on;

    # 场景 GLB 独立目录(alias serve)
    location ^~ /scene_glbs/ {
        alias /var/www/slm2viewer/scene_glbs/;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    # PVS 模型(no-cache,训练迭代后会更新)
    location ^~ /assets/neural_instance_culling/ {
        add_header Cache-Control "no-cache";
        try_files $uri =404;
    }

    # config(no-cache)
    location = /assets/config.json {
        add_header Cache-Control "no-cache";
        try_files $uri =404;
    }

    # JS/CSS/WASM(长期缓存,文件名含hash自动更新)
    location ~* \.(js|css|wasm|mjs)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        try_files $uri =404;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### 3) 验证

```bash
ssh root@<服务器> "nginx -t && systemctl reload nginx"
# 测试
curl -k https://<域名>/assets/config.json       # 应 200
curl -k https://<域名>/scene_glbs/hkust-v3/task-0/glb/LOD0/sub_0.glb  # 应 200
```

### 4) 访问

```
https://<域名>/?scene=hkust-v3
https://<域名>/?scene=ifcbench_fantasy_metropolis_instanced_v2
```

不传 scene 参数通常默认展示 `hkust-v3`；只有该场景不存在时才回退到 `default_config` 或第一个可用场景。

## 怎么加/删场景

新增场景需要同步以下资源和代码映射:

1. 把新场景元数据放进 `public_deploy/assets/scenes/<新场景>/`(sceneWeb.json + glbIndex.json + runtimeVisibilityMeta.json + proxy)
2. 在 `public_deploy/assets/config.json` 注册新场景(只写 `./` 相对路径):
   ```json
   "<新场景>": {
     "cameraPostion": [x, y, z],
     "cameraTarget": [x, y, z],
     "loaderConfig": {
       "resourcesBaseUrl": "./assets/scenes/<新场景>",
       "glbResourcesBaseUrl": "./scene_glbs/<新场景>",
       "schedulingStrategy": "auto"
     }
   }
   ```
3. 把新场景 GLB 传到 `scene_glbs/<新场景>/`。
4. 在 `scripts/package_deploy.mjs` 注册场景到神经模型的映射。
5. 在 `src/neuralCullingBackendMode.js` 注册前端场景到模型的映射，并更新 `verify_deploy_assets.sh` 的检查路径。

删场景:从 `public_deploy/assets/config.json` 里删掉对应条目即可。
