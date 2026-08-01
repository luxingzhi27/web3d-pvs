说明：
1. 将封装由sceneMgr迁移为slm2loader
2. 支持进行静态slm2资源的加载，如下：
    slm2Loader.loadStatic("http://127.0.0.1:8080/sceneWeb.json", scope.renderer, scope.activeCamera, ...) 