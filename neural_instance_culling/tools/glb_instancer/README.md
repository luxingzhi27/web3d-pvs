# SLM GLB Instancer

该工具提供两种输入模式。当前推荐模式从原始 `sub_*.glb` 出发,保留完整构件边界和稳定构件 ID,再识别严格相同的可复用构件。单体 GLB 连通区域模式仅保留为几何分析和历史对照,不作为当前实例化主线。

## 输入与输出

原始构件模式输入要求:

- 场景资产目录包含 `glbIndex.json` 和 `task-*/glb/LOD0/sub_*.glb`。
- 每个 `sub_*.glb` 表示一个完整构件；工具不会再将其拆成连通小块。
- 仅在完整几何、顶点颜色、拓扑和刚体配准均通过验证时共享原型。

单体网格模式输入要求:

- glTF 2.0 二进制 GLB。
- 当前版本要求一个 indexed triangle primitive。
- 顶点位置为浮点 `POSITION`。
- 可选归一化 8 位 `COLOR_0`;没有颜色时使用白色。

原始构件模式处理流程:

1. 读取 `glbIndex.json`,保留原始构件 ID 和构件边界。
2. 对每个完整构件计算平移和旋转不敏感的几何、颜色及拓扑指纹。
3. 对候选同型构件执行 Horn 四元数刚体配准和逐顶点误差验证。
4. 每个同型组导出一个原型 GLB,重复构件写入 `EXT_mesh_gpu_instancing`。
5. 重建 `sceneWeb.json`、`glbIndex.json` 和 `runtimeVisibilityMeta.json`,构件 ID 保持不变。

单体网格模式处理流程:

1. 根据三角形共享顶点关系建立并查集,提取几何连通区域。
2. 为每个连通区域计算对平移和三维旋转不敏感的几何指纹。
3. 对候选相似区域执行 Horn 四元数刚体配准和逐顶点误差验证。
4. 每个相似组导出一个原型 GLB;重复区域写入 `EXT_mesh_gpu_instancing` 矩阵。
5. 生成 `sceneWeb.json`、`glbIndex.json`、`runtimeVisibilityMeta.json` 和 `conversionManifest.json`。

原始构件模式中的 component ID 与源场景一致。只有单体网格模式的 component ID 才表示几何连通区域。

## 从 IFCBench 原始 sub GLB 分析

```bash
cd neural_instance_culling/tools/glb_instancer
npm run analyze -- \
  --source-assets ../../../ifcbench_fantasy_metropolis_source/assets \
  --scene-name ifcbench_fantasy_metropolis_instanced_v2 \
  --report ./out/ifcbench_fantasy_metropolis_instanced_v2_analysis.json
```

## 从 IFCBench 原始 sub GLB 构建

```bash
cd neural_instance_culling/tools/glb_instancer
npm run build -- \
  --source-assets ../../../ifcbench_fantasy_metropolis_source/assets \
  --output-assets ../../../ifcbench_fantasy_metropolis_instanced_v2/assets \
  --scene-name ifcbench_fantasy_metropolis_instanced_v2 \
  --report ./out/ifcbench_fantasy_metropolis_instanced_v2_build.json \
  --overwrite
```

该命令把每个原始 `sub_*.glb` 当作一个完整构件，保留原始构件 ID，再对严格通过几何和刚体配准验证的构件共享原型。它不会把当前展示场景降级为按连通小块拆分的单体网格方案。

## 单体 GLB 分析（历史对照）

```bash
cd neural_instance_culling/tools/glb_instancer
npm run analyze -- \
  --input /path/to/scene.glb \
  --report ./out/monolithic_scene_analysis.json
```

## 单体 GLB 构建（历史对照）

```bash
cd neural_instance_culling/tools/glb_instancer
npm run build -- \
  --input /path/to/scene.glb \
  --output-assets /path/to/instanced_scene/assets \
  --scene-name instanced_scene \
  --report ./out/monolithic_scene_build.json \
  --overwrite
```

默认匹配是保守的刚体复用:拓扑、顶点颜色和顶点到质心的距离指纹必须一致,并且三维配准后的均方根误差和最大误差必须低于阈值。默认不把仅 AABB 相似或视觉上大致相似的区域合并。

## 验证

```bash
npm test
```

测试会构造包含两个刚体重复区域和一个独立区域的单体 GLB,检查连通区域数量、原型数量、实例映射和输出 GLB 结构。
