# 视线关系场与遮挡生存场文献对照矩阵

日期：2026-08-11
用途：把计划中的论文依据落实到本项目的数据结构、监督信号和前端成本，不把相关工作名称直接当作本项目方法。

| 工作 | 原论文解决的问题 | 本项目借鉴的原则 | 本项目不直接复用的部分 | 可验证的项目实现 |
|---|---|---|---|---|
| Superpoint Transformer | 以超点和层次化关系注意力进行大型三维语义分割 | 关系应按结构化区域和层次组织，而不是无方向的全局池化 | 语义分割任务、在线大规模注意力和原始网络骨干 | 离线聚合实例、遮挡来源和深度层，导出固定 32 维上下文系数 |
| NeuRay | 用单调累计分布表达射线上的遮挡/可见性变化 | 沿深度方向使用单调生存/累计表示 | 输入图像依赖的神经射线表示，不直接预测实例级 PVS | 4×7 低秩实例生存场、事件/右删失似然和前端一次查询 |
| FastNeRF 与预计算辐射传输 | 将场景系数和方向基函数分解并缓存 | 重型场景表征离线计算，运行时只做方向查询 | 辐射传输、颜色和光照预测 | 视角无关实例表 + 共享 `3→16→4` 方向基函数 |
| Neural Visibility of Point Sets | 用视角无关点集表征和共享查询网络预测可见性 | 固定离线表征与轻量视角条件化查询 | 不把点集编码器放进浏览器，不改变本项目实例级候选 CSR | 96 维离线几何、32 维上下文、28 维生存场和九维 ray 输入 |
| NeuralPVS | 用视锥体网格和三维卷积做区域可见性预测 | 需要比较固定空间表示对运行时成本的影响 | 不采用其在线体素化和本项目中昂贵的体素到实例遍历作为主线 | 单独的 NeuralPVS 运行时对照；主线保持固定实例表 |
| NVGS | 在高斯场景中用固定特征和共享网络辅助可见性/渲染 | 固定特征加小查询网络可产生实际渲染收益 | 高斯表示、训练目标和渲染管线不同 | 以实例可见性、GLB 字节和浏览器延迟为共同评价对象 |
| Proxy-Lagrangian | 将约束优化转化为模型目标和对偶变量更新 | 用非负对偶变量将画面安全约束直接纳入训练 | 不把 test 数据用于约束或阈值，亦不以任意综合分数替代原始指标 | pose miss、weighted miss、CVaR 约束及 calibration 冻结阈值 |

## 本项目的创新边界

本项目的主张限定为：将真实三角形遮挡关系按球面方向和深度层离线压缩为连续、深度单调的实例级表示，并在 weighted recall 安全门下联合优化可见性和 GLB 下载成本。上下文编码、遮挡生存场和安全约束损失分别通过 2×2×2 消融和 paired bootstrap 验证，不能仅凭结构新颖性写成贡献。

## 成本边界

所有关系注意力、PointNet++ 几何编码和深度证据生成均属于离线阶段。浏览器运行时只保留固定半精度实例表、九维 ray 查询、方向基函数和小型 MLP；任何需要在线遍历邻居、在线点云编码、全量 HZB 或全量 AABB 投影的实现都不属于本实验主线。运行时结论必须同时记录特征表字节、输入维度、CUDA/WebGPU 延迟、主线程时间和内存，不能只比较离线准确率。

## 参考链接

- [Superpoint Transformer](https://openaccess.thecvf.com/content/ICCV2023/papers/Robert_Efficient_3D_Semantic_Segmentation_with_Superpoint_Transformer_ICCV_2023_paper.pdf)
- [NeuRay](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Neural_Rays_for_Occlusion-Aware_Image-Based_Rendering_CVPR_2022_paper.pdf)
- [FastNeRF](https://arxiv.org/abs/2103.10380)
- [Neural Visibility of Point Sets](https://arxiv.org/abs/2509.24150)
- [NeuralPVS](https://windingwind.github.io/neuralpvs)
- [NVGS](https://openaccess.thecvf.com/content/CVPR2026/papers/Zoomers_NVGS_Neural_Visibility_for_Occlusion_Culling_in_3D_Gaussian_Splatting_CVPR_2026_paper.pdf)
- [Proxy-Lagrangian](https://jmlr.org/papers/v20/18-616.html)
