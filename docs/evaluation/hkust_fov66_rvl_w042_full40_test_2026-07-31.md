# HKUST FOV66 采样、训练与正式评测报告

日期：2026-07-31

## 1. 目的与变更

本次实验修正 HKUST 场景的视场角口径，并重新构建与训练当前方向遮挡代理模型。前端真实渲染相机固定为垂直视场角 60°，采样相机、后退候选相机和模型查询相机统一为垂直视场角 66°。旧版 Pose CSR 数据集和旧版方向遮挡证据已经删除；原始非均匀采样点的代表表被保留为独立 JSONL，后续可以在不依赖旧 CSR 的情况下重建采样计划。

本次变更不改变场景实例组织、实例到 GLB 的映射和前端实例级渲染逻辑，变化集中在采样视场角、view-cell 子视点和对应训练数据语义。

同时清理了数据集构建脚本中的历史 FOV 缺省值；后续缺少显式 FOV 字段时只回退到当前协议的 66°模型视场角，并按采样宽高比计算水平视场角，不再隐式使用旧的扩大视场角。

## 2. View-cell 采样

View-cell 表示一个具有固定观察方向的局部空间区域。每个 view-cell 内生成多个位置不同、方向相同的子相机，采样结果在 view-cell 内取可见实例并集。这一语义对应 NeuralPVS 的 from-region PVS 思路，模型学习的是位置扰动范围内的保守可见集合。

本次直接复用已核验的 HKUST 非均匀代表点分布，没有改成均匀网格：

| 区域类别 | View-cell 数量 |
|---|---:|
| 街道缝隙 | 2,799 |
| 建筑近旁 | 2,000 |
| 广场 | 960 |
| 场景外围 | 800 |
| 天空俯视 | 800 |
| 远景 | 640 |
| 合计 | 7,999 |

每个普通区域 view-cell 生成 32 个子视点，天空和远景生成 48 个子视点，共 279,008 个子视点。View-cell 使用半径 2 m 的水平圆盘采样；每个子视点沿相同的 view-cell 前向方向观察，未加入 yaw/pitch 方向扰动。

采样计划检查发现，旧策略对天空和远景点跳过了碰撞检查。本次对这些点统一执行 AABB 碰撞检查，并向上抬升落入几何体的代表点：天空区域修正 377 个，远景区域修正 227 个；子视点发生碰撞时有 25 个回退到安全中心；没有丢弃 view-cell。

相机参数为：

| 参数 | 数值 |
|---|---:|
| 宽高 | 512 × 288 |
| 宽高比 | 16:9 |
| 采样/模型垂直 FOV | 66° |
| 对应水平 FOV | 98.2035° |
| 前端真实渲染垂直 FOV | 60° |
| 后退距离 | 3.4641 m |

后退距离由 2 m 的位置扰动半径和真实渲染相机的 60° 垂直 FOV 计算，模型本身仍使用 66° FOV 进行候选查询和训练监督。

## 3. 数据集构建

采样使用 Three.js Color-ID 光栅化。Color-ID 权重表示实例在屏幕上的覆盖率，单位为 parts-per-million；它用于可见性重要性监督，不被解释为深度缓冲或严格的遮挡深度。

当前正式数据集为：

```text
neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66
```

其统计如下：

| 项目 | 数值 |
|---|---:|
| 实例数 | 18,831 |
| View-cell 总数 | 7,999 |
| Train / Val / Test | 6,585 / 730 / 684 |
| 子视点数 | 279,008 |
| 平均候选实例数 | 5,029.42 |
| 平均可见实例数 | 107.53 |
| 可见实例不在候选集合中的数量 | 0 |
| 可见集合子集检查失败数 | 0 |

候选集合由所有密集子视点的后退视锥 AABB 候选并集构成，再显式补入采样得到的可见正样本，保证 `visible_ids ⊆ candidate_ids`。该补入数量为 0，说明本次 AABB 候选与 Color-ID 可见结果在数据构建阶段一致。

方向遮挡证据由训练可见实例和不可见候选实例的屏幕矩形重叠、深度前后关系和视线方向离线计算得到，不使用 dynamic-pool 作为 teacher。训练部分共扫描约 11.52 亿个候选源-目标投影对，形成约 5,414.7 万个有效遮挡关系。

## 4. 训练配置

训练模型是方向遮挡代理编码器。它离线编码每个实例的几何、场景上下文和方向遮挡代理；前端只读取固定实例特征，并用当前相机到实例的视线方向进行轻量查询。

本次训练使用：

```text
实验名：pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66
训练轮数：40
每轮步数：900
批大小：每次 2 个 pose 集合
优化器：AdamW
初始学习率：2e-4
损失配置：legacy 参数快照 + RVL evidence
RVL 权重：0.42
设备：CUDA + AMP
```

`legacy` 在这里仅表示正式 checkpoint 的原始损失参数快照，不表示回退到 dynamic-pool。RVL 负责在可见性集合损失中提高画面安全约束，并根据离线遮挡证据加强对高风险不可见候选的抑制。

训练没有出现非有限 loss；AMP 训练期间累计跳过 21 次非有限梯度更新。该问题没有导致训练中止，但后续若继续扩大数据规模，应继续监控梯度范围。`best.pt` 的验证集最佳 checkpoint 来自第 34 个 epoch，最终正式 test 工作点由完整 test split 的阈值扫描确定。

固定运行时特征为每实例 352 个 FP16 数值，包含 96 维几何特征、64 维上下文特征和 192 维方向遮挡代理特征。浏览器不运行 PointNet、图传播或动态遮挡池。

## 5. 正式 Test 结果

正式评测遍历 test split 的全部 684 个唯一 view-cell，候选数量不做截断。当前统一规则为先要求 `weighted recall > 0.99`，再选择 `pose precision` 最高的工作点；普通 pose recall、useful cull 和 bad cull 作为并行诊断指标。阈值 `0.99` 本身不满足严格条件。

### 5.1 安全约束下的剔除效率

| 模型 | 阈值 | 逐实例准确率 | 平衡准确率 | Pose Recall | Weighted Recall | Utility Recall | Useful Cull | Bad Cull | 原始削减率 | 平均候选 | 平均 GT | 平均预测 | 过预测倍数 | GLB 字节削减 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AABB/HZB 基线 | 0.001 | 0.788 | 0.624 | 0.439 | 0.435 | 0.448 | 0.753 | 0.04866 | 0.978 | 4640.57 | 103.25 | 102.42 | 0.99 | 0.732 |
| FOV66 方向遮挡代理模型（weighted-safe precision） | 0.640 | 0.953 | 0.905 | 0.868 | 0.990 | 0.879 | 0.00959 | 0.960 | 4640.57 | 103.25 | 186.63 | 1.81 | 0.858 |

其中：

- `Useful Cull = TN / candidate`，只统计正确剔除的不可见候选；
- `Bad Cull = FN / candidate`，统计被错误剔除的 GT 可见实例；
- `Raw reduction` 将 TN 与 FN 混合，只作为辅助量，不能替代 useful cull；
- `Weighted Recall` 按 Color-ID 屏幕覆盖率加权，只惩罚漏掉的重要可见实例，不惩罚 false positive；
- `GLB 字节削减` 以候选 GLB 字节为分母，反映下载资源减少量。

该工作点的 pose precision 为 `0.8156`，pose F1 为 `0.8160`，pose Jaccard 为 `0.7303`；跨所有 pose 合并后的 aggregate precision 为 `0.4183`、aggregate recall 为 `0.7560`。两者统计粒度不同，不能混用。统一 benchmark 在同一阈值下得到 useful cull `0.87897`、bad cull `0.00959`、GLB 字节削减 `0.85837`；这些指标来自完整 684 个 test view-cell，而不是训练阶段的简化阈值扫描。

best-F1 工作点阈值约为 `0.76`，pose precision 为 `0.8267`，pose recall 为 `0.856`，weighted recall 为 `0.988`，未满足当前严格安全目标，因此没有用于前端默认推理。旧高普通召回工作点阈值约为 `0.01`，precision 为 `0.6785`，weighted recall 为 `0.9962`，用于诊断模型在召回与过预测之间的权衡。

### 5.2 GLB 下载预算

| GLB 预算 | Utility Recall | Required Recall | 平均选中 GLB | 平均选中字节 | 字节削减 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.897 | 0.862 | 40.98 | 10,125,724 | 0.543 |
| 100 | 0.925 | 0.890 | 72.74 | 17,409,636 | 0.466 |
| 200 | 0.948 | 0.922 | 132.01 | 33,303,082 | 0.370 |
| 384 | 0.970 | 0.954 | 229.63 | 59,894,791 | 0.256 |

当 utility recall 达到 0.98 时，模型平均需要约 187.26 个 GLB、44,572,653 字节；对应的候选字节削减为 0.794。下载头因此可以在实例可见性表征基础上输出资源优先级，但实例渲染过滤仍保持实例粒度。

## 6. 前端导出与验证

新模型导出目录为：

```text
slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best
```

运行时模型二进制大小为 `13,844,856` bytes，元数据大小为 `197,034` bytes。当前导出元数据应写入 weighted-safe precision 工作点阈值 `0.6399999857`（展示时记为 `0.64`）、模型 FOV 66°、真实渲染 FOV 60°、viewcell-back-camera 模式、18,831 个实例和 3,273 个 GLB；旧导出目录中的 `0.01` 仅对应历史高普通召回阈值，不能继续作为当前默认前端阈值。

前端映射已切换到新模型目录，`npm test` 通过。Parcel 构建和单场景混淆部署包也已完成：

```text
slm2viewer/public_deploy_hkust_fov66
```

部署包只包含 HKUST 场景和新模型，JavaScript 混淆已开启；场景 GLB 仍按原有远程地址加载，不复制进部署包。

## 7. 保留与删除

已删除：

```text
neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1
neural_instance_culling/dataset/out/directional_occlusion_evidence_v1
```

已保留：

```text
neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/representatives.jsonl
neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/subposes.jsonl
neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source
neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66
neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66
neural_instance_culling/benchmark/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_test_20260731
```

因此当前采样点选择、采样原始结果、聚合数据集、遮挡证据、训练权重和正式评测均可独立追溯，默认代码不再依赖已删除的旧 CSR 数据集。
