# M3：方向遮挡代理推理期干预协议

日期：2026-08-01  
状态：M3 runner 已完成审计和 v2 实现，静态检查与合成 fixture 已通过；HKUST 正式 checkpoint 干预已完成，Metropolis 和三随机种子重训练仍待执行。

## 目的

当前模型把每个实例的固定表征拆成几何特征、上下文特征和方向遮挡代理。方向遮挡代理只有在干预后仍然改变可见性决策，且这种改变改善安全约束下的有效剔除或下载效用，才具有论文中的独立解释价值。本实验只修改推理期固定特征，不更新模型参数，因此可以区分“模型参数中存在该分支”与“该分支在当前视角查询中实际被使用”。

## 固定条件

- 使用同一个正式空间 split checkpoint；
- 先物化一份固定的 pose 批次计划，所有变体复用同一计划；
- 候选严格读取 CSR 中保存的 `frustum_ids.bin`，不补入 GT 可见实例，也不裁剪候选；
- 运行结束时逐 pose 比较候选 ID 的 SHA-256，候选集合发生变化会直接失败，而不是静默丢弃不匹配 pose；
- 阈值只读取 checkpoint 对应的 calibration 冻结工作点；
- 默认只评估 validation 或 calibration，禁止使用 test 重新选阈值；test 诊断也不接受 `--threshold` 覆盖，只能读取冻结阈值；
- 每个 pose 记录 precision、recall、F1、Jaccard、weighted recall、accuracy、balanced accuracy、specificity、useful cull、bad cull、预测数量、候选数量和 GT 数量；
- 每个 pose 还记录基础 logit、最终 logit、抑制量、选中代理绝对值和方向-深度门控熵的 count/mean/std/min/P05/median/P95/max；旧版的均值字段仍保留，但结构化字段是权威记录。

## 干预变体

| 变体 | 操作 | 解释目标 |
|---|---|---|
| `baseline` | 原始固定特征 | 参照 |
| `proxy_zero` | 所有方向/深度代理置零 | 判断代理是否被使用 |
| `proxy_random_same_distribution` | 按每一维均值和标准差生成随机代理 | 排除固定数值幅度的偶然作用 |
| `proxy_direction_mean` | 八个方向替换为方向均值 | 判断方向分辨率是否重要 |
| `proxy_direction_roll` | 方向索引循环移位 | 破坏方向对应关系但保持分布 |
| `proxy_cross_instance_permutation` | 在实例维度置换完整代理表；实例数大于 1 时保证不是恒等置换 | 判断代理是否与具体实例绑定 |
| `context_zero` | 上下文特征置零 | 区分上下文与遮挡代理作用 |
| `context_cross_instance_permutation` | 在实例维度置换完整上下文表；实例数大于 1 时保证不是恒等置换 | 判断上下文是否被实例化记忆 |
| `context_proxy_zero` | 同时清零上下文和代理 | 几何+视线退化诊断 |

旧命令中的 `proxy_instance_permutation` 和 `context_instance_permutation` 仍可作为兼容输入，但会被规范化为上表中的跨实例名称，不会产生额外实验。

`proxy_random_same_distribution` 使用固定随机种子，并按每个方向-深度切片在整张实例表上计算均值和标准差后生成随机值；它不是从另一实例复制代理。`proxy_direction_mean` 只平均方向维，保留深度壳层；`proxy_direction_roll` 只移位方向维，保留每个切片的数值分布。

## 统计与准入

`evaluate_proxy_interventions.py` 以 pose/view-cell 为统计单位，按照同一候选集合的同一 pose 变体差值执行默认 10,000 次 paired bootstrap，输出均值差和双侧 95% 区间。主比较是完整模型相对于 `context_zero` 或 `context_proxy_zero`（几何+视线参照）的：

- `useful cull` 增益；
- 在安全约束下的 weighted recall 与 bad cull；
- 达到相同图像效用时的下载字节变化（由后续调度实验补充）。

逐 pose 的 `base_logit` 是可见性头在显式抑制之前的 logit；`suppression` 是模型抑制头输出的非负抑制量，最终 logit 满足 `final_logit = base_logit - suppression`。`gate_entropy` 是方向-深度代理门控的归一化 Shannon 熵，0 表示门控集中，1 表示近似均匀。它们是诊断量，不会被用来重新选择阈值。

汇总同时给出逐 pose 平均和跨 pose 合并的指标。`weighted_recall` 使用数据集中的 `visible_weights`，这些权重是弱重要性权重，不宣称为真实像素覆盖率；`useful_cull = TN / candidate`，`bad_cull = FN / candidate`。另外报告 specificity、balanced accuracy、accuracy、平均预测数、平均候选数和预测/候选比，避免只用 F1 或 weighted recall 解释代理作用。

论文路线 A 的预注册 Go 条件是：完整模型相对几何+上下文+视线参照提高 useful cull 至少 2 个百分点，或在相同图像效用下减少至少 10% 下载字节，并且三个随机种子的 paired 95% 区间不跨零。否则把方向代理降级为辅助表征，转入路线 B，不通过措辞掩盖干预失败。

## 复现命令模板

先执行不需要 checkpoint 的 fixture：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_proxy_interventions.py \
  --self-test
```

正式运行：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_proxy_interventions.py \
  --checkpoint <formal-model>/best.pt \
  --runtime-features <formal-model>/instance_runtime_features_fp16.bin \
  --dataset-dir <formal-spatial-dataset> \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --eval-summary <formal-model>/eval_summary.json \
  --split validation \
  --poses-per-batch 2 \
  --output neural_instance_culling/benchmark/out/<experiment>/m3_proxy_interventions.json \
  --device cuda
```

正式执行前必须确认 `eval_summary.json` 的协议为 `frozen_calibration_one_shot_test`，并记录 checkpoint、数据集、特征表和 runtime meta 的 SHA-256。输出的 `interventionManifest` 记录每个干预的种子、特征布局以及跨实例置换的 SHA-256；`posePlan` 记录固定 pose 计划和候选语义。HKUST 的正式输出记录在 `m3_formal_execution_2026-08-01.md`；Metropolis 仍未形成正式结果，三随机种子路线判定也未完成。

## 2026-08-01 实现审计与验证记录

### 变更目的

补齐推理期代理干预的可复核性，确认清零、随机化、方向聚合、方向移位、跨实例置换以及上下文干预不会隐式修改 checkpoint、候选集合或阈值。原 runner 已有这些操作的基本分支，但缺少规范化名称、候选配对硬校验、完整的逐 pose 诊断结构和若干标准指标。

### 修改文件

- `neural_instance_culling/benchmark/evaluate_proxy_interventions.py`
- 本协议文档

未修改 M5/M6 文件、训练主脚本、前端或 test 阈值。

### 已完成验证

```text
py_compile: passed
--self-test: passed
```

fixture 已验证全部九个输出变体、两个跨实例置换的非恒等性、方向均值/移位、context zero、weighted recall、accuracy、门控熵归一化以及别名规范化。HKUST 正式 checkpoint 运行已完成；Metropolis、三随机种子干预和路线 A/B 判定仍待完成。

## 2026-08-01 探索性运行记录（不进入正式主表）

为先验证真实 CSR、固定特征表和 paired bootstrap 链路，在 M0 空间训练完成前使用了旧的
`pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66` checkpoint。该 checkpoint
是在旧随机 view-cell 数据集上训练的，本次只在空间 validation 的 664 个 view-cell 上运行，使用显式
阈值 `0.64`，因此不能作为正式空间模型、正式阈值或 test 结论。

输出：
`neural_instance_culling/benchmark/out/m3_proxy_interventions_exploratory_hkust_spatial_old_w042_validation_20260801.json`。

关键观察如下：

| 干预 | 相对 baseline 的 weighted recall 差值 | 相对 baseline 的 useful cull 差值 | 诊断 |
|---|---:|---:|---|
| `proxy_zero` | -0.00132 | +0.00145 | 代理清零有轻微质量变化，但不代表正式安全收益 |
| `proxy_direction_mean` | -0.00022 | +0.00020 | 方向细节影响很小，需正式 checkpoint 重复 |
| `proxy_direction_roll` | -0.00009 | +0.00018 | 方向对应关系的探索性影响接近零 |
| `proxy_cross_instance_permutation` | -0.00017 | +0.00035 | 跨实例绑定影响未形成稳定收益证据 |
| `context_zero` | -0.02546 | -0.22891 | 上下文表征在旧模型中具有明显作用 |
| `context_cross_instance_permutation` | -0.80044 | +0.00080 | 置换上下文破坏实例对应关系，质量显著下降 |

所有 paired 区间均以 view-cell 为 bootstrap 单位。完整模型的显式 inhibition 平均抑制量约为
`10^-8`，而上下文清零时才升到约 `1.5×10^-3`；这说明当前旧模型的主要作用来自固定表征和查询交互，
不能把 inhibition 分支本身描述为已经验证有效。正式空间 checkpoint 完成后必须按同一协议重跑；如果
方向代理的三种子干预仍未达到预注册的 2 个百分点 useful-cull 或 10% 下载收益门槛，论文路线应转为
固定表征/系统调度路线 B，并删除方向代理的独立贡献主张。
