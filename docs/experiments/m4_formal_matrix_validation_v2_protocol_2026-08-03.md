# M4-v2 完整 2×2 因子消融评价协议

日期：2026-08-03  
状态：执行前冻结；本协议提交前不启动 M4-v2 新训练或正式评价  
对应计划：`docs/experiments/neuralstreamweb3d_submission_plan_2026-09.md` 第 21.25 节

## 1. 目的

M4-v2 解决旧 M4 只用 useful cull 和 bad cull 对 `full` 与 `geometry_context_ray` 做单差值比较的问题。新协议把方向遮挡代理和显式遮挡抑制头作为两个独立因子，形成完整的 2×2 设计，并同时检查画面安全、分类诊断、有效剔除、GLB 资源和运行时成本。

本协议不会修改旧 M4 的任何 JSON、Markdown、模型或前端资产。新评价目录为：

```text
neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2/
```

路线文件和报告使用独立的 `*_v2` 名称。

## 2. 正式成员和复用规则

| 代号 | 变体 | 方向代理 | 显式抑制 | 处理 |
|---|---|---:|---:|---|
| A | `geometry_context_ray_no_inhibition` | 关闭 | 关闭 | 缺失，独立训练三 seed |
| B | `geometry_context_proxy_ray_no_inhibition` | 开启 | 关闭 | 复用旧 M4 checkpoint/interventions |
| C | `geometry_context_ray` | 关闭 | 开启 | 复用旧 M4 checkpoint/interventions |
| D | `full` | 开启 | 开启 | 复用旧 M4 checkpoint/interventions |

`aabb_ray` 和 `geometry_ray` 是逐级输入参考，不纳入因子代数。复用前必须逐文件验证 checkpoint SHA、runtime feature SHA、dataset 路径、阈值来源和候选哈希。A 不能用 C 的 `base_logit` 直接伪造正式成员：C 的 base logit 仍可能受训练时抑制分支相关参数影响，且没有独立无抑制训练语义。

## 3. 不可变数据协议

- 场景：HKUST v3 空间 validation 数据。
- FOV：模型和后退候选相机 66°；真实渲染相机 60°。
- 每个 seed/变体：同一 validation pose 集，预期 664 个 pose。
- 候选：数据集保存的后退相机候选；不补 GT、不截断、不使用前端白名单。
- GT：数据集保存的实例可见集合和 `visible_weights`。
- 阈值：仅从对应 checkpoint 的独立 calibration 冻结；validation/test 不重新选阈值。
- test：M4-v2 validation/calibration 阶段不读取；正式 test 若后续执行只能在模型、阈值和路线冻结后一次完成。

任何成员若 pose 数、pose digest、candidate hash、candidate count 或阈值 provenance 不一致，必须退出矩阵，不能用交集或补集修复。

## 4. 指标

对候选集合 C、GT 集合 G、预测集合 P，定义 `TP=P∩G`、`FP=P-G`、`FN=G-P`、`TN=C-(P∪G)`。

### 4.1 画面安全

报告 pose-level 宏平均和 aggregate 合并值：pose recall、aggregate recall、weighted recall、visual utility recall、bad cull=`FN/candidate`。`visual utility recall` 必须来自已有统一视觉效用评价器；若没有同口径结果，输出 `not_available`，不以 weighted recall 替代。miss-pixel、wrong-ID pixel、extra-pixel 同理。

### 4.2 分类诊断

报告 pose/aggregate precision、F1、Jaccard、accuracy=`(TP+TN)/candidate`、balanced accuracy=`(recall+specificity)/2`、specificity=`TN/(TN+FP)`。

### 4.3 剔除、资源和成本

报告 useful cull=`TN/candidate`、平均 TP/FP/FN/TN、平均预测数、预测/候选、预测/GT、GLB 数量削减、GLB 字节削减、同视觉效用所需 GLB 字节、单 pose forward 延迟、固定特征表大小和推理输入大小。没有真实数据的项必须保留字段并标记 `not_available` 与原因。

## 5. 工作点

每个成员从自己的 calibration 结果中选择安全工作点：pose recall `>=0.95`、weighted recall `>0.99`、weighted recall 下置信界 `>0.99`（若已计算）。在所有约束满足的阈值中最大化 useful cull。无合格阈值时记录 `no_qualified_safety_workpoint`，不参与安全排名。

同时保存 best F1、最高 precision 和固定 checkpoint 阈值作为诊断工作点，明确它们不属于安全主工作点。保存 safety factor 和 safety-adjusted useful cull，但不使用它替代原始指标。

## 6. 因子效应和 bootstrap

逐 pose 配对计算：

```text
proxy_without_inhibition = B - A
inhibition_without_proxy = C - A
proxy_with_inhibition = D - C
interaction = D - B - C + A
inhibition_with_proxy = D - B
```

每个指标均输出 observed delta、10,000 次分层 paired bootstrap 95% CI、正负方向和 `crossesZero`。bootstrap 先重采样三种 seed 聚类，再在选中的 seed 内对 664 个 pose 有放回重采样；变体间始终使用同一 pose。

## 7. 路线判定

为使“明显下降”可执行且在结果产生前固定，本轮将方向代理的 recall 和 weighted recall 差值 95% 区间下界低于 `-0.01` 视为安全层失败；bad-cull 差值区间上界固定为不超过 `+0.002`。两个限制同时在 pose 宏平均和 aggregate 口径检查。

路线脚本不接受任意综合分数。判定顺序为：

1. 安全层：代理不得使 recall/weighted recall 出现预注册的明显下降，bad-cull 增量必须满足上限。
2. 分类/剔除层：在共同安全工作点下，precision、balanced accuracy、F1、useful cull 或平均预测/GLB 成本至少一项的 95% 区间下界大于 0，才记录预测贡献。
3. 系统层：同视觉效用下 GLB 字节/首屏时间下降，或 miss-pixel/p95 miss-pixel 改善，才记录系统贡献。

若只观察到 logits 发生变化，记录为“模型使用了该输入，但效果证据不足”，不能升级为独立贡献。

## 8. 执行命令和验收

计划冻结提交后，先执行只读审计和旧结果 v2 复汇总：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/summarize_formal_m4_matrix_v2.py --self-test
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/audit_m4_formal_matrix_v2.py \
  --output neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2/input_manifest.json
```

然后仅在 input manifest 确认 A 缺失且依赖资源通过时启动 A 的独立训练入口。所有长任务必须写入 stdout/stderr；旧 M4 目录只读。

验收至少包括 M4-v2 相关 unittest、汇总 self-test、schema 校验、6 个变体/3 个 seed/664 pose 检查、候选 hash 检查、阈值 provenance 检查和 `git diff --check`。报告必须解释 useful cull 高但 recall/weighted recall 低时的反例，不得把该情况描述为性能更好。

## 9. 保留规则

M4-v2 结果全部保留在独立目录并写入正式报告。无合格安全工作点或没有稳定因子效应的成员可作为诊断/失败证据保留，但不改变默认模型、默认前端资产和旧 Route B 文档结论。
