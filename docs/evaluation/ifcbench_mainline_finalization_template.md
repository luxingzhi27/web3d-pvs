# IFCBench Mainline Finalization

日期：YYYY-MM-DD

状态：待正式 finalize 后回填。本文只记录冻结决策、正式 test 和 runtime 导出，不把 dry-run 或未完成任务写成结果。

## 目的与冻结规则

本次流程读取 confirmation summary，对三个确认 seed 逐一检查 validation 的 weighted recall（按 visible weight 汇总的 GT 找回率）和单侧置信下界。只有三个 seed 的两个数都严格大于 `0.99`，才晋级微调族；否则保留原三 seed Full V4。正式 test 不重新选择阈值，也不从 test 反向校准。

runtime seed 只从最终族的 validation 安全池选择，排序第一关键字为 useful cull（`TN / candidate`，正确剔除不可见候选的比例），之后记录 balanced accuracy、specificity、precision、weighted recall 下界和平均预测数作为诊断。

## 输入与产物

- confirmation summary：`<finetune-benchmark-root>/confirmation_summary.json`
- IFCBench 数据：`<data-root>/neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1`
- 原 Full V4：`<full-model-root>/full_seed<seed>_e40/`
- 微调 confirmation：`<finetune-model-root>/confirm/<selectedConfig>/seed<seed>_e8/`
- 正式 test：`paper_results/test_metrics/ifcbench_fantasy_metropolis/`
- runtime 资产：`<selected-model-root>/runtime_final_<method>_seed<seed>_v1/`
- 编排日志和最终 manifest：`paper_results/logs/finalize_ifcbench_mainline/` 与 `paper_results/finalize/ifcbench_fantasy_metropolis/`

所有 test JSON、typed sidecar、日志、manifest 和 runtime 目录必须是新路径；目标已存在时流程中止，不覆盖已有结果或资产。

## 运行命令

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/finalize_ifcbench_mainline.py \
  preflight \
  --data-root /mnt/sda/rhyang/slm \
  --confirmation-summary /mnt/sda/rhyang/slm-wt-ifc/neural_instance_culling/benchmark/out/pvs_ifcbench_v4_calibration_finetune_v1/confirmation_summary.json \
  --finetune-model-root /mnt/sda/rhyang/slm-wt-ifc/neural_instance_culling/model/out/pvs_ifcbench_v4_calibration_finetune_v1 \
  --finetune-benchmark-root /mnt/sda/rhyang/slm-wt-ifc/neural_instance_culling/benchmark/out/pvs_ifcbench_v4_calibration_finetune_v1
```

确认路径、输出路径和依赖资源通过后，先查看完整命令与日志计划：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/finalize_ifcbench_mainline.py \
  dry-run \
  --confirmation-summary <confirmation-summary> \
  --finetune-model-root <finetune-model-root> \
  --finetune-benchmark-root <finetune-benchmark-root>
```

正式执行：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/finalize_ifcbench_mainline.py \
  finalize \
  --confirmation-summary <confirmation-summary> \
  --finetune-model-root <finetune-model-root> \
  --finetune-benchmark-root <finetune-benchmark-root>
```

本脚本本身不启动浏览器采样，也不生成图像级指标；正式模型评估设备由 `--device` 控制，正式硬件运行按项目 GPU policy 单独留存证据。

## 冻结决策

| 项目 | 值 |
|---|---|
| confirmation summary | `<path>` |
| selectedConfig | `<value>` |
| seed 20260801 WR / LCB | `<value>` / `<value>` |
| seed 20260802 WR / LCB | `<value>` / `<value>` |
| seed 20260803 WR / LCB | `<value>` / `<value>` |
| 最终族 | `finetune` 或 `full_v4` |
| fallback 原因 | `<value>` |

## Formal Test

每个最终族 seed 只运行一次 `evaluate_pvs.py`，并使用该成员自己的 calibration：微调使用 `exact_calibration.json`，原 Full 使用成员目录内的 `calibration_ready_summary`。test 结果必须记录 `testRead=true`、`testEvaluationCount=1`、实际 test pose 数 `2710`，以及普通集合指标和 weighted 指标。

| method | seed | threshold | WR | LCB | recall | precision | balanced accuracy | useful cull | bad cull | avg prediction | GLB byte reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `<method>` | `<seed>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |

画面损失指标（image PER、miss pixel rate、wrong-ID pixel rate）若未接入，必须明确写“未实现”，不能从集合 weighted recall 推断。

## Runtime Export

| 选择依据 | seed | validation WR | validation LCB | useful cull | asset directory | explicit calibration |
|---|---:|---:|---:|---:|---|---|
| validation safe pool, useful cull first | `<seed>` | `<value>` | `<value>` | `<value>` | `<path>` | `true` for fine-tune / `false` for native Full |

runtime 资产只记录导出器生成的 `model_meta.json` 和固定表、查询权重、频率表、查表数据的 schema/shape/字节大小；不把 checkpoint、关系 CSR 或训练端数据放入部署包。

## 验证与风险

- CPU 单测：`<command>`，结果 `<count>`。
- preflight/dry-run：`<result>`。
- formal test：三 seed、每个 `testEvaluationCount=1`。
- 前端 smoke、图像 PER 和硬件延迟：`未实现` 或填写独立报告路径。
- 后续风险：`<value>`。

是否保留为主线：`由逐 seed confirmation 决策填写，不由单个 test seed 或单个 precision 指标决定`。
