# M1：正式空间数据训练执行记录

日期：2026-08-01  
状态：资源语义审计通过；两场景正式训练进行中，尚未形成 checkpoint 准入结论。

## 目的

在已经通过资源语义审计和空间隔离 split 的数据上，使用统一的 `rvl_strong_v2` 损失配置训练
正式模型。训练必须使用保存的后退视锥候选，不补入 GT 可见实例，不裁剪候选上限；校准阈值、
validation checkpoint 选择和 one-shot test 仍遵循 M0 协议。

## 固定配置

- 模型相机 FOV：66°；前端真实渲染 FOV：60°。
- 精度：FP32，无 AMP。
- 训练轮数：40 epoch；每 epoch 900 个姿态批次。
- 损失：`rvl_strong_v2`，RVL 默认开启。
- 校准目标：weighted recall `>0.99`，校准点估计下限 `0.9925`，最终校准使用 view-cell 聚类单侧 95% 下置信界 `>0.99`。
- 候选语义：`stored_candidate_set_strict`，`max_candidates_per_pose=0`。
- 数据：固定空间四路 split，`train/validation/calibration/test/guard`。
- 当前运行的实际随机种子：`20260610`。输出目录中的 `seed20260801` 是启动时实验标签，不能作为随机种子证据；正式报告以 checkpoint `args.seed` 和 `protocolSplit.selectionSeed` 为准。

## 正式命令

HKUST 使用 GPU 0：

```bash
conda run -n slm_pvs python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_spatial_raw_fov66_v1 \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json --glb-root hkust-v3/assets \
  --epochs 40 --steps-per-epoch 900 --pose-set-batch-size 2 --eval-every 2 \
  --loss-profile rvl_strong_v2 --target-weighted-recall 0.99 \
  --calibration-point-floor 0.9925 --calibration-lcb-floor 0.99 \
  --calibration-bootstrap-replicates 10000 --seed 20260610 --device cuda
```

Metropolis 使用同一配置和 GPU 1；由于候选规模更大，姿态批次改为 1。该改变不修改候选、
损失或数据 split，只改变显存峰值和梯度更新的姿态批次粒度，必须在最终报告中作为训练配置
记录，不能与 batch=2 的结果直接当作同一训练运行。

日志分别写入各自输出目录的 `train_stdout.log` 和 `train_stderr.log`。

## 执行异常与处理

### 第一次 Metropolis 运行：OOM

输出目录：

`neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801`

在第 1 个 epoch、第 266/900 步发生 `torch.OutOfMemoryError`。当时 `pose_set_batch_size=2`，
上下文/遮挡代理编码将每个目标的 `8×3×8` 个证据边一次性送入边编码器，并为联合训练保留
完整反向图。报错发生在边编码器线性层申请约 2.65 GiB 时，GPU 0 已被 HKUST 进程占用，
因此这次运行没有产生可用 checkpoint，也没有被标记为训练失败的模型结果。

### 第二次运行：错误的 GPU 绑定

独立输出目录：

`..._seed20260801_bs1_retry`

虽然改为 batch=1，但没有设置 `CUDA_VISIBLE_DEVICES`，进程仍落到 GPU 0，与 HKUST 共同占用
显存后立即 OOM。该目录只保留错误记录，不参与模型比较。

### 当前重试

独立输出目录：

`..._seed20260801_bs1_gpu1`

已显式设置 `CUDA_VISIBLE_DEVICES=1`、`pose_set_batch_size=1` 和
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。当前 GPU 1 正常运行，未观察到非有限
loss；实际 `args.seed=20260610`，训练完成后必须检查 40 个 epoch、validation 选择、calibration
下置信界和 one-shot test。该目录在正式归档时应改为反映实际 seed 的稳定名称，或在 artifact
manifest 中显式记录旧目录标签与实际 seed 的对应关系。

## 种子命名审计

2026-08-01 检查 `model_meta.json` 得到 HKUST 与 Metropolis 当前运行的
`args.seed=20260610`、`protocolSplit.selectionSeed=20260610`。因此这两次运行可以作为
`seed20260610` 的候选结果，但不能称为 `seed20260801`。训练过程不因目录标签问题中断；完成后
将输出移动到稳定的实际 seed 目录并记录 SHA-256，不覆盖其他正式 seed。后续 seed 必须在命令行
显式传入，不能依赖默认值。

## 当前门控判断

- 数据语义门：通过，见 `docs/evaluation/m1_m2_formal_resource_audit_2026-08-01.md`。
- OOM 处理门：通过重启策略，尚未证明训练完成。
- checkpoint 门：未通过，`best.pt` 尚未产生。
- 阈值/test 门：未通过，不能把旧版本 `0.64` 或测试扫描结果带入正式主表。

## 2026-08-01 运行监控更新

本次检查只读取 tmux、GPU、stdout/stderr 和输出目录，没有修改 CSR、候选集合或 test 数据。

| 场景 | 会话 | 当前进度 | GPU | 当前判断 |
|---|---|---:|---:|---|
| HKUST | `m1_train_hkust_seed20260801` | epoch 9/40，约 61/900 step | GPU 0，约 33.6 GiB | 正常运行 |
| Metropolis | `m1_train_metropolis_seed20260801_gpu1` | epoch 4/40，约 655/900 step | GPU 1，约 38.5 GiB | 正常运行 |

截至本次检查，两个 stderr 均未出现 `NaN`、非有限 loss、OOM 或 CUDA out-of-memory；训练 stdout 中的
`trainSkippedNonFiniteLoss` 和 `trainSkippedNonFiniteGrad` 仍为 0。按最近 epoch 速度粗略估计，HKUST
还需要约 4 小时，Metropolis 还需要约 8 小时，验证阶段和机器负载会使该估计变化，不能作为完成时间承诺。

当前阻塞仍是 checkpoint/test 门：HKUST 只产生到 epoch 8 的中间 checkpoint，Metropolis 只产生到
epoch 2 的中间 checkpoint，两个目录都尚未出现 `instance_runtime_features_fp16.bin` 和最终
`eval_summary.json`。因此现在不能生成冻结 manifest，也不能运行 one-shot test。

另一个需要在训练结束时核对的协议边界是：现有训练程序的最终收尾路径会在写入 `eval_summary.json`
前执行一次单阈值 test。训练结束后若该路径正常完成，这一次应作为正式 one-shot test 处理；不能再用
独立 benchmark 入口重复请求同一个 test split。只有在确认训练收尾没有执行 test、且已生成完整 calibration
bootstrap provenance 时，才可以使用 `evaluate_frozen_test.py evaluate`。

建议监控命令（只读）：

```bash
tmux ls
tail -n 5 neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801/train_stdout.log
tail -n 5 neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_bs1_gpu1/train_stdout.log
nvidia-smi
```

训练完成后由 agent 执行只读核验；用户当前不需要执行重启、采样或评估命令。不要运行旧的
`evaluate_unified_pvs_metrics.py --exploratory-test-threshold-scan`，也不要运行任何会修改
`dataset/out/pose_csr_*` 的候选重建命令。

## 2026-08-01 07:30 后续执行记录

### Metropolis 正式协议重启

此前目录 `..._seed20260801_protocolfix` 只有空的 stdout/stderr，未产生训练进程或 checkpoint，
不能作为失败模型，也不能作为正式运行证据。一次 `/tmp` 启动探针使用同一数据、同一环境、
`1 epoch/1 step` 在 GPU 2 完成了前向、反向和验证入口，证明 CUDA 与资源读取链路可用；探针不写入
正式输出。

随后在独立 tmux 会话 `slm_metro_formal_gpu2` 中启动了新的正式运行：

`neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_gpu2`

命令显式设置 `CUDA_VISIBLE_DEVICES=2`、`--seed 20260801`、FP32、`pose-set-batch-size=2`、
`rvl_strong_v2`、40 epoch、900 steps/epoch、`--skip-final-test`，并保留原始 CSR candidate 集合。
训练进程已进入第 1 epoch，stderr 未见 OOM、NaN 或非有限 loss。该目录是当前唯一可作为
`seed20260801` 的 Metropolis 正式训练候选；GPU 1 上的 `..._bs1_gpu1` 仍是实际 seed
`20260610` 的旧诊断运行，完成后只能作为历史诊断，不能覆盖或冒充本次正式结果。

### 同时执行的验证

完整 benchmark 单元测试于本次记录前运行完成：

```text
PYTHONDONTWRITEBYTECODE=1 /home/data/rhyang/miniconda3/envs/slm_pvs/bin/python \
  -m unittest discover -s neural_instance_culling/benchmark/tests -p 'test_*.py' -v
Ran 23 tests in 51.431s
OK
```

该测试只验证评估器、冻结 test 入口、实例 ID 图像 schema、输入消融和基线 runner 契约，
不构成模型质量门或 test 结果。

### 当前状态边界

截至本次记录，HKUST 正式运行约在 epoch 10，Metropolis GPU2 正式运行尚未完成第 1 epoch；
两者都没有最终 calibration bootstrap、冻结阈值 manifest 或 one-shot test。任何中间 checkpoint、
旧 seed20260610 运行和 `/tmp` 探针均不得进入论文主表。

## 后续改进方向

如果 batch=1 仍无法覆盖 Metropolis 的最大候选姿态，不能继续盲目降低候选或修改 GT。应实现
真正的离线特征阶段：先在无梯度模式下按候选分块生成固定几何、上下文和遮挡代理表，再冻结
该表训练视线查询头；或者实现带梯度检查点的分块边编码，并验证其输出与未分块路径一致。两者
都必须保留完整候选语义和独立实验目录，不能用前端或评测阶段的补丁掩盖训练显存问题。

## 2026-08-01 07:54 运行状态纠正与 batch=2 失败记录补充

上一节“Metropolis 正式协议重启”中的目录状态已经过时，现按实际日志更正如下：

- `..._protocolfix_gpu2` 使用 `CUDA_VISIBLE_DEVICES=2`、`pose_set_batch_size=2`，在第 1 个
  epoch 约第 359/900 步因边编码器线性层申请额外约 2.75 GiB 而 OOM 退出。这里的错误信息
  显示 `GPU 0` 是 CUDA 可见设备重映射后的逻辑编号，物理卡为 GPU 2；该目录没有可用
  `best.pt`，只作为失败实验记录保留。
- 当前正式重试使用独立目录
  `..._protocolfix_bs1_gpu2`，显式设置 `CUDA_VISIBLE_DEVICES=2`、
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`--seed 20260801` 和
  `--pose-set-batch-size 1`。截至 07:54，日志到第 1/40 epoch、约 554/900 步，未出现
  OOM、NaN 或非有限梯度；训练仍在运行。
- GPU1 上的 `..._bs1_gpu1` 仍是早先未显式使用本次正式 seed 的诊断运行，不能与当前
  `seed20260801` 结果合并，也不能替代正式 Metropolis 结果。
- 同时检查到 HKUST 正式目录日志已进入第 14/40 epoch（约 680/900 步），当前无异常。

本次修正没有修改候选集合、GT 标签、空间 split 或阈值；batch=1 只降低单次反向传播的
显存峰值，正式结论仍需等待完整 40 epoch、校准 bootstrap 和冻结 test。若 batch=1 最终仍
因最大姿态失败，下一步必须采用保持完整候选语义的边分块/梯度检查点或离线固定特征方案，
并以独立实验目录和等价性测试验证，不能用候选截断规避问题。

## 2026-08-01 HKUST 收尾失败审计与正式重试

### 失败原因

原 HKUST 运行目录
`model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix`
在第 24/40 epoch 的中间验证完成后停止。日志显示训练 loss、非有限 loss/gradient 计数均正常，
停止发生在最终收尾写入 calibration-ready 结果时，旧进程报出：

```text
TypeError: 'NoneType' object is not subscriptable
```

该错误没有把“没有可用安全工作点”和“收尾返回对象非法”区分开，导致结果目录只有中间 checkpoint，
不能作为正式模型或正式阈值证据。该目录保留为失败审计，不读取其旧 `best.pt` 进入 test 或主表。

### 修复

训练器在最终 calibration 返回后新增显式检查：

- `calibration_workpoint` 必须是满足注册 weighted-recall/置信下界规则的字典；
- validation workpoint 必须包含冻结阈值；
- 任一条件不满足时抛出明确的安全门错误，不导出含糊的 calibration-ready 文件。

该修复不修改候选集合、GT、损失、阈值网格或 checkpoint 选择规则；`py_compile` 已通过，独立复现
探针也验证了 runtime feature 导出和 calibration/validation 调用契约可正常完成。

### 正式重试

新任务使用独立目录
`model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2`，
GPU0、显式 seed `20260801`、FP32、pose-set batch size 2、40 epoch、900 steps/epoch、
`rvl_strong_v2`、`--skip-final-test`。stdout/stderr 分别写入该目录；test 仍需由冻结测试入口唯一执行。
启动时第 1 epoch 正常运行，未出现 OOM、NaN 或非有限梯度。

当前结论：原运行是收尾协议失败，不是模型质量通过；重试完成并通过 calibration-ready 检查前，
M0 总门和 HKUST 正式 checkpoint 门保持未通过。
