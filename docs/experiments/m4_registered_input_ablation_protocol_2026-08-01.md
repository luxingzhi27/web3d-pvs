# M4：注册输入分支消融协议

日期：2026-08-01
状态：代码与 fixture 已完成，正式三随机种子训练尚未开始。
目的：为核心消融提供与完整模型相同的候选集合、模型容量、训练协议和阈值冻结流程，分别检验几何、上下文和方向遮挡代理输入的独立作用。

## 变更目的

此前的推理期代理干预只能回答“已经训练好的模型是否使用了某个输入”，不能替代从头训练的架构/输入消融。训练器现增加一个注册的运行时输入屏蔽选项。它保留同一网络结构和固定特征表布局，只在查询头之前屏蔽指定分支；这样候选集合、视角输入、输出头和统计协议不变，结果可与完整模型配对比较。

屏蔽分支的辅助监督同时关闭：方向代理被屏蔽时，不再用代理证据、代理可见保护、代理稀疏和代理排序损失训练一个不会进入可见性查询的分支。可见性集合损失、RVL、数量/预算约束、视觉效用和 GLB 优先级损失保持不变。

## 变体定义

| 计划名称 | `--runtime-feature-ablation` | 查询头可用的固定特征 | 解释 |
|---|---|---|---|
| AABB + ray | `geo_context_proxy_zero` | AABB 派生的射线/屏幕标量 | 仅使用轻量实例包围盒和当前视线查询 |
| geometry + ray | `context_proxy_zero` | 几何特征和当前视线查询 | 检验固定几何是否提供超出 AABB 的信息 |
| geometry + context + ray | `proxy_zero` | 几何、上下文和当前视线查询 | 检验方向代理的增量作用 |
| geometry + context + proxy + ray（无显式抑制） | `none` + `--disable-explicit-inhibition` | 几何、上下文、方向代理和当前视线查询；抑制输出固定为零 | 区分代理输入与显式抑制头的贡献 |
| full | `none` | 几何、上下文、方向代理、当前视线查询和显式抑制 | 当前正式主线 |

固定特征文件仍完整导出，以便报告冷启动字节和 schema 成本；这不是把特征从资产中删除，而是明确禁止相应分支参与该消融的运行时决策。该语义会写入 checkpoint 的 `config.runtimeFeatureAblation` 和训练参数。

## 修改文件

- `neural_instance_culling/model/directional_occlusion_proxy_encoder_model.py`
  - 增加注册的运行时分支屏蔽，并把屏蔽语义写入模型配置。
- `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py`
  - 增加 CLI 选项和辅助损失屏蔽，保证训练与评估使用同一分支语义。
- `neural_instance_culling/benchmark/model_runners.py`
- `neural_instance_culling/benchmark/evaluate_proxy_interventions.py`
  - 从 checkpoint 配置恢复消融状态，避免评估时静默恢复完整输入。

模型配置同时记录 `usesExplicitInhibition`。默认值为 `true`，只有正式消融显式传入
`--disable-explicit-inhibition` 才会将抑制输出置零；该开关不删除参数，保证 state-dict schema 与主线一致。

## 运行模板

每个场景和每个变体必须使用独立输出目录、显式 seed、完整 validation/calibration/test 协议。以下只给出参数片段，正式运行时沿用 M1 的场景资源参数：

```bash
--loss-profile rvl_strong_v2 \
--runtime-feature-ablation context_proxy_zero \
--seed 20260811 \
--device cuda
```

其中 `context_proxy_zero` 只是示例；完整矩阵必须包含上述五个变体，至少三个随机种子。正式 test 仍只能读取 calibration 冻结的阈值，不能为某个消融单独扫描 test 阈值。

## 已完成验证

```text
Python compile: passed
四种消融的合成前向 shape/finite 检查: passed
evaluate_proxy_interventions.py --self-test: passed
git diff --check: passed
```

当前没有正式指标。正在运行的 M1 两场景 40 epoch 进程在本次代码修改前启动，未受影响；它们完成后先完成 M0 冻结审计，再启动正式 M3 干预和 M4 多种子消融。

## 2026-08-01 RVL-off 控制变量修正

审计发现训练器在解析参数后无条件应用 loss profile，导致 `rvl_strong_v2` 会覆盖显式的
`--rvl-mode off`。这会使 RVL-off 变体无法真正关闭 RVL，破坏 M4 的因果比较。

已修改 `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py`：loss profile 现在只提供默认值，显式命令行 `--rvl-mode` 优先；省略该参数时仍使用所选 profile 的默认模式。新增 `resolve_loss_profile` 并在
`test_training_calibration_control.py` 中覆盖显式 `off` 和 profile 默认两条路径。

验证结果：针对性训练控制测试 5 项通过，训练脚本编译通过，帮助信息确认 `--rvl-mode` 为可用覆盖项。该修复不改变已经启动的训练进程；正式 RVL-off 消融必须在新进程中显式传入 `--rvl-mode off`，并使用独立输出目录和 seed。

## 质量门与解释边界

M4 的主比较以 `geometry + context + ray` 为代理增量母基线，按相同 calibration 安全工作点比较 useful cull、bad cull、weighted recall、普通集合指标和图像/下载指标。方向代理只有在三种子 paired bootstrap 的安全工作点上达到投稿计划规定的 useful cull 或同效用字节收益，并且区间不跨零时，才能升级为独立论文贡献；否则将其降级为固定场景表征中的辅助分支，不通过命名或阈值调整掩盖失败。

## 正式矩阵收尾入口

`neural_instance_culling/benchmark/run_formal_m4_ablation_matrix.sh` 已登记五个变体和三个随机种子。它等待当前主线训练与已启动的 AABB+ray seed 结束后，默认使用 GPU 0、1、2 按三个并发槽排队运行剩余成员；每个成员拥有独立目录、stdout/stderr、checkpoint 和 calibration-ready 记录，不覆盖已有结果。

矩阵名称为 `aabb_ray`、`geometry_ray`、`geometry_context_ray`、
`geometry_context_proxy_ray_no_inhibition`、`full`，种子默认是 `20260801`、`20260802`、`20260803`。
其中首个 `aabb_ray/20260801` 就是本节后面的已启动目录，其余成员由收尾入口补齐。若机器上有其他 GPU 作业，使用
`SLM_M4_GPUS="1"` 可以改为单槽运行；脚本不会自动终止外部进程。

## 2026-08-01 正式矩阵启动记录

在 GPU1 上启动第一个正式矩阵成员，使用与当前 HKUST 主线相同的空间数据、证据目录、点云缓存、FP32 和
40 epoch 训练协议：

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n slm_pvs \
  python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_spatial_raw_fov66_v1 \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/model/out/pvs_m4_ablation_aabb_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40 \
  --experiment-name pvs_m4_ablation_aabb_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40 \
  --epochs 40 --steps-per-epoch 900 --pose-set-batch-size 2 --eval-every 2 \
  --loss-profile rvl_strong_v2 --runtime-feature-ablation geo_context_proxy_zero \
  --target-weighted-recall 0.99 --calibration-point-floor 0.9925 --calibration-lcb-floor 0.99 \
  --calibration-bootstrap-replicates 10000 --seed 20260801 --device cuda --skip-final-test \
  > neural_instance_culling/model/out/pvs_m4_ablation_aabb_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40/train_stdout.log \
  2> neural_instance_culling/model/out/pvs_m4_ablation_aabb_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40/train_stderr.log
```

启动时尚无指标结论；必须等待 `calibration_ready_summary.json` 和 `best.pt` 生成后，才可把该成员纳入
M4 比较。其余变体和随机种子仍需使用不同的稳定输出目录，不能复用该目录或把中间 checkpoint 当成正式结果。
