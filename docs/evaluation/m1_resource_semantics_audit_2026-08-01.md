# M1：训练资源语义审计

日期：2026-08-01  
状态：历史旧数据审计不通过；同日重建的正式空间数据已通过 M1 资源语义门控。

## 正式空间数据复审

上一版审计针对旧的 view-cell CSR：候选文件可能已经补入 GT 可见实例，HKUST 点云存在
fallback，Metropolis 点云索引也指向未实例化源资源。该结果仍保留在下文作为失败记录，不能
与正式数据混用。

本次重新审计的正式资源为：

| 场景 | 正式 CSR | 运行时实例 / GLB | view-cell | 原始候选漏正样本 | 点云失败 / fallback | hash 匹配 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| HKUST | `pose_csr_hkust_v3_spatial_fov66_v1` | 18,831 / 3,273 | 7,999 | 0 | 0 / 0 | 3,273 / 3,273 | **pass** |
| Metropolis | `pose_csr_metropolis_spatial_dense_subpose_union_fov66_v1` | 41,298 / 3,669 | 27,252 | 0 | 0 / 0 | 3,669 / 3,669 | **pass** |

正式审计输出：

- `docs/evaluation/m1_hkust_spatial_resource_audit_2026-08-01.json`
- `docs/evaluation/m1_metropolis_spatial_resource_audit_2026-08-01.json`

两个正式数据集都保存了独立的 `raw_candidate_ids.bin`/offsets，训练和评测默认严格使用
存储候选；`candidateVisibleUnionAdded=0`。Metropolis 的运行时点云来自 3,669 个实例化
GLB 原型，不再使用 41,298 个源场景 GLB 的旧行号。

正式资源 smoke 命令：

```bash
conda run -n slm_pvs bash -c 'PYTHONPATH=neural_instance_culling/model python - <<"PY"
# 对两个正式数据集调用 validate_training_resources(strict_semantics=True)，
# 并抽查每个 split 的 pose，确认 pose-set batch 与存储候选逐项一致。
PY'
```

正式数据资源门已通过；M1 仍不等于模型性能门通过，后续必须使用这些资源重建遮挡证据、
训练模型并按 M0 协议进行评测。

## 历史旧数据审计（保留记录）

## 审计方法

新增 `neural_instance_culling/benchmark/audit_pvs_resources.py`，逐 pose 检查可见集合、候选文件、运行时实例到 GLB 映射和离线点云缓存。脚本明确区分“文件中可见集合属于候选集合”和“原始 AABB 候选阶段已经安全”这两个命题：如果数据元信息表明候选文件是在加入可见正样本之后保存的，就不会把事后子集检查当作候选安全证据。

训练入口的正式模式现在默认执行同等严格的资源语义检查。旧点云或候选文件若只用于探索性复现，必须显式加入 `--allow-invalid-resource-semantics`，该标记不得用于投稿主结果。

运行命令：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/audit_pvs_resources.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3.bin \
  --glb-index hkust-v3/assets/glbIndex.json \
  --output docs/evaluation/m1_hkust_resource_audit_2026-08-01.json

conda run -n slm_pvs python neural_instance_culling/benchmark/audit_pvs_resources.py \
  --dataset-dir neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4 \
  --runtime-meta ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json \
  --glb-points neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin \
  --glb-index ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json \
  --output docs/evaluation/m1_metropolis_resource_audit_2026-08-01.json
```

退出码为 `2` 表示审计失败；JSON 仍会保存完整检查结果，便于记录失败原因。

## 结果

| 项目 | HKUST | Metropolis |
|---|---:|---:|
| 实例数 / 运行时 GLB 数 | 18,831 / 3,273 | 41,298 / 3,669 |
| view-cell / pose 数 | 7,999 | 27,252 |
| 文件中可见引用 / 候选引用 | 860,150 / 40,230,293 | 25,991,985 / 269,836,224 |
| 文件元信息中的候选语义 | AABB 候选加可见正样本 | AABB 候选加可见正样本 |
| 原始候选是否可独立审计 | 否 | 否 |
| 文件中可见集合是否属于保存的候选 | 是 | 是 |
| 点云缓存行数 | 3,273 | 41,298 |
| 运行时原型数是否匹配 | 是 | 否 |
| 解码失败 / fallback | 77 / 80 | 0 / 0 |
| 点云 hash 与运行时 GLB 索引 | 3,273 / 3,273 | 10 / 3,669 |
| 总状态 | **fail** | **fail** |

## HKUST 结论

HKUST 的实例 ID 范围、AABB 映射、运行时 GLB 原型行数和 hash 均一致。当前数据仍有两个阻断项：

1. 构建器先计算 AABB 候选，再把 dense subpose 可见正样本补入候选文件；因此 `candidateMissVisible=0` 只证明保存后的候选覆盖 GT，不能证明前端可执行的原始 AABB 候选没有漏正样本。原始候选没有被单独保存，必须重建并保存 `raw_candidate_ids` 后复核。
2. GLB 点云缓存有解码失败和 fallback。即使行索引与 GLB hash 对齐，也不能把 fallback 点云当作完整几何编码输入。正式训练应重建并要求 `failed=0`、`fallback=0`。

HKUST 可以继续作为协议开发和探索性复现数据，但当前 checkpoint 不能升级为投稿主表证据。

## Metropolis 结论

Metropolis 的运行时元数据是 3,669 个实例化 GLB 原型，而当前点云缓存是 41,298 个源场景独立 GLB。模型训练按运行时 `instance_to_glb` 索引前 3,669 行，因此绝大多数实例取到的不是对应原型几何；只有 10/3,669 行 hash 对齐。该结果不能用于性能比较，必须从实例化场景原型重新生成点云缓存。

候选文件同样记录了加入 GT 后的集合，当前 `candidateMissVisible=0` 不构成原始候选安全证明。重建时需要同时保存：

- 66° 后退相机下由与前端一致的 AABB 候选算法产生的原始候选；
- dense subpose 可见集合；
- 仅用于诊断的补入集合和漏正样本列表。

若原始候选确实漏正样本，必须先修复采样/候选口径，再构造正式 CSR；不能继续依靠构建器补入。

## 资源准入条件

后续重建必须满足：

- 原始 AABB 候选阶段 `candidateMissVisible=0`，并可独立审计；
- `visible_ids`、AABB、`instance_to_glb` 和运行时实例数一致；
- 点云缓存行数等于运行时 GLB 原型数，逐行 hash 全匹配；
- `failed=0`、`fallback=0`；
- 记录输入文件 SHA-256、FOV（采样/模型 66°，真实渲染 60°）、后退距离、采样 seed 和构建命令。

当前 M1 不通过，暂不启动 Metropolis 正式训练，也不把现有两个场景 checkpoint 写入投稿主结果。
