# View-cell GT 采样收敛初检

日期：2026-09-09

## 目的与口径

本实验检查有限 subpose 并集能否稳定近似区域 PVS。每个场景从 validation 按场景类别、candidate 数和 GT 数分层选择 100 个 view-cell；嵌套采样顺序从中心最近点开始，再用最远点覆盖扩展。`G_N` 是前 N 个真实 Color-ID subpose 的实例并集，参考是该 cell 当前所有已采样 subpose 的并集。

该初检不读取 test。实例覆盖和 `visible_weights` 加权覆盖同时报告；后者是采样器的重要性权重，不称为真实像素覆盖。零 GT cell 的空集对空集覆盖定义为 1。

## 当前结果

| 场景 | 当前 subpose | N | 实例覆盖均值 | 实例覆盖 5% 分位 | 加权覆盖均值 | 从 N 到 2N 新增比例 |
|---|---:|---:|---:|---:|---:|---:|
| HKUST | 32/48 | 16 | 98.401% | 91.402% | 99.9929% | 1.311% |
| HKUST | 32/48 | 32 | 99.668% | 98.166% | 99.9973% | 未观测 |
| IFCBench | 4 | 1 | 70.539% | 60.835% | 94.983% | 16.525% |
| IFCBench | 4 | 2 | 84.382% | 77.403% | 97.692% | 15.618% |
| IFCBench | 4 | 4 | 100%（相对自身） | 100% | 100% | 未观测 |

HKUST 当前 32 点已接近当前参考并集，但仍不能证明 64/128 点不再发现新实例。IFCBench 从 2 点增加到 4 点仍新增约 15.62% 实例，因此 4 点只定义了当前训练/评价 GT，不能作为区域采样收敛证据。

## 后续正式动作

正式 supplementary 实验仍需对同一 100 个 validation cell 构造嵌套的 `1/2/4/8/16/32/64/128` 真实硬件 Color-ID 采样。补采样必须使用登记的 Chrome Vulkan 硬件路径及 GPU evidence；完成后以 128 点并集为参考重跑本工具。若 64 到 128 的新增实例比例或加权遗漏仍明显，继续增加参考采样，不能提前宣称收敛。

实现入口：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_gt_convergence.py \
  --scene <scene> --viewcell-dataset <viewcell-source> \
  --pose-csr <pose-csr> --raw-dir <color-id-jsonl> \
  --output-dir neural_instance_culling/benchmark/out/paper_results/gt_convergence/<scene>
```

当前输出位于 `paper_results/gt_convergence/hkust_existing/` 和 `ifcbench_existing/`，明确标记为现有采样初检，不进入最终 128 点论文表。
