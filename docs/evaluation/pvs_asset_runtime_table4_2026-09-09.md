# 资产与运行时 Table 4

日期：2026-09-09。目的：把 rank 容量、神经 runtime 资产、Geometry-shell HZB 资产和已有正式端侧推理摘要收敛到论文 Table 4 与一张 Pareto 图；本任务只读现有结果，不启动 GPU、不读取浏览器上传原始记录、不计算哈希。

## 口径

生成器为 `neural_instance_culling/benchmark/generate_asset_runtime_table.py`。它读取：

- `paper_results/rank_sweep/rank_capacity.csv`：R=2/4/8/12 的 validation 容量汇总；Pareto 横轴为固定运行特征表 MiB，纵轴为 `aggregateUsefulCull`。只在源 CSV 的全部 seed 已通过 weighted-recall 安全状态且 aggregate weighted recall `>0.99` 的点上判定 Pareto；本次四点均为安全点，Pareto 点为 R=2、R=4。
- `paper_results/mobile_runtime/runtime_paper_summary.csv`：只接受 status 为 `formal`、5 个 session 且 latency 字段完整的正式行。源行 reason 含 `smoke-excluded` 的并发 A6000 记录保留为 `smoke-excluded`，其所有 latency 单元清空；没有正式上传的组合保留为 `unavailable`，不以 smoke 数值补齐。
- HZB 两场景的四个 `shell_meta.json` 与四个 `.offline.json`：核对 schema、场景/变体、offline report 的 `runtimeAsset=false`、压缩流字节和实际 runtime 目录字节。HZB 展开几何内存是 positions、indices、transforms 三类 meshopt segment 的 decoded byte 总和；展开运行时内存再加固定 AABB、instance-to-GLB 和 occluder payload。
- 两场景实际神经 runtime asset 目录：递归求和文件 `stat().st_size`，包含 `model_meta.json`。该值是实际传输字节，不使用 runtime 摘要中只计部分文件的 MiB 作为替代。

表格 CSV 使用 `status` 区分 `available`、`formal`、`smoke-excluded` 和 `unavailable`。Markdown 将资产和运行延迟分成两节；HZB `lossless` 与 `equal-asset` 均保留。`equal-asset` 是按神经资产预算删除完整 primitive 的资产敏感性对照，不是 lossless 的替代，也不宣称独立的保守可见性结论。

## 指标含义与边界

- 传输字节是启动资产的实际文件字节，属于资源效率指标；展开几何/运行时内存是 HZB 解码后的运行资源成本。
- prototype 三角形、展开三角形和 occluder 实例描述 HZB 的遮挡查询工作量，不等价于画面质量。
- formal latency 是已有协议定义的模型前向时间，不包含下载、初始化、候选构造、渲染或 GLB 调度。
- 画面损失指标（image PER、miss pixel、wrong-ID pixel）本表未实现，原因是本任务没有新增 HZB 全量 test visibility；剔除效率指标（useful cull、bad cull）也不从资产大小推断，继续以现有 visibility 报告为准。

修改文件：`neural_instance_culling/benchmark/generate_asset_runtime_table.py`、`neural_instance_culling/benchmark/tests/test_generate_asset_runtime_table.py`、本报告和 `docs/README.md`。代码只依赖 Python 标准库与已有 `slm_pvs` 环境中的 Matplotlib。

## 结果

### 资产

| 场景 | 方案/变体 | 实际传输字节 | 展开几何 MiB | 展开运行时 MiB | prototype | prototype 三角形 | 展开三角形 | occluder 实例 | neural / lossless |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HKUST | neural / pvs-mainline-v4 | 5,479,213 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | **1.37%** |
| HKUST | geometry-shell-hzb / lossless | 399,388,459 | 1,697.30 | 1,697.87 | 3,042 | 54,771,324 | 55,537,431 | 18,566 | 1.37% |
| HKUST | geometry-shell-hzb / equal-asset | 5,476,306 | 18.77 | 19.34 | 91 | 723,248 | 822,121 | 1,257 | 1.37% |
| IFCBench | neural / pvs-mainline-v4 | 11,883,919 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | **19.78%** |
| IFCBench | geometry-shell-hzb / lossless | 60,089,560 | 153.30 | 154.56 | 3,669 | 8,648,833 | 25,556,160 | 41,298 | 19.78% |
| IFCBench | geometry-shell-hzb / equal-asset | 11,835,015 | 24.93 | 26.20 | 1,412 | 1,312,974 | 6,267,776 | 31,909 | 19.78% |

神经资产相对 lossless shell 的实际传输比例为 HKUST `1.37%`、IFCBench `19.78%`。HZB equal-asset 的实际总量分别为神经资产预算下的 `5,476,306 B` 和 `11,835,015 B`；它们没有被合并到 lossless 行。

### 正式运行

| 场景 | 设备 | 后端 | 状态 | session | 样本 | mean ms | p50 ms | p95 ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| HKUST | MacBook Air M2 | WebGPU | formal | 5 | 3,420 | 4.183 (4.010, 4.353) | 2.163 (2.097, 2.163) | 13.242 (12.390, 13.966) |
| HKUST | vivo X200 Pro mini | WebGPU | formal | 5 | 3,420 | 6.466 (6.031, 6.891) | 3.473 (3.080, 3.801) | 18.350 (17.826, 18.809) |
| HKUST / IFCBench | RTX A6000 | WebGPU/WASM | smoke-excluded | 0 | 0 | unavailable | unavailable | unavailable |
| IFCBench | 移动设备 | WebGPU/WASM | unavailable | 0 | 0 | unavailable | unavailable | unavailable |

完整机器可读文件会保留 runtime summary 中每个设备/后端组合，因此 A6000 的四个 smoke-excluded 组合和 IFCBench/移动端的缺失组合均可审计。正式延迟只使用 HKUST 两组五 session WebGPU 结果；不能把并发 smoke 的 timing 当作正式值。

## 产物与复现

在 CPU 环境执行，使用独立 worktree 的结果目录和现有实际资产目录：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/generate_asset_runtime_table.py \
  --paper-results-dir /mnt/sda/rhyang/slm/neural_instance_culling/benchmark/out/paper_results \
  --hzb-dir /mnt/sda/rhyang/slm/neural_instance_culling/benchmark/out/paper_results/hzb \
  --neural-asset hkust=/mnt/sda/rhyang/slm/slm2viewer/assets/neural_instance_culling/pvs_mainline_v4 \
  --neural-asset ifcbench=/mnt/sda/rhyang/slm/slm2viewer/assets/neural_instance_culling/pvs_mainline_v4_ifcbench \
  --output-dir /mnt/sda/rhyang/slm-wt-table4/neural_instance_culling/benchmark/out/paper_results/figures
```

输出目录：

- `rank_capacity_pareto.csv/.pdf/.svg/.png`；
- `table4_asset_runtime.csv/.md`。

测试命令：

```bash
conda run -n slm_pvs python -m unittest neural_instance_culling.benchmark.tests.test_generate_asset_runtime_table
```

## 状态与风险

该表和 Pareto 图作为论文结果整理产物保留，不改变训练、导出、前端或 runtime benchmark 入口。rank 输入只有容量汇总列，没有逐点 LCB 字段；生成器沿用源 `safeSeedCount` 与 aggregate weighted recall 的安全状态，不重新推断置信区间。HZB 当前仍没有正式完整 test visibility 或独占 GPU latency，因此本表只报告已有 CPU 资产统计，并明确保留 smoke/unavailable 状态。
