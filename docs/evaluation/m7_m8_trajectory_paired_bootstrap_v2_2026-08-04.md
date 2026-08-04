# M7/M8 轨迹配对 bootstrap 汇总

状态：离线轨迹配对统计完成；不替代真实网络、移动设备或像素级效用质量门。

## 口径

current cascade 与 independent ranker 只在相同场景、相同导航轨迹和相同 pose 顺序上比较。bootstrap 外层按轨迹聚类重采样，内层按该轨迹的 pose 重采样，共 `10,000` 次。
候选集合哈希在旧 replay schema 中没有保存，因此本次只校验 pose index、时间、候选数量、GLB 候选数量和弱效用需求；结果不能声称完成候选哈希审计。

## 配对差值

`improvementPositive` 已按指标方向统一：召回提高为正，缺失效用、时间和字节减少为正。

| 指标 | 改善点估计 | 95% CI | 是否跨零 |
|---|---:|---:|---|
| `pose.deadlineUtilityRecall` | -0.126118 | [-0.206147, -0.0494915] | 否 |
| `pose.meanMissingUtilityRatio` | -0.124356 | [-0.202961, -0.0500412] | 否 |
| `pose.missingWeakUtilityIntegralMs` | -82060.5 | [-151457, -20989.3] | 否 |
| `trajectory.finalTrajectoryUtilityRecall` | -0.128326 | [-0.212509, -0.0475759] | 否 |
| `trajectory.meanMissingUtilityRatio` | -0.125336 | [-0.202591, -0.0527515] | 否 |
| `trajectory.downloadedBytes` | 37289.3 | [-7.89132e+06, 8.23121e+06] | 是 |
| `trajectory.requestedBytes` | 1.09274e+07 | [-4.33942e+06, 2.75477e+07] | 是 |
| `trajectory.invalidDownloadBytes` | -1.08168e+07 | [-1.80928e+07, -4.34185e+06] | 否 |
| `trajectory.lateUsefulDownloadBytes` | -615101 | [-1.99555e+06, 557304] | 是 |
| `trajectory.firstUsefulFrameMs` | 16.9029 | [1.31227, 33.9904] | 否 |

## 解释边界

该汇总只说明在固定离线网络假设下，两种排序路线在相同轨迹上的配对差异；它不把独立 ranker 的优势解释成联合可见性头的因果贡献，也不把弱可见权重解释成真实像素覆盖率。M7/M8 的真实网络、设备解码/上传和移动端 p95 质量门仍需独立证据。
