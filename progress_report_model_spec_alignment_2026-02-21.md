# DCMRTA 进度说明（对齐 `model_spec_mrta_dynamic_coalition_v0_1.md`）

更新时间：2026-02-21
范围：基于当前 `DCMRTA` 代码库，对照你的建模文档，评估“已实现 / 部分实现 / 未实现”。

## 1. 结论摘要

- 当前仓库已经稳定实现了 ICRA 版本 DCMRTA 的核心流程：`Attention + REINFORCE + 事件驱动决策 + 动态组队阈值`。
- 你的 `model_spec` 是明显更先进的下一阶段目标模型，和当前代码存在结构性差距，尤其是“可分解工作量 + 连续进度动力学 + 协同效率函数 + 模式成本”。
- 从已保存结果看，现有 RL 在 makespan 上优于 OR-Tools 和 CTAS-D 基线，但 travel distance 更高，且 reward 设计还没有覆盖你希望的多目标权衡。

## 2. 与 model_spec 的逐项对照

| model_spec 条目 | 当前代码状态 | 证据 |
|---|---|---|
| 静态任务集（episode 开始给定） | 已实现 | `env/task_env.py:57`, `model_spec_mrta_dynamic_coalition_v0_1.md:11` |
| 同构机器人、同一 depot、欧氏距离、恒速 | 已实现 | `env/task_env.py:66`, `env/task_env.py:99`, `env/task_env.py:111`, `env/task_env.py:162` |
| ST 假设（同一时刻单任务） | 已实现 | `env/task_env.py:300` |
| 允许等待与改主意 | 部分实现 | 到点等待由 `max_waiting_time` 控制：`env/task_env.py:30`, `env/task_env.py:245`; 改主意主要发生在任务不可行或等待超时后的再决策 |
| 终止时全员回 depot | 已实现 | `env/task_env.py:279`, `env/task_env.py:370` |
| 上层 MARL 决策“下一任务（含 IDLE）” | 已实现（版本 A） | `worker.py:69`, `worker.py:73`, `env/task_env.py:307` |
| 下层 mTSP 与上层解耦（分层接口） | 部分实现 | 训练主流程未接入 mTSP；mTSP/VRP 主要在 baseline：`baselines/OR-Tools.py:136` |
| 任务可分解 workload `R_j(t)` 连续演化 | 未实现 | 当前任务是“人数达阈值后执行固定时长 `time`”：`env/task_env.py:252`, `env/task_env.py:257` |
| 协同效率函数 `g(n)`（边际递减） | 未实现 | 未见 `g(n)` 或 `r_j(n)`；仅有阈值式可行性 |
| 模式成本 `h(n)` 惩罚 | 未实现 | 奖励未包含规模成本项 |
| 过程 shaping（travel/wait/mode） | 未实现（仅终局 makespan） | `env/task_env.py:424` |
| 动作升级到 `(task, strength)` | 未实现 | 仅单动作任务选择：`worker.py:70` |

## 3. 当前训练与评估管线成熟度

## 3.1 训练主线（可用）

- 使用 Ray 并行采样与 REINFORCE 更新：`driver.py:19`, `driver.py:99`, `driver.py:175`。
- 经验中优势估计由“当前策略回报 - 贪心基线回报”给出：`worker.py:89`, `worker.py:92`。
- 超参数集中在 `parameters.py`，当前规模是 10-20 agents / 20-50 tasks：`parameters.py:15`, `parameters.py:16`。

## 3.2 测试与基线（可用）

- RL 离线测试脚本：`RL_test.py:34`。
- OR-Tools baseline：`baselines/OR-Tools.py:163`。
- CTAS-D 结果读取与回放：`baselines/CTAS-D.py:26`。
- 统一指标汇总与统计检验：`results_plotting.py:23`, `results_plotting.py:52`。

## 4. 现有结果快照（来自仓库已有文件）

来源：`testSet_20A_50T_CONDET/metrics/metrics.csv`

- CTAS-D_300s: success 1.0, makespan 36.908, travel_dist 42.027
- OR-Tools: success 1.0, makespan 41.996, travel_dist 49.590
- REINFORCE_IS: success 1.0, makespan 35.017, travel_dist 50.001
- REINFORCE_LF: success 1.0, makespan 34.827, travel_dist 49.629

解读：

- 现有 RL 在 makespan 上最好（约 34.8-35.0）。
- 但 travel distance 没有优于 CTAS-D，说明当前 reward 几乎只在优化完工时间。

## 5. 关键差距与风险

- 语义差距：你文档中的任务是“可分解工作量系统”，当前代码是“达人数阈值后固定服务时长”。
- 目标差距：你计划的多目标/多成本 reward 还未进入训练闭环。
- 分层差距：当前 RL 和 mTSP 没有形成在线协同接口，mTSP 仅用于基线。
- 指标定义差距：代码中的 `efficiency` 实际等于任务平均等待时间，不是资源利用率定义。

## 6. 建议的落地顺序（最小改动到最大收益）

1. P0：把任务模型从“阈值+固定时长”改为“`R_j` 连续衰减 + `g(n)`”，先保持动作空间不变。
2. P0：reward 增加 travel/wait/mode 三项 shaping，但保留终局 makespan 主导。
3. P1：把动作从 `task` 扩展为 `(task, mode)`，实现你文档里的动态协同规模控制。
4. P1：抽象下层调度接口，允许同一训练流程下切换“直接执行 / 局部路径优化 / 近似 mTSP”。
5. P2：补资源利用率指标（`U_exec/U_travel/U_wait`）并替换现有 `efficiency` 命名，避免语义混淆。

## 7. 当前项目阶段判断

当前建议阶段标签：

- 阶段 A（已完成）：DCMRTA baseline 与 RL 主干可复现实验。
- 阶段 B（进行中）：向 `model_spec` 的“动态可分解任务模型”迁移设计已明确，但代码尚未开始系统改造。

如果后续以 `DCMRTA` 作为唯一 codebase，这份文档可作为迁移任务的起始基线。
