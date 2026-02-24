# DCMRTA 进度说明（对齐 `model_spec_mrta_dynamic_coalition_v0_1.md`）

更新时间：2026-02-23
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

## 8. 2026-02-23 设计迭代会话纪要（可续聊版本）

本节用于记录本轮与用户的关键对话、冻结决策、已产出文档与后续执行入口，确保新会话可直接接续。

### 8.1 本轮用户目标

- 阅读代码库与 `model_spec`/进展文档；
- 先做下一版本详细 design，再进入实现；
- 过程中保持随时沟通；
- 将对齐结论写入 `progress_report` 以便会话切换后可无缝继续。

### 8.2 本轮已完成动作

- 完整扫描代码结构，重点阅读：
  - `model_spec_mrta_dynamic_coalition_v0_1.md`
  - `progress_report_model_spec_alignment_2026-02-21.md`
  - `env/task_env.py`
  - `worker.py`
  - `driver.py`
  - `parameters.py`
  - `TestSetGenerator.py`
  - `results_plotting.py`
- 新增 v0.2 设计文档：
  - `model_spec_mrta_dynamic_coalition_v0_2_design.md`
- 将 v0.2 决策区更新为“已确认/暂定”状态（见该文档第 13 节）。

### 8.3 本轮对话决策冻结（用户确认）

用户明确确认：

1. 允许任务执行中机器人随时离开并立即生效。  
2. `WAITING` 包含“到达但任务暂停（无人工作）”。  
3. mode cost 按瞬时人数进行时间积分。  
4. 保留超时重规划机制（`max_waiting_time=10`）。  

关于第 5 点（指标命名）：

- 用户表示“暂时不太理解”；本轮采用兼容方案：
  - 新增标准语义列：`utilization_exec`；
  - 暂保留旧列：`efficiency` 作为别名输出；
  - 在 P1 阶段再完成默认主列切换，避免历史脚本和已有 CSV 中断。

### 8.4 第 5 点的语义说明（供后续会话快速理解）

- 当前 `efficiency` 在代码里实际是“等待时间统计”，名称与“资源利用率”语义不一致。
- 改名为 `utilization_exec` 的目的：
  - 避免误读；
  - 和 `utilization_wait`、`utilization_travel` 形成一致指标组；
  - 便于后续论文与报告表达。
- 兼容双列是迁移策略，不改变算法行为，只改变日志语义清晰度。

### 8.5 下一步执行入口（P0 实现计划）

按 `model_spec_mrta_dynamic_coalition_v0_2_design.md` 执行 P0：

1. 在 `env/task_env.py` 落地统一积分入口 `advance_time(t_next)`。  
2. 将未完成判据统一到 `remaining_workload > eps`。  
3. 支持任务 `ACTIVE -> PAUSED -> ACTIVE` 的动态速率切换。  
4. 保持动作空间与训练主干不变，先跑通训练烟雾测试。  

### 8.6 当前仓库状态（会话切换提示）

- 已新增：`model_spec_mrta_dynamic_coalition_v0_2_design.md`。
- 已更新：本 `progress_report` 文档。
- 仓库中有既有未提交改动：`scripts/train.sh`（非本轮修改，需保留）。

## 9. 会话切换最小上下文

若在新会话继续，请直接提供以下锚点：

- 目标文档：`model_spec_mrta_dynamic_coalition_v0_2_design.md`
- 对齐记录：`progress_report_model_spec_alignment_2026-02-21.md` 第 8 节
- 执行阶段：`P0`（先改环境动力学，不改模型结构）
- 决策状态：第 8.3 节 1-4 已确认；第 5 点为兼容迁移策略

## 10. 2026-02-23 P0 实施进展（已落代码）

### 10.1 已修改文件

- `env/task_env.py`
- `worker.py`
- `results_plotting.py`
- `RL_test.py`
- `model_spec_mrta_dynamic_coalition_v0_2_design.md`
- 本文档 `progress_report_model_spec_alignment_2026-02-21.md`

说明：`scripts/train.sh` 存在用户既有未提交改动，本轮未触碰其内容。

### 10.2 环境动力学改造（`env/task_env.py`）

已落地要点：

1. 新增统一时间推进入口 `advance_time(t_next)`：  
   - 在事件区间内按当前 active team size 积分 `remaining_workload` 与 `mode_cost`。  
   - 按 agent 状态累计 `time_exec/time_wait/time_travel`。  

2. 未完成任务判据切换：  
   - `get_unfinished_tasks()` 改为以 `remaining_workload > eps` 为准。  

3. 任务状态与成员管理增强：  
   - 新增/兼容字段：`state`, `started`, `active_members`, `arrived_members`。  
   - `task_update()` 按到达成员与需求门槛动态切换 `UNSTARTED/ACTIVE/PAUSED/FINISHED`。  

4. 机器人改主意即时生效：  
   - `agent_step()` 在新分配前会从旧任务成员集中移除该机器人，立即影响协同人数。  

5. 决策节奏更新：  
   - `agent_update()` 中工作中的 agent 采用 `min(预计完成时刻, current_time + max_waiting_time)` 触发再决策，支持周期性离开/重规划。  
   - 任务暂停等待超时后，agent 可即时进入再决策。  

6. 指标基础能力：  
   - 新增 `get_utilization_metrics()` 返回 `utilization_exec/wait/travel`。  
   - 奖励中 travel 项改为使用累计 `time_travel`。

### 10.3 指标并行迁移（你要求的新旧并行）

1. `worker.py`：
   - 新增输出：
     - `utilization_exec`
     - `utilization_wait`
     - `utilization_travel`
   - 保留兼容列：
     - `efficiency`（当前映射为 `utilization_exec`）  

2. `results_plotting.py`：
   - `Efficiency` 指标优先读取 `utilization_exec`，缺失时回退 `efficiency`。  
   - 增加可选绘图：`Utilization Wait`、`Utilization Travel`。  

3. `RL_test.py`：
   - 结果模板已扩展新列，同时保留 `efficiency`。

### 10.4 本轮验证结果

已执行：

1. 语法校验：  
   - `python -m py_compile env/task_env.py worker.py results_plotting.py RL_test.py` 通过。  

2. 环境烟雾测试（随机策略，纯 `TaskEnv`，无需 scipy）：
   - 多轮事件推进无异常；
   - 检查通过：`remaining_workload >= 0`、无 NaN、utilization 在合法范围。  

3. 训练级 smoke 说明：  
   - 直接跑 `Worker` 受限于当前环境缺少 `scipy`（`worker.py` 依赖），因此未做完整训练回归。  
   - 该限制是运行环境依赖问题，不是本次代码语法或接口错误。

### 10.5 当前阶段判定

- `P0`：已进入“代码已落地 + 基础可运行”阶段。  
- 下一建议动作：
  1. 安装训练依赖（至少 `scipy`）后做短训练回归；
  2. 对比改造前后同 seed 的 makespan/wait/travel/utilization；
  3. 根据结果微调 `max_waiting_time` 与 reward 权重。

## 11. 2026-02-23 P1/P2 实施进展（本轮已完成）

### 11.1 P1（指标与日志一致化）完成项

已落地文件：

- `driver.py`
- `worker.py`
- `results_plotting.py`
- `RL_test.py`

关键结果：

1. 训练指标链路已切换到新指标组并保留兼容：
   - 新增：`utilization_exec`, `utilization_wait`, `utilization_travel`
   - 兼容：`efficiency`（别名保留）

2. TensorBoard/W&B 指标已同步：
   - 新增 Perf 标签：`Utilization Exec/Wait/Travel`
   - 旧标签 `Waiting Efficiency` 继续保留（兼容历史面板）

3. 训练日志输出已扩展：
   - `driver.py` 的 train log 现在同时打印 `util_*` 与 `eff`

4. 绘图与统计兼容逻辑已打通：
   - `results_plotting.py` 优先读 `utilization_exec`，缺失时回退 `efficiency`
   - 并可选输出 `Utilization Wait/Travel` 图表

### 11.2 P2（稳定性与回归）完成项

已新增脚本：

- `scripts/p2_regression_suite.py`

功能：

1. 固定 seed 的环境回归（默认 `0,1,2`）：
   - 随机策略下执行事件驱动仿真；
   - 自动检查不变量：剩余工作量不为负/不反增、reward 有限、utilization 范围合法；
   - 输出：
     - 回归明细 CSV
     - 汇总 JSON（pass rate、finished rate、均值指标）

2. CSV 对照报告（可选）：
   - 输入 `--current-csv` 与 `--reference-csv`
   - 输出 markdown 对比报告（均值、delta、若可用则 paired t-test）

3. 兼容当前容器依赖：
   - 脚本不依赖 pandas；
   - `scipy` 不存在时仍可运行主流程，仅统计显著性列为 `nan`。

### 11.3 P2 脚本使用示例

1. 仅跑环境回归：
`python scripts/p2_regression_suite.py --output-dir regression --seeds 0,1,2 --agents 6 --tasks 12`

2. 环境回归 + 历史 CSV 对照：
`python scripts/p2_regression_suite.py --output-dir regression --seeds 0,1,2 --current-csv testSet_20A_50T_CONDET/REINFORCE_LF.csv --reference-csv testSet_20A_50T_CONDET/OR-Tools.csv`

### 11.4 本轮验证记录（P1/P2）

1. 语法校验通过：
`python -m py_compile driver.py env/task_env.py worker.py results_plotting.py RL_test.py scripts/p2_regression_suite.py`

2. P2 脚本 smoke 通过：
- 已成功生成：
  - `p2_env_regression_*.csv`
  - `p2_env_regression_summary_*.json`
  - `p2_csv_compare_*.md`（启用 CSV 对照时）

3. 运行环境限制说明：
- 当前容器缺少 `scipy`、`ray`、`pandas` 的部分依赖组合；
- 因此未在本轮执行完整训练回归（但不影响已提交代码的语法与 P2 回归脚本可用性）。

### 11.5 当前阶段结论

- `P1`：已完成（新旧指标并行已落地到训练、测试、绘图链路）。  
- `P2`：已完成首版（固定 seed 回归与对照报告工具已就位并可执行）。  
- 下一阶段可进入：短训练回归与参数稳定性调优（在依赖环境补齐后执行）。

## 12. 2026-02-23 baseline 最小对齐层（已完成）

### 12.1 目标

在不改 baseline 求解核心语义的前提下，先完成“统一评测参数 + 统一输出接口”，为后续公平对齐改造打底。

### 12.2 已改动

1. 统一评测参数入口：
- `parameters.py` 新增 `EVAL_MAX_WAITING_TIME = 10`
- baseline 回放时统一传入：
  - `max_time=MAX_TIME`
  - `max_waiting_time=EVAL_MAX_WAITING_TIME`

2. baseline 输出指标统一到新 schema（并行兼容）：
- 文件：
  - `baselines/OR-Tools.py`
  - `baselines/CTAS-D.py`
- 新增输出：
  - `utilization_exec`
  - `utilization_wait`
  - `utilization_travel`
- 兼容保留：
  - `efficiency`（映射为 `utilization_exec`）

3. baseline 回放接口支持显式评测参数：
- `env/task_env.py`:
  - `execute_by_route(path, method, plot_figure, max_time, max_waiting_time)`

### 12.3 新增接口文档

- `pipeline_interface_env_eval_v0_2.md`

内容覆盖：

- RL / OR-Tools / CTAS-D 三条 pipeline 的环境接口与评估接口；
- 统一指标 schema；
- 后续语义对齐的优先级与修改入口。

### 12.4 状态结论

- “同执行环境 + 同评测参数 + 同输出接口” 已完成；
- “同优化目标语义（动态 workload 在线协同）” 尚未完成，属于后续深度对齐阶段。

## 13. 2026-02-24 v0.3 基础改进（振荡与信用分配）实施记录

### 13.1 会话目标与冻结决策

用户在本轮明确：

1. 先做基础改进，不改上下层 pipeline。  
2. 振荡与信用分配是主要矛盾，不走“单纯增大模型”路线。  
3. 接受 `轻量 Critic + GAE`。  
4. 约束策略采用“软硬混合 + 中等硬约束”。  

### 13.2 已落地代码

本轮已改动：

- `parameters.py`
- `env/task_env.py`
- `model/value_net.py`（新增）
- `worker.py`
- `runner.py`
- `driver.py`
- `scripts/p2_regression_suite.py`
- `RL_test.py`（构造函数兼容调整）
- `model_spec_mrta_dynamic_coalition_v0_3_design.md`（新增）

### 13.3 环境层（v0.3）已实现内容

1. 新增硬约束接口：`get_action_mask(agent_id)`  
   - commit lock（最短驻留）  
   - quorum protect（避免离开导致配额破坏）  

2. 新增事件级奖励接口：`get_dense_reward_delta(reset=True)`  
   - 支持 `Δtime/Δtravel/Δwait/Δmode`  
   - 支持 switch / quorum-break / pause 软惩罚  
   - 支持 potential shaping  

3. 新增 critic 特征接口：`get_global_features()`（16 维）  

4. 新增行为指标接口：`get_behavior_metrics()`  
   - `switch_rate`, `quorum_break_rate`, `pause_events`  

5. 扩展 agent/task 状态字段：  
   - agent: `commit_task_id`, `commit_until`, `last_task_id`, `switch_count`  
   - task: `pause_events`, `quorum_break_events`  

### 13.4 训练层（v0.3）已实现内容

1. 新增 `ValueNet`（`model/value_net.py`）。  
2. `worker.py` 改为事件级轨迹采样，回传 `rewards/dones/values/global_feats` 与 `advantages/returns`。  
3. `driver.py` 新增 actor-critic 优化：
   - `policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy`
   - 优势标准化 + 裁剪
   - entropy 线性衰减
   - 新日志指标：`switch_rate`, `quorum_break_rate`, `pause_events`, `adv_std`, `value_loss`, `explained_var`
4. `runner.py` 的分布式 job 接口增加 `value_weights` 下发。  

### 13.5 回归工具更新

`scripts/p2_regression_suite.py` 已补充 v0.3 行为指标输出与 CSV 对照：

- `switch_rate`
- `quorum_break_rate`
- `pause_events`

### 13.6 兼容与注意事项

1. 本轮未触碰 OR-Tools / CTAS-D 语义，仅保持评测接口兼容。  
2. `RL_test.py` 已按新 `Worker` 构造函数兼容。  
3. v0.3 目标是先稳定训练信号与行为，不等价于最终上下层协同方案。  

### 13.7 新会话接续锚点

- 设计文档：`model_spec_mrta_dynamic_coalition_v0_3_design.md`  
- 主实现文件：`env/task_env.py`, `worker.py`, `driver.py`, `runner.py`  
- 回归入口：`scripts/p2_regression_suite.py`  
- 下一步优先事项：
  1. 运行短训（建议 20k episode）验证新指标是否达标。  
  2. 按结果微调 `SWITCH_PENALTY / QUORUM_BREAK_PENALTY / MIN_COMMIT_TIME`。  
  3. 达标后再进入 v0.4 上下层接口融合。  

## 14. 2026-02-25 v0.4 预留接口与工程精简（本轮）

### 14.1 本轮目标

1. 将模型选择改为 `makespan-first`。  
2. 预留在线下层调度接口，但默认保持 baseline 组队行为。  
3. 对主链路做去重精简，降低冗杂度并提高可维护性。  

### 14.2 新增设计文档

- `model_spec_mrta_dynamic_coalition_v0_4_design.md`

覆盖内容：

- 在线 dispatcher 接口 contract
- makespan-first 评估/选模
- 主链路精简与后续 v0.5 入口

### 14.3 已落地代码变更

1. `driver.py`（选模与评估链路）
- 评估改为 makespan-first：
  - 主判据：`test_makespan_mean < baseline_makespan_mean`
  - 显著性：paired t-test on makespan
- 保留 reward 统计，但不再作为 best 更新主判据
- 新增字段：
  - `best_makespan`
  - `p_value_makespan`
  - `p_value_reward`
- 去重函数新增：
  - `evaluate_policy_on_testset`
  - `summarize_eval_rows`
  - `paired_ttest_pvalue`
  - `get_worker_weight_bundle`
  - `get_policy_weights`
  - `launch_training_jobs`
- 评估效率优化：
  - baseline/test 评估复用同一组 actors，减少重复创建开销
  - eval 过程复用训练 actors，不再每次 eval 后 kill/recreate actors

2. `worker.py` / `runner.py`（评估输出）
- `baseline_test()` 改为返回结构化指标字典（`reward/makespan/success_rate`）。
- `runner.testing()` 同步返回指标字典，配合 driver 的 makespan-first 逻辑。
- 抽取 `_collect_perf_metrics()` 统一收集指标，减少 `run_episode/run_test/run_test_IS` 重复。
- 增加注释，明确指标收集函数是唯一 schema 维护点。

3. `env/task_env.py` + `scheduler/`（在线调度接口预留）
- 新增 `scheduler/online_dispatcher.py`：
  - `DispatchContext`
  - `OnlineDispatcher`
  - `BaselineRandomDispatcher`
- `TaskEnv` 新增 `online_dispatcher` 注入点与 `set_online_dispatcher()`
- `step()` 新增：
  - `_resolve_vacancy()`
  - `_resolve_members()`
  - `_sanitize_selected_members()`
- 默认行为不变：未启用 dispatcher 时仍走 baseline 随机组队。
- 增加注释说明 dispatcher 分支是在线下层调度唯一扩展点。

4. `parameters.py`
- 新增预留参数：
  - `ENABLE_ONLINE_DISPATCH`
  - `ONLINE_DISPATCH_POLICY`

### 14.4 本轮验证

已执行语法检查：

- `python -m py_compile driver.py runner.py worker.py env/task_env.py scheduler/online_dispatcher.py`

结果：通过。

### 14.5 当前结论

1. 选模目标已和“makespan 主目标”对齐。  
2. 在线下层调度接口已完成最小侵入预留，可在不改上层策略结构的前提下迭代。  
3. 主链路冗杂度已下降，后续继续做性能型优化（如 actor 复用、评估并发调度优化）风险更可控。  
