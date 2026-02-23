# DCMRTA Pipeline Interface（Env + Eval）v0.2

更新时间：2026-02-23  
目的：定义 RL 与 baseline（OR-Tools / CTAS-D）在当前代码中的统一接口，便于后续会话继续做“语义对齐改造”和“公平评测”。

---

## 1. 总览

当前有 3 条主 pipeline：

1. RL 训练/测试 pipeline（Attention + REINFORCE + TaskEnv）
2. OR-Tools baseline pipeline（离线路径 -> TaskEnv 执行评测）
3. CTAS-D baseline pipeline（外部求解结果 -> TaskEnv 执行评测）

统一目标：

- 统一执行环境：`env/task_env.py`
- 统一评测时间与等待参数：`MAX_TIME`, `EVAL_MAX_WAITING_TIME`
- 统一输出指标 schema：`utilization_*` + `efficiency` 兼容别名

---

## 2. 环境接口（TaskEnv Contract）

核心文件：`env/task_env.py`

### 2.1 关键状态字段

- Task 侧：
  - `workload`, `remaining_workload`
  - `active_members`, `arrived_members`
  - `state` (`UNSTARTED/ACTIVE/PAUSED/FINISHED`)
  - `mode_cost`
- Agent 侧：
  - `state` (`TRAVELING/WAITING/WORKING/IDLE_AT_DEPOT/RETURNING`)
  - `time_exec`, `time_wait`, `time_travel`
  - `pre_set_route`

### 2.2 关键方法（供上层 pipeline 调用）

- `reset(test_env=None, seed=None)`  
  重置环境（可注入给定任务集）

- `clear_decisions()`  
  清空 episode 决策痕迹并重置任务/机器人执行态

- `next_decision() -> (decision_agents, current_time)`  
  返回下一决策时刻的 agent 集合与事件时间

- `task_update()` / `agent_update()`  
  在当前时间点推进环境状态

- `step(group, leader_id, action, current_action_index=0)`  
  RL 训练/测试使用的在线动作接口

- `pre_set_route(routes, agent_id)`  
  baseline 注入离线路径

- `execute_by_route(path='./', method=0, plot_figure=False, max_time=200, max_waiting_time=100)`  
  baseline 路径回放执行接口（已支持传参统一）

- `get_episode_reward(max_time=100) -> (reward, finished_tasks)`  
  终局奖励与任务完成状态

- `get_utilization_metrics() -> (util_exec, util_wait, util_travel)`  
  资源利用率指标

---

## 3. 评测接口（Metrics Contract）

统一指标字段（CSV/字典）：

- `success_rate`
- `makespan`
- `time_cost`
- `waiting_time`
- `travel_dist`
- `utilization_exec`
- `utilization_wait`
- `utilization_travel`
- `efficiency`（兼容别名，当前映射为 `utilization_exec`）

说明：

- 新代码应优先读 `utilization_exec`
- 旧脚本仍可读 `efficiency`

---

## 4. RL Pipeline 接口

### 4.1 训练入口

- `driver.py` -> `Runner` -> `Worker.run_episode()`
- 关键调用：
  - `TaskEnv.step(...)`
  - `TaskEnv.get_episode_reward(...)`
  - `TaskEnv.get_utilization_metrics()`

### 4.2 测试入口

- `RL_test.py` -> `Worker.run_test()` / `Worker.run_test_IS()`

### 4.3 参数入口

文件：`parameters.py`

- `MAX_TIME`: 评测/训练最大时间窗
- `EVAL_MAX_WAITING_TIME`: 统一 baseline 回放等待阈值
- reward 权重与动态协同参数同文件维护

---

## 5. OR-Tools Baseline Pipeline 接口

文件：`baselines/OR-Tools.py`

流程：

1. 读取 `testSet_*/env_*.pkl`
2. `env.reset(...)` + `env.clear_decisions()`
3. `solver.VRP(env)` 生成并注入 `pre_set_route`
4. `env.execute_by_route(..., max_time=MAX_TIME, max_waiting_time=EVAL_MAX_WAITING_TIME)`
5. `env.get_episode_reward(MAX_TIME)` + `env.get_utilization_metrics()`
6. 输出统一指标 schema 到 CSV

已完成最小对齐：

- 与统一评测参数对齐（`MAX_TIME`, `EVAL_MAX_WAITING_TIME`）
- 输出新旧并行指标

未完成语义对齐（后续）：

- OR-Tools 求解代价仍是静态路由代价，不是动态 workload 在线协同优化

---

## 6. CTAS-D Baseline Pipeline 接口

文件：

- 求解：`CTAS-D_test.bash`
- 结果读取与回放：`baselines/CTAS-D.py`
- 测试集生成：`TestSetGenerator.py`

流程：

1. `TestSetGenerator.py` 生成 `planner_param.yaml / graph.yaml / task_param.yaml`
2. `CTAS-D_test.bash` 调用 CTAS-D 可执行程序产出 `results.yaml`
3. `baselines/CTAS-D.py` 读 `results.yaml` 注入 `pre_set_route`
4. `env.execute_by_route(..., max_time=MAX_TIME, max_waiting_time=EVAL_MAX_WAITING_TIME)`
5. 统一指标输出到 CSV

已完成最小对齐：

- 与统一评测参数对齐
- 输出新旧并行指标

未完成语义对齐（后续）：

- CTAS-D 输入图仍主要依赖静态任务时长/需求，不含动态协同在线决策语义

---

## 7. 统一改造优先级（后续会话直接接）

### P-A（已完成）

- 统一评测参数入口（baseline 回放阶段）
- 统一指标 schema（新旧并行）

### P-B（建议下一步）

- 统一 baseline 求解输入语义（从静态 `task['time']` 迁移到动态 workload 近似模型）
- 明确“公平对比定义”：
  - 同执行环境（已完成）
  - 同评测参数（已完成）
  - 同优化目标语义（未完成）

### P-C（高级）

- 抽象公共 evaluator（单脚本调度 RL/OR/CTAS-D，统一出表）
- 为 baseline 增加“动态重规划版本”（分段求解 + 在线回放）

---

## 8. 关键文件速查（修改入口）

- 环境动力学：`env/task_env.py`
- RL 训练日志与指标：`driver.py`
- RL 测试：`worker.py`, `RL_test.py`
- OR-Tools baseline：`baselines/OR-Tools.py`
- CTAS-D baseline：`baselines/CTAS-D.py`, `CTAS-D_test.bash`, `TestSetGenerator.py`
- 结果聚合：`results_plotting.py`
- 全局参数：`parameters.py`

---

## 9. 兼容性约束

- 保持 `efficiency` 列至少一个版本周期（兼容历史 CSV / 绘图脚本）。
- 新代码统一写 `utilization_*`。
- 若修改 `execute_by_route` 行为，必须同步更新本文件和 `progress_report`。
