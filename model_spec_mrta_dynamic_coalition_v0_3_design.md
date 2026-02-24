# 同构多机器人 MRTA（动态协同规模）v0.3 设计与实现说明

更新时间：2026-02-24  
目标：在不改上下层分配 pipeline 的前提下，优先修复 `行为振荡` 与 `信用分配失败`。

---

## 1. 范围

### 1.1 In Scope

- 环境层反振荡约束（软硬混合，中等硬约束）
- 事件级稠密奖励（增量成本 + 行为惩罚 + potential shaping）
- 轻量 critic（`ValueNet`）+ GAE
- 训练日志新增行为诊断指标

### 1.2 Out of Scope

- OR-Tools / CTAS-D 求解语义升级
- RL 与 mTSP/MILP 的上下层在线耦合接口
- MAPPO / 多智能体算法替换

---

## 2. 关键决策

1. Actor 保持现有 `AttentionNet` 结构。  
2. 新增独立 `ValueNet(input_dim=16, hidden_dim=128)`。  
3. 约束采用“硬限制 + 软惩罚”：
- 硬限制：commit lock、quorum protect
- 软惩罚：switch/quorum-break/pause event

---

## 3. 环境层变更（`env/task_env.py`）

### 3.1 新增状态

- Agent：
  - `commit_task_id`, `commit_until`
  - `last_task_id`, `switch_count`
- Task：
  - `pause_events`, `quorum_break_events`
- 全局计数：
  - `total_switch_events`, `total_quorum_break_events`, `total_pause_events`, `total_actions`

### 3.2 新增接口

- `get_action_mask(agent_id)`
  - 基于未完成任务 mask，叠加硬约束
  - Commit lock：`current_time < commit_until` 且锁定任务未完成时，只允许继续该任务
  - Quorum protect：离开会破坏当前任务配额且未到超时重规划点，则禁止离开

- `get_dense_reward_delta(reset=True)`
  - 事件区间增量奖励：
  - `-(w_time*dt + w_travel*Δtravel + w_wait*Δwait + w_mode*Δmode)`
  - `- SWITCH_PENALTY*I_switch`
  - `- QUORUM_BREAK_PENALTY*I_quorum_break`
  - `- PAUSE_EVENT_PENALTY*I_pause`
  - `+ POTENTIAL_SHAPING_COEF*(phi_{t+1}-phi_t)`

- `get_global_features()`
  - 输出 16 维 critic 输入（进度、任务状态占比、utilization、行为事件率等）

- `get_behavior_metrics()`
  - 输出 `switch_rate`, `quorum_break_rate`, `pause_events`

### 3.3 状态转移事件计数

- `agent_step` 中检测切换与破坏配额离开，更新 switch/quorum-break 计数。
- `task_update` 中检测 `ACTIVE -> PAUSED`，更新 pause 计数。

---

## 4. 训练层变更

### 4.1 新模型文件

- `model/value_net.py`：三层 MLP，输出标量价值。

### 4.2 采样与回传（`worker.py`）

- 经验从终局稀疏回报改为事件级轨迹：
  - `agent_inputs, task_inputs, actions, masks`
  - `rewards, dones, values, global_feats`
  - `advantages, returns`
- 每步调用 `get_dense_reward_delta()` 获取 `reward_t`。
- 使用本地 critic 值估计计算 GAE/returns 后回传。

### 4.3 优化与日志（`driver.py`）

- 新增 `value_network` 与 `value_optimizer`。
- 损失：
  - `policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy`
- 优势标准化并裁剪到 `[-5, 5]`。
- entropy 系数线性衰减（`ENTROPY_COEF_START -> ENTROPY_COEF_END`）。
- 新增日志指标：
  - `switch_rate`, `quorum_break_rate`, `pause_events`
  - `adv_std`, `value_loss`, `explained_var`

### 4.4 分布式接口（`runner.py`）

- `job(...)` 新增 `value_weights` 参数并下发到 worker。

---

## 5. 参数新增（`parameters.py`）

- 约束与惩罚：`ENABLE_COMMIT_LOCK`, `MIN_COMMIT_TIME`, `ENABLE_QUORUM_PROTECT`, `SWITCH_PENALTY`, `QUORUM_BREAK_PENALTY`, `PAUSE_EVENT_PENALTY`
- 稠密奖励：`USE_DENSE_EVENT_REWARD`, `USE_POTENTIAL_SHAPING`, `POTENTIAL_SHAPING_COEF`
- Critic 训练：`USE_CRITIC`, `GAE_LAMBDA`, `VALUE_COEF`, `MAX_GRAD_NORM`
- 熵调度：`ENTROPY_COEF_START`, `ENTROPY_COEF_END`, `ENTROPY_DECAY_STEPS`
- Critic 结构：`CRITIC_INPUT_DIM`, `CRITIC_HIDDEN_DIM`

---

## 6. 回归脚本

- `scripts/p2_regression_suite.py` 已扩展输出与对比：
  - `switch_rate`
  - `quorum_break_rate`
  - `pause_events`

---

## 7. 验收建议

1. 机制验收：
- commit lock 生效
- quorum protect 生效
- 事件计数与动作行为一致

2. 训练验收（短训 20k）：
- `switch_rate` 相对 v0.2 显著下降
- `success_rate` / `makespan` 不退化
- 无 NaN，`utilization_exec` 不塌陷

3. 下一阶段（v0.4）：
- 再推进上下层（RL + mTSP/MILP）接口重构与语义对齐
