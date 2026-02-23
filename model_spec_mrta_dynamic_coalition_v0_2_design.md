# 同构多机器人 MRTA（动态协同规模）v0.2 详细设计

更新时间：2026-02-23  
设计目标：在保留现有 `Attention + REINFORCE + 事件驱动决策` 训练主干的前提下，把环境从“阈值触发后固定服务时长”升级为“可抢占、可变协同规模、连续工作量衰减”的一致动力学系统。

---

## 1. 设计范围

### 1.1 In Scope（本轮）

- 保持动作空间不变：`a_i in {IDLE, task_1..task_M}`。
- 环境语义升级为连续工作量模型：`R_j(t)` 按当前参与人数 `n_j(t)` 动态衰减。
- 支持任务执行中协同人数变化（机器人离开/加入）并即时改变 `work_rate`。
- 统一奖励与指标定义：makespan 主目标 + travel/wait/mode shaping。
- 提供向后兼容策略，允许复用现有训练脚本与 checkpoint 管线。

### 1.2 Out of Scope（本轮不做）

- 不引入 `(task, mode)` 复合动作。
- 不接入在线 mTSP 求解器（仅保留接口占位）。
- 不改 Attention 网络结构与 loss 形式。
- 不引入障碍物、通信约束、时间窗与任务前后置依赖。

---

## 2. 当前实现与目标语义差距

基于代码阅读：

- `env/task_env.py` 当前在任务启动时一次性计算 `time_finish = time_start + workload/work_rate`，之后不再随人数变化重算，无法表达“动态协同规模可变”。
- `get_unfinished_tasks()` 仍使用 `status/feasible_assignment` 语义，不完全等价于 `remaining_workload > 0`。
- 指标 `efficiency` 当前实质是等待时间统计，名称与资源利用率定义不一致。
- `TestSetGenerator.py` 仍以 `task['time']` 作为静态节点负载，没有显式体现 `workload` 字段语义。

结论：v0.1 已有字段铺垫，但“动力学闭环”尚未真正落地。

---

## 3. v0.2 核心语义

### 3.1 任务动力学

对每个任务 `j`：

- 初始：`R_j(0) = W_j`（默认 `W_j = task['time']`）。
- 瞬时速率：`r_j(t) = alpha_j * g(n_j(t))`。
- 协同效率：`g(n) = n / (1 + beta * (n - 1))`，`beta=0.8`（本版固定）。
- 演化：`dR_j/dt = -r_j(t)`，`R_j(t)<=0` 即完成。

### 3.2 模式成本与奖励

- 模式成本瞬时函数：`h(n)`，默认线性 `h(n)=n`。
- 对每个任务按时间积分：`mode_cost_j += h(n_j(t)) * dt`。
- 终局奖励：
  `reward = -(w_makespan*T + w_travel*travel_time + w_wait*wait_time + w_mode*sum(mode_cost_j))`。

### 3.3 动态协同规模

- 机器人到达任务后可进入 waiting/working。
- 当其他机器人离开或加入任务，`n_j(t)` 改变，任务立即切换到新 `work_rate`。
- 任务可以被抢占：若 `n_j(t)` 降至 0，则进度暂停（`R_j` 保持不变）直到再次有人投入。

---

## 4. 环境状态机设计

### 4.1 Agent 状态

- `TRAVELING`：前往目标点。
- `WAITING`：已到达任务点但暂未参与有效工作（例如策略等待或任务未激活）。
- `WORKING`：计入任务当前 `n_j(t)`。
- `IDLE_AT_DEPOT`：在 depot 等待。
- `RETURNING`：回 depot 路上。

### 4.2 Task 状态

- `UNSTARTED`：`R_j == W_j` 且 `n_j=0`。
- `ACTIVE`：`R_j>0` 且 `n_j>0`。
- `PAUSED`：`0<R_j<W_j` 且 `n_j=0`。
- `FINISHED`：`R_j<=eps`。

### 4.3 事件推进（Event-Driven）

下一时刻由最早事件决定：

- 任一机器人到达目标点；
- 任一任务完成（由 `R_j/work_rate` 推导）；
- 任一机器人等待超时（若保留 `max_waiting_time` 机制）；
- 全任务完成后的返航与回仓完成。

时间推进统一走 `advance_time(t_next)`：

1. 对所有 ACTIVE 任务积分进度与 mode cost。  
2. 对所有 agent 累积 travel/wait/exec 时间分量。  
3. 更新全局时间到 `t_next`。  
4. 处理该时刻事件导致的成员变化与状态转移。  

---

## 5. 数据结构改造（`env/task_env.py`）

### 5.1 Task 字段（新增/重定义）

- `workload`: float，任务总工作量 `W_j`。
- `remaining_workload`: float，任务剩余量 `R_j`。
- `alpha`: float，任务系数。
- `active_members`: list[int]，当前有效参与者（计入 `n_j`）。
- `arrived_members`: list[int]，到达该任务但不一定在工作的人。
- `work_rate`: float，当前速率。
- `last_update_time`: float，上次进度积分时刻。
- `mode_cost`: float，模式成本累计值。
- `state`: str，`UNSTARTED/ACTIVE/PAUSED/FINISHED`。

### 5.2 Agent 字段（新增/重定义）

- `state`: str，`TRAVELING/WAITING/WORKING/IDLE_AT_DEPOT/RETURNING`。
- `target_task_id`: int，当前目标任务（`-1` 为 depot）。
- `decision_time`: float，下次需策略决策时刻。
- `time_exec`: float，累计执行时长。
- `time_wait`: float，累计等待时长。
- `time_travel`: float，累计路途时长。

### 5.3 兼容字段保留

为兼容现有可视化与统计，以下字段暂保留并由新逻辑回填：

- task 侧：`members`, `sum_waiting_time`, `time_start`, `time_finish`
- agent 侧：`route`, `arrival_time`, `sum_waiting_time`, `travel_dist`

---

## 6. 关键接口重构

### 6.1 `advance_time(t_next)`（新增）

职责：

- 统一做连续积分，替代分散在 `task_update/agent_update` 的局部更新；
- 确保每个事件区间 `[t_cur, t_next]` 的状态转移可追溯。

伪代码：

```text
dt = t_next - current_time
for each active task j:
    n = len(active_members_j)
    rate = alpha_j * g(n)
    dW = rate * dt
    remaining_workload_j = max(0, remaining_workload_j - dW)
    mode_cost_j += h(n) * dt

for each agent i:
    accumulate time_travel/time_wait/time_exec by state over dt

current_time = t_next
```

### 6.2 `task_update()`（重构）

改为事件处理器：

- 响应到达事件，更新 `arrived_members`；
- 依据策略/规则把 agent 标记为 waiting 或 working；
- 动态重建 `active_members`，刷新 `work_rate` 与 task state；
- 若 `remaining_workload<=eps`，立刻标记完成并释放成员。

### 6.3 `next_decision()`（重构）

返回最早决策事件时间与 agents：

- 到达且未分配下一目标的 agent；
- 等待超时需重选目标的 agent；
- 任务完成后被释放、需重新决策的 agent；
- 全任务完成后，仍未回仓且需下发 `depot` 的 agent。

### 6.4 `get_unfinished_tasks()`（修正）

以 `remaining_workload > eps` 作为唯一未完成判据。  
不再依赖 `status/feasible_assignment`。

---

## 7. 观测与动作兼容策略

### 7.1 动作空间

不变：`IDLE + task_id`。

### 7.2 任务观测（维持 `TASK_INPUT_DIM=5`）

建议映射：

- `x0`: `remaining_workload / workload`（或原 `status` 占位）  
- `x1`: `current_team_size`（归一化）  
- `x2`: `estimated_rate`（归一化）  
- `x3,x4`: 相对坐标

说明：不增加维度，避免立刻改模型结构；先替换语义。

### 7.3 机器人观测（维持 `AGENT_INPUT_DIM=6`）

建议保留原框架，替换内部统计来源：

- 余下行程时间、余下执行时间、当前等待时间；
- 相对位姿；
- 是否已分配标志。

---

## 8. 训练与评测改造点

### 8.1 `worker.py`

- 训练循环中继续使用事件驱动框架；
- 奖励读取仍走 `env.get_episode_reward()`，但内部已是新定义；
- `perf_metrics['efficiency']` 改名为 `utilization_exec`（保留旧字段别名一段时间）。

### 8.2 `results_plotting.py`

- 指标列从 `efficiency` 迁移到：
  - `utilization_exec`
  - `utilization_wait`
  - `utilization_travel`
- 旧 CSV 兼容：若无新列则回退到旧列，打印 warning。

### 8.3 `TestSetGenerator.py`

- 保持 `W_j <- task['time']`；
- 在输出 YAML 时可选新增 `node_workload` 注释字段，避免与旧基线脚本冲突；
- 修复路径写入一致性（当前创建目录与 dump 路径存在相对路径不一致风险）。

---

## 9. 迁移计划（分阶段）

### Phase P0（语义落地，最小侵入）

- 实现 `advance_time`；
- 用 `remaining_workload` 替代 `status` 判定；
- 让 `work_rate` 可随 `n_j(t)` 动态变化；
- 保持现有动作和训练脚本可运行。

DoD：

- 训练可跑通，`makespan` 与奖励无 NaN；
- 任务可出现 `ACTIVE -> PAUSED -> ACTIVE`；
- `remaining_workload` 单调非增且最终到 0。

### Phase P1（指标与日志一致化）

- 引入三类 utilization 指标；
- 调整 `results_plotting.py` 与 TensorBoard 标签；
- 旧指标兼容导出。

DoD：

- 新旧测试脚本都能产出 CSV；
- 指标字段语义与文档一致。

### Phase P2（稳定性与回归）

- 增加回归测试脚本（固定 seed）；
- 对比改造前后在小规模任务集下的统计波动；
- 评估 travel/wait/mode 对策略行为影响。

DoD：

- 至少 3 组种子稳定跑完；
- 输出对照报告（均值/方差/显著性）。

---

## 10. 测试方案

### 10.1 单元级不变量

- `remaining_workload >= 0`；
- 若 `n_j == 0`，则该时间段 `remaining_workload` 不变化；
- 若 `n_j` 增加，在相同 `dt` 下工作量消耗不减；
- `sum(time_exec + time_wait + time_travel)` 与 `agents_num * makespan` 一致（容忍微小浮点误差）。

### 10.2 场景级回归

- Case A：单任务、多机器人，验证 `g(n)` 边际递减；
- Case B：机器人中途撤离，验证任务进入 `PAUSED`；
- Case C：全部任务完成后返航，验证终止条件。

### 10.3 训练级烟雾测试

- `SAMPLE_SIZE=16`、短回合先跑通；
- 再恢复默认训练规模。

---

## 11. 风险与规避

- 风险 1：事件时间与连续积分耦合，易出现“跨事件积分重复/漏计”。
  - 规避：统一仅由 `advance_time` 执行积分，其他函数禁止改 `remaining_workload`。
- 风险 2：兼容旧字段导致状态双写不一致。
  - 规避：定义单一真值源（`state/remaining_workload`），旧字段只做派生。
- 风险 3：奖励尺度变化导致训练震荡。
  - 规避：先冻结权重，监控 reward 分量并做归一化日志。

---

## 12. 实施文件清单（预计）

- `env/task_env.py`：核心状态机与动力学实现。
- `worker.py`：指标字段与兼容读取。
- `results_plotting.py`：新旧指标兼容逻辑。
- `TestSetGenerator.py`：数据语义与路径一致性小修。
- `parameters.py`：新增兼容开关（如 `USE_DYNAMIC_WORKLOAD=True`）。

---

## 13. 决策冻结（2026-02-23 对齐结果）

1. 任务进行中允许“随时离开并立即生效”。
- 状态：已确认。

2. `WAITING` 包含“到达任务点但任务暂停（无人工作）”场景。
- 状态：已确认。

3. mode cost 按瞬时 active team size 做时间积分 `∫ h(n(t)) dt`。
- 状态：已确认。

4. 第一轮保留 `max_waiting_time=10` 的超时重规划机制。
- 状态：已确认。

5. 指标命名迁移（`efficiency` -> `utilization_exec`）先采用兼容策略。
- 状态：暂定兼容双列（新列 + 旧列别名），避免旧脚本和历史 CSV 断裂；
- 后续在 P1 完成后再将新列设为默认主列。

---

## 14. 本文档与现有文档关系

- `model_spec_mrta_dynamic_coalition_v0_1.md`：定义目标语义与参数基线。
- `progress_report_model_spec_alignment_2026-02-21.md`：描述 v0.1 对齐现状。
- 本文档：给出 v0.2 具体工程落地设计与迁移步骤，作为下一轮代码改造执行清单。
