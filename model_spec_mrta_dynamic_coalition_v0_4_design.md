# 同构多机器人 MRTA（在线下层调度接口 + 工程精简）v0.4 设计

更新时间：2026-02-25  
目标：在保持 v0.3 行为语义不回退的前提下，预留在线下层调度接口，并降低训练主链路复杂度。

---

## 1. 设计范围

### 1.1 In Scope

- 预留在线下层调度接口（dispatcher contract）
- 默认策略保持 baseline 组队逻辑（行为兼容）
- 训练/评估主链路去重与代码精简
- makespan-first 模型选择落地

### 1.2 Out of Scope

- 不引入完整在线 mTSP/MILP 求解器
- 不改 actor/critic 网络结构
- 不改变现有 reward 的数学定义

---

## 2. 在线下层调度接口（预留）

### 2.1 目标语义

将“上层决策意图（选任务）”与“下层成员编组/执行调度”解耦。

- 上层 RL：输出 action（任务或 IDLE）
- 下层 dispatcher：决定本次 group 的成员子集（谁跟随 leader 执行该 action）

### 2.2 接口定义

新增模块：`scheduler/online_dispatcher.py`

- `DispatchContext`
  - `leader_id`
  - `action`
  - `vacancy`
  - `group`
- `OnlineDispatcher.select_members(context, env) -> List[int]`
- `BaselineRandomDispatcher`
  - 与当前 baseline 随机 follower 行为一致

### 2.3 环境接入点

`env/task_env.py`：

- `TaskEnv(..., online_dispatcher=None)`
- `set_online_dispatcher(dispatcher)`
- `step(...)` 内新增：
  - `_resolve_vacancy(...)`
  - `_resolve_members(...)`
  - `_sanitize_selected_members(...)`

默认 `online_dispatcher=None`，等价于原始 baseline 随机组队行为。

### 2.4 参数预留

`parameters.py` 新增：

- `ENABLE_ONLINE_DISPATCH`
- `ONLINE_DISPATCH_POLICY`

当前仅支持 `baseline_random`，用于行为兼容验证。

---

## 3. makespan-first 模型选择

`driver.py` 评估逻辑改为：

1. 主判据：`test_makespan_mean < baseline_makespan_mean`
2. 显著性：paired t-test on makespan（`p < 0.05`）
3. 次要记录：reward 仍记录并输出显著性，不作为 best 更新主判据

新增持久化字段：

- `best_makespan`
- `p_value_makespan`
- `p_value_reward`

---

## 4. 主代码精简策略

### 4.1 已落地去重

1. `driver.py` 抽取评估函数：
   - `evaluate_policy_on_testset`
   - `summarize_eval_rows`
   - `paired_ttest_pvalue`
   - baseline/test 评估复用同一组 Ray actors
   - 评估阶段复用训练 actors，避免每次 eval 的 actor kill/recreate
2. `driver.py` 抽取权重同步函数：
   - `get_worker_weight_bundle`
   - `get_policy_weights`
   - `launch_training_jobs`
3. `worker.py` 抽取统一指标收集：
   - `_collect_perf_metrics`

### 4.2 预期收益

- 减少重复逻辑，降低后续改动引发回归的概率
- 评估链路可读性提升，便于后续切换到其他判据
- 局部减少重复张量搬运和重复指标组装开销

---

## 5. 验收建议

1. 行为兼容性：
- `ENABLE_ONLINE_DISPATCH=false` 与 v0.3 行为一致
- `ENABLE_ONLINE_DISPATCH=true` 且 `baseline_random` 时指标分布无系统性偏移

2. 模型选择一致性：
- best 更新由 makespan 主导
- reward 不再与 best 判据混淆

3. 工程质量：
- 训练主链路通过语法检查
- 评估日志字段完整且可回溯

---

## 6. v0.5 入口（建议）

1. 在 dispatcher 中新增 `greedy_eta` 策略（轻量在线调度）
2. 定义统一 `dispatch debug trace`（便于分析 coalition 行为）
3. 将 dispatcher 接口逐步与 baseline 的 OR/CTAS 局部重规划能力对齐
