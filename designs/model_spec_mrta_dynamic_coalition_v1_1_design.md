# 同构多机器人 MRTA 动态协同模型 v1.1 设计说明（澄清版）

- 文件名: `model_spec_mrta_dynamic_coalition_v1_1_design.md`
- 更新时间: 2026-02-27
- 定位: 文档澄清与实现收敛（在 v1.0 基础上消歧，不再延续 v0.1-v0.4 迭代链）
- 目标: 明确唯一问题定义、统一术语、解释当前 baseline 对接问题，并给出新 clone 代码库的直接落地路径

---

## 0. 核心结论（先看这个）

1. **Problem Formulation 以 v1.0 简化设定为唯一最终版本**。  
2. 后续实现不再按 v0.1-v0.4 的历史路线重复演进。  
3. 当前代码中保留的开关/参数，定位是“兼容与可控迁移”，不是重新引入复杂问题定义。  
4. OR-Tools / CTAS-D 必须在**同一仿真语义**下评测，否则与 RL 对比不公平。  
5. 目前出现的 `mean_makespan=nan`、`deadlock guard triggered`，本质是“route 生成与 v1.0 仿真执行语义未完全对齐”，不是单一命令问题。

---

## 1. 为什么需要 v1.1（消除歧义）

### 1.1 已暴露的歧义点

1. “语义收敛”与“新增参数”在文档表达上容易被理解为冲突。  
2. `execution_semantics`/`simplified_setting` 命名不统一，导致理解成本高。  
3. baseline 仍沿用“路径先验 + 旧执行假设”，与新环境动力学的契合性不明确。  
4. 历史版本（v0.x）与最终版本（v1.0）在同一仓库并存，阅读路径混乱。

### 1.2 v1.1 的作用

- 不是新增建模版本，而是把“v1.0 已定稿”的工程解释写清楚。  
- 为你即将新 clone 的代码库提供“直接从最终设定起步”的实施规范。

---

## 2. 最终问题定义（冻结）

本节结论：**v1.0 简化设定冻结，不再回退到 fully-preemptive 通用语义。**

### 2.1 保留的核心建模要素

1. 静态任务集、同构机器人、单 depot。  
2. 工作量动力学：`remaining_workload` 连续衰减。  
3. 动态协同规模 + 边际递减效率 `g(n)`。  
4. 事件驱动推进（不是固定 duration 一次算死）。  
5. makespan-first 评估。

### 2.2 执行设定（Simplified）

- 采用 `growth_only_strict` 作为默认执行语义：  
1. 执行中 active 成员不可任意切走（直到任务完成）。  
2. 允许增援加入（growth-only）。  
3. `quorum_protect` 继续保留，作为可行性保护。  
4. `commit_lock/min_commit_time` 继续保留，作为兼容开关，不强制绑定 strict。

---

## 3. 参数与接口为何仍存在（不是“复杂化”）

### 3.1 原则

- 目标是“简化问题定义”，不是“删除全部工程控制面”。  
- 参数存在的目的：  
1. 显式记录实验 setting。  
2. 保证训练可复现（写入 `train.yaml`）。  
3. 给 baseline / RL / ablation 使用同一入口。

### 3.2 术语统一决议（v1.1）

- 对外统一称为 **`SIMPLIFIED_SETTING`**。  
- 可选值暂保留：`growth_only_strict`、`growth_only_relaxed`。  
- 当前实验默认且主线仅使用：`growth_only_strict`。  

说明：`growth_only_relaxed` 保留为扩展入口，不代表当前要同时研究两套 formulation。

### 3.3 接口解释

- 原环境接口可以看作“更宽泛超集”；  
- v1.0/v1.1 是在该超集上通过 `SIMPLIFIED_SETTING + mask/guard` 约束为最终设定；  
- 这是一种“保留可维护性 + 收敛行为语义”的工程折中，不是建模倒退。

---

## 4. 三类方法是否在同一 setting 下评测（判定标准）

需要同时满足以下条件才算“同 setting 公平对比”：

1. 使用同一批测试环境实例（同一 `env_*.pkl` 集合）。  
2. 使用同一 `TaskEnv` 仿真逻辑（同一 `task_env.py` 版本）。  
3. 使用同一终止与统计逻辑（success/makespan/time/wait/travel/utilization）。  
4. baseline 的 route 必须能被仿真端正确执行到完成，而不是提前卡死或退出。

结论：只要 baseline 最终也是通过当前 `TaskEnv.execute_by_route(...)` 落地评估，就在同一环境语义下；但 route->simulation 对接若不完整，结果会出现 `NaN`，这属于实现问题，不是评测框架自动保证的。

---

## 5. OR-Tools 当前 test 逻辑与问题根因

### 5.1 当前 OR-Tools baseline 流程

1. 读取 `env_*.pkl`。  
2. `env.reset(...)` + `env.clear_decisions()`。  
3. `solver.VRP(env)` 生成预设路线并写入 `env.pre_set_route(...)`。  
4. `env.execute_by_route(...)` 在 `TaskEnv` 中按事件推进。  
5. `env.get_episode_reward(...)` 与指标落 CSV。

### 5.2 现象

- 你的日志中出现：  
1. `vrp_done` 很快完成；  
2. 仿真阶段长时间 `sim_t` 不增长或提前触发 deadlock guard；  
3. `mean_makespan=nan`, `success_rate=0`。

### 5.3 根因（代码语义层）

`execute_by_route` 的 deadlock guard 触发条件是：
- 没有 active task；  
- 没有 future decision；  
- 当前时间不再前进。  

这意味着 route 执行后，系统进入“无人可继续决策且任务未完成”的状态。典型原因：

1. 预设路线未能覆盖到可执行完成所需的有效协同（特别是 quorum 约束下）。  
2. 某些 agent 早早 `route_exhausted` 回 depot，剩余任务无人形成可行执行。  
3. route 仅是路径解，不天然满足 v1.0 动态执行可行性（工作量+人数+等待时限的联动）。

结论：这是 **route->simulation 对接不充分**，不是“命令卡死”本身。

---

## 6. CTAS-D 在当前 setting 下的同类风险

CTAS-D 也是“读结果路线 -> 写 pre_set_route -> execute_by_route”的链路。  
因此与 OR-Tools 同样受制于：

1. 路线是否满足 v1.0 执行可行性；  
2. 是否与当前 `max_waiting_time`、quorum 机制匹配；  
3. 是否能在同一仿真终止条件下完成全部任务。

如果不满足，同样会出现 success<1、makespan=nan。

---

## 7. 新 clone 代码库的实施策略（直接从 v1.0 起步）

### 7.1 总策略

1. 以干净仓库为起点。  
2. 不再按 v0.1-v0.4 历史步骤重复改造。  
3. 直接实现“简化版 v1.0（本文件定义）”。

### 7.2 最小实现清单（必须）

1. 单一主设定：`SIMPLIFIED_SETTING=growth_only_strict`。  
2. `TaskEnv` 中 strict lock + quorum protect + workload dynamics 全量生效。  
3. 统一指标输出：success/makespan/time/wait/travel/utilization。  
4. baseline 统一经过同一 `execute_by_route` 评测。  
5. deadlock/progress/timeout 日志作为默认诊断能力保留。

### 7.3 建议清理项（减少歧义）

1. 文档首页明确“v1.0 final formulation”。  
2. 将 v0.x 文档移入 `archive/`（仅历史参考，不作实现入口）。  
3. 配置注释显式写明：`SIMPLIFIED_SETTING` 是“语义固定入口”，不是研究多版本 formulation。

---

## 8. v1.1 验收标准

1. 文档层：团队成员能在 5 分钟内明确“唯一最终设定是什么”。  
2. 代码层：新 run 的 `config/train.yaml` 中可见 `SIMPLIFIED_SETTING: growth_only_strict`。  
3. 实验层：RL / OR-Tools / CTAS-D 均在同一测试集、同一 `TaskEnv` 逻辑下产出可比较指标。  
4. 诊断层：若 baseline 失败，日志能区分“超时”与“deadlock（无 future decision）”。

---

## 9. 一句话版本（给后续协作者）

> 从现在开始，DCMRTA 的 Problem Formulation 只认“v1.0 简化版”；v1.1 只负责把这个事实与工程边界写清楚，并确保所有方法在同一仿真语义下公平对比。

