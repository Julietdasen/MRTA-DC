# 同构多机器人MRTA动态协同规模模型 v1.0 设计说明

- 文件名: `model_spec_mrta_dynamic_coalition_v1_0_design.md`
- 更新时间: 2026-02-25
- 定位: 问题定义收敛版本 (Problem Definition Consolidation)
- 目标: 在保留“工作量动力学 + 边际递减协同效率”的前提下，收敛任务执行语义，减少fully-preemptive setting导致的高切换/暂停振荡，为后续PPO与dispatcher迭代提供稳定环境基线。

---

## 0. 版本背景与定位

### 0.1 v1.0 的核心定位

v1.0 是在 v0.2-v0.4 基础上的问题定义收敛版本，重点是先把环境语义做稳定，再推进算法迭代。

保留内容:
- 静态任务集
- 同构机器人
- 工作量动力学 (`remaining_workload`)
- 动态协同规模
- 边际递减协同效率函数 `g(n)`
- 事件驱动环境推进
- makespan-first 目标
- 分层结构（上层MARL + 下层dispatcher接口）

收敛内容:
- 从广义可抢占（任意加入/离开/暂停）收缩为 **Growth-Only / Quorum-Safe** 执行语义
- 允许任务开始后新增机器人加入提升速度
- 限制执行中成员离开，避免任务频繁跌破配额与暂停

### 0.2 设计动机

在此前版本中，fully-preemptive 自由度会诱发以下病态行为:
- 高频切换 (switch)
- quorum break（抽走关键成员导致任务不可执行）
- pause/resume 振荡
- reward credit assignment 噪声增大

v1.0 的目标不是否定动态协同，而是将其约束到更符合现实的半可抢占语义:
- 允许增援加速
- 不允许随意撤离破坏执行连续性

---

## 1. 问题定义 (Problem Definition)

### 1.1 实体与集合

- 机器人集合: $\mathcal{A}=\{1,\dots,N\}$，同构
- 任务集合: $\mathcal{T}=\{1,\dots,M\}$，静态任务集（episode开始时一次性给定）
- Depot: $p_0 \in \mathbb{R}^2$
- 任务位置: $p_j \in \mathbb{R}^2$

### 1.2 运动模型

- 机器人恒速 $v>0$
- 欧氏距离旅行时间:
\[
\tau(u \rightarrow v) = \frac{\|p_u-p_v\|_2}{v}
\]

v1.0 不考虑障碍，后续版本可扩展到图路网或带障碍路径规划。

### 1.3 任务参数（每个任务 $j$）

- $p_j$: 任务位置
- $W_j>0$: 总工作量 (workload)
- $R_j(t)$: 剩余工作量，初始 $R_j(0)=W_j$
- $\alpha_j>0$: 任务处理系数（默认可设为1）
- $q_j \in \mathbb{Z}_+$: 最低可执行人数 (quorum)
- （可选）$\eta_j \in \{0,1\}$: 是否允许任务开始后新增成员加入
  - v1.0 默认 $\eta_j = 1$

---

## 2. 核心任务动力学（工作量模型）

### 2.1 协同效率函数 `g(n)`

令 $n_j(t)\in\{0,1,\dots,N\}$ 为任务 $j$ 在时刻 $t$ 的有效执行人数（active working members）。

任务瞬时处理速率定义为:
\[
r_j(t)=\alpha_j \, g(n_j(t))
\]

函数 `g(n)` 需满足:
1. $g(0)=0$
2. 单调递增: $g(n+1)\ge g(n)$
3. 边际递减: $g(n+1)-g(n)$ 随 $n$ 增大而减小

v1.0 推荐默认形式（饱和型）:
\[
g(n)=\frac{n}{1+\beta(n-1)},\quad \beta \ge 0
\]

含义:
- 增员可提高速度
- 边际收益递减
- 避免线性堆人带来的不现实策略

### 2.2 剩余工作量演化

\[
\frac{dR_j(t)}{dt} = -\alpha_j g(n_j(t))
\]

完成条件:
\[
R_j(t) \le \varepsilon
\]

其中 $\varepsilon$ 为数值容忍阈值（例如 `1e-6`）。

### 2.3 动态人数变化下的事件处理原则（重要）

当任务成员加入/离开导致 $n_j(t)$ 变化时:
- 不应沿用“固定 time_finish 一次性算死”的旧语义
- 应基于当前 `remaining_workload` 与新速率重新推导下一次可能完成事件时间

这是 v1.0 必须保留的核心语义一致性要求。

---

## 3. v1.0 执行语义（Growth-Only / Quorum-Safe）

这是 v1.0 的关键定义，用于替代过于自由的 fully-preemptive 执行语义。

### 3.1 任务启动条件（Quorum启动）

任务 $j$ 进入执行态（ACTIVE）的必要条件:
\[
n_j(t) \ge q_j \quad \text{and} \quad R_j(t) > 0
\]

### 3.2 Growth-Only（允许增援加速）

当任务已经处于 ACTIVE 且任务允许 join-after-start（$\eta_j=1$）时:
- 允许新增机器人加入该任务
- 新增后 $n_j(t)$ 增加
- 任务速率 $r_j(t)$ 即时提升（由 `g(n)` 给出，边际递减）

直觉:
- 开工后可以增援加速
- 保留动态协同规模建模价值
- 不再依赖固定duration语义

### 3.3 Quorum-Safe（离开限制，保护执行连续性）

v1.0 默认不允许任意撤离导致任务跌破最低配额。

#### 约束 A: Quorum Protect（最低配额保护）
若某机器人离开会使任务人数低于 quorum，则该离开动作无效:
\[
n_j(t)-1 < q_j \Rightarrow \text{leave action invalid}
\]

#### 约束 B: 执行成员离开限制（Execution Leave Restriction）
v1.0 提供两档模式（推荐先实现Strict）:

1. `growth_only_strict`（默认更稳）
   - 执行中的成员在任务完成前不可离开
   - 允许新成员加入，但不允许撤离

2. `growth_only_relaxed`（后续扩展）
   - 核心成员不可离开
   - 增援成员可在满足 commit lock（若启用）后离开
   - 离开仍需满足 quorum protect

> 推荐工程策略: 先用 strict 跑通环境与训练，再做 relaxed 版本对比。

### 3.4 关于暂停（PAUSED）的定位

v1.0 不再鼓励任务频繁进入 `PAUSED` 状态:
- `PAUSED` 状态保留（用于兼容与异常情形）
- 通过硬约束（quorum protect + 执行离开限制）使其成为少数事件
- `pause_events` 应从常态行为变为诊断信号

---

## 4. 任务状态机 (Task State Machine)

每个任务的状态定义如下:

- `UNSTARTED`: 尚未进入有效执行，通常 `R_j = W_j`
- `ACTIVE`: 正在执行，`R_j > \varepsilon` 且 `n_j \ge q_j`
- `PAUSED`: 已开始但暂不可执行（兼容保留）
- `FINISHED`: `R_j \le \varepsilon`

### 4.1 状态转移规则

- `UNSTARTED -> ACTIVE`: 当人数达到 quorum 且任务未完成
- `ACTIVE -> FINISHED`: `remaining_workload <= eps`
- `ACTIVE -> PAUSED`: 仅在异常或特殊放宽策略时发生（应尽量少）
- `PAUSED -> ACTIVE`: 再次满足 quorum 且继续执行

---

## 5. 机器人状态机 (Agent State Machine)

机器人状态建议保留以下枚举:

- `TRAVELING`
- `WAITING`（到达任务点但未进入有效执行）
- `WORKING`
- `RETURNING`
- `IDLE_AT_DEPOT`

### 5.1 单任务约束（ST约束）

任一时刻，一个机器人只能处于一个任务或一个移动/等待状态中:
\[
\forall i,\ \text{agent } i \text{ cannot contribute to multiple tasks simultaneously}
\]

### 5.2 成员角色划分（建议实现）

为支持 quorum-safe 与后续 relaxed 模式，建议每个任务维护两类成员:

- `core_members`: 维持最低配额的核心成员
- `support_members`: 额外增援成员

有效执行人数:
\[
n_j(t)=|core\_members_j(t)|+|support\_members_j(t)|
\]

该划分有助于:
- 更清晰地实现 quorum protect
- 在 relaxed 模式中区分谁能离开
- 提升日志可解释性

---

## 6. 目标函数与评估指标（Makespan-First）

### 6.1 主目标

定义:
- $T_{done}$: 所有任务完成的时刻
- $T$: 所有机器人回到 depot 的时刻（标准 makespan）

主目标:
\[
\min T
\]

### 6.2 次级过程代价（训练塑形与诊断）

用于训练阶段 shaping 或日志分析，不改变 makespan-first 的主评判优先级:
- 总旅行时间/距离
- 总等待时间
- 规模使用成本（mode cost）
- 行为振荡统计（switch / quorum break / pause）

### 6.3 建议长期追踪的KPI

- `Perf/Makespan`（主）
- `Perf/Success rate`
- `Perf/Time cost`
- `Perf/Traveling distance`
- `Perf/Utilization Exec/Travel/Wait`
- `Perf/Switch Rate`
- `Perf/Quorum Break Rate`
- `Perf/Pause Events`

---

## 7. 规模成本（Mode Cost / Resource Occupancy Cost）

v1.0 保留规模成本，用于表达“更多机器人占用系统资源”的代价与偏好。

### 7.1 瞬时成本

对任务 $j$:
\[
c_j(t)=\lambda h(n_j(t))
\]

其中 `h(n)` 单调递增，推荐:
- 线性: $h(n)=n$
- 凸型: $h(n)=n^2$（更强抑制堆人）

### 7.2 时间积分形式

\[
C_{mode}=\sum_{j=1}^M \int_0^T h(n_j(t)) dt
\]

说明:
- 该项可进入 reward shaping
- 不建议替代 makespan 主目标
- 建议在报告中明确其属于“次目标/偏好项”

---

## 8. RL形式化（用于实现接口）

### 8.1 上层动作（v1.0 保持简化）

每个机器人在决策事件点选择:
\[
a_i \in \mathcal{T}\cup\{\text{IDLE}\}
\]

即选择下一目标任务或 idle。v1.0 不强制引入 `(task, intent_strength)` 复合动作。

### 8.2 下层在线调度接口（保持v0.4方向）

保持 dispatcher contract:
- 上层输出任务意图（leader action）
- 下层 dispatcher 负责成员子集选择（follower selection / allocation refinement）
- 初始可使用 `baseline_random`
- 后续可替换为 `greedy_eta` 等更稳定策略

### 8.3 观测（建议兼容现有代码）

任务侧建议至少包含:
- `remaining_workload / workload`
- `current_team_size`
- `estimated_rate`
- `quorum`
- 相对位置特征

机器人侧建议包含:
- 当前状态（traveling/waiting/working/returning）
- 当前目标任务（若有）
- 剩余旅行时间
- 累计 exec/travel/wait 时间等统计

---

## 9. 硬约束 vs 软惩罚（重要定位）

### 9.1 约束定位（不是算法本体）

以下机制应归类为 Problem Definition / Feasibility Constraints，而非“学习算法本体”:
- `quorum protect`
- `commit lock`（若启用）
- 执行期离开限制（strict / relaxed）

它们在RL中的实现方式通常是 `action mask`。

### 9.2 软惩罚定位（训练偏好）

以下项属于训练时行为正则（soft preferences）:
- `switch penalty`
- `pause penalty`
- `quorum_break penalty`

软惩罚不能替代硬约束，因为:
- 可能被短期收益“买通”
- 无法保证动作可行性
- 易在训练前期产生病态行为

### 9.3 推荐实现优先级

1. 先保证硬约束 `action mask` 正确
2. 再调软惩罚系数
3. 再调PPO超参与critic

---

## 10. 事件驱动仿真设计（Event-Driven Simulation）

v1.0 继续采用事件驱动环境推进。

### 10.1 关键事件类型

- 机器人到达任务点
- 机器人到达depot
- 任务完成事件（由当前 `remaining_workload / rate` 推导）
- 等待超时重规划事件（若启用）
- 全任务完成后的返航事件

### 10.2 统一时间推进 `advance_time(t_next)`

在区间 `[t, t_next]` 内:
1. 对所有 ACTIVE 任务积分 `remaining_workload` 与 `mode_cost`
2. 对所有 agent 按状态累计 `travel/wait/exec` 时间
3. 更新时间到 `t_next`
4. 处理事件触发的状态转移与成员变化
5. 若速率变化，重新注册相关任务完成事件时间

### 10.3 v1.0 关键正确性要求

- `remaining_workload` 单调非增（允许浮点误差）
- 任务完成后不得再次进入 ACTIVE
- 事件处理顺序在同一时间戳应固定（保证可复现）
- 日志中可追踪成员加入/离开与被mask拦截原因

---

## 11. Action Mask 规则（可直接编码）

定义 `valid_actions(i, s_t)` 为 agent `i` 在状态 `s_t` 的可行动作集合。

### 11.1 基础可行域

- `unfinished_tasks = {j | R_j > eps}`
- 基础动作集合:
  - `unfinished_tasks ∪ {IDLE}`

### 11.2 执行期离开限制（Growth-Only语义）

若 agent `i` 当前正在任务 `j` 执行:

#### 模式1: `growth_only_strict`
- 仅允许继续当前任务（任务完成前不能切走）
- 等价于 mask 掉所有离开动作

#### 模式2: `growth_only_relaxed`
- 若 `i in core_members[j]`:
  - 仅允许继续当前任务
- 若 `i in support_members[j]`:
  - 离开需同时满足:
    - `commit_lock` 到期（若启用）
    - 离开后不破坏 `quorum protect`
    - （可选）达到重规划点/等待超时点

### 11.3 Quorum Protect（通用）

若离开当前任务会使人数低于 quorum，则所有离开动作被mask:
\[
n_j-1<q_j \Rightarrow \text{mask leave}
\]

### 11.4 建议记录的mask原因日志

为便于调试和训练诊断，建议记录:
- `mask_reason_commit_lock`
- `mask_reason_quorum_protect`
- `mask_reason_execution_strict_lock`
- `mask_reason_invalid_task_finished`

---

## 12. 奖励设计（训练用，保持 makespan-first）

### 12.1 终局奖励（主）

\[
R_{terminal} = -T
\]

其中 `T` 为标准 makespan（含返航完成时刻）。

### 12.2 过程shaping（推荐形式）

按事件区间增量定义:
\[
r_t =
-w_{time}\Delta t
-w_{travel}\Delta travel
-w_{wait}\Delta wait
-w_{mode}\Delta mode
-w_{switch}I_{switch}
-w_{qb}I_{quorum\_break}
-w_{pause}I_{pause}
+w_{\phi}(\phi_{t+1}-\phi_t)
\]

说明:
- 可保留 `switch/quorum_break/pause` 惩罚作为行为正则
- 即使 v1.0 通过硬约束减少了它们发生频率，保留惩罚仍有助于稳定训练
- best checkpoint 选择建议继续以验证集 `makespan` 为主，而非训练reward

---

## 13. 与其他setting的关系（对照定位）

### 13.1 相对原论文式（固定duration）

原论文式 setting 更接近:
- 达到需求人数后执行固定时长
- 任务执行期语义相对简化

v1.0 则是:
- 任务以工作量定义（非固定duration）
- 允许任务开始后新增成员提升速率
- 边际效应显式进入动力学

因此:
- 绝对 makespan 不应直接横向比较
- 应在同一setting下比较算法表现，或使用相对基线改进率

### 13.2 相对 v0.2/v0.3 广义可抢占

v1.0 的变化:
- 保留连续工作量与动态协同规模
- 收紧离开语义，减少pause/switch振荡
- 更适合作为PPO和dispatcher迭代的稳定环境基线

---

## 14. CodeX/Agent实现任务拆解（建议顺序）

### Phase A: 环境语义落地（优先，不改网络）
1. 增加/确认任务字段:
   - `workload`, `remaining_workload`, `alpha`, `quorum`
   - `core_members`, `support_members`, `active_members`
2. 新增配置开关:
   - `SIMPLIFIED_SETTING = growth_only_strict | growth_only_relaxed`
3. 在 `get_action_mask(agent_id)` 中实现:
   - 执行期离开限制
   - quorum protect
   - mask原因日志
4. 保持 `advance_time(t_next)` 连续积分逻辑（不要退回固定finish语义）
5. 增加事件日志:
   - 成员加入/离开
   - 被拦截离开次数（按原因统计）

### Phase B: 回归与机制验收（短训）
1. 小规模场景 smoke test:
   - 无NaN、无死循环、无负工作量
2. 机制验收:
   - 开工后允许增援，速率即时提升
   - strict模式下执行成员无法切走
   - quorum protect 生效
3. 指标验收（相对旧setting）:
   - `pause_events` 下降
   - `switch_rate` 下降或不恶化
   - `success_rate` 保持高位

### Phase C: 算法迭代（在v1.0稳定后）
1. PPO超参（KL/clip/value_coef/grad_norm）
2. dispatcher 从 `baseline_random` 升级到 `greedy_eta`
3. 再尝试 `growth_only_relaxed` 与 commit lock 组合

---

## 15. 验收清单（v1.0）

### 15.1 机制正确性
- [ ] 任务以 `remaining_workload` 推进，不使用固定duration完成语义
- [ ] 开工后允许新增成员加入并提升速率
- [ ] 执行期离开动作按strict/relaxed规则被mask
- [ ] `quorum protect` 正确阻止破坏配额的离开
- [ ] `remaining_workload` 单调非增且不出现明显负值

### 15.2 指标行为
- [ ] `pause_events` 明显下降
- [ ] `quorum_break_rate` 明显下降
- [ ] `switch_rate` 下降（或至少在更严格可行域内稳定）
- [ ] `success_rate` 保持高位
- [ ] makespan方差下降（同配置多seed）

### 15.3 工程兼容
- [ ] 与 v0.4 dispatcher接口兼容
- [ ] REINFORCE/PPO 分支均可运行
- [ ] 日志与可视化面板新增字段完整

---

## 16. 后续版本建议（v1.1 / v1.2）

### v1.1
- 实现 `growth_only_relaxed`
- 增援成员可在满足条件后离开
- 增加 core/support 可视化与trace

### v1.2
- 引入任务级 preemption 类型参数:
  - non-preemptive
  - growth-only
  - semi-preemptive
- 引入任务级 `join-after-start` 开关 `eta_j`

---

## 17. 附录: 下一次Session快速背景摘要（可复制）

- 项目主目标: 同构多机器人MRTA，makespan-first
- 当前框架: 上层MARL（Attention actor，REINFORCE/PPO）+ 下层dispatcher接口（v0.4已预留）
- 建模演进:
  - 从固定duration迁移到连续工作量动力学
  - 引入边际递减协同效率 `g(n)`
  - v0.3加入反振荡约束与行为指标
  - v0.4引入PPO分支与dispatcher contract
- v1.0关键决策:
  - 保留工作量动力学与动态协同规模
  - 收敛为 growth-only / quorum-safe 执行语义
  - 允许开工后增援加速，不允许随意撤离破坏quorum
- commit lock / quorum protect 定位:
  - 属于 Problem Definition / Feasibility Constraints
  - 在RL中通过 action mask 实现，不是算法本体
- 实施优先级:
  1) 先改setting到v1.0
  2) 回归短训验证机制与指标
  3) 再做PPO与dispatcher迭代
