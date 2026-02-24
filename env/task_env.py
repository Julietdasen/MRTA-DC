import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scheduler.online_dispatcher import DispatchContext, OnlineDispatcher


class TaskEnv:
    def __init__(self, agents_range=(10, 10), tasks_range=(10, 10), traits_dim=1, max_coalition_size=3, max_duration=5,
                 seed=None, plot_figure=False, task_alpha=1.0, coalition_beta=0.8, mode_cost_type='linear',
                 reward_w_makespan=1.0, reward_w_travel=0.05, reward_w_wait=0.1, reward_w_mode=0.05,
                 enable_commit_lock=True, min_commit_time=2.0, enable_quorum_protect=True,
                 switch_penalty=0.15, quorum_break_penalty=0.5, pause_event_penalty=0.2,
                 use_dense_event_reward=True, use_potential_shaping=True, potential_shaping_coef=1.0,
                 online_dispatcher=None):
        """
        :param traits_dim: number of capabilities in this problem, e.g. 3 traits
        :param seed: seed to generate pseudo random problem instance
        """
        self.rng = None
        self.agents_range = agents_range
        self.tasks_range = tasks_range
        self.max_coalition_size = max_coalition_size
        self.max_duration = max_duration
        self.plot_figure = plot_figure
        # v0.1 model parameters (workload dynamics + reward shaping)
        self.task_alpha = task_alpha
        self.coalition_beta = coalition_beta
        self.mode_cost_type = mode_cost_type
        self.reward_w_makespan = reward_w_makespan
        self.reward_w_travel = reward_w_travel
        self.reward_w_wait = reward_w_wait
        self.reward_w_mode = reward_w_mode
        # v0.3 anti-oscillation and dense reward controls
        self.enable_commit_lock = bool(enable_commit_lock)
        self.min_commit_time = float(min_commit_time)
        self.enable_quorum_protect = bool(enable_quorum_protect)
        self.switch_penalty = float(switch_penalty)
        self.quorum_break_penalty = float(quorum_break_penalty)
        self.pause_event_penalty = float(pause_event_penalty)
        self.use_dense_event_reward = bool(use_dense_event_reward)
        self.use_potential_shaping = bool(use_potential_shaping)
        self.potential_shaping_coef = float(potential_shaping_coef)
        self.online_dispatcher = online_dispatcher
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.traits_dim = traits_dim
        self.task_dic, self.agent_dic, self.depot = self.generate_env()
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        self.current_time = 0
        self.last_update_time = 0
        self.dt = 0.1
        self.eps = 1e-8
        self.max_waiting_time = 10
        self.finished = False
        self.force_wait = True
        self.force_waiting = True
        self.reactive_planning = False
        self.visible_length = 0
        # v0.3 event counters and reward snapshots
        self.total_switch_events = 0
        self.total_quorum_break_events = 0
        self.total_pause_events = 0
        self.total_actions = 0
        self._reward_snapshot = None
        self._event_snapshot = None
        self.reset_dense_reward_snapshot()

    def set_online_dispatcher(self, dispatcher):
        if dispatcher is None:
            self.online_dispatcher = None
            return
        if not isinstance(dispatcher, OnlineDispatcher) and not callable(getattr(dispatcher, 'select_members', None)):
            raise TypeError('dispatcher must implement OnlineDispatcher')
        self.online_dispatcher = dispatcher

    def random_int(self, low, high, size=None):
        if self.rng is not None:
            integer = self.rng.integers(low, high, size)
        else:
            integer = np.random.randint(low, high, size)
        return integer

    def random_value(self, row, col):
        if self.rng is not None:
            value = self.rng.random((row, col))
        else:
            value = np.random.rand(row, col)
        return value

    def random_choice(self, a, size=None, replace=True):
        if self.rng is not None:
            choice = self.rng.choice(a, size, replace)
        else:
            choice = np.random.choice(a, size, replace)
        return choice

    def generate_env(self):
        if type(self.tasks_range) is tuple:
            tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1)
        else:
            tasks_num = self.tasks_range
        if type(self.agents_range) is tuple:
            agents_num = self.random_int(self.agents_range[0], self.agents_range[1] + 1)
        else:
            agents_num = self.agents_range
        agents_ini = np.ones((agents_num, self.traits_dim))
        depot = self.random_value(1, 2)
        cost_ini = self.random_value(agents_num, 1)
        tasks_loc = self.random_value(tasks_num, 2)
        tasks_time = np.ones((tasks_num, 1)) * self.max_duration
        tasks_ini = self.random_int(1, self.max_coalition_size + 1, tasks_num).reshape(-1, self.traits_dim)

        task_dic = dict()
        agent_dic = dict()
        for i in range(tasks_num):
            task_dic[i] = {'ID': i,
                           'requirements': tasks_ini[i, :],  # requirements of the task
                           'members': [],  # members of the task
                           'cost': [],  # cost of each agent
                           'location': tasks_loc[i, :],  # location of the task
                           'feasible_assignment': False,  # whether the task assignment is feasible
                           'finished': False,
                           'time_start': 0,
                           'time_finish': 0,
                           'status': tasks_ini[i, :],
                           'time': float(tasks_time[i, :]),
                           # Workload state: initialized from legacy task time.
                           'workload': float(tasks_time[i, :]),
                           'remaining_workload': float(tasks_time[i, :]),
                           'state': 'UNSTARTED',
                           'started': False,
                           'work_rate': 0.0,
                           'alpha': float(self.task_alpha),
                           'start_team_size': 0,
                           'mode_cost': 0.0,
                           'active_members': [],
                           'arrived_members': [],
                           'pause_events': 0,
                           'quorum_break_events': 0,
                           'sum_waiting_time': 0,
                           'efficiency': 0,
                           'abandoned_agent': []}
        for i in range(agents_num):
            agent_dic[i] = {'ID': i,
                            'abilities': agents_ini[i, :],
                            'location': depot[0, :],
                            'next_location': depot[0, :],
                            'route': [],
                            'arrival_time': [],
                            'cost': cost_ini[i],
                            'travel_time': 0,
                            'velocity': 0.2,
                            'next_decision': 0,
                            'depot': depot[0, :],
                            'travel_dist': 0,
                            'sum_waiting_time': 0,
                            'current_action_index': 0,
                            'working_condition': 0,
                            'trajectory': [],
                            'angle': 0,
                            'returned': False,
                            'assigned': False,
                            'pre_set_route': None,
                            'state': 'IDLE_AT_DEPOT',
                            'target_task_id': -1,
                            'time_exec': 0.0,
                            'time_wait': 0.0,
                            'time_travel': 0.0,
                            'commit_task_id': -1,
                            'commit_until': 0.0,
                            'last_task_id': -1,
                            'switch_count': 0}
        depot = {'location': depot[0, :],
                 'members': [],
                 'ID': -1}
        return task_dic, agent_dic, depot

    def reset(self, test_env=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None
        if test_env is not None:
            self.task_dic, self.agent_dic, self.depot = test_env
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        self.current_time = 0
        self.last_update_time = 0
        self.finished = False
        self.total_switch_events = 0
        self.total_quorum_break_events = 0
        self.total_pause_events = 0
        self.total_actions = 0
        for task in self.task_dic.values():
            if 'workload' not in task:
                task['workload'] = float(task['time'])
            if 'remaining_workload' not in task:
                task['remaining_workload'] = float(task['workload'])
            if 'state' not in task:
                task['state'] = 'UNSTARTED'
            if 'started' not in task:
                task['started'] = False
            if 'work_rate' not in task:
                task['work_rate'] = 0.0
            if 'alpha' not in task:
                task['alpha'] = float(self.task_alpha)
            if 'start_team_size' not in task:
                task['start_team_size'] = 0
            if 'mode_cost' not in task:
                task['mode_cost'] = 0.0
            if 'active_members' not in task:
                task['active_members'] = []
            if 'arrived_members' not in task:
                task['arrived_members'] = []
            if 'pause_events' not in task:
                task['pause_events'] = 0
            if 'quorum_break_events' not in task:
                task['quorum_break_events'] = 0
            if 'abandoned_agent' not in task:
                task['abandoned_agent'] = []
        for agent in self.agent_dic.values():
            if 'state' not in agent:
                agent['state'] = 'IDLE_AT_DEPOT'
            if 'target_task_id' not in agent:
                agent['target_task_id'] = -1
            if 'time_exec' not in agent:
                agent['time_exec'] = 0.0
            if 'time_wait' not in agent:
                agent['time_wait'] = 0.0
            if 'time_travel' not in agent:
                agent['time_travel'] = 0.0
            if 'commit_task_id' not in agent:
                agent['commit_task_id'] = -1
            if 'commit_until' not in agent:
                agent['commit_until'] = 0.0
            if 'last_task_id' not in agent:
                agent['last_task_id'] = -1
            if 'switch_count' not in agent:
                agent['switch_count'] = 0
        self.reset_dense_reward_snapshot()

    def clear_decisions(self):
        for task in self.task_dic.values():
            task.update(members=[], cost=[], finished=False, status=task['requirements'],feasible_assignment=False,
                        time_start=0, time_finish=0, remaining_workload=float(task['workload']), work_rate=0.0,
                        alpha=float(task.get('alpha', self.task_alpha)), start_team_size=0, mode_cost=0.0,
                        state='UNSTARTED', started=False, active_members=[], arrived_members=[],
                        pause_events=0, quorum_break_events=0,
                        sum_waiting_time=0, efficiency=0, abandoned_agent=[])
        for agent in self.agent_dic.values():
            agent.update(route=[], location=self.depot['location'], next_location=self.depot['location'],
                         next_decision=0, travel_time=0, travel_dist=0, arrival_time=[], assigned=False,
                         sum_waiting_time=0, working_condition=0, current_action_index=0,
                         trajectory=[], angle=0, returned=False, pre_set_route=None, depot=self.depot['location'],
                         state='IDLE_AT_DEPOT', target_task_id=-1, time_exec=0.0, time_wait=0.0, time_travel=0.0,
                         commit_task_id=-1, commit_until=0.0, last_task_id=-1, switch_count=0)
        self.depot.update(members=[], ID=-1)
        self.current_time = 0
        self.last_update_time = 0
        self.finished = False
        self.total_switch_events = 0
        self.total_quorum_break_events = 0
        self.total_pause_events = 0
        self.total_actions = 0
        self.reset_dense_reward_snapshot()

    @staticmethod
    def find_by_key(data, target):
        for key, value in data.items():
            if isinstance(value, dict):
                yield from TaskEnv.find_by_key(value, target)
            elif key == target:
                yield value

    @staticmethod
    def get_matrix(dictionary, key):
        """
        :param key: the key to index
        :param dictionary: the dictionary for key to index
        """
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    @staticmethod
    def calculate_eulidean_distance(agent, task):
        return np.linalg.norm(agent['location'] - task['location'])

    def coalition_efficiency(self, n):
        n = int(max(n, 0))
        if n == 0:
            return 0.0
        # Saturating coalition gain: g(n)=n/(1+beta*(n-1))
        return float(n / (1.0 + self.coalition_beta * (n - 1)))

    def mode_cost(self, n):
        n = int(max(n, 0))
        if self.mode_cost_type == 'quadratic':
            return float(n ** 2)
        # Current default is linear mode cost h(n)=n.
        return float(n)

    @staticmethod
    def _task_requirement(task):
        return int(max(1, np.ceil(np.sum(task['requirements']))))

    def _potential_value(self):
        ratios = []
        for task in self.task_dic.values():
            workload = float(max(task.get('workload', task.get('time', 1.0)), self.eps))
            remaining = float(max(task.get('remaining_workload', workload), 0.0))
            ratios.append(remaining / workload)
        if not ratios:
            return 0.0
        return -float(np.mean(ratios))

    def _collect_reward_components(self):
        total_travel_time = float(np.sum([agent.get('time_travel', 0.0) for agent in self.agent_dic.values()]))
        total_waiting_time = float(np.sum([agent.get('time_wait', 0.0) for agent in self.agent_dic.values()]))
        total_mode_cost = float(np.sum([task.get('mode_cost', 0.0) for task in self.task_dic.values()]))
        return {
            'time': float(self.current_time),
            'travel': total_travel_time,
            'wait': total_waiting_time,
            'mode': total_mode_cost,
            'potential': self._potential_value(),
            'switch_events': int(self.total_switch_events),
            'quorum_break_events': int(self.total_quorum_break_events),
            'pause_events': int(self.total_pause_events),
        }

    def reset_dense_reward_snapshot(self):
        snapshot = self._collect_reward_components()
        self._reward_snapshot = snapshot.copy()
        self._event_snapshot = snapshot.copy()

    def get_dense_reward_delta(self, reset=True):
        current = self._collect_reward_components()
        if self._reward_snapshot is None:
            self.reset_dense_reward_snapshot()
        prev = self._reward_snapshot

        dt = float(current['time'] - prev['time'])
        d_travel = float(current['travel'] - prev['travel'])
        d_wait = float(current['wait'] - prev['wait'])
        d_mode = float(current['mode'] - prev['mode'])
        d_switch = int(current['switch_events'] - prev['switch_events'])
        d_quorum_break = int(current['quorum_break_events'] - prev['quorum_break_events'])
        d_pause = int(current['pause_events'] - prev['pause_events'])

        if self.use_dense_event_reward:
            reward = -(
                self.reward_w_makespan * dt
                + self.reward_w_travel * d_travel
                + self.reward_w_wait * d_wait
                + self.reward_w_mode * d_mode
            )
        else:
            reward = 0.0

        reward -= self.switch_penalty * d_switch
        reward -= self.quorum_break_penalty * d_quorum_break
        reward -= self.pause_event_penalty * d_pause

        if self.use_potential_shaping:
            reward += self.potential_shaping_coef * float(current['potential'] - prev['potential'])

        out = {
            'reward': float(reward),
            'dt': dt,
            'delta_travel': d_travel,
            'delta_wait': d_wait,
            'delta_mode': d_mode,
            'switch_events': d_switch,
            'quorum_break_events': d_quorum_break,
            'pause_events': d_pause,
            'potential': float(current['potential']),
        }
        if reset:
            self._reward_snapshot = current.copy()
        return out

    def _can_leave_task(self, agent_id, from_task_id):
        if not self.enable_quorum_protect:
            return True
        if from_task_id not in self.task_dic:
            return True
        task = self.task_dic[from_task_id]
        if task['finished']:
            return True
        if agent_id not in task.get('active_members', []):
            return True
        requirement = self._task_requirement(task)
        n_active = len(task.get('active_members', []))
        if n_active > requirement:
            return True

        # Allow leaving when this task has already waited beyond the replanning timeout.
        waiting_deadline = float(self.get_arrival_time(agent_id, from_task_id) + self.max_waiting_time)
        return bool(self.current_time >= waiting_deadline - self.eps)

    def get_action_mask(self, agent_id):
        mask = self.get_unfinished_task_mask()
        if np.sum(mask) == self.tasks_num:
            mask = np.insert(mask, 0, False)
        else:
            mask = np.insert(mask, 0, True)
        mask = mask.astype(bool)

        agent = self.agent_dic[agent_id]
        current_target = int(agent.get('target_task_id', -1))
        locked_task = int(agent.get('commit_task_id', -1))
        lock_active = (
            self.enable_commit_lock
            and locked_task in self.task_dic
            and not self.task_dic[locked_task]['finished']
            and float(self.current_time) < float(agent.get('commit_until', 0.0)) - self.eps
        )
        if lock_active:
            constrained = np.ones_like(mask, dtype=bool)
            constrained[locked_task + 1] = mask[locked_task + 1]
            return constrained

        if current_target in self.task_dic and not self._can_leave_task(agent_id, current_target):
            constrained = np.ones_like(mask, dtype=bool)
            constrained[current_target + 1] = mask[current_target + 1]
            return constrained

        return mask

    def get_global_features(self):
        unfinished = np.array(self.get_unfinished_tasks(), dtype=np.float32)
        unfinished_ratio = float(np.mean(unfinished)) if unfinished.size else 0.0
        done_ratio = 1.0 - unfinished_ratio

        rem_ratio = []
        active_count = 0
        paused_count = 0
        team_sizes = []
        for task in self.task_dic.values():
            workload = float(max(task.get('workload', task.get('time', 1.0)), self.eps))
            rem_ratio.append(float(task.get('remaining_workload', workload)) / workload)
            if task.get('state') == 'ACTIVE':
                active_count += 1
            if task.get('state') == 'PAUSED':
                paused_count += 1
            team_sizes.append(float(len(task.get('active_members', []))))

        rem_ratio = np.array(rem_ratio, dtype=np.float32) if rem_ratio else np.zeros((1,), dtype=np.float32)
        team_sizes = np.array(team_sizes, dtype=np.float32) if team_sizes else np.zeros((1,), dtype=np.float32)
        active_ratio = float(active_count / max(self.tasks_num, 1))
        paused_ratio = float(paused_count / max(self.tasks_num, 1))
        team_mean = float(np.mean(team_sizes) / max(self.max_coalition_size, 1))

        util_exec, util_wait, util_travel = self.get_utilization_metrics()
        mean_wait = float(np.mean([a.get('time_wait', 0.0) for a in self.agent_dic.values()])) if self.agent_dic else 0.0
        mean_travel = float(np.mean([a.get('time_travel', 0.0) for a in self.agent_dic.values()])) if self.agent_dic else 0.0
        mean_exec = float(np.mean([a.get('time_exec', 0.0) for a in self.agent_dic.values()])) if self.agent_dic else 0.0
        makespan = float(max(self.current_time, self.eps))

        action_denom = float(max(self.total_actions, 1))
        switch_rate = float(self.total_switch_events / action_denom)
        quorum_break_rate = float(self.total_quorum_break_events / action_denom)
        pause_rate = float(self.total_pause_events / max(self.tasks_num, 1))

        features = np.array([
            float(self.current_time / max(self.max_duration * max(self.tasks_num, 1), 1.0)),
            unfinished_ratio,
            done_ratio,
            float(np.mean(rem_ratio)),
            float(np.std(rem_ratio)),
            active_ratio,
            paused_ratio,
            team_mean,
            float(util_exec),
            float(util_wait),
            float(util_travel),
            float(mean_wait / makespan),
            float(mean_travel / makespan),
            switch_rate,
            quorum_break_rate,
            pause_rate,
        ], dtype=np.float32)
        return features

    def get_behavior_metrics(self):
        action_denom = float(max(self.total_actions, 1))
        return {
            'switch_rate': float(self.total_switch_events / action_denom),
            'quorum_break_rate': float(self.total_quorum_break_events / action_denom),
            'pause_events': float(self.total_pause_events),
        }

    def get_current_agent_status(self, agent):
        status = []
        for a in self.agent_dic.values():
            if len(a['route']) > 0 and a['route'][-1] in self.task_dic.keys():
                travel_time = np.clip(self.get_arrival_time(a['ID'], a['route'][-1]) - self.current_time, a_min=0, a_max=None)
                current_waiting_time = np.clip(self.current_time - self.get_arrival_time(a['ID'], a['route'][-1]), a_min=0, a_max=None) if self.current_time <= self.task_dic[a['route'][-1]]['time_start'] else 0
                remaining_working_time = np.clip(self.task_dic[a['route'][-1]]['time_finish'] - self.current_time, a_min=0, a_max=None) if self.current_time >= self.task_dic[a['route'][-1]]['time_start'] else 0
            else:
                travel_time = 0
                current_waiting_time = 0
                remaining_working_time = 0
            temp_status = np.hstack([travel_time, remaining_working_time, current_waiting_time,
                                     agent['location'] - a['location'], a['assigned']])
            status.append(temp_status)
        current_agents = np.vstack(status)
        return current_agents

    def get_current_task_status(self, agent):
        status = []
        for t in self.task_dic.values():
            temp_status = np.hstack([t['status'], t['requirements'], t['remaining_workload'],
                                     t['location'] - agent['location']])
            status.append(temp_status)
        status = [np.hstack([0, 0, 0, self.depot['location'] - agent['location']])] + status
        current_tasks = np.vstack(status)
        return current_tasks

    def get_unfinished_task_mask(self):
        mask = np.logical_not(self.get_unfinished_tasks())
        return mask

    def get_unfinished_tasks(self):
        unfinished_tasks = []
        for task in self.task_dic.values():
            unfinished_tasks.append(float(task['remaining_workload']) > self.eps and not task['finished'])
        return unfinished_tasks

    def get_arrival_time(self, agent_id, task_id):
        arrival_time = self.agent_dic[agent_id]['arrival_time']
        arrival_for_task = np.where(np.array(self.agent_dic[agent_id]['route']) == task_id)[0][-1]
        return float(arrival_time[arrival_for_task])

    def _arrived_to_task(self, agent_id, task_id):
        agent = self.agent_dic[agent_id]
        if not agent['route']:
            return False
        if agent['route'][-1] != task_id:
            return False
        return float(agent['arrival_time'][-1]) <= float(self.current_time) + self.eps

    def _refresh_task_members(self):
        for task in self.task_dic.values():
            task_id = task['ID']
            arrived_members = []
            for agent in self.agent_dic.values():
                if self._arrived_to_task(agent['ID'], task_id):
                    arrived_members.append(agent['ID'])
            task['arrived_members'] = arrived_members
            task['members'] = arrived_members.copy()

    def _predict_task_finish(self, task):
        if task['work_rate'] <= self.eps:
            return np.inf
        return float(self.current_time + task['remaining_workload'] / task['work_rate'])

    def advance_time(self, t_next):
        dt = float(t_next - self.last_update_time)
        if dt <= self.eps:
            self.last_update_time = float(t_next)
            return

        for task in self.task_dic.values():
            if task['finished']:
                continue
            active_members = task.get('active_members', [])
            n_active = len(active_members)
            if n_active <= 0:
                continue
            rate = float(task['alpha']) * self.coalition_efficiency(n_active)
            if rate <= self.eps:
                continue
            remaining = float(task['remaining_workload'])
            time_to_finish = remaining / rate
            effective_dt = min(dt, time_to_finish)
            task['remaining_workload'] = float(np.clip(remaining - rate * effective_dt, a_min=0.0, a_max=None))
            task['mode_cost'] += float(self.mode_cost(n_active) * effective_dt)
            task['work_rate'] = rate
            if task['remaining_workload'] <= self.eps:
                task['remaining_workload'] = 0.0
                task['finished'] = True
                task['feasible_assignment'] = True
                task['state'] = 'FINISHED'
                task['time_finish'] = float(self.last_update_time + effective_dt)

        for agent in self.agent_dic.values():
            state = agent.get('state', 'IDLE_AT_DEPOT')
            if state == 'TRAVELING':
                agent['time_travel'] += dt
            elif state == 'WORKING':
                agent['time_exec'] += dt
            elif state == 'WAITING':
                agent['time_wait'] += dt

        self.last_update_time = float(t_next)

    def agent_update(self):
        for agent in self.agent_dic.values():
            if len(agent['arrival_time']) == 0:
                agent['state'] = 'IDLE_AT_DEPOT'
                agent['target_task_id'] = -1
                if not np.all(self.get_matrix(self.task_dic, 'finished')):
                    agent['next_decision'] = float(self.current_time)
                else:
                    agent['next_decision'] = np.nan
                agent['working_condition'] = 0.0
                continue

            latest_arrival = float(agent['arrival_time'][-1])
            target_task_id = int(agent['route'][-1])
            agent['target_task_id'] = target_task_id

            if latest_arrival > float(self.current_time) + self.eps:
                agent['state'] = 'TRAVELING'
                agent['assigned'] = False
                agent['returned'] = False
                agent['next_decision'] = latest_arrival
                agent['working_condition'] = latest_arrival - float(self.current_time)
                continue

            if target_task_id == -1:
                agent['state'] = 'IDLE_AT_DEPOT'
                agent['assigned'] = False
                if np.all(self.get_matrix(self.task_dic, 'finished')):
                    agent['returned'] = True
                    agent['next_decision'] = np.nan
                elif self.reactive_planning:
                    if np.all(self.get_matrix(self.task_dic, 'finished')[:self.visible_length]):
                        agent['next_decision'] = np.nan
                    else:
                        if agent['pre_set_route'] is not None and not agent['pre_set_route']:
                            agent['next_decision'] = np.nan
                        else:
                            next_action = agent['pre_set_route'][0]
                            next_decision_time = (next_action - 1) // 20 * 10
                            agent['next_decision'] = np.max(
                                [self.get_arrival_time(agent['ID'], -1), next_decision_time, self.current_time]
                            )
                            if agent['ID'] in self.depot['members']:
                                self.depot['members'].remove(agent['ID'])
                else:
                    agent['returned'] = False
                    agent['next_decision'] = float(self.current_time)
                agent['working_condition'] = 0.0
                continue

            current_task = self.task_dic[target_task_id]
            if current_task['finished']:
                agent['state'] = 'WAITING'
                agent['assigned'] = False
                agent['returned'] = False
                agent['next_decision'] = float(self.current_time)
            elif agent['ID'] in current_task.get('active_members', []):
                agent['state'] = 'WORKING'
                agent['assigned'] = True
                agent['returned'] = False
                predicted_finish = self._predict_task_finish(current_task)
                periodic_replan = float(self.current_time + self.max_waiting_time)
                agent['next_decision'] = float(min(predicted_finish, periodic_replan))
            elif agent['ID'] in current_task.get('arrived_members', []):
                agent['state'] = 'WAITING'
                agent['assigned'] = False
                agent['returned'] = False
                waiting_deadline = float(self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time)
                if self.current_time >= waiting_deadline - self.eps:
                    agent['next_decision'] = float(self.current_time)
                else:
                    agent['next_decision'] = waiting_deadline
            else:
                agent['state'] = 'TRAVELING'
                agent['assigned'] = False
                agent['returned'] = False
                agent['next_decision'] = latest_arrival

            if np.isnan(agent['next_decision']) or np.isinf(agent['next_decision']):
                agent['working_condition'] = 0.0
            else:
                agent['working_condition'] = float(np.clip(agent['next_decision'] - self.current_time, a_min=0.0, a_max=None))

    def task_update(self):
        finished_tasks = []
        self.advance_time(self.current_time)
        self._refresh_task_members()

        for task in self.task_dic.values():
            prev_state = task.get('state', 'UNSTARTED')
            was_finished = bool(task['finished'])
            if float(task['remaining_workload']) <= self.eps:
                task['remaining_workload'] = 0.0
                task['finished'] = True
                task['state'] = 'FINISHED'

            arrived_members = task.get('arrived_members', [])
            requirement = self._task_requirement(task)
            task['status'] = np.clip(task['requirements'] - len(arrived_members), a_min=0, a_max=None)

            if task['finished']:
                task['feasible_assignment'] = True
                task['active_members'] = []
                task['work_rate'] = 0.0
                task['state'] = 'FINISHED'
                if task['time_finish'] == 0:
                    task['time_finish'] = float(self.current_time)
                if not was_finished:
                    finished_tasks.append(task['ID'])
                continue

            if len(arrived_members) >= requirement:
                task['active_members'] = arrived_members.copy()
                task['feasible_assignment'] = True
                task['work_rate'] = float(task['alpha'] * self.coalition_efficiency(len(task['active_members'])))
                if not task.get('started', False):
                    task['started'] = True
                    task['time_start'] = float(self.current_time)
                    task['start_team_size'] = len(task['active_members'])
                task['time_finish'] = self._predict_task_finish(task)
                task['state'] = 'ACTIVE'
            else:
                task['active_members'] = []
                task['feasible_assignment'] = False
                task['work_rate'] = 0.0
                task['time_finish'] = 0
                if task.get('started', False) and float(task['remaining_workload']) > self.eps:
                    task['state'] = 'PAUSED'
                else:
                    task['state'] = 'UNSTARTED'
                for member in arrived_members:
                    waited = self.current_time - self.get_arrival_time(member, task['ID'])
                    if waited >= self.max_waiting_time and member not in task['abandoned_agent']:
                        task['abandoned_agent'].append(member)
                        self.agent_dic[member]['next_decision'] = float(self.current_time)

            if prev_state == 'ACTIVE' and task['state'] == 'PAUSED':
                task['pause_events'] = int(task.get('pause_events', 0)) + 1
                self.total_pause_events += 1

        all_finished = bool(np.all(self.get_matrix(self.task_dic, 'finished')))
        for member in self.depot['members']:
            if self.current_time >= self.get_arrival_time(member, -1) and all_finished:
                self.agent_dic[member]['returned'] = True
                self.agent_dic[member]['next_decision'] = np.nan

        return finished_tasks

    def next_decision(self):
        decision_time = np.array(self.get_matrix(self.agent_dic, 'next_decision'))
        if np.all(np.isnan(decision_time)):
            return [], max(map(lambda x: max(x) if x else 0, self.get_matrix(self.agent_dic, 'arrival_time')))
        next_decision = np.nanmin(decision_time)
        agents = np.where(decision_time == next_decision)[0]
        return agents, next_decision

    def get_unique_group(self, agents):
        location = np.array(self.get_matrix(self.agent_dic, 'location'))[agents]
        unique_location = np.unique(location, axis=0)
        # agent group in each unique location
        unique_group = []
        for loc in unique_location:
            unique_group.append(agents[np.where(np.all(location == loc, axis=1))[0].tolist()].tolist())
        return unique_group

    def agent_step(self, agent_id, task_id):
        """
        :param agent_id: the id of agent
        :param task_id: the id of task
        :return: end_episode, finished_tasks
        """
        #  choose any task
        task_id = task_id - 1
        if task_id not in self.task_dic and task_id != -1:
            task_id = -1
        agent = self.agent_dic[agent_id]
        previous_target = int(agent['route'][-1]) if agent['route'] else -1

        # Leave current target/task immediately when replanning.
        if agent['route']:
            if previous_target != -1 and previous_target in self.task_dic:
                previous_task = self.task_dic[previous_target]
                if previous_target != task_id and agent_id in previous_task.get('active_members', []):
                    requirement = self._task_requirement(previous_task)
                    n_active = len(previous_task.get('active_members', []))
                    if n_active <= requirement and float(previous_task.get('remaining_workload', 0.0)) > self.eps:
                        previous_task['quorum_break_events'] = int(previous_task.get('quorum_break_events', 0)) + 1
                        self.total_quorum_break_events += 1
                if agent_id in previous_task['members']:
                    previous_task['members'].remove(agent_id)
                if agent_id in previous_task.get('arrived_members', []):
                    previous_task['arrived_members'].remove(agent_id)
                if agent_id in previous_task.get('active_members', []):
                    previous_task['active_members'].remove(agent_id)
            elif previous_target == -1 and agent_id in self.depot['members']:
                self.depot['members'].remove(agent_id)

        if previous_target != -1 and previous_target != task_id:
            agent['switch_count'] = int(agent.get('switch_count', 0)) + 1
            self.total_switch_events += 1
        self.total_actions += 1

        if task_id != -1:
            task = self.task_dic[task_id]
            agent['returned'] = False
            agent['commit_task_id'] = task_id
            if self.enable_commit_lock:
                agent['commit_until'] = float(self.current_time + self.min_commit_time)
            else:
                agent['commit_until'] = float(self.current_time)
        else:
            task = self.depot
            agent['commit_task_id'] = -1
            agent['commit_until'] = float(self.current_time)
        agent['last_task_id'] = previous_target
        agent['route'].append(task_id)
        travel_time = self.calculate_eulidean_distance(agent, task) / agent['velocity']
        agent['travel_time'] = travel_time
        agent['travel_dist'] += self.calculate_eulidean_distance(agent, task)
        agent['arrival_time'] += [self.current_time + travel_time]
        # calculate the angle from current location to next location
        agent['location'] = task['location']
        agent['target_task_id'] = task_id
        agent['state'] = 'TRAVELING' if travel_time > self.eps else ('IDLE_AT_DEPOT' if task_id == -1 else 'WAITING')
        if agent_id not in task['members']:
            task['members'].append(agent_id)

        return - travel_time

    def step(self, group, leader_id, action, current_action_index=0):
        vacancy = self._resolve_vacancy(action, group)
        members = self._resolve_members(group, leader_id, action, vacancy)
        reward = 0
        for member in members:
            reward += self.agent_step(member, action)
            self.agent_dic[member]['current_action_index'] = current_action_index
        reward = reward / len(members)
        return group, reward

    def _resolve_vacancy(self, action, group):
        if action - 1 in self.task_dic.keys():
            return int(np.maximum(1, np.ceil(np.sum(self.task_dic[action - 1]['status']))))
        return int(len(group))

    def _resolve_members(self, group, leader_id, action, vacancy):
        group_copy = list(group)
        if leader_id in group_copy:
            group_copy.remove(leader_id)

        if self.online_dispatcher is not None:
            # Dispatcher mode: keep this branch as the single extension point for online lower-level scheduling.
            context = DispatchContext(
                leader_id=int(leader_id),
                action=int(action),
                vacancy=int(vacancy),
                group=list(group),
            )
            selected = self.online_dispatcher.select_members(context, self)
            selected = [int(x) for x in selected]
            members = self._sanitize_selected_members(selected, group, leader_id)
        else:
            # Baseline-compatible fallback: random followers based on current vacancy.
            members = [int(leader_id)]
            if vacancy > 1 and len(group_copy) > 0:
                followers = self.random_choice(group_copy, np.minimum(vacancy - 1, len(group_copy)), False).tolist()
                members += [int(x) for x in followers]

        for member in members:
            if member in group:
                group.remove(member)
        return members

    @staticmethod
    def _sanitize_selected_members(selected, group, leader_id):
        valid = set(group)
        members = []
        if leader_id in valid:
            members.append(int(leader_id))
        for aid in selected:
            if aid == leader_id:
                continue
            if aid not in valid:
                continue
            if aid in members:
                continue
            members.append(int(aid))
        if not members:
            if leader_id in valid:
                members = [int(leader_id)]
            elif len(group) > 0:
                members = [int(group[0])]
        return members

    def calculate_waiting_time(self):
        for agent in self.agent_dic.values():
            agent['sum_waiting_time'] = float(agent.get('time_wait', 0.0))
        for task in self.task_dic.values():
            waiting = 0.0
            for member in task.get('members', []):
                waiting += float(self.agent_dic[member].get('time_wait', 0.0))
            task['sum_waiting_time'] = waiting

    def check_finished(self):
        decision_agents, current_time = self.next_decision()
        if len(decision_agents) == 0:
            self.current_time = current_time
            self.task_update()
            self.agent_update()
            finished = np.all(self.get_matrix(self.agent_dic, 'returned')) and np.all(self.get_matrix(self.task_dic, 'finished'))
        else:
            finished = False
        return finished

    def generate_traj(self):
        for agent in self.agent_dic.values():
            # save the location of the agent as trajectory
            time_step = 0
            for i in range(len(agent['route'])):
                previous_task = self.task_dic[agent['route'][i - 1]] if i > 0 and agent['route'][i - 1] != -1 else self.depot
                current_task = self.task_dic[agent['route'][i]] if agent['route'][i] != -1 else self.depot
                angle = np.arctan2(current_task['location'][1] - previous_task['location'][1],
                                   current_task['location'][0] - previous_task['location'][0])
                distance = self.calculate_eulidean_distance(previous_task, current_task)
                total_time = distance / agent['velocity']
                arrival_time_current = agent['arrival_time'][i]
                arrival_time_prev = agent['arrival_time'][i - 1] if previous_task['ID'] != -1 else 0
                if current_task['ID'] != -1 and agent['ID'] in current_task['members'] \
                        and current_task['feasible_assignment']:
                    if current_task['time_start'] - arrival_time_current <= self.max_waiting_time:
                        next_decision = current_task['time_finish']
                    else:
                        next_decision = arrival_time_current + self.max_waiting_time
                else:
                    next_decision = arrival_time_current + self.max_waiting_time
                if previous_task['ID'] == -1:
                    prev_decision = 0
                else:
                    if agent['ID'] in previous_task['members'] \
                            and previous_task['time_start'] - arrival_time_prev <= self.max_waiting_time \
                            and previous_task['feasible_assignment']:
                        prev_decision = previous_task['time_finish']
                    else:
                        prev_decision = arrival_time_prev + self.max_waiting_time
                while time_step < next_decision:
                    time_step += self.dt
                    if time_step < arrival_time_current:
                        fraction_of_time = (time_step - prev_decision) / total_time
                        x = previous_task['location'][0] + fraction_of_time * (
                                    current_task['location'][0] - previous_task['location'][0])
                        y = previous_task['location'][1] + fraction_of_time * (
                                    current_task['location'][1] - previous_task['location'][1])
                        agent['trajectory'].append(np.hstack([x, y, angle]))
                    else:
                        agent['trajectory'].append(np.array([current_task['location'][0], current_task['location'][1], angle]))
            while time_step < self.current_time:
                time_step += self.dt
                agent['trajectory'].append(np.array([self.depot['location'][0], self.depot['location'][1], angle]))

    def get_episode_reward(self, max_time=100):
        self.calculate_waiting_time()
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        total_waiting_time = float(np.sum(self.get_matrix(self.agent_dic, 'sum_waiting_time')))
        total_travel_time = float(np.sum([agent.get('time_travel', 0.0) for agent in self.agent_dic.values()]))
        total_mode_cost = float(np.sum(self.get_matrix(self.task_dic, 'mode_cost')))
        # Terminal reward uses weighted sum per v0.1 spec.
        reward = - (
            self.reward_w_makespan * float(self.current_time)
            + self.reward_w_travel * total_travel_time
            + self.reward_w_wait * total_waiting_time
            + self.reward_w_mode * total_mode_cost
        )
        return reward, finished_tasks

    def get_utilization_metrics(self):
        makespan = float(self.current_time)
        if makespan <= self.eps:
            return 0.0, 0.0, 0.0
        denom = float(self.agents_num * makespan)
        total_exec = float(np.sum([agent.get('time_exec', 0.0) for agent in self.agent_dic.values()]))
        total_wait = float(np.sum([agent.get('time_wait', 0.0) for agent in self.agent_dic.values()]))
        total_travel = float(np.sum([agent.get('time_travel', 0.0) for agent in self.agent_dic.values()]))
        return total_exec / denom, total_wait / denom, total_travel / denom

    def stack_trajectory(self):
        for agent in self.agent_dic.values():
            agent['trajectory'] = np.vstack(agent['trajectory'])

    def plot_animation(self, path, n):
        self.generate_traj()
        plot_robot_icon = False
        if plot_robot_icon:
            drone = plt.imread('env/drone.png')
            drone_oi = OffsetImage(drone, zoom=0.05)

        # Set up the plot
        self.stack_trajectory()
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        finished_rate = np.sum(finished_tasks) / len(finished_tasks)
        gif_len = int(self.current_time/self.dt)
        fig, ax = plt.subplots(dpi=100)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0, right=0.85, top=0.87, bottom=0.02)
        lines = [ax.plot([], [], color='teal', zorder=0)[0] for _ in self.agent_dic.values()]
        ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                     f'\nCurrent time is {0:.2f}min')
        green_patch = patches.Patch(color='g', label='Finished task')
        blue_patch = patches.Patch(color='b', label='Unfinished task')
        red_patch = patches.Patch(color='r', label='Single agent')
        yellow_patch = patches.Patch(color='y', label='Two agents')
        cyan_patch = patches.Patch(color='c', label='Three agents')
        magenta_patch = patches.Patch(color='m', label='>= Four agents')
        if plot_robot_icon:
            ax.legend(handles=[green_patch, blue_patch], bbox_to_anchor=(0.99, 0.7))
        else:
            ax.legend(handles=[green_patch, blue_patch, red_patch, yellow_patch, cyan_patch, magenta_patch],
                      bbox_to_anchor=(0.99, 0.7))
        task_squares = [ax.add_patch(patches.RegularPolygon(xy=(task['location'][0] * 10,
                                                             task['location'][1] * 10),
                                                            numVertices=int(task['requirements'].sum()) + 3,
                                                            radius=0.3, color='b')) for task in self.task_dic.values()]
        ax.add_patch(patches.Circle((self.depot['location'][0] * 10, self.depot['location'][1] * 10),
                                    0.2,
                                    color='r'))
        if plot_robot_icon:
            agent_triangles = []
            for a in self.agent_dic.values():
                agent_triangles.append(ax.add_artist(AnnotationBbox(drone_oi, (self.depot['location'][0] * 10,
                                                     self.depot['location'][1] * 10),
                                       frameon=False)))
        else:
            agent_triangles = [ax.add_patch(patches.RegularPolygon(xy=(self.depot['location'][0] * 10,
                                                                       self.depot['location'][1] * 10), numVertices=3,
                                                                   radius=0.2, color='r'))
                               for _ in self.agent_dic.values()]

        # Define the update function for the animation
        def update(frame):
            ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                         f'\nCurrent time is {frame * self.dt:.2f}min')
            pos = np.round([agent['trajectory'][frame, 0:2] for agent in self.agent_dic.values()], 4)
            unq, count = np.unique(pos, axis=0, return_counts=True)
            for agent in self.agent_dic.values():
                repeats = int(count[np.argwhere(np.all(unq == np.round(agent['trajectory'][frame, 0:2], 4), axis=1))])
                agent_triangles[agent['ID']].xy = tuple(agent['trajectory'][frame, 0:2] * 10)
                if plot_robot_icon:
                    agent_triangles[agent['ID']].xyann = tuple(agent['trajectory'][frame, 0:2] * 10)
                    agent_triangles[agent['ID']].xybox = tuple(agent['trajectory'][frame, 0:2] * 10)
                else:
                    agent_triangles[agent['ID']].set_color('m' if repeats >= 4 else 'c' if repeats == 3
                                                           else 'y' if repeats == 2 else 'r')
                agent_triangles[agent['ID']].orientation = agent['trajectory'][frame, 2] - np.pi / 2
                # Add the current frame's data point to the plot for each trajectory
                if frame > 40:
                    lines[agent['ID']].set_data(agent['trajectory'][frame - 40:frame + 1, 0] * 10,
                                                agent['trajectory'][frame - 40:frame + 1, 1] * 10)
                else:
                    lines[agent['ID']].set_data(agent['trajectory'][:frame + 1, 0] * 10,
                                                agent['trajectory'][:frame + 1, 1] * 10)

            for task in self.task_dic.values():
                if self.reactive_planning:
                    if task['ID'] > np.clip(frame * self.dt//10 * 20 + 20, 20, 100):
                        task_squares[task['ID']].set_color('w')
                        task_squares[task['ID']].set_zorder(0)
                    else:
                        task_squares[task['ID']].set_color('b')
                        task_squares[task['ID']].set_zorder(1)
                if frame * self.dt >= task['time_finish'] > 0:
                    task_squares[task['ID']].set_color('g')
            return lines

        # Set up the animation
        ani = FuncAnimation(fig, update, frames=gif_len, interval=100, blit=True)
        ani.save(f'{path}/episode_{n}_{self.current_time:.1f}.gif')

    def get_grouped_tasks(self):
        grouped_tasks = dict()
        groups = list(set(np.array(self.get_matrix(self.task_dic, 'requirements')).squeeze(1).tolist()))
        for task_requirement in groups:
            grouped_tasks[task_requirement] = dict()
        index = np.zeros_like(groups)
        for i, task in self.task_dic.items():
            requirement = int(task['requirements'])
            ind = index[groups.index(requirement)]
            grouped_tasks[requirement].update({ind: task})
            index[groups.index(requirement)] += 1
        grouped_tasks = {key: value for key, value in grouped_tasks.items() if len(value) > 0}
        agent_v = {}
        if np.sum(list(grouped_tasks.keys())) > self.agents_num:
            agent_num = self.agents_num * 2
        else:
            agent_num = self.agents_num
        for keys, values in grouped_tasks.items():
            agent_v[keys] = (len(values) / self.tasks_num + np.sum(self.get_matrix(values, 'time')) / np.sum(self.get_matrix(self.task_dic, 'time'))) * keys
        agent_v_ = np.array(list(agent_v.values()))
        agent_v_ = agent_v_ / np.sum(agent_v_) * agent_num
        agent_v_ = np.clip(agent_v_, list(agent_v.keys()), None)
        agent_v_r = agent_v_ // np.array(list(agent_v.keys()))
        remainder = agent_v_ % np.array(list(agent_v.keys())) / np.array(list(agent_v.keys()))
        rest_ = agent_num - np.dot(agent_v_r, np.array(list(agent_v.keys())))
        sort_ = np.argsort(remainder)[::-1]
        while rest_ != 0 and np.any(np.array(list(agent_v.keys())) // rest_ == 0):
            for idx in sort_:
                add_ = np.clip(rest_ // list(agent_v.keys())[idx], 0, 1)
                agent_v_r[idx] += add_
                rest_ -= add_ * list(agent_v.keys())[idx]
        grouped_agents = {}
        for i, (keys, values) in enumerate(grouped_tasks.items()):
            if agent_v_r[i] == 0 and len(grouped_tasks[keys]) != 0:
                grouped_agents[keys] = 1
            else:
                grouped_agents[keys] = int(agent_v_r[i])
        return grouped_tasks, grouped_agents

    def execute_by_route(self, path='./', method=0, plot_figure=False, max_time=200, max_waiting_time=100):
        self.plot_figure = plot_figure
        if max_waiting_time is not None:
            self.max_waiting_time = float(max_waiting_time)
        while not self.finished and self.current_time < float(max_time):
            if self.reactive_planning:
                self.visible_length = int(np.clip(self.current_time//10 * 20 + 20, 20, 100))
            decision_agents, current_time = self.next_decision()
            self.current_time = current_time
            self.task_update()
            self.agent_update()
            for agent in decision_agents:
                if self.agent_dic[agent]['pre_set_route'] is None or not self.agent_dic[agent]['pre_set_route']:
                    self.agent_step(agent, 0)
                    self.task_update()
                    self.agent_update()
                    continue
                if self.reactive_planning:
                    if self.agent_dic[agent]['pre_set_route']:
                        if self.agent_dic[agent]['pre_set_route'][0] > self.visible_length:
                            self.agent_step(agent, 0)
                            self.task_update()
                            self.agent_update()
                            continue
                self.agent_step(agent, self.agent_dic[agent]['pre_set_route'].pop(0))
                self.task_update()
                self.agent_update()
            self.finished = self.check_finished()
        if self.plot_figure:
            self.plot_animation(path, method)
        print(self.current_time)
        # self.process_map(path)
        return self.current_time

    def pre_set_route(self, routes, agent_id):
        if not self.agent_dic[agent_id]['pre_set_route']:
            self.agent_dic[agent_id]['pre_set_route'] = routes
        else:
            self.agent_dic[agent_id]['pre_set_route'] += routes

    def process_map(self, path):
        import pandas as pd
        grouped_tasks = dict()
        groups = list(set(np.array(self.get_matrix(self.task_dic, 'requirements')).squeeze(1).tolist()))
        for task_requirement in groups:
            grouped_tasks[task_requirement] = dict()
        index = np.zeros_like(groups)
        for i, task in self.task_dic.items():
            requirement = int(task['requirements'])
            ind = index[groups.index(requirement)]
            grouped_tasks[requirement].update({ind: task})
            index[groups.index(requirement)] += 1
        grouped_tasks = {key: value for key, value in grouped_tasks.items() if len(value) > 0}
        time_finished = [self.get_matrix(dic, 'time_finish') for dic in grouped_tasks.values()]
        t = 0
        time_tick_stamp = dict()
        while t <= self.current_time:
            time_tick_stamp[t] = [np.sum(np.array(ratio) < t)/len(ratio) for ratio in time_finished]
            t += 0.1
            t = np.round(t, 1)
        pd = pd.DataFrame(time_tick_stamp)
        pd.to_csv(f'{path}time_RL.csv')
