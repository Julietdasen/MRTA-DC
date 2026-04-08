import copy
import numpy as np

from env.task_env import TaskEnv


class TaskEnvSpecial(TaskEnv):
    """Dual-mode extension of ``TaskEnv``.

    A special parent task is expanded into two alternative task entries that share
    the same location but have different coalition sizes and durations. During an
    episode, once one mode is selected, its sibling mode is disabled.
    """

    def __init__(
        self,
        agents_range=(10, 10),
        tasks_range=(20, 20),
        traits_dim=1,
        max_coalition_size=7,
        max_duration=100,
        seed=None,
        plot_figure=False,
        enable_special_modes=True,
        num_special_tasks=1,
        mode_size_low=2,
        special_time_range=(80, 140),
        special_speedup_range=(10, 30),
    ):
        self.enable_special_modes = enable_special_modes
        self.num_special_tasks = num_special_tasks
        self.mode_size_low = mode_size_low
        self.special_time_range = special_time_range
        self.special_speedup_range = special_speedup_range
        self.runtime_locked_parent_modes = {}

        super().__init__(
            agents_range=agents_range,
            tasks_range=tasks_range,
            traits_dim=traits_dim,
            max_coalition_size=max_coalition_size,
            max_duration=max_duration,
            seed=seed,
            plot_figure=plot_figure,
        )
        self._sync_mode_metadata()

    @staticmethod
    def _copy_requirement(requirements):
        return np.array(requirements, copy=True)

    def sample_two_mode_sizes(self):
        """Sample two coalition sizes for a special task."""
        if self.max_coalition_size <= 1:
            return 1, 1

        min_size = max(1, min(int(self.mode_size_low), int(self.max_coalition_size)))
        candidates = np.arange(min_size, int(self.max_coalition_size) + 1)
        if candidates.size < 2:
            candidates = np.arange(1, int(self.max_coalition_size) + 1)
        if candidates.size < 2:
            only_size = int(candidates[0]) if candidates.size == 1 else 1
            return only_size, only_size

        mode_sizes = sorted(
            self.random_choice(candidates, size=2, replace=False).tolist()
        )
        return int(mode_sizes[0]), int(mode_sizes[1])

    def sample_special_mode_times(self, small_k, large_k):
        """Sample durations for the two modes, with the larger coalition faster."""
        t_low, t_high = self.special_time_range
        s_low, s_high = self.special_speedup_range

        base_time = float(self.random_int(t_low, t_high + 1))
        size_gap = max(int(large_k) - int(small_k), 1)
        min_speedup = min(max(int(s_low), size_gap), int(s_high))
        if min_speedup > int(s_high):
            speedup = float(size_gap)
        else:
            speedup = float(self.random_int(min_speedup, int(s_high) + 1))

        small_time = base_time
        large_time = max(10.0, base_time - speedup)
        return float(small_time), float(large_time)

    def _sync_mode_metadata(self):
        """Normalize task metadata so old pickles and compacted envs still work."""
        parent_to_task_ids = {}
        for task_id, task in self.task_dic.items():
            parent_id = int(task.get("parent_task_id", task["ID"]))
            task["parent_task_id"] = parent_id
            task.setdefault("mode_name", "default_mode")
            task.setdefault("base_time", float(task["time"]))
            task.setdefault("origin_is_special_mode", bool(task.get("is_special_mode", False)))
            task.setdefault("disabled", bool(task.get("disabled", False)))

            if "preset_disabled" not in task:
                inferred_disabled = (
                    bool(task.get("disabled", False))
                    and bool(task.get("finished", False))
                    and not np.any(np.array(task.get("status", [1])) > 0)
                )
                task["preset_disabled"] = inferred_disabled

            parent_to_task_ids.setdefault(parent_id, []).append(task_id)

        for task_ids in parent_to_task_ids.values():
            mode_group = sorted(task_ids)
            has_multiple_modes = len(mode_group) > 1
            for task_id in mode_group:
                task = self.task_dic[task_id]
                task["mode_group"] = mode_group.copy()
                task["is_special_mode"] = has_multiple_modes

    def _mark_task_pending(self, task):
        requirements = self._copy_requirement(task["requirements"])
        task.update(
            members=[],
            cost=[],
            finished=False,
            status=requirements.copy(),
            feasible_assignment=False,
            time_start=0,
            time_finish=0,
            sum_waiting_time=0,
            efficiency=0,
            abandoned_agent=[],
            disabled=False,
            time=float(task.get("base_time", task["time"])),
        )

    def _mark_task_disabled(self, task):
        requirements = self._copy_requirement(task["requirements"])
        task.update(
            members=[],
            cost=[],
            finished=True,
            status=np.zeros_like(requirements),
            feasible_assignment=True,
            time_start=0,
            time_finish=0,
            sum_waiting_time=0,
            efficiency=0,
            abandoned_agent=[],
            disabled=True,
            time=float(task.get("base_time", task["time"])),
        )

    def _reset_agent_runtime(self):
        for agent in self.agent_dic.values():
            agent.update(
                route=[],
                location=self.depot["location"],
                next_location=self.depot["location"],
                next_decision=0,
                travel_time=0,
                travel_dist=0,
                arrival_time=[],
                assigned=False,
                sum_waiting_time=0,
                working_condition=0,
                current_action_index=0,
                trajectory=[],
                angle=0,
                returned=False,
                pre_set_route=None,
                depot=self.depot["location"],
            )

        self.depot.update(members=[], ID=-1)
        self.current_time = 0
        self.finished = False

    def get_special_parent_ids(self):
        self._sync_mode_metadata()
        return sorted(
            {
                task["parent_task_id"]
                for task in self.task_dic.values()
                if task.get("is_special_mode", False)
            }
        )

    def get_special_mode_tasks(self):
        self._sync_mode_metadata()
        return [
            task for task in self.task_dic.values()
            if task.get("is_special_mode", False)
        ]

    def describe_special_modes(self):
        self._sync_mode_metadata()
        info = []
        for task in self.get_special_mode_tasks():
            requirements = np.array(task["requirements"]).reshape(-1)
            info.append(
                {
                    "env_task_id": int(task["ID"]),
                    "parent_task_id": int(task["parent_task_id"]),
                    "mode_name": task["mode_name"],
                    "requirements": int(requirements[0]),
                    "time": float(task["time"]),
                    "location": np.array(task["location"]).tolist(),
                    "disabled": bool(task.get("disabled", False)),
                    "preset_disabled": bool(task.get("preset_disabled", False)),
                }
            )
        return info

    def get_parent_finished_vector(self):
        parent_finished = {}
        for task in self.task_dic.values():
            parent_id = task.get("parent_task_id", task["ID"])
            parent_finished[parent_id] = parent_finished.get(parent_id, False) or task["finished"]
        return [parent_finished[parent_id] for parent_id in sorted(parent_finished.keys())]

    def get_parent_task_success_rate(self):
        parent_finished = self.get_parent_finished_vector()
        if len(parent_finished) == 0:
            return 0.0
        return float(np.mean(parent_finished))

    def generate_env(self):
        """Generate an env where selected parent tasks own two alternative modes."""
        if isinstance(self.tasks_range, tuple):
            real_tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1)
        else:
            real_tasks_num = self.tasks_range

        if isinstance(self.agents_range, tuple):
            agents_num = self.random_int(self.agents_range[0], self.agents_range[1] + 1)
        else:
            agents_num = self.agents_range

        agents_ini = np.ones((agents_num, self.traits_dim))
        depot = self.random_value(1, 2)
        cost_ini = self.random_value(agents_num, 1)
        tasks_loc = self.random_value(real_tasks_num, 2)

        task_dic = {}
        agent_dic = {}

        if self.enable_special_modes and self.num_special_tasks > 0:
            special_parent_ids = set(
                self.random_choice(
                    np.arange(real_tasks_num),
                    size=min(self.num_special_tasks, real_tasks_num),
                    replace=False,
                ).tolist()
            )
        else:
            special_parent_ids = set()

        env_task_id = 0
        for parent_id in range(real_tasks_num):
            location = tasks_loc[parent_id, :]

            if parent_id in special_parent_ids:
                small_k, large_k = self.sample_two_mode_sizes()
                small_time, large_time = self.sample_special_mode_times(small_k, large_k)
                mode_specs = [
                    (small_k, small_time, "small_mode", True),
                    (large_k, large_time, "large_mode", True),
                ]
            else:
                req = int(self.random_int(1, self.max_coalition_size + 1))
                mode_specs = [(req, float(self.max_duration), "default_mode", False)]

            current_mode_ids = []
            for req, duration, mode_name, is_special in mode_specs:
                task_dic[env_task_id] = {
                    "ID": env_task_id,
                    "parent_task_id": parent_id,
                    "mode_name": mode_name,
                    "requirements": np.array([req]),
                    "members": [],
                    "cost": [],
                    "location": location.copy(),
                    "feasible_assignment": False,
                    "finished": False,
                    "time_start": 0,
                    "time_finish": 0,
                    "status": np.array([req]),
                    "time": float(duration),
                    "base_time": float(duration),
                    "sum_waiting_time": 0,
                    "efficiency": 0,
                    "abandoned_agent": [],
                    "disabled": False,
                    "preset_disabled": False,
                    "is_special_mode": is_special,
                    "origin_is_special_mode": is_special,
                }
                current_mode_ids.append(env_task_id)
                env_task_id += 1

            for task_id in current_mode_ids:
                task_dic[task_id]["mode_group"] = current_mode_ids.copy()

        for agent_id in range(agents_num):
            agent_dic[agent_id] = {
                "ID": agent_id,
                "abilities": agents_ini[agent_id, :],
                "location": depot[0, :],
                "next_location": depot[0, :],
                "route": [],
                "arrival_time": [],
                "cost": cost_ini[agent_id],
                "travel_time": 0,
                "velocity": 0.2,
                "next_decision": 0,
                "depot": depot[0, :],
                "travel_dist": 0,
                "sum_waiting_time": 0,
                "current_action_index": 0,
                "working_condition": 0,
                "trajectory": [],
                "angle": 0,
                "returned": False,
                "assigned": False,
                "pre_set_route": None,
            }

        depot = {
            "location": depot[0, :],
            "members": [],
            "ID": -1,
        }
        return task_dic, agent_dic, depot

    def reset(self, test_env=None, seed=None):
        super().reset(test_env=test_env, seed=seed)
        self.runtime_locked_parent_modes = {}
        self._sync_mode_metadata()

    def lock_mode_group(self, selected_task_id):
        """Disable sibling modes once one task mode has been committed."""
        self._sync_mode_metadata()
        selected_task = self.task_dic[selected_task_id]
        siblings = selected_task.get("mode_group", [selected_task_id])
        if len(siblings) <= 1:
            return

        self.runtime_locked_parent_modes[selected_task["parent_task_id"]] = selected_task_id
        for sibling_id in siblings:
            if sibling_id == selected_task_id:
                continue
            self._mark_task_disabled(self.task_dic[sibling_id])

    def force_special_mode(self, chosen_mode_name, parent_id=None):
        """Persistently keep only one mode active and reset episode state."""
        self._sync_mode_metadata()
        parent_ids = [parent_id] if parent_id is not None else self.get_special_parent_ids()

        for current_parent_id in parent_ids:
            parent_tasks = [
                task for task in self.task_dic.values()
                if task["parent_task_id"] == current_parent_id
            ]
            if len(parent_tasks) <= 1:
                continue

            mode_names = {task["mode_name"] for task in parent_tasks}
            if chosen_mode_name not in mode_names:
                raise ValueError(
                    f"Parent task {current_parent_id} does not have mode {chosen_mode_name}. "
                    f"Available modes: {sorted(mode_names)}"
                )

            for task in parent_tasks:
                task["preset_disabled"] = task["mode_name"] != chosen_mode_name

        self.clear_decisions()

    def clear_forced_special_mode(self, parent_id=None):
        """Re-enable all modes for the selected parent tasks and reset state."""
        self._sync_mode_metadata()
        parent_ids = [parent_id] if parent_id is not None else self.get_special_parent_ids()

        for current_parent_id in parent_ids:
            for task in self.task_dic.values():
                if task["parent_task_id"] == current_parent_id:
                    task["preset_disabled"] = False

        self.clear_decisions()

    def make_compact_copy(self, chosen_mode_name=None, parent_id=None):
        """Return a clean compact copy, useful for later export scripts."""
        env_copy = copy.deepcopy(self)
        if chosen_mode_name is not None:
            env_copy.force_special_mode(chosen_mode_name, parent_id=parent_id)
        else:
            env_copy.clear_decisions()
        env_copy.compact_task_dic()
        return env_copy

    def compact_task_dic(self):
        """Drop disabled tasks, re-index the rest, and rebuild a clean env state."""
        self._sync_mode_metadata()
        active_tasks = [
            copy.deepcopy(task) for task in self.task_dic.values()
            if not task.get("disabled", False) and not task.get("preset_disabled", False)
        ]
        active_tasks.sort(key=lambda task: task["ID"])

        new_task_dic = {}
        old_to_new = {task["ID"]: new_id for new_id, task in enumerate(active_tasks)}

        for task in active_tasks:
            old_id = task["ID"]
            task["ID"] = old_to_new[old_id]
            task["mode_group"] = [
                old_to_new[mode_id]
                for mode_id in task.get("mode_group", [old_id])
                if mode_id in old_to_new
            ]
            task["preset_disabled"] = False
            self._mark_task_pending(task)
            new_task_dic[task["ID"]] = task

        self.task_dic = new_task_dic
        self.tasks_num = len(self.task_dic)
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        self.runtime_locked_parent_modes = {}
        self._sync_mode_metadata()
        self._reset_agent_runtime()

    def clear_decisions(self):
        """Reset runtime state while preserving special-mode metadata and forced modes."""
        self._sync_mode_metadata()
        for task in self.task_dic.values():
            if task.get("preset_disabled", False):
                self._mark_task_disabled(task)
            else:
                self._mark_task_pending(task)

        self.runtime_locked_parent_modes = {}
        self._reset_agent_runtime()

    def get_unfinished_tasks(self):
        """Exclude disabled sibling modes from action candidates."""
        unfinished_tasks = []
        for task in self.task_dic.values():
            unfinished_tasks.append(
                (not task.get("disabled", False))
                and (task["feasible_assignment"] is False)
                and np.any(task["status"] > 0)
            )
        return unfinished_tasks

    def agent_step(self, agent_id, task_id):
        """Mirror ``TaskEnv.agent_step`` while adding mode locking."""
        task_id = task_id - 1
        agent = self.agent_dic[agent_id]

        if task_id != -1:
            task = self.task_dic[task_id]
            if task.get("disabled", False):
                raise ValueError(f"Task mode {task_id} is disabled and cannot be selected.")

            parent_id = task.get("parent_task_id", task_id)
            has_alternative_mode = len(task.get("mode_group", [])) > 1
            if (
                has_alternative_mode
                and parent_id not in self.runtime_locked_parent_modes
                and len(task["members"]) == 0
            ):
                self.lock_mode_group(task_id)
            task = self.task_dic[task_id]
        else:
            task = self.depot

        agent["route"].append(task_id)
        travel_time = self.calculate_eulidean_distance(agent, task) / agent["velocity"]
        agent["travel_time"] = travel_time
        agent["travel_dist"] += self.calculate_eulidean_distance(agent, task)
        agent["arrival_time"] += [self.current_time + travel_time]
        agent["location"] = task["location"]

        if agent_id not in task["members"]:
            task["members"].append(agent_id)

        return -travel_time
