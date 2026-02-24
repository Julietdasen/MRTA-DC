import copy
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from attention import AttentionNet
from env.task_env import TaskEnv
from parameters import *


class Worker:
    def __init__(
        self,
        mete_agent_id,
        local_network,
        local_baseline,
        local_value=None,
        global_step=0,
        device='cuda',
        save_image=False,
        agents_num=AGENTS_RANGE,
        tasks_num=TASKS_RANGE,
        seed=None,
    ):
        self.device = device
        self.metaAgentID = mete_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.env = TaskEnv(
            agents_num,
            tasks_num,
            TRAIT_DIM,
            COALITION_SIZE,
            seed=seed,
            plot_figure=save_image,
            task_alpha=TASK_ALPHA,
            coalition_beta=COALITION_BETA,
            mode_cost_type=MODE_COST_TYPE,
            reward_w_makespan=REWARD_W_MAKESPAN,
            reward_w_travel=REWARD_W_TRAVEL,
            reward_w_wait=REWARD_W_WAIT,
            reward_w_mode=REWARD_W_MODE,
            enable_commit_lock=ENABLE_COMMIT_LOCK,
            min_commit_time=MIN_COMMIT_TIME,
            enable_quorum_protect=ENABLE_QUORUM_PROTECT,
            switch_penalty=SWITCH_PENALTY,
            quorum_break_penalty=QUORUM_BREAK_PENALTY,
            pause_event_penalty=PAUSE_EVENT_PENALTY,
            use_dense_event_reward=USE_DENSE_EVENT_REWARD,
            use_potential_shaping=USE_POTENTIAL_SHAPING,
            potential_shaping_coef=POTENTIAL_SHAPING_COEF,
        )
        self.baseline_env = copy.deepcopy(self.env)
        self.local_net = local_network
        self.local_baseline = local_baseline
        self.local_value = local_value
        self.experience = None
        self.episode_number = None
        self.perf_metrics = {}

    @staticmethod
    def _compute_gae(rewards, values, dones, gamma, gae_lambda):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = float(rewards[t]) + gamma * next_value * mask - float(values[t])
            gae = delta + gamma * gae_lambda * mask * gae
            adv[t] = gae
            next_value = float(values[t])
        returns = adv + values
        return adv, returns

    def _build_step_tensors(self, env, agent):
        mask_np = env.get_action_mask(agent['ID'])
        total_agents = torch.FloatTensor(env.get_current_agent_status(agent)).unsqueeze(0).to(self.device)
        task_info = torch.FloatTensor(env.get_current_task_status(agent)).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(self.device)
        global_feat = torch.FloatTensor(env.get_global_features()).unsqueeze(0).to(self.device)
        return total_agents, task_info, mask, global_feat

    def _sample_action(self, logp_list, mask, tasks_num):
        action = Categorical(logp_list.exp()).sample()
        n_tries = 0
        while action.item() > tasks_num or mask[:, action.item()].all().item():
            action = Categorical(logp_list.exp()).sample()
            n_tries += 1
            if n_tries >= 512:
                valid_logp = logp_list.clone()
                valid_logp[mask] = float('-inf')
                if tasks_num + 1 < valid_logp.shape[-1]:
                    valid_logp[..., tasks_num + 1:] = float('-inf')
                action = valid_logp.argmax(dim=-1)
                break
        return action

    def run_episode(self, episode_number):
        episode = {
            'agent_inputs': [],
            'task_inputs': [],
            'actions': [],
            'masks': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'global_feats': [],
            'advantages': [],
            'returns': [],
        }
        self.env.reset_dense_reward_snapshot()
        current_action_index = 0

        while (not self.env.finished) and self.env.current_time < MAX_TIME:
            with torch.no_grad():
                decision_agents, current_time = self.env.next_decision()
                groups = self.env.get_unique_group(decision_agents) if len(decision_agents) > 0 else []
                self.env.current_time = current_time
                self.env.task_update()
                self.env.agent_update()

                for group in groups:
                    while len(group) > 0:
                        leader_id = int(np.random.choice(group))
                        agent = self.env.agent_dic[leader_id]
                        if agent['returned']:
                            group.remove(leader_id)
                            continue

                        total_agents, task_info, mask, global_feat = self._build_step_tensors(self.env, agent)
                        logp_list = self.local_net(task_info, total_agents, mask)
                        action = self._sample_action(logp_list, mask, self.env.tasks_num)
                        if USE_CRITIC and self.local_value is not None:
                            value_t = self.local_value(global_feat).squeeze(-1)
                        else:
                            value_t = torch.zeros((1,), dtype=torch.float32, device=self.device)

                        group, _ = self.env.step(group, leader_id, int(action.item()), current_action_index)
                        self.env.task_update()
                        self.env.agent_update()
                        reward_info = self.env.get_dense_reward_delta(reset=True)

                        episode['agent_inputs'].append(total_agents)
                        episode['task_inputs'].append(task_info)
                        episode['actions'].append(action.unsqueeze(0))
                        episode['masks'].append(mask)
                        episode['rewards'].append(torch.tensor([[float(reward_info['reward'])]], dtype=torch.float32, device=self.device))
                        episode['dones'].append(torch.tensor([[0.0]], dtype=torch.float32, device=self.device))
                        episode['values'].append(value_t.view(1, 1))
                        episode['global_feats'].append(global_feat)
                        current_action_index += 1

                self.env.finished = self.env.check_finished()
                tail_delta = self.env.get_dense_reward_delta(reset=True)
                if episode['rewards']:
                    episode['rewards'][-1] = episode['rewards'][-1] + torch.tensor(
                        [[float(tail_delta['reward'])]], dtype=torch.float32, device=self.device
                    )

        reward, finished_tasks = self.env.get_episode_reward(MAX_TIME)

        if episode['dones']:
            episode['dones'][-1] = torch.tensor([[1.0]], dtype=torch.float32, device=self.device)
            rewards_np = np.array([x.item() for x in episode['rewards']], dtype=np.float32)
            values_np = np.array([x.item() for x in episode['values']], dtype=np.float32)
            dones_np = np.array([x.item() for x in episode['dones']], dtype=np.float32)
            advantages, returns = self._compute_gae(rewards_np, values_np, dones_np, GAMMA, GAE_LAMBDA)
            for adv, ret in zip(advantages, returns):
                episode['advantages'].append(torch.tensor([[float(adv)]], dtype=torch.float32, device=self.device))
                episode['returns'].append(torch.tensor([[float(ret)]], dtype=torch.float32, device=self.device))

        perf_metrics = {
            'success_rate': float(np.sum(finished_tasks) / len(finished_tasks)),
            'makespan': float(self.env.current_time),
            'time_cost': float(np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))),
            'waiting_time': float(np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))),
            'travel_dist': float(np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))),
            'reward': float(reward),
            'episode_steps': float(len(episode['rewards'])),
        }
        util_exec, util_wait, util_travel = self.env.get_utilization_metrics()
        perf_metrics['utilization_exec'] = float(util_exec)
        perf_metrics['utilization_wait'] = float(util_wait)
        perf_metrics['utilization_travel'] = float(util_travel)
        perf_metrics['efficiency'] = float(util_exec)
        perf_metrics.update(self.env.get_behavior_metrics())

        if self.save_image:
            self.env.plot_animation(gifs_path, episode_number)
        self.experience = episode
        return perf_metrics

    def _greedy_policy_eval(self, env):
        while not env.finished and env.current_time < MAX_TIME:
            with torch.no_grad():
                decision_agents, current_time = env.next_decision()
                groups = env.get_unique_group(decision_agents)
                env.current_time = current_time
                env.task_update()
                env.agent_update()
                for group in groups:
                    while len(group) > 0:
                        leader_id = int(np.random.choice(group))
                        agent = env.agent_dic[leader_id]
                        if agent['returned']:
                            group.remove(leader_id)
                            continue
                        total_agents, task_info, mask, _ = self._build_step_tensors(env, agent)
                        logp_list = self.local_net(task_info, total_agents, mask)
                        masked_scores = logp_list.exp().clone()
                        masked_scores[mask] = -1e9
                        action = torch.argmax(masked_scores, dim=1)
                        group, _ = env.step(group, leader_id, action.item(), 0)
                        env.task_update()
                        env.agent_update()
                env.finished = env.check_finished()

    def run_test(self, test_episode, test_env, image_path=None):
        perf_metrics = {}
        self.baseline_env = copy.copy(test_env)
        self.baseline_env.plot_figure = False
        self._greedy_policy_eval(self.baseline_env)
        _, finished_tasks = self.baseline_env.get_episode_reward(MAX_TIME)

        perf_metrics['success_rate'] = float(np.sum(finished_tasks) / len(finished_tasks))
        perf_metrics['makespan'] = float(self.baseline_env.current_time)
        perf_metrics['time_cost'] = float(np.nanmean(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'time_start')))
        perf_metrics['waiting_time'] = float(np.mean(self.baseline_env.get_matrix(self.baseline_env.agent_dic, 'sum_waiting_time')))
        perf_metrics['travel_dist'] = float(np.sum(self.baseline_env.get_matrix(self.baseline_env.agent_dic, 'travel_dist')))
        util_exec, util_wait, util_travel = self.baseline_env.get_utilization_metrics()
        perf_metrics['utilization_exec'] = float(util_exec)
        perf_metrics['utilization_wait'] = float(util_wait)
        perf_metrics['utilization_travel'] = float(util_travel)
        perf_metrics['efficiency'] = float(util_exec)
        if image_path is not None:
            self.baseline_env.plot_animation(image_path, 'RL')
        return perf_metrics

    def run_test_IS(self, test_episode, test_env):
        perf_metrics = {}
        self.baseline_env = copy.copy(test_env)
        self.baseline_env.plot_figure = False

        while not self.baseline_env.finished and self.baseline_env.current_time < MAX_TIME:
            with torch.no_grad():
                decision_agents, current_time = self.baseline_env.next_decision()
                self.baseline_env.current_time = current_time
                self.baseline_env.task_update()
                self.baseline_env.agent_update()
                for aid in decision_agents:
                    agent = self.baseline_env.agent_dic[aid]
                    if agent['returned']:
                        continue
                    total_agents, task_info, mask, _ = self._build_step_tensors(self.baseline_env, agent)
                    logp_list = self.local_net(task_info, total_agents, mask)
                    masked_scores = logp_list.exp().clone()
                    masked_scores[mask] = -1e9
                    action = torch.argmax(masked_scores, dim=1)
                    self.baseline_env.agent_step(int(aid), int(action.item()))
                    self.baseline_env.task_update()
                    self.baseline_env.agent_update()
                self.baseline_env.finished = self.baseline_env.check_finished()

        _, finished_tasks = self.baseline_env.get_episode_reward(MAX_TIME)
        perf_metrics['success_rate'] = float(np.sum(finished_tasks) / len(finished_tasks))
        perf_metrics['makespan'] = float(self.baseline_env.current_time)
        perf_metrics['time_cost'] = float(np.nanmean(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'time_start')))
        perf_metrics['waiting_time'] = float(np.mean(self.baseline_env.get_matrix(self.baseline_env.agent_dic, 'sum_waiting_time')))
        perf_metrics['travel_dist'] = float(np.sum(self.baseline_env.get_matrix(self.baseline_env.agent_dic, 'travel_dist')))
        util_exec, util_wait, util_travel = self.baseline_env.get_utilization_metrics()
        perf_metrics['utilization_exec'] = float(util_exec)
        perf_metrics['utilization_wait'] = float(util_wait)
        perf_metrics['utilization_travel'] = float(util_travel)
        perf_metrics['efficiency'] = float(util_exec)
        return perf_metrics

    def baseline_test(self):
        self.baseline_env.plot_figure = False
        self._greedy_policy_eval(self.baseline_env)
        reward, _ = self.baseline_env.get_episode_reward(MAX_TIME)
        return reward

    def work(self, episode_number):
        self.episode_number = episode_number
        self.perf_metrics = self.run_episode(episode_number)

    def generate_route(self):
        route = self.baseline_env.get_matrix(self.baseline_env.agent_dic, 'route')
        for i in range(len(route)):
            route[i] = [iterator + 1 for iterator in route[i]]
        route = dict(enumerate(route))
        import yaml
        with open('route_ros_large.yaml', 'w') as f:
            yaml.dump(route, f, sort_keys=False)

    @staticmethod
    def zero_padding(a, max_len=AGENTS_RANGE[1]):
        return F.pad(a, (0, 0, 0, max_len - a.shape[1]), 'constant', -1)

    @staticmethod
    def true_padding(a, max_len=TASKS_RANGE[1]):
        return F.pad(a, (0, max_len - a.shape[1]), 'constant', True)


if __name__ == '__main__':
    device = torch.device('cpu')
    localNetwork = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
    for i in range(3):
        worker = Worker(1, localNetwork, localNetwork, None, 0, device=device, seed=i, save_image=False)
        worker.run_episode(i)
        print(i)
