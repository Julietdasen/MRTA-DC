import argparse
import copy
import json
import logging
import os

_omp = os.environ.get('OMP_NUM_THREADS', '')
try:
    if int(_omp) < 1:
        os.environ['OMP_NUM_THREADS'] = '1'
except (ValueError, TypeError):
    pass

import random
import socket
import subprocess
from datetime import datetime

import numpy as np
import ray
import torch
import torch.optim as optim
import wandb
from scipy.stats import ttest_rel
from torch.utils.tensorboard import SummaryWriter

import parameters as params
from attention import AttentionNet
from model.value_net import ValueNet
from parameters import *
from runner import RLRunner


def parse_args():
    parser = argparse.ArgumentParser(description='DCMRTA trainer')
    parser.add_argument('--tag', type=str, default=os.getenv('DCMRTA_RUN_TAG', ''), help='Optional run tag for logging.')
    parser.add_argument('--comment', type=str, default=os.getenv('DCMRTA_RUN_COMMENT', ''), help='Optional run comment.')
    return parser.parse_args()


def infer_run_dir():
    run_dir = os.getenv('DCMRTA_RUN_DIR', '')
    if run_dir:
        return run_dir
    model_dir = os.path.dirname(model_path)
    if model_dir:
        return model_dir
    return '.'


def setup_logger(log_file):
    logger = logging.getLogger('dcmrta.train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    disable_file_log = os.getenv('DCMRTA_DISABLE_FILE_LOG', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    if not disable_file_log:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def append_jsonl(path, payload):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=True) + '\n')


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def to_yaml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return '[' + ', '.join(to_yaml_value(v) for v in value) + ']'
    text = str(value)
    safe = text.replace('"', '\\"')
    return f'"{safe}"'


def write_yaml(path, config):
    with open(path, 'w', encoding='utf-8') as f:
        for key in sorted(config.keys()):
            f.write(f'{key}: {to_yaml_value(config[key])}\n')


def collect_config(run_name, run_dir, args):
    cfg = {}
    for key, value in vars(params).items():
        if key.isupper() or key in ['FOLDER_NAME', 'model_path', 'train_path', 'gifs_path']:
            if isinstance(value, (str, int, float, bool, tuple, list)) or value is None:
                cfg[key] = value
    cfg['RUN_NAME'] = run_name
    cfg['RUN_DIR'] = run_dir
    cfg['RUN_TAG'] = args.tag
    cfg['RUN_COMMENT'] = args.comment
    cfg['HOSTNAME'] = socket.gethostname()
    cfg['GIT_COMMIT'] = get_git_commit()
    cfg['START_TIME'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return cfg


def find_resume_checkpoint():
    explicit = os.getenv('DCMRTA_RESUME_CKPT', '')
    if explicit and os.path.exists(explicit):
        return explicit
    numbered = []
    if os.path.exists(model_path):
        for name in os.listdir(model_path):
            if name.startswith('checkpoint_ep') and name.endswith('.pth'):
                numbered.append(os.path.join(model_path, name))
    if numbered:
        numbered.sort()
        return numbered[-1]
    fallback = os.path.join(model_path, 'checkpoint.pth')
    if os.path.exists(fallback):
        return fallback
    return ''


def save_checkpoint(checkpoint, episode, best=False):
    if best:
        named = os.path.join(model_path, f'best_ep{episode:06d}.pth')
        latest = os.path.join(model_path, 'best_model_checkpoint.pth')
    else:
        named = os.path.join(model_path, f'checkpoint_ep{episode:06d}.pth')
        latest = os.path.join(model_path, 'checkpoint.pth')
    torch.save(checkpoint, named)
    torch.save(checkpoint, latest)
    return named


def entropy_coef_by_step(step):
    if ENTROPY_DECAY_STEPS <= 0:
        return ENTROPY_COEF_END
    ratio = np.clip(step / float(ENTROPY_DECAY_STEPS), 0.0, 1.0)
    return float(ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * ratio)


def explained_variance(y_pred, y_true):
    var_y = float(torch.var(y_true, unbiased=False).item())
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False).item() / (var_y + 1e-8))


def merge_metrics(rows):
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out = {}
    for k in keys:
        vals = [float(r.get(k, np.nan)) for r in rows]
        out[k] = float(np.nanmean(vals))
    return out


def write_to_tensorboard(writer, train_rows, curr_episode):
    summary = merge_metrics(train_rows)
    writer.add_scalar('Losses/Policy Loss', summary.get('policy_loss', np.nan), curr_episode)
    writer.add_scalar('Losses/Value Loss', summary.get('value_loss', np.nan), curr_episode)
    writer.add_scalar('Losses/Entropy', summary.get('entropy', np.nan), curr_episode)
    writer.add_scalar('Losses/Grad Norm', summary.get('grad_norm', np.nan), curr_episode)
    writer.add_scalar('Losses/Adv Std', summary.get('adv_std', np.nan), curr_episode)
    writer.add_scalar('Losses/Explained Variance', summary.get('explained_var', np.nan), curr_episode)

    writer.add_scalar('Perf/Reward', summary.get('reward', np.nan), curr_episode)
    writer.add_scalar('Perf/Makespan', summary.get('makespan', np.nan), curr_episode)
    writer.add_scalar('Perf/Success rate', summary.get('success_rate', np.nan), curr_episode)
    writer.add_scalar('Perf/Time cost', summary.get('time_cost', np.nan), curr_episode)
    writer.add_scalar('Perf/Waiting time', summary.get('waiting_time', np.nan), curr_episode)
    writer.add_scalar('Perf/Traveling distance', summary.get('travel_dist', np.nan), curr_episode)
    writer.add_scalar('Perf/Utilization Exec', summary.get('utilization_exec', np.nan), curr_episode)
    writer.add_scalar('Perf/Utilization Wait', summary.get('utilization_wait', np.nan), curr_episode)
    writer.add_scalar('Perf/Utilization Travel', summary.get('utilization_travel', np.nan), curr_episode)
    writer.add_scalar('Perf/Switch Rate', summary.get('switch_rate', np.nan), curr_episode)
    writer.add_scalar('Perf/Quorum Break Rate', summary.get('quorum_break_rate', np.nan), curr_episode)
    writer.add_scalar('Perf/Pause Events', summary.get('pause_events', np.nan), curr_episode)
    writer.add_scalar('Perf/Waiting Efficiency', summary.get('efficiency', np.nan), curr_episode)

    if WANDB_LOG:
        wandb.log({'train': summary}, step=curr_episode)
    return summary


def empty_experience_buffer():
    return {
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


def take_batch(experience_buffer, batch_size):
    rollouts = {}
    for k, v in experience_buffer.items():
        rollouts[k] = v[:batch_size]
        experience_buffer[k] = v[batch_size:]
    return rollouts


def evaluate_policy_on_testset(policy_weights, test_set, test_agent_list=None):
    # Reuse caller-provided actors when possible to avoid repeated actor startup overhead.
    own_actors = test_agent_list is None
    if own_actors:
        test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(NUM_META_AGENT)]
    try:
        # Keep local policy/baseline weights aligned for deterministic eval semantics.
        set_refs = []
        for test_agent in test_agent_list:
            set_refs.append(test_agent.set_weights.remote(policy_weights))
            set_refs.append(test_agent.set_baseline_weights.remote(policy_weights))
        ray.get(set_refs)

        rows = []
        for i in range(test_set.shape[0]):
            sample_job_list = []
            for j, test_agent in enumerate(test_agent_list):
                sample_job_list.append(test_agent.testing.remote(seed=int(test_set[i][j])))
            rows += ray.get(sample_job_list)
        return rows
    finally:
        if own_actors:
            for actor in test_agent_list:
                ray.kill(actor)


def summarize_eval_rows(rows):
    if not rows:
        empty = np.array([], dtype=np.float64)
        return {
            'reward': empty,
            'makespan': empty,
            'success_rate': empty,
            'reward_mean': np.nan,
            'makespan_mean': np.nan,
            'success_rate_mean': np.nan,
        }

    reward = np.array([float(r.get('reward', np.nan)) for r in rows], dtype=np.float64)
    makespan = np.array([float(r.get('makespan', np.nan)) for r in rows], dtype=np.float64)
    success_rate = np.array([float(r.get('success_rate', np.nan)) for r in rows], dtype=np.float64)
    return {
        'reward': reward,
        'makespan': makespan,
        'success_rate': success_rate,
        'reward_mean': float(np.nanmean(reward)),
        'makespan_mean': float(np.nanmean(makespan)),
        'success_rate_mean': float(np.nanmean(success_rate)),
    }


def paired_ttest_pvalue(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 2:
        return None
    _, p_value = ttest_rel(a[valid], b[valid])
    if not np.isfinite(p_value):
        return None
    return float(p_value)


def get_worker_weight_bundle(global_network, baseline_network, value_network, device, local_device):
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        value_weights = value_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
        value_network.to(device)
        return weights, baseline_weights, value_weights
    return global_network.state_dict(), baseline_network.state_dict(), value_network.state_dict()


def get_policy_weights(global_network, device, local_device):
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        global_network.to(device)
        return weights
    return global_network.state_dict()


def launch_training_jobs(meta_agents, weights, baseline_weights, value_weights, curr_episode, agents_num, tasks_num):
    # Submit one rollout job per actor and advance episode counter in lockstep.
    job_list = []
    for meta_agent in meta_agents:
        job_list.append(meta_agent.job.remote(weights, baseline_weights, value_weights, curr_episode, agents_num, tasks_num))
        curr_episode += 1
    return job_list, curr_episode


def main():
    args = parse_args()
    run_dir = infer_run_dir()
    run_name = os.path.basename(os.path.normpath(run_dir))

    config_dir = os.path.join(run_dir, 'config')
    logs_dir = os.path.join(run_dir, 'logs')
    train_dir = os.path.dirname(train_path) or train_path

    for path in [run_dir, model_path, train_path, train_dir, gifs_path, config_dir, logs_dir]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logger(os.path.join(logs_dir, 'train.log'))
    logger.info('Run started | run_name=%s | run_dir=%s', run_name, run_dir)
    logger.info('Paths | model=%s | tb=%s | gifs=%s', model_path, train_path, gifs_path)

    config = collect_config(run_name, run_dir, args)
    write_yaml(os.path.join(config_dir, 'train.yaml'), config)

    env_log = {
        'python': os.sys.version,
        'torch': torch.__version__,
        'ray': ray.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'num_gpu_config': NUM_GPU,
        'num_meta_agent': NUM_META_AGENT,
        'hostname': socket.gethostname(),
        'git_commit': get_git_commit(),
        'run_name': run_name,
        'run_tag': args.tag,
        'run_comment': args.comment,
    }
    with open(os.path.join(logs_dir, 'env.log'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(env_log, ensure_ascii=True, indent=2))

    ray.init(num_gpus=NUM_GPU, runtime_env={'env_vars': {'OMP_NUM_THREADS': '1'}})
    logger.info('Ray resources: %s', ray.cluster_resources())

    writer = SummaryWriter(train_path)
    if WANDB_LOG:
        wandb.init(project='CF', name=run_name)

    train_metrics_jsonl = os.path.join(train_dir, 'train_metrics.jsonl')
    eval_metrics_jsonl = os.path.join(train_dir, 'eval_metrics.jsonl')

    use_cuda_global = USE_GPU_GLOBAL and torch.cuda.is_available()
    use_cuda_local = USE_GPU and torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda_global else torch.device('cpu')
    local_device = torch.device('cuda') if use_cuda_local else torch.device('cpu')

    global_network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
    baseline_network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
    value_network = ValueNet(CRITIC_INPUT_DIM, CRITIC_HIDDEN_DIM).to(device)

    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    value_optimizer = optim.Adam(value_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.98)

    if WANDB_LOG:
        wandb.watch(global_network)

    curr_episode = 0
    curr_level = 0
    best_perf = -100.0
    best_makespan = float('inf')
    update_step = 0

    if LOAD_MODEL or os.getenv('DCMRTA_RESUME_CKPT', ''):
        resume_checkpoint = find_resume_checkpoint()
        if resume_checkpoint:
            logger.info('Loading model from %s', resume_checkpoint)
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            global_network.load_state_dict(checkpoint['model'])
            baseline_network.load_state_dict(checkpoint.get('baseline_model', checkpoint['model']))
            if 'value_model' in checkpoint:
                value_network.load_state_dict(checkpoint['value_model'])
            if 'optimizer' in checkpoint:
                global_optimizer.load_state_dict(checkpoint['optimizer'])
            if 'value_optimizer' in checkpoint:
                value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            if 'lr_decay' in checkpoint:
                lr_decay.load_state_dict(checkpoint['lr_decay'])
            curr_episode = checkpoint.get('episode', 0)
            curr_level = checkpoint.get('level', 0)
            best_perf = checkpoint.get('best_perf', best_perf)
            best_makespan = float(checkpoint.get('best_makespan', best_makespan))
            if np.isfinite(best_makespan):
                best_perf = -best_makespan
            update_step = checkpoint.get('update_step', 0)
            logger.info('Resumed | episode=%s | best_perf=%.6f | best_makespan=%s', curr_episode, best_perf, f'{best_makespan:.6f}' if np.isfinite(best_makespan) else 'NA')
        else:
            logger.warning('LOAD_MODEL enabled but no checkpoint found under %s', model_path)

    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    weights, baseline_weights, value_weights = get_worker_weight_bundle(
        global_network, baseline_network, value_network, device, local_device
    )

    agents_num = np.random.randint(AGENTS_RANGE[0], AGENTS_RANGE[1] + 1)
    tasks_num = np.random.randint(TASKS_RANGE[0], TASKS_RANGE[1] + 1)
    jobList, curr_episode = launch_training_jobs(
        meta_agents, weights, baseline_weights, value_weights, curr_episode, agents_num, tasks_num
    )

    metric_name = [
        'success_rate',
        'makespan',
        'time_cost',
        'waiting_time',
        'travel_dist',
        'utilization_exec',
        'utilization_wait',
        'utilization_travel',
        'efficiency',
        'switch_rate',
        'quorum_break_rate',
        'pause_events',
    ]
    training_rows = []
    experience_buffer = empty_experience_buffer()
    test_set = np.random.randint(low=0, high=1e8, size=[256 // NUM_META_AGENT, NUM_META_AGENT])
    baseline_eval = None

    try:
        while True:
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)

            perf_metrics = {name: [] for name in metric_name}
            for job in done_jobs:
                job_results, metrics, info = job
                for k in experience_buffer.keys():
                    experience_buffer[k] += job_results[k]
                for n in metric_name:
                    perf_metrics[n].append(float(metrics.get(n, np.nan)))

            update_done = False
            while len(experience_buffer['agent_inputs']) >= BATCH_SIZE:
                rollouts = take_batch(experience_buffer, BATCH_SIZE)
                if len(experience_buffer['agent_inputs']) < BATCH_SIZE:
                    update_done = True
                    experience_buffer = empty_experience_buffer()

                agent_inputs = torch.stack(rollouts['agent_inputs'], dim=0)
                task_inputs = torch.stack(rollouts['task_inputs'], dim=0)
                action_batch = torch.stack(rollouts['actions'], dim=0)
                mask_batch = torch.stack(rollouts['masks'], dim=0)
                advantage_batch = torch.stack(rollouts['advantages'], dim=0)
                return_batch = torch.stack(rollouts['returns'], dim=0)
                global_feat_batch = torch.stack(rollouts['global_feats'], dim=0)
                reward_batch = torch.stack(rollouts['rewards'], dim=0)

                if agent_inputs.dim() == 4 and agent_inputs.shape[1] == 1:
                    agent_inputs = agent_inputs.squeeze(1)
                if task_inputs.dim() == 4 and task_inputs.shape[1] == 1:
                    task_inputs = task_inputs.squeeze(1)
                if mask_batch.dim() == 3 and mask_batch.shape[1] == 1:
                    mask_batch = mask_batch.squeeze(1)
                if global_feat_batch.dim() == 3 and global_feat_batch.shape[1] == 1:
                    global_feat_batch = global_feat_batch.squeeze(1)
                if action_batch.dim() == 3 and action_batch.shape[1] == 1:
                    action_batch = action_batch.squeeze(1)
                if advantage_batch.dim() == 3 and advantage_batch.shape[1] == 1:
                    advantage_batch = advantage_batch.squeeze(1)
                if return_batch.dim() == 3 and return_batch.shape[1] == 1:
                    return_batch = return_batch.squeeze(1)
                if reward_batch.dim() == 3 and reward_batch.shape[1] == 1:
                    reward_batch = reward_batch.squeeze(1)

                if device != local_device:
                    agent_inputs = agent_inputs.to(device)
                    task_inputs = task_inputs.to(device)
                    action_batch = action_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    advantage_batch = advantage_batch.to(device)
                    return_batch = return_batch.to(device)
                    global_feat_batch = global_feat_batch.to(device)
                    reward_batch = reward_batch.to(device)

                adv_mean = advantage_batch.mean()
                adv_std = advantage_batch.std(unbiased=False)
                norm_adv = (advantage_batch - adv_mean) / (adv_std + 1e-8)
                norm_adv = torch.clamp(norm_adv, min=-5.0, max=5.0)

                logp_list = global_network(task_inputs, agent_inputs, mask_batch)
                logp = torch.gather(logp_list, 1, action_batch.long())
                entropy = -(logp_list.exp() * logp_list).nansum(dim=-1).mean()
                policy_loss = (-logp * norm_adv.detach()).mean()

                value_pred = value_network(global_feat_batch)
                value_loss = torch.nn.functional.mse_loss(value_pred, return_batch)
                ent_coef = entropy_coef_by_step(update_step)
                total_loss = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy

                global_optimizer.zero_grad()
                value_optimizer.zero_grad()
                total_loss.backward()
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=MAX_GRAD_NORM, norm_type=2)
                grad_norm_value = torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=MAX_GRAD_NORM, norm_type=2)
                global_optimizer.step()
                value_optimizer.step()
                lr_decay.step()
                update_step += 1

                ev = explained_variance(value_pred.detach(), return_batch.detach())
                perf_data = {n: float(np.nanmean(perf_metrics[n])) for n in metric_name}
                training_rows.append({
                    'reward': float(reward_batch.mean().item()),
                    'value_loss': float(value_loss.item()),
                    'policy_loss': float(policy_loss.item()),
                    'entropy': float(entropy.item()),
                    'grad_norm': float(max(grad_norm_actor.item(), grad_norm_value.item())),
                    'adv_std': float(adv_std.item()),
                    'explained_var': float(ev),
                    **perf_data,
                })

            if update_done:
                weights, baseline_weights, value_weights = get_worker_weight_bundle(
                    global_network, baseline_network, value_network, device, local_device
                )

            next_jobs, curr_episode = launch_training_jobs(
                meta_agents, weights, baseline_weights, value_weights, curr_episode, agents_num, tasks_num
            )
            jobList += next_jobs

            if len(training_rows) >= SUMMARY_WINDOW:
                summary = write_to_tensorboard(writer, training_rows, curr_episode)
                lr_now = global_optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(
                    '[TRAIN] ep=%d lr=%.8f reward=%.4f success=%.4f makespan=%.4f time=%.4f wait=%.4f dist=%.4f '
                    'util_exec=%.4f util_wait=%.4f util_travel=%.4f switch=%.4f quorum=%.4f pause=%.2f '
                    'policy=%.4f value=%.4f entropy=%.4f adv_std=%.4f ev=%.4f grad=%.4f',
                    curr_episode,
                    lr_now,
                    summary.get('reward', np.nan),
                    summary.get('success_rate', np.nan),
                    summary.get('makespan', np.nan),
                    summary.get('time_cost', np.nan),
                    summary.get('waiting_time', np.nan),
                    summary.get('travel_dist', np.nan),
                    summary.get('utilization_exec', np.nan),
                    summary.get('utilization_wait', np.nan),
                    summary.get('utilization_travel', np.nan),
                    summary.get('switch_rate', np.nan),
                    summary.get('quorum_break_rate', np.nan),
                    summary.get('pause_events', np.nan),
                    summary.get('policy_loss', np.nan),
                    summary.get('value_loss', np.nan),
                    summary.get('entropy', np.nan),
                    summary.get('adv_std', np.nan),
                    summary.get('explained_var', np.nan),
                    summary.get('grad_norm', np.nan),
                )
                append_jsonl(train_metrics_jsonl, {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'episode': curr_episode,
                    'lr': float(lr_now),
                    **summary,
                })
                training_rows = []

            if curr_episode % 512 == 0:
                checkpoint = {
                    'model': global_network.state_dict(),
                    'baseline_model': baseline_network.state_dict(),
                    'value_model': value_network.state_dict(),
                    'optimizer': global_optimizer.state_dict(),
                    'value_optimizer': value_optimizer.state_dict(),
                    'episode': curr_episode,
                    'lr_decay': lr_decay.state_dict(),
                    'level': curr_level,
                    'best_perf': best_perf,
                    'best_makespan': best_makespan,
                    'run_name': run_name,
                    'update_step': update_step,
                }
                ckpt_path = save_checkpoint(checkpoint, curr_episode, best=False)
                logger.info('[CHECKPOINT] saved %s', ckpt_path)

            if EVALUATE and curr_episode % 1024 == 0:
                # Drain in-flight rollout jobs before running evaluation on the same actors.
                ray.get(jobList)
                jobList = []
                logger.info('[EVAL] start ep=%d', curr_episode)
                # Reuse the same actors for evaluation to avoid repeated actor init/teardown cost.
                if baseline_eval is None:
                    baseline_eval_rows = evaluate_policy_on_testset(baseline_weights, test_set, test_agent_list=meta_agents)
                    baseline_eval = summarize_eval_rows(baseline_eval_rows)
                test_eval_rows = evaluate_policy_on_testset(weights, test_set, test_agent_list=meta_agents)
                test_eval = summarize_eval_rows(test_eval_rows)

                test_reward_mean = test_eval['reward_mean']
                baseline_reward_mean = baseline_eval['reward_mean']
                test_makespan_mean = test_eval['makespan_mean']
                baseline_makespan_mean = baseline_eval['makespan_mean']

                makespan_p_value = None
                reward_p_value = paired_ttest_pvalue(test_eval['reward'], baseline_eval['reward'])
                best_updated = False

                if test_makespan_mean < baseline_makespan_mean:
                    makespan_p_value = paired_ttest_pvalue(test_eval['makespan'], baseline_eval['makespan'])
                    if makespan_p_value is not None and makespan_p_value < 0.05:
                        weights = get_policy_weights(global_network, device, local_device)
                        baseline_weights = copy.deepcopy(weights)
                        baseline_network.load_state_dict(baseline_weights)
                        test_set = np.random.randint(low=0, high=1e8, size=[256 // NUM_META_AGENT, NUM_META_AGENT])
                        baseline_eval = None
                        best_makespan = float(test_makespan_mean)
                        best_perf = -best_makespan
                        checkpoint = {
                            'model': global_network.state_dict(),
                            'baseline_model': baseline_network.state_dict(),
                            'value_model': value_network.state_dict(),
                            'optimizer': global_optimizer.state_dict(),
                            'value_optimizer': value_optimizer.state_dict(),
                            'episode': curr_episode,
                            'lr_decay': lr_decay.state_dict(),
                            'best_perf': best_perf,
                            'best_makespan': best_makespan,
                            'run_name': run_name,
                            'update_step': update_step,
                        }
                        best_path = save_checkpoint(checkpoint, curr_episode, best=True)
                        logger.info('[BEST] updated best model %s', best_path)
                        best_updated = True

                logger.info(
                    '[EVAL] ep=%d reward(test/base)=%.6f/%.6f makespan(test/base)=%.6f/%.6f '
                    'p_makespan=%s p_reward=%s best_updated=%s',
                    curr_episode,
                    test_reward_mean,
                    baseline_reward_mean,
                    test_makespan_mean,
                    baseline_makespan_mean,
                    f'{makespan_p_value:.6f}' if makespan_p_value is not None else 'NA',
                    f'{reward_p_value:.6f}' if reward_p_value is not None else 'NA',
                    best_updated,
                )
                append_jsonl(eval_metrics_jsonl, {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'episode': curr_episode,
                    'test_mean': float(test_reward_mean),
                    'baseline_mean': float(baseline_reward_mean),
                    'test_reward_mean': float(test_reward_mean),
                    'baseline_reward_mean': float(baseline_reward_mean),
                    'test_makespan_mean': float(test_makespan_mean),
                    'baseline_makespan_mean': float(baseline_makespan_mean),
                    'p_value': makespan_p_value,
                    'p_value_makespan': makespan_p_value,
                    'p_value_reward': reward_p_value,
                    'best_updated': best_updated,
                    'best_makespan': float(best_makespan) if np.isfinite(best_makespan) else None,
                    'best_perf': float(best_perf),
                })

                weights, baseline_weights, value_weights = get_worker_weight_bundle(
                    global_network, baseline_network, value_network, device, local_device
                )

                # Resume rollout collection immediately after eval with fresh/global weights.
                jobList, curr_episode = launch_training_jobs(
                    meta_agents, weights, baseline_weights, value_weights, curr_episode, agents_num, tasks_num
                )

    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt detected, stopping training')
    finally:
        if WANDB_LOG:
            wandb.finish()
        for actor in locals().get('meta_agents', []):
            ray.kill(actor)
        writer.close()
        logger.info('Run finished')


if __name__ == '__main__':
    main()
