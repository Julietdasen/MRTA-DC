import argparse
import copy
import json
import logging
import os
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
from parameters import *
from runner import RLRunner


def parse_args():
    parser = argparse.ArgumentParser(description='DCMRTA trainer')
    parser.add_argument('--tag', type=str, default=os.getenv('DCMRTA_RUN_TAG', ''),
                        help='Optional run tag for logging.')
    parser.add_argument('--comment', type=str, default=os.getenv('DCMRTA_RUN_COMMENT', ''),
                        help='Optional run comment.')
    return parser.parse_args()


def sanitize_tag(text):
    if not text:
        return ''
    allowed = []
    for ch in text.strip():
        if ch.isalnum() or ch in ['-', '_', '.']:
            allowed.append(ch)
        elif ch.isspace():
            allowed.append('_')
    compact = ''.join(allowed).strip('_.')
    return compact[:80]


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


def writeToTensorBoard(writer, tensorboardData, curr_episode, plotMeans=True):
    if plotMeans:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    else:
        tensorboardData = list(tensorboardData)

    reward = tensorboardData[0]
    valueLoss = tensorboardData[1]
    policyLoss = tensorboardData[2]
    entropy = tensorboardData[3]
    gradNorm = tensorboardData[4]
    success_rate = tensorboardData[5]
    time_cost_makespan = tensorboardData[6]
    time_cost = tensorboardData[7]
    waiting = tensorboardData[8]
    distance = tensorboardData[9]
    # New metrics (v0.2): utilization_exec/wait/travel + legacy efficiency alias.
    if len(tensorboardData) >= 14:
        util_exec = tensorboardData[10]
        util_wait = tensorboardData[11]
        util_travel = tensorboardData[12]
        effi = tensorboardData[13]
    else:
        # Backward compatibility with old training data layout.
        effi = tensorboardData[10]
        util_exec = effi
        util_wait = np.nan
        util_travel = np.nan

    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Grad Norm', scalar_value=gradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Value Loss', scalar_value=valueLoss, global_step=curr_episode)

    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Makespan', scalar_value=time_cost_makespan, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Time cost', scalar_value=time_cost, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Waiting time', scalar_value=waiting, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Traveling distance', scalar_value=distance, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Utilization Exec', scalar_value=util_exec, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Utilization Wait', scalar_value=util_wait, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Utilization Travel', scalar_value=util_travel, global_step=curr_episode)
    # Backward-compatible tag kept for old dashboards.
    writer.add_scalar(tag='Perf/Waiting Efficiency', scalar_value=effi, global_step=curr_episode)

    if WANDB_LOG:
        wandb.log({
            'Losses': {
                'Grad Norm': gradNorm,
                'Policy Loss': policyLoss,
                'Entropy': entropy,
                'Value Loss': valueLoss,
            },
            'Perf': {
                'Reward': reward,
                'Makespan': time_cost_makespan,
                'Success Rate': success_rate,
                'Time Cost': time_cost,
                'Waiting Time': waiting,
                'Traveling Distance': distance,
                'Utilization Exec': util_exec,
                'Utilization Wait': util_wait,
                'Utilization Travel': util_travel,
                'Waiting Efficiency': effi,
            },
        }, step=curr_episode)

    return {
        'reward': float(reward),
        'value_loss': float(valueLoss),
        'policy_loss': float(policyLoss),
        'entropy': float(entropy),
        'grad_norm': float(gradNorm),
        'success_rate': float(success_rate),
        'makespan': float(time_cost_makespan),
        'time_cost': float(time_cost),
        'waiting_time': float(waiting),
        'travel_dist': float(distance),
        'utilization_exec': float(util_exec),
        'utilization_wait': float(util_wait),
        'utilization_travel': float(util_travel),
        'efficiency': float(effi),
    }


def main():
    args = parse_args()
    run_dir = infer_run_dir()
    run_name = os.path.basename(os.path.normpath(run_dir))

    config_dir = os.path.join(run_dir, 'config')
    logs_dir = os.path.join(run_dir, 'logs')
    train_dir = os.path.dirname(train_path)
    if not train_dir:
        train_dir = train_path

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

    ray.init(num_gpus=NUM_GPU)
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
    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.98)

    if WANDB_LOG:
        wandb.watch(global_network)

    curr_episode = 0
    best_perf = -100
    curr_level = 0

    if LOAD_MODEL or os.getenv('DCMRTA_RESUME_CKPT', ''):
        resume_checkpoint = find_resume_checkpoint()
        if resume_checkpoint:
            logger.info('Loading model from %s', resume_checkpoint)
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            global_network.load_state_dict(checkpoint['model'])
            baseline_network.load_state_dict(checkpoint['model'])
            global_optimizer.load_state_dict(checkpoint['optimizer'])
            lr_decay.load_state_dict(checkpoint['lr_decay'])
            curr_episode = checkpoint.get('episode', 0)
            curr_level = checkpoint.get('level', 0)
            best_perf = checkpoint.get('best_perf', best_perf)
            logger.info('Resumed | episode=%s | best_perf=%.6f', curr_episode, best_perf)

            best_latest = os.path.join(model_path, 'best_model_checkpoint.pth')
            if os.path.exists(best_latest):
                best_model_checkpoint = torch.load(best_latest, map_location=device)
                best_perf = best_model_checkpoint.get('best_perf', best_perf)
                baseline_network.load_state_dict(best_model_checkpoint['model'])
                logger.info('Best performance loaded: %.6f', best_perf)

            if RESET_OPT:
                global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
                lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.98)
                curr_episode = 0
                logger.info('Optimizer reset and episode reset to 0')
        else:
            logger.warning('LOAD_MODEL is enabled but no checkpoint file found under %s', model_path)

    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
    else:
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()

    jobList = []
    agents_num = np.random.randint(AGENTS_RANGE[0], AGENTS_RANGE[1] + 1)
    tasks_num = np.random.randint(TASKS_RANGE[0], TASKS_RANGE[1] + 1)
    for _, meta_agent in enumerate(meta_agents):
        jobList.append(meta_agent.job.remote(weights, baseline_weights, curr_episode, agents_num, tasks_num))
        curr_episode += 1

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
    ]
    trainingData = []
    experience_buffer = [[] for _ in range(9)]
    test_set = np.random.randint(low=0, high=1e8, size=[256 // NUM_META_AGENT, NUM_META_AGENT])
    baseline_value = None

    try:
        while True:
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)

            perf_metrics = {name: [] for name in metric_name}
            for job in done_jobs:
                jobResults, metrics, info = job
                for i in range(9):
                    experience_buffer[i] += jobResults[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            update_done = False
            while len(experience_buffer[0]) >= BATCH_SIZE:
                agents_num = np.random.randint(AGENTS_RANGE[0], AGENTS_RANGE[1] + 1)
                tasks_num = np.random.randint(TASKS_RANGE[0], TASKS_RANGE[1] + 1)
                rollouts = copy.copy(experience_buffer)
                for i in range(len(rollouts)):
                    rollouts[i] = rollouts[i][:BATCH_SIZE]
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]
                if len(experience_buffer[0]) < BATCH_SIZE:
                    update_done = True
                if update_done:
                    experience_buffer = [[] for _ in range(9)]

                agent_inputs = torch.stack(rollouts[0], dim=0)
                task_inputs = torch.stack(rollouts[1], dim=0)
                action_batch = torch.stack(rollouts[2], dim=0)
                mask_batch = torch.stack(rollouts[3], dim=0)
                advantage_batch = torch.stack(rollouts[6], dim=0)
                reward_batch = torch.stack(rollouts[4], dim=0)

                if device != local_device:
                    agent_inputs = agent_inputs.to(device)
                    task_inputs = task_inputs.to(device)
                    action_batch = action_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    advantage_batch = advantage_batch.to(device)

                logp_list = global_network(task_inputs, agent_inputs, mask_batch)
                logp = torch.gather(logp_list, 1, action_batch)
                entropy = (logp_list * logp_list.exp()).nansum(dim=-1).mean()
                policy_loss = (-logp * advantage_batch.detach()).mean()

                global_optimizer.zero_grad()
                policy_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=10, norm_type=2)
                global_optimizer.step()
                lr_decay.step()

                perf_data = [np.nanmean(perf_metrics[n]) for n in metric_name]
                data = [
                    reward_batch.mean().item(),
                    0,
                    policy_loss.item(),
                    entropy.item(),
                    grad_norm.item(),
                    *perf_data,
                ]
                trainingData.append(data)

            for _, meta_agent in enumerate(meta_agents):
                jobList.append(meta_agent.job.remote(weights, baseline_weights, curr_episode, agents_num, tasks_num))
                curr_episode += 1

            if len(trainingData) >= SUMMARY_WINDOW:
                summary = writeToTensorBoard(writer, trainingData, curr_episode)
                lr_now = global_optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(
                    '[TRAIN] ep=%d lr=%.8f reward=%.4f success=%.4f makespan=%.4f time=%.4f wait=%.4f dist=%.4f '
                    'util_exec=%.4f util_wait=%.4f util_travel=%.4f eff=%.4f policy=%.4f entropy=%.4f grad=%.4f',
                    curr_episode,
                    lr_now,
                    summary['reward'],
                    summary['success_rate'],
                    summary['makespan'],
                    summary['time_cost'],
                    summary['waiting_time'],
                    summary['travel_dist'],
                    summary['utilization_exec'],
                    summary['utilization_wait'],
                    summary['utilization_travel'],
                    summary['efficiency'],
                    summary['policy_loss'],
                    summary['entropy'],
                    summary['grad_norm'],
                )
                append_jsonl(train_metrics_jsonl, {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'episode': curr_episode,
                    'lr': float(lr_now),
                    **summary,
                })
                trainingData = []

            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()

            if curr_episode % 512 == 0:
                checkpoint = {
                    'model': global_network.state_dict(),
                    'optimizer': global_optimizer.state_dict(),
                    'episode': curr_episode,
                    'lr_decay': lr_decay.state_dict(),
                    'level': curr_level,
                    'best_perf': best_perf,
                    'run_name': run_name,
                }
                ckpt_path = save_checkpoint(checkpoint, curr_episode, best=False)
                logger.info('[CHECKPOINT] saved %s', ckpt_path)

            if EVALUATE and curr_episode % 1024 == 0:
                ray.wait(jobList, num_returns=NUM_META_AGENT)
                for actor in meta_agents:
                    ray.kill(actor)
                torch.cuda.empty_cache()

                logger.info('[EVAL] start ep=%d', curr_episode)

                if baseline_value is None:
                    test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(NUM_META_AGENT)]
                    for _, test_agent in enumerate(test_agent_list):
                        ray.get(test_agent.set_baseline_weights.remote(baseline_weights))
                    rewards = []
                    for i in range(256 // NUM_META_AGENT):
                        sample_job_list = []
                        for j, test_agent in enumerate(test_agent_list):
                            sample_job_list.append(test_agent.testing.remote(seed=test_set[i][j]))
                        sample_done_id, _ = ray.wait(sample_job_list, num_returns=NUM_META_AGENT)
                        rewards += ray.get(sample_done_id)
                    baseline_value = np.stack(rewards)
                    for actor in test_agent_list:
                        ray.kill(actor)

                test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(NUM_META_AGENT)]
                for _, test_agent in enumerate(test_agent_list):
                    ray.get(test_agent.set_baseline_weights.remote(weights))
                rewards = []
                for i in range(256 // NUM_META_AGENT):
                    sample_job_list = []
                    for j, test_agent in enumerate(test_agent_list):
                        sample_job_list.append(test_agent.testing.remote(seed=test_set[i][j]))
                    sample_done_id, _ = ray.wait(sample_job_list, num_returns=NUM_META_AGENT)
                    rewards += ray.get(sample_done_id)
                test_value = np.stack(rewards)
                for actor in test_agent_list:
                    ray.kill(actor)

                meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

                test_mean = float(test_value.mean())
                baseline_mean = float(baseline_value.mean())
                p_value = None
                best_updated = False

                if test_mean > baseline_mean:
                    _, p = ttest_rel(test_value, baseline_value)
                    p_value = float(p)
                    if p < 0.05:
                        if device != local_device:
                            weights = global_network.to(local_device).state_dict()
                            global_network.to(device)
                        else:
                            weights = global_network.state_dict()
                        baseline_weights = copy.deepcopy(weights)
                        baseline_network.load_state_dict(baseline_weights)
                        test_set = np.random.randint(low=0, high=1e8, size=[256 // NUM_META_AGENT, NUM_META_AGENT])
                        baseline_value = None
                        best_perf = test_mean
                        checkpoint = {
                            'model': global_network.state_dict(),
                            'optimizer': global_optimizer.state_dict(),
                            'episode': curr_episode,
                            'lr_decay': lr_decay.state_dict(),
                            'best_perf': best_perf,
                            'run_name': run_name,
                        }
                        best_path = save_checkpoint(checkpoint, curr_episode, best=True)
                        logger.info('[BEST] updated best model %s', best_path)
                        best_updated = True

                logger.info(
                    '[EVAL] ep=%d test_mean=%.6f baseline_mean=%.6f p_value=%s best_updated=%s',
                    curr_episode,
                    test_mean,
                    baseline_mean,
                    f'{p_value:.6f}' if p_value is not None else 'NA',
                    best_updated,
                )
                append_jsonl(eval_metrics_jsonl, {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'episode': curr_episode,
                    'test_mean': test_mean,
                    'baseline_mean': baseline_mean,
                    'p_value': p_value,
                    'best_updated': best_updated,
                    'best_perf': float(best_perf),
                })

                jobList = []
                for _, meta_agent in enumerate(meta_agents):
                    jobList.append(meta_agent.job.remote(weights, baseline_weights, curr_episode, agents_num, tasks_num))
                    curr_episode += 1

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
