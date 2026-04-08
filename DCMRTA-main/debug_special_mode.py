import copy
from env.task_env import TaskEnv

env = TaskEnv(
    agents_range=(10, 10),
    tasks_range=(20, 20),
    traits_dim=1,
    max_coalition_size=7,
    max_duration=100,
    seed=0,
    enable_special_modes=True,
    num_special_tasks=1,
    mode_size_low=2,
    special_time_range=(80, 140),
    special_speedup_range=(10, 30),
)

print("===== Special modes sampled =====")
for item in env.describe_special_modes():
    print(item)

env_small = copy.deepcopy(env)
env_large = copy.deepcopy(env)

env_small.force_special_mode('small_mode')
env_large.force_special_mode('large_mode')

print("===== Forced small =====")
for item in env_small.describe_special_modes():
    print(item)

print("===== Forced large =====")
for item in env_large.describe_special_modes():
    print(item)