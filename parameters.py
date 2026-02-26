import os

# =============================================================================
# Runtime / Hardware
# =============================================================================
USE_GPU = True
USE_GPU_GLOBAL = True
NUM_GPU = 1
NUM_META_AGENT = 8 #number of parallel rollout actors.

# =============================================================================
# Core Training Schedule
# =============================================================================
# ALGO: algorithm switch.
# Supported values: 'reinforce', 'ppo'
ALGO = os.getenv('DCMRTA_ALGO', 'reinforce').strip().lower()
LR = 3e-5 # learning rate
GAMMA = 0.99 # discount factor.
DECAY_STEP = 2e3 # StepLR decay interval (in optimizer updates)
# RESET_OPT: reserved flag for optimizer reset behavior when resuming.
RESET_OPT = False
EVALUATE = True
# CURRICULUM_LEARNING / INCREASE_DIFFICULTY: reserved curriculum hooks.
CURRICULUM_LEARNING = False
INCREASE_DIFFICULTY = 2000
# SUMMARY_WINDOW: how many update rows are merged before one log/tensorboard emit.
SUMMARY_WINDOW = 8
# DEMON_RATE / IL_DECAY: legacy knobs kept for compatibility with previous runs.
DEMON_RATE = 0.5
IL_DECAY = -1e-5  # tuning note: -1e-6/700k, -1e-5/70k, -1e-4/7k

# =============================================================================
# Problem Scale / Episode Horizon
# =============================================================================
# AGENTS_RANGE / TASKS_RANGE: instance size range used by random sampling.
AGENTS_RANGE = (10, 20)
TASKS_RANGE = (20, 50)
COALITION_SIZE = 5 # max initial requirement level sampled in env generation.
MAX_TIME = 100 # hard episode time horizon.
EVAL_MAX_WAITING_TIME = 10 # waiting timeout used in baseline replay/evaluation
TRAIT_DIM = 1 # capability dimension (homogeneous setup uses 1)

# =============================================================================
# Task Dynamics (v0.1 model)
# =============================================================================
# TASK_ALPHA: base task execution coefficient.
TASK_ALPHA = 1.0
# COALITION_BETA: diminishing-return parameter in g(n)=n/(1+beta*(n-1)).
COALITION_BETA = 0.8
# MODE_COST_TYPE: team-size cost function h(n).
MODE_COST_TYPE = 'linear'  # options: 'linear' or 'quadratic'

# =============================================================================
# Reward Weights
# =============================================================================
# Weighted objective ratio:
# makespan : travel : wait : mode = 1.0 : 0.05 : 0.1 : 0.05
REWARD_W_MAKESPAN = 1.0
REWARD_W_TRAVEL = 0.05
REWARD_W_WAIT = 0.1
REWARD_W_MODE = 0.05

# =============================================================================
# Anti-Oscillation Constraints + Event Reward (v0.3)
# =============================================================================
# SIMPLIFIED_SETTING: v1.0 simplified execution switch.
# Supported values: 'growth_only_strict', 'growth_only_relaxed'
SIMPLIFIED_SETTING = os.getenv('DCMRTA_SIMPLIFIED_SETTING', 'growth_only_strict').strip().lower()
# Commit lock: once assigned, agent remains committed for MIN_COMMIT_TIME.
ENABLE_COMMIT_LOCK = True
MIN_COMMIT_TIME = 2.0
# Quorum protect: prevents harmful departures that break minimum required team size.
ENABLE_QUORUM_PROTECT = True
# Event penalties: discourage unstable behavior.
SWITCH_PENALTY = 0.15
QUORUM_BREAK_PENALTY = 0.5
PAUSE_EVENT_PENALTY = 0.2
# Dense event reward / potential shaping toggles.
USE_DENSE_EVENT_REWARD = True
USE_POTENTIAL_SHAPING = True
POTENTIAL_SHAPING_COEF = 1.0

# =============================================================================
# Online Dispatch Interface Reservation (v0.4)
# =============================================================================
# ENABLE_ONLINE_DISPATCH: enable lower-level dispatcher hook in env.step().
ENABLE_ONLINE_DISPATCH = False
# ONLINE_DISPATCH_POLICY: currently implemented 'baseline_random'.
ONLINE_DISPATCH_POLICY = 'baseline_random'

# =============================================================================
# Critic / GAE / Entropy
# =============================================================================
USE_CRITIC = True
GAE_LAMBDA = 0.95
VALUE_COEF = 0.5 # critic loss weight in total loss.
# Entropy schedule: linearly decays from START to END over DECAY_STEPS updates.
ENTROPY_COEF_START = 0.02
ENTROPY_COEF_END = 0.002
ENTROPY_DECAY_STEPS = 300000
MAX_GRAD_NORM = 1.0 # global gradient clipping threshold
CRITIC_INPUT_DIM = 16
CRITIC_HIDDEN_DIM = 128

# =============================================================================
# PPO Hyperparameters (active only when ALGO='ppo')
# =============================================================================
# PPO_EPOCHS: number of passes over one sampled rollout batch.
PPO_EPOCHS = int(os.getenv('DCMRTA_PPO_EPOCHS', '4'))
# PPO_MINIBATCH_SIZE: mini-batch size inside each PPO epoch.
PPO_MINIBATCH_SIZE = int(os.getenv('DCMRTA_PPO_MINIBATCH_SIZE', '256'))
# PPO_CLIP_EPS: clip epsilon for policy ratio.
PPO_CLIP_EPS = float(os.getenv('DCMRTA_PPO_CLIP_EPS', '0.15'))
# PPO_VALUE_CLIP_EPS: optional clip epsilon for value updates.
PPO_VALUE_CLIP_EPS = float(os.getenv('DCMRTA_PPO_VALUE_CLIP_EPS', '0.2'))
# PPO_TARGET_KL: early-stop threshold when approximate KL becomes too large.
PPO_TARGET_KL = float(os.getenv('DCMRTA_PPO_TARGET_KL', '0.01'))

# ===============================
# Paths / Runtime IO
# ===============================
FOLDER_NAME = 'REINFORCE'
# Paths are usually overridden by scripts/train.sh via environment variables.
model_path = os.getenv('DCMRTA_MODEL_PATH', f'model/{FOLDER_NAME}')
train_path = os.getenv('DCMRTA_TRAIN_PATH', f'train/{FOLDER_NAME}')
gifs_path = os.getenv('DCMRTA_GIFS_PATH', f'gifs/{FOLDER_NAME}')
# LOAD_MODEL: whether to restore checkpoints at startup.
LOAD_MODEL = False
# SAVE_IMG/SAVE_IMG_GAP: GIF generation controls.
SAVE_IMG = True
SAVE_IMG_GAP = 10000
# WANDB_LOG: external experiment tracking toggle.
WANDB_LOG = False

# =============================================================================
# Batch / Model Shape
# =============================================================================
# BATCH_SIZE: number of transitions collected before one learner update trigger.
BATCH_SIZE = 1024
# Input and embedding dimensions (must stay aligned with env feature builders).
AGENT_INPUT_DIM = 6
TASK_INPUT_DIM = 5
EMBEDDING_DIM = 128
# SAMPLE_SIZE / PADDING_SIZE: legacy compatibility parameters.
SAMPLE_SIZE = 200
PADDING_SIZE = 50
