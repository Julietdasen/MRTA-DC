import os

USE_GPU = True
USE_GPU_GLOBAL = True
NUM_GPU = 1
NUM_META_AGENT = 8
LR = 1e-5
GAMMA = 1
DECAY_STEP = 2e3
RESET_OPT = False
EVALUATE = True
CURRICULUM_LEARNING = False
INCREASE_DIFFICULTY = 2000
SUMMARY_WINDOW = 8
DEMON_RATE = 0.5
IL_DECAY = -1e-5  # -1e-6 700k decay 0.5, -1e-5 70k decay 0.5, -1e-4 7k decay 0.5
AGENTS_RANGE = (10, 20)
TASKS_RANGE = (20, 50)
COALITION_SIZE = 5
MAX_TIME = 100
TRAIT_DIM = 1

# v0.1 dynamic coalition/task settings
TASK_ALPHA = 1.0
# Fixed beta for the first implementation (no sweep yet).
COALITION_BETA = 0.8
MODE_COST_TYPE = 'linear'  # 'linear' or 'quadratic'

# v0.1 reward weights
# Reward ratio follows: makespan : travel : wait : mode = 1.0 : 0.05 : 0.1 : 0.05
REWARD_W_MAKESPAN = 1.0
REWARD_W_TRAVEL = 0.05
REWARD_W_WAIT = 0.1
REWARD_W_MODE = 0.05

FOLDER_NAME = 'REINFORCE'
model_path = os.getenv('DCMRTA_MODEL_PATH', f'model/{FOLDER_NAME}')
train_path = os.getenv('DCMRTA_TRAIN_PATH', f'train/{FOLDER_NAME}')
gifs_path = os.getenv('DCMRTA_GIFS_PATH', f'gifs/{FOLDER_NAME}')
LOAD_MODEL = False
SAVE_IMG = True
SAVE_IMG_GAP = 10000
WANDB_LOG = False

BATCH_SIZE = 1024
AGENT_INPUT_DIM = 6
TASK_INPUT_DIM = 5
EMBEDDING_DIM = 128
SAMPLE_SIZE = 200
PADDING_SIZE = 50
