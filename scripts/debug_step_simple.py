import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.envs.tw_lob_trading_env import TaiwanLOBTradingEnv
import traceback
import logging
import numpy as np

logging.basicConfig(level=logging.ERROR)

config = {
    'data_dir': 'data/processed_v7/npz',
    'sample_ratio': 0.1,
    'max_steps': 500,
    'data_mode': 'train'
}

env = TaiwanLOBTradingEnv(config)
obs, info = env.reset()
print('Reset OK')

# 手動執行 _get_observation 邏輯
print('\n--- Manually constructing observation ---')
current_lob = np.array(env.lob_history[-1], dtype=np.float32)
print(f'current_lob: {current_lob.shape}')

print(f'env.deeplob_model is None: {env.deeplob_model is None}')

# 檢查 state_features 中每個值的類型
val_list = [
    env.position / env.max_position if env.max_position > 0 else 0.0,
    env.inventory / env.initial_balance,
    env.total_cost / env.initial_balance,
    env.current_step / env.max_steps,
    env.prev_action / 2.0
]

for i, val in enumerate(val_list):
    print(f'val[{i}]: {val} (type: {type(val)}, shape: {getattr(val, "shape", "N/A")})')

print('\n--- Trying to create state_features array ---')
try:
    state_features = np.array(val_list, dtype=np.float32)
    print(f'Success! state_features shape: {state_features.shape}')
except Exception as e:
    print(f'Failed! Error: {e}')
    traceback.print_exc()
