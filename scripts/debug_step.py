import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.envs.tw_lob_trading_env import TaiwanLOBTradingEnv
import traceback
import logging

logging.basicConfig(level=logging.ERROR)  # 只顯示 ERROR

config = {
    'data_dir': 'data/processed_v7/npz',
    'sample_ratio': 0.1,
    'max_steps': 500,
    'data_mode': 'train'
}

env = TaiwanLOBTradingEnv(config)
obs, info = env.reset()
print('Reset OK')
print(f'Obs shape: {obs.shape}')
print(f'Obs dtype: {obs.dtype}')

# 調試 step 內部
import numpy as np
import torch

try:
    # 模擬 step 的觀測構建
    current_lob = np.array(env.lob_history[-1], dtype=np.float32)
    print(f'\ncurrent_lob shape: {current_lob.shape}')

    if env.deeplob_model is not None:
        with torch.no_grad():
            lob_seq = torch.FloatTensor(env.lob_history).unsqueeze(0)
            print(f'lob_seq shape: {lob_seq.shape}')
            deeplob_probs = env.deeplob_model.predict_proba(lob_seq)[0].numpy()
            print(f'deeplob_probs shape: {deeplob_probs.shape}')
            print(f'deeplob_probs dtype: {deeplob_probs.dtype}')
            print(f'deeplob_probs: {deeplob_probs}')

    # 逐個檢查 state 值
    print(f'\nenv.position: {env.position} (type: {type(env.position)})')
    print(f'env.max_position: {env.max_position} (type: {type(env.max_position)})')
    print(f'env.inventory: {env.inventory} (type: {type(env.inventory)})')
    print(f'env.initial_balance: {env.initial_balance} (type: {type(env.initial_balance)})')
    print(f'env.total_cost: {env.total_cost} (type: {type(env.total_cost)})')
    print(f'env.current_step: {env.current_step} (type: {type(env.current_step)})')
    print(f'env.max_steps: {env.max_steps} (type: {type(env.max_steps)})')
    print(f'env.prev_action: {env.prev_action} (type: {type(env.prev_action)})')

    val1 = env.position / env.max_position if env.max_position > 0 else 0.0
    val2 = env.inventory / env.initial_balance
    val3 = env.total_cost / env.initial_balance
    val4 = env.current_step / env.max_steps
    val5 = env.prev_action / 2.0

    print(f'\nval1: {val1} (type: {type(val1)})')
    print(f'val2: {val2} (type: {type(val2)})')
    print(f'val3: {val3} (type: {type(val3)})')
    print(f'val4: {val4} (type: {type(val4)})')
    print(f'val5: {val5} (type: {type(val5)})')

    state_features = np.array([val1, val2, val3, val4, val5], dtype=np.float32)
    print(f'\nstate_features shape: {state_features.shape}')
    print(f'state_features dtype: {state_features.dtype}')

    # 嘗試串接
    obs = np.concatenate([current_lob, deeplob_probs, state_features])
    print(f'\nConcatenated obs shape: {obs.shape}')
    print('Observation constructed successfully!')

except Exception as e:
    print(f'\nConstruction Error: {e}')
    traceback.print_exc()

print('\n--- Now testing actual step() ---')
try:
    obs, reward, terminated, truncated, info = env.step(0)
    print('Step OK')
    print(f'Reward: {reward}')
    print(f'Terminated: {terminated}')
except Exception as e:
    print(f'Step Error: {e}')
    traceback.print_exc()
