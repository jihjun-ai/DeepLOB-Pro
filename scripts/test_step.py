import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.envs.tw_lob_trading_env import TaiwanLOBTradingEnv
import logging

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
print(f'Obs shape: {obs.shape}')

# Test step
obs, reward, terminated, truncated, info = env.step(0)
print('Step 1 OK')
print(f'Reward: {reward:.4f}')
print(f'Obs shape: {obs.shape}')

# Test multiple steps
for i in range(2, 6):
    obs, reward, terminated, truncated, info = env.step(1)
    print(f'Step {i} OK, Reward: {reward:.4f}')

print('\nAll tests passed!')
