"""環境驗證腳本 - 驗證 TaiwanLOBTradingEnv 符合 Gymnasium 標準

此腳本用於驗證台股交易環境是否正確實作並符合 Stable-Baselines3 的要求。

驗證項目：
    1. 環境創建成功
    2. 觀測空間和動作空間正確
    3. reset() 方法正常
    4. step() 方法正常
    5. 完整 episode 運行成功
    6. SB3 check_env() 驗證通過

使用範例：
    python scripts/verify_env.py
    python scripts/verify_env.py --config configs/sb3_config.yaml
    python scripts/verify_env.py --episodes 5

作者: SB3-DeepLOB 專案團隊
日期: 2025-10-24
版本: v1.0
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
import yaml
import logging

from src.envs.tw_lob_trading_env import TaiwanLOBTradingEnv

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """載入配置文件"""
    if config_path is None:
        # 使用預設配置
        return {
            'data_dir': 'data/processed_v7/npz',
            'max_steps': 500,
            'initial_balance': 10000.0,
            'transaction_cost_rate': 0.001,
            'max_position': 1,
            'deeplob_checkpoint': 'checkpoints/v5/deeplob_v5_best.pth',
            'reward_config': {
                'pnl_scale': 1.0,
                'cost_penalty': 1.0,
                'inventory_penalty': 0.01,
                'risk_penalty': 0.005
            },
            'data_mode': 'train'
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config.get('env_config', config)


def verify_env_creation(config: dict) -> TaiwanLOBTradingEnv:
    """驗證環境創建"""
    logger.info("=" * 60)
    logger.info("1️⃣  驗證環境創建")
    logger.info("=" * 60)

    try:
        env = TaiwanLOBTradingEnv(config)
        logger.info("✅ 環境創建成功")
        return env
    except Exception as e:
        logger.error(f"❌ 環境創建失敗: {e}")
        raise


def verify_spaces(env: TaiwanLOBTradingEnv):
    """驗證觀測和動作空間"""
    logger.info("\n" + "=" * 60)
    logger.info("2️⃣  驗證觀測和動作空間")
    logger.info("=" * 60)

    # 檢查觀測空間
    obs_space = env.observation_space
    logger.info(f"觀測空間: {obs_space}")
    logger.info(f"  - 類型: {type(obs_space)}")
    logger.info(f"  - 形狀: {obs_space.shape}")
    logger.info(f"  - 數據類型: {obs_space.dtype}")

    expected_obs_dim = 28  # LOB(20) + DeepLOB(3) + State(5)
    if obs_space.shape[0] == expected_obs_dim:
        logger.info(f"✅ 觀測維度正確 ({expected_obs_dim})")
    else:
        logger.error(f"❌ 觀測維度錯誤: 預期 {expected_obs_dim}, 實際 {obs_space.shape[0]}")
        raise ValueError(f"觀測維度不匹配")

    # 檢查動作空間
    action_space = env.action_space
    logger.info(f"\n動作空間: {action_space}")
    logger.info(f"  - 類型: {type(action_space)}")
    logger.info(f"  - 動作數量: {action_space.n}")

    if action_space.n == 3:
        logger.info("✅ 動作空間正確 (Discrete(3): Hold, Buy, Sell)")
    else:
        logger.error(f"❌ 動作空間錯誤: 預期 3, 實際 {action_space.n}")
        raise ValueError("動作空間不匹配")


def verify_reset(env: TaiwanLOBTradingEnv):
    """驗證 reset() 方法"""
    logger.info("\n" + "=" * 60)
    logger.info("3️⃣  驗證 reset() 方法")
    logger.info("=" * 60)

    try:
        obs, info = env.reset()
        logger.info(f"✅ reset() 執行成功")
        logger.info(f"  - 觀測形狀: {obs.shape}")
        logger.info(f"  - 觀測類型: {type(obs)}")
        logger.info(f"  - 觀測範圍: [{obs.min():.4f}, {obs.max():.4f}]")
        logger.info(f"  - Info 鍵: {list(info.keys())}")

        # 驗證觀測格式
        if obs.shape != (28,):
            logger.error(f"❌ reset() 返回的觀測形狀錯誤: {obs.shape}")
            raise ValueError("reset() 觀測形狀不匹配")

        if not isinstance(obs, np.ndarray):
            logger.error(f"❌ reset() 返回的觀測類型錯誤: {type(obs)}")
            raise ValueError("reset() 觀測類型不是 numpy array")

        return obs, info

    except Exception as e:
        logger.error(f"❌ reset() 執行失敗: {e}")
        raise


def verify_step(env: TaiwanLOBTradingEnv):
    """驗證 step() 方法"""
    logger.info("\n" + "=" * 60)
    logger.info("4️⃣  驗證 step() 方法")
    logger.info("=" * 60)

    try:
        env.reset()

        # 測試所有動作
        actions = [0, 1, 2]  # Hold, Buy, Sell
        action_names = ['Hold', 'Buy', 'Sell']

        for action, name in zip(actions, action_names):
            obs, reward, terminated, truncated, info = env.step(action)

            logger.info(f"\n動作 {action} ({name}):")
            logger.info(f"  - 觀測形狀: {obs.shape}")
            logger.info(f"  - 獎勵: {reward:.6f}")
            logger.info(f"  - Terminated: {terminated}")
            logger.info(f"  - Truncated: {truncated}")
            logger.info(f"  - Info 鍵: {list(info.keys())}")

            # 驗證返回值
            if obs.shape != (28,):
                logger.error(f"❌ step() 返回的觀測形狀錯誤: {obs.shape}")
                raise ValueError("step() 觀測形狀不匹配")

            if not isinstance(reward, (int, float, np.number)):
                logger.error(f"❌ step() 返回的獎勵類型錯誤: {type(reward)}")
                raise ValueError("step() 獎勵類型錯誤")

        logger.info("\n✅ step() 執行成功（所有動作）")

    except Exception as e:
        logger.error(f"❌ step() 執行失敗: {e}")
        raise


def verify_full_episode(env: TaiwanLOBTradingEnv, n_episodes: int = 3):
    """驗證完整 episode 運行"""
    logger.info("\n" + "=" * 60)
    logger.info(f"5️⃣  驗證完整 Episode 運行 (n={n_episodes})")
    logger.info("=" * 60)

    episode_rewards = []
    episode_lengths = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0

            while True:
                # 隨機動作
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

            logger.info(f"  Episode {ep+1}: 獎勵={episode_reward:.4f}, 步數={steps}")

        logger.info(f"\n✅ 完整 Episode 運行成功")
        logger.info(f"  - 平均獎勵: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
        logger.info(f"  - 平均步數: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    except Exception as e:
        logger.error(f"❌ Episode 運行失敗: {e}")
        raise


def verify_sb3_compatibility(env: TaiwanLOBTradingEnv):
    """驗證 SB3 兼容性"""
    logger.info("\n" + "=" * 60)
    logger.info("6️⃣  驗證 Stable-Baselines3 兼容性")
    logger.info("=" * 60)

    try:
        check_env(env, warn=True)
        logger.info("✅ SB3 check_env() 驗證通過")
    except Exception as e:
        logger.error(f"❌ SB3 check_env() 驗證失敗: {e}")
        raise


def verify_gpu_availability():
    """驗證 GPU 可用性"""
    logger.info("\n" + "=" * 60)
    logger.info("7️⃣  驗證 GPU 可用性")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        logger.info("✅ CUDA 可用")
        logger.info(f"  - GPU 數量: {torch.cuda.device_count()}")
        logger.info(f"  - 當前設備: {torch.cuda.current_device()}")
        logger.info(f"  - GPU 名稱: {torch.cuda.get_device_name(0)}")

        # 顯示顯存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  - 總顯存: {total_memory:.2f} GB")
    else:
        logger.warning("⚠️  CUDA 不可用，將使用 CPU 訓練")


def main():
    parser = argparse.ArgumentParser(description='驗證台股交易環境')
    parser.add_argument('--config', type=str, default=None,
                      help='配置文件路徑 (預設使用內建配置)')
    parser.add_argument('--episodes', type=int, default=3,
                      help='測試 episode 數量 (預設 3)')
    parser.add_argument('--no-sb3-check', action='store_true',
                      help='跳過 SB3 check_env() 驗證')

    args = parser.parse_args()

    logger.info("🚀 開始環境驗證")
    logger.info("=" * 60)

    try:
        # 載入配置
        config = load_config(args.config)
        logger.info(f"配置來源: {'內建預設' if args.config is None else args.config}")

        # 1. 創建環境
        env = verify_env_creation(config)

        # 2. 驗證空間
        verify_spaces(env)

        # 3. 驗證 reset
        verify_reset(env)

        # 4. 驗證 step
        verify_step(env)

        # 5. 驗證完整 episode
        verify_full_episode(env, n_episodes=args.episodes)

        # 6. 驗證 SB3 兼容性
        if not args.no_sb3_check:
            verify_sb3_compatibility(env)

        # 7. 驗證 GPU
        verify_gpu_availability()

        # 總結
        logger.info("\n" + "=" * 60)
        logger.info("🎉 所有驗證通過！")
        logger.info("=" * 60)
        logger.info("✅ 環境已準備好用於 Stable-Baselines3 訓練")
        logger.info("\n下一步:")
        logger.info("  1. 運行基礎訓練: python scripts/train_sb3.py --timesteps 10000 --test")
        logger.info("  2. 運行完整訓練: python scripts/train_sb3_deeplob.py --timesteps 1000000")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("❌ 驗證失敗")
        logger.error("=" * 60)
        logger.error(f"錯誤: {e}")
        logger.error("\n請檢查:")
        logger.error("  1. 數據文件是否存在於 data/processed_v7/npz/")
        logger.error("  2. DeepLOB 檢查點是否存在於 checkpoints/v5/deeplob_v5_best.pth")
        logger.error("  3. 環境配置是否正確")
        return 1


if __name__ == "__main__":
    exit(main())
