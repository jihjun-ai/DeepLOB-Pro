"""基礎 PPO 訓練腳本 - 使用 MlpPolicy

此腳本用於快速測試 SB3 訓練管線，使用簡單的 MlpPolicy（不整合 DeepLOB）。

功能：
    1. 載入台股交易環境
    2. 創建 PPO 模型（MlpPolicy）
    3. 設置訓練回調（Checkpoint, Eval, TensorBoard）
    4. 執行訓練
    5. 保存最終模型

使用範例：
    # 快速測試（10K steps，5-10 分鐘）
    python scripts/train_sb3.py --timesteps 10000 --test

    # 完整訓練（500K steps，2-4 小時）
    python scripts/train_sb3.py --timesteps 500000

    # 自定義配置
    python scripts/train_sb3.py --config configs/sb3_config.yaml --timesteps 1000000

    # 監控訓練
    tensorboard --logdir logs/sb3/

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
import yaml
import logging
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor

from src.envs.tw_lob_trading_env import TaiwanLOBTradingEnv

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def make_env(env_config: dict, rank: int = 0):
    """創建環境工廠函數（用於向量化環境）"""
    def _init():
        env = TaiwanLOBTradingEnv(env_config)
        env = Monitor(env)  # 包裝 Monitor 用於記錄
        return env
    return _init


def create_vec_env(config: dict, n_envs: int = 1, vec_type: str = "dummy"):
    """創建向量化環境"""
    env_config = config['env_config']

    if n_envs == 1:
        # 單環境
        env = TaiwanLOBTradingEnv(env_config)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        logger.info("✅ 創建單一環境")
    else:
        # 多環境並行
        env_fns = [make_env(env_config, i) for i in range(n_envs)]

        if vec_type == "subproc":
            env = SubprocVecEnv(env_fns)
            logger.info(f"✅ 創建 SubprocVecEnv ({n_envs} 個環境)")
        else:
            env = DummyVecEnv(env_fns)
            logger.info(f"✅ 創建 DummyVecEnv ({n_envs} 個環境)")

    return env


def create_eval_env(config: dict):
    """創建評估環境（使用驗證集）"""
    eval_config = config.get('evaluation', {}).get('eval_env_config', {})

    # 合併環境配置和評估配置
    env_config = config['env_config'].copy()
    env_config.update(eval_config)

    env = TaiwanLOBTradingEnv(env_config)
    env = Monitor(env)
    logger.info("✅ 創建評估環境（驗證集）")

    return env


def create_callbacks(config: dict, eval_env):
    """創建訓練回調"""
    callbacks = []

    # Checkpoint Callback
    checkpoint_config = config.get('callbacks', {}).get('checkpoint', {})
    if checkpoint_config.get('enabled', True):
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_config.get('save_freq', 50000),
            save_path=checkpoint_config.get('save_path', 'checkpoints/sb3/ppo_basic'),
            name_prefix=checkpoint_config.get('name_prefix', 'ppo_model'),
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)
        logger.info(f"✅ Checkpoint Callback (每 {checkpoint_config.get('save_freq', 50000)} steps)")

    # Eval Callback
    eval_config = config.get('callbacks', {}).get('eval', {})
    if eval_config.get('enabled', True) and eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=eval_config.get('eval_freq', 10000),
            n_eval_episodes=eval_config.get('n_eval_episodes', 10),
            best_model_save_path=eval_config.get('best_model_save_path', 'checkpoints/sb3/ppo_basic'),
            log_path=eval_config.get('log_path', 'logs/sb3_eval'),
            deterministic=eval_config.get('deterministic', True),
            render=False,
        )
        callbacks.append(eval_callback)
        logger.info(f"✅ Eval Callback (每 {eval_config.get('eval_freq', 10000)} steps)")

    if len(callbacks) > 0:
        return CallbackList(callbacks)
    else:
        return None


def create_ppo_model(env, config: dict, device: str = "cuda"):
    """創建 PPO 模型（基礎 MlpPolicy）"""
    ppo_config = config.get('ppo', {})

    # 網絡架構
    net_arch = ppo_config.get('net_arch', dict(pi=[256, 128], vf=[256, 128]))

    # 創建模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=config.get('training', {}).get('tensorboard_log', 'logs/sb3'),
        verbose=ppo_config.get('verbose', 1),
        seed=ppo_config.get('seed', 42),
        device=device
    )

    logger.info("✅ PPO 模型創建成功")
    logger.info(f"  - Policy: MlpPolicy")
    logger.info(f"  - Learning Rate: {ppo_config.get('learning_rate', 3e-4)}")
    logger.info(f"  - Gamma: {ppo_config.get('gamma', 0.99)}")
    logger.info(f"  - N Steps: {ppo_config.get('n_steps', 2048)}")
    logger.info(f"  - Batch Size: {ppo_config.get('batch_size', 64)}")
    logger.info(f"  - Device: {device}")

    return model


def train_model(model, total_timesteps: int, callbacks=None, log_interval: int = 10):
    """訓練模型"""
    logger.info("=" * 60)
    logger.info("🚀 開始訓練")
    logger.info("=" * 60)
    logger.info(f"總步數: {total_timesteps:,}")

    start_time = datetime.now()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
        progress_bar=True
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("✅ 訓練完成")
    logger.info("=" * 60)
    logger.info(f"訓練時間: {duration/60:.2f} 分鐘")
    logger.info(f"訓練速度: {total_timesteps/duration:.1f} steps/sec")

    return model


def save_final_model(model, config: dict):
    """保存最終模型"""
    save_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints/sb3/ppo_basic')
    model_name = config.get('training', {}).get('final_model_name', 'ppo_basic_final')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    model.save(save_path)
    logger.info(f"✅ 最終模型已保存: {save_path}.zip")

    return save_path


def main():
    parser = argparse.ArgumentParser(description='基礎 PPO 訓練（MlpPolicy）')
    parser.add_argument('--config', type=str, default='configs/sb3_config.yaml',
                      help='配置文件路徑')
    parser.add_argument('--timesteps', type=int, default=None,
                      help='訓練步數（覆蓋配置文件）')
    parser.add_argument('--device', type=str, default='cuda',
                      help='設備: cuda / cpu')
    parser.add_argument('--test', action='store_true',
                      help='測試模式（使用 test_mode 配置）')
    parser.add_argument('--n-envs', type=int, default=1,
                      help='並行環境數量')
    parser.add_argument('--vec-type', type=str, default='dummy',
                      help='向量化類型: dummy / subproc')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 基礎 PPO 訓練腳本")
    logger.info("=" * 60)

    # 檢查 CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️  CUDA 不可用，改用 CPU")
        args.device = 'cpu'
    else:
        logger.info(f"✅ 使用設備: {args.device}")
        if args.device == 'cuda':
            logger.info(f"  - GPU: {torch.cuda.get_device_name(0)}")

    try:
        # 1. 載入配置
        logger.info(f"\n📄 載入配置: {args.config}")
        config = load_config(args.config)

        # 測試模式
        if args.test:
            logger.info("🧪 測試模式啟用")
            test_config = config.get('test_mode', {})
            config['training']['total_timesteps'] = test_config.get('total_timesteps', 10000)
            config['callbacks']['checkpoint']['save_freq'] = test_config.get('save_freq', 5000)
            config['callbacks']['eval']['eval_freq'] = test_config.get('eval_freq', 5000)
            config['ppo']['n_steps'] = test_config.get('n_steps', 512)
            config['ppo']['batch_size'] = test_config.get('batch_size', 32)

        # 命令行參數覆蓋
        if args.timesteps is not None:
            config['training']['total_timesteps'] = args.timesteps

        total_timesteps = config['training']['total_timesteps']
        logger.info(f"✅ 配置載入成功（訓練 {total_timesteps:,} steps）")

        # 2. 創建訓練環境
        logger.info("\n🏗️  創建訓練環境")
        env = create_vec_env(config, n_envs=args.n_envs, vec_type=args.vec_type)

        # 3. 創建評估環境
        logger.info("\n🏗️  創建評估環境")
        eval_env = create_eval_env(config)

        # 4. 創建回調
        logger.info("\n🔔 設置訓練回調")
        callbacks = create_callbacks(config, eval_env)

        # 5. 創建 PPO 模型
        logger.info("\n🤖 創建 PPO 模型")
        model = create_ppo_model(env, config, device=args.device)

        # 6. 開始訓練
        model = train_model(
            model,
            total_timesteps=total_timesteps,
            callbacks=callbacks,
            log_interval=config.get('training', {}).get('log_interval', 10)
        )

        # 7. 保存最終模型
        logger.info("\n💾 保存最終模型")
        save_path = save_final_model(model, config)

        # 8. 總結
        logger.info("\n" + "=" * 60)
        logger.info("🎉 訓練流程完成")
        logger.info("=" * 60)
        logger.info(f"✅ 最終模型: {save_path}.zip")
        logger.info(f"✅ 日誌目錄: {config.get('training', {}).get('tensorboard_log', 'logs/sb3')}")
        logger.info("\n下一步:")
        logger.info(f"  1. 查看訓練日誌: tensorboard --logdir {config.get('training', {}).get('tensorboard_log', 'logs/sb3')}")
        logger.info(f"  2. 評估模型: python scripts/evaluate_sb3.py --model {save_path}")
        logger.info(f"  3. 運行完整訓練: python scripts/train_sb3_deeplob.py --timesteps 1000000")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("❌ 訓練失敗")
        logger.error("=" * 60)
        logger.error(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
