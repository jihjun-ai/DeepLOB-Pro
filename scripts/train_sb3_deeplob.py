"""PPO + DeepLOB 完整訓練腳本 - 雙層學習架構

此腳本實現完整的 DeepLOB + SB3 PPO 整合訓練：
    第一層: DeepLOB 提取 LOB 深層特徵（凍結權重）
    第二層: PPO 學習最優交易策略

功能：
    1. 載入預訓練 DeepLOB 模型
    2. 創建帶 DeepLOB 特徵提取器的 PPO 模型
    3. 執行完整訓練（推薦 1M steps）
    4. 持續評估與保存最佳模型

使用範例：
    # 完整訓練（1M steps，推薦，4-8 小時 RTX 5090）
    python scripts/train_sb3_deeplob.py

    # 快速測試（10K steps，10 分鐘）
    python scripts/train_sb3_deeplob.py --test

    # 指定配置文件
    python scripts/train_sb3_deeplob.py --config configs/sb3_deeplob_config.yaml

    # 指定 DeepLOB 檢查點
    python scripts/train_sb3_deeplob.py \
        --deeplob-checkpoint checkpoints/v5/deeplob_v5_best.pth

    # 高性能訓練（大 batch size + 並行環境）
    python scripts/train_sb3_deeplob.py --n-envs 4 --device cuda

    # 監控訓練
    tensorboard --logdir logs/sb3_deeplob/

作者: SB3-DeepLOB 專案團隊
日期: 2025-10-26
版本: v2.0 (移除所有硬編碼，使用 YAMLManager)
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
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
from src.models.deeplob_feature_extractor import (
    DeepLOBExtractor,
    make_deeplob_policy_kwargs
)
from src.utils.yaml_manager import YAMLManager

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """載入配置文件（使用 YAMLManager）"""
    try:
        yaml_manager = YAMLManager(config_path)
        config = yaml_manager.as_dict()
        logger.info(f"✅ 配置載入成功: {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    except Exception as e:
        raise RuntimeError(f"配置載入失敗: {e}")


def verify_deeplob_checkpoint(checkpoint_path: str, config: dict):
    """驗證 DeepLOB 檢查點存在"""
    # 檢查是否需要驗證
    if not config.get('validation', {}).get('verify_checkpoint', True):
        logger.info("⏭️  跳過 DeepLOB 檢查點驗證")
        return

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"DeepLOB 檢查點不存在: {checkpoint_path}\n"
            f"請確認文件路徑或先訓練 DeepLOB 模型"
        )

    # 嘗試載入檢查點
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info(f"✅ DeepLOB 檢查點驗證成功: {checkpoint_path}")

        # 顯示檢查點信息
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                logger.info(f"  - Epoch: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                logger.info(f"  - 驗證準確率: {checkpoint['val_acc']:.4f}")
            if 'test_acc' in checkpoint:
                logger.info(f"  - 測試準確率: {checkpoint['test_acc']:.4f}")

    except Exception as e:
        raise RuntimeError(f"DeepLOB 檢查點載入失敗: {e}")


def make_env(env_config: dict, rank: int = 0):
    """創建環境工廠函數（用於向量化環境）"""
    def _init():
        env = TaiwanLOBTradingEnv(env_config)
        env = Monitor(env)
        return env
    return _init


def create_vec_env(config: dict, n_envs: int = None, vec_type: str = None):
    """創建向量化環境"""
    env_config = config['env_config']

    # 從配置讀取預設值（如果命令行未指定）
    if n_envs is None:
        n_envs = config.get('vec_env', {}).get('n_envs', 1)
    if vec_type is None:
        vec_type = config.get('vec_env', {}).get('vec_type', 'dummy')

    if n_envs == 1:
        env = TaiwanLOBTradingEnv(env_config)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        logger.info("✅ 創建單一環境")
    else:
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
            save_path=checkpoint_config.get('save_path', 'checkpoints/sb3/ppo_deeplob'),
            name_prefix=checkpoint_config.get('name_prefix', 'ppo_model'),
            save_replay_buffer=checkpoint_config.get('save_replay_buffer', False),
            save_vecnormalize=checkpoint_config.get('save_vecnormalize', False),
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
            best_model_save_path=eval_config.get('best_model_save_path', 'checkpoints/sb3/ppo_deeplob'),
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


def create_ppo_deeplob_model(env, config: dict, deeplob_checkpoint: str, device: str = None):
    """創建整合 DeepLOB 的 PPO 模型"""
    ppo_config = config.get('ppo', {})
    deeplob_config = config.get('deeplob_extractor', {})

    # 從配置讀取設備（如果命令行未指定）
    if device is None:
        device = config.get('device', {}).get('default', 'cuda')

    logger.info("🔨 構建 PPO + DeepLOB 模型")

    # 使用 DeepLOB 特徵提取器
    if deeplob_config.get('use_deeplob', True):
        logger.info("  - 模式: DeepLOB 特徵提取器")

        # 創建 policy_kwargs
        policy_kwargs = make_deeplob_policy_kwargs(
            deeplob_checkpoint=deeplob_checkpoint,
            features_dim=deeplob_config.get('features_dim', 128),
            use_lstm_hidden=deeplob_config.get('use_lstm_hidden', False),
            net_arch=ppo_config.get('net_arch', dict(pi=[256, 128], vf=[256, 128]))
        )

        logger.info(f"  - DeepLOB 檢查點: {deeplob_checkpoint}")
        logger.info(f"  - 特徵維度: {deeplob_config.get('features_dim', 128)}")
        logger.info(f"  - 凍結 DeepLOB: {deeplob_config.get('freeze_deeplob', True)}")

    else:
        # 不使用 DeepLOB（回退到基礎 MlpPolicy）
        logger.warning("⚠️  未啟用 DeepLOB 特徵提取器，使用基礎 MlpPolicy")
        policy_kwargs = dict(
            net_arch=ppo_config.get('net_arch', dict(pi=[256, 128], vf=[256, 128]))
        )

    # 創建 PPO 模型
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
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.get('training', {}).get('tensorboard_log', 'logs/sb3_deeplob'),
        verbose=ppo_config.get('verbose', 1),
        seed=ppo_config.get('seed', 42),
        device=device
    )

    logger.info("✅ PPO + DeepLOB 模型創建成功")
    logger.info(f"  - Policy: MlpPolicy + DeepLOBExtractor")
    logger.info(f"  - Learning Rate: {ppo_config.get('learning_rate', 3e-4)}")
    logger.info(f"  - Gamma: {ppo_config.get('gamma', 0.99)}")
    logger.info(f"  - N Steps: {ppo_config.get('n_steps', 2048)}")
    logger.info(f"  - Batch Size: {ppo_config.get('batch_size', 64)}")
    logger.info(f"  - Device: {device}")

    return model


def train_model(model, config: dict, callbacks=None):
    """訓練模型"""
    training_config = config.get('training', {})
    total_timesteps = training_config.get('total_timesteps', 1000000)
    log_interval = training_config.get('log_interval', 10)
    progress_bar = training_config.get('progress_bar', True)

    logger.info("=" * 60)
    logger.info("🚀 開始訓練 (PPO + DeepLOB)")
    logger.info("=" * 60)
    logger.info(f"總步數: {total_timesteps:,}")

    start_time = datetime.now()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
        progress_bar=progress_bar
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("✅ 訓練完成")
    logger.info("=" * 60)
    logger.info(f"訓練時間: {duration/60:.2f} 分鐘 ({duration/3600:.2f} 小時)")
    logger.info(f"訓練速度: {total_timesteps/duration:.1f} steps/sec")

    return model


def save_final_model(model, config: dict):
    """保存最終模型"""
    training_config = config.get('training', {})
    save_dir = training_config.get('checkpoint_dir', 'checkpoints/sb3/ppo_deeplob')
    model_name = training_config.get('final_model_name', 'ppo_deeplob_final')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    model.save(save_path)
    logger.info(f"✅ 最終模型已保存: {save_path}.zip")

    return save_path


def apply_test_mode(config: dict):
    """應用測試模式配置"""
    test_config = config.get('test_mode', {})

    # 更新訓練配置
    config['training']['total_timesteps'] = test_config.get('total_timesteps', 10000)
    config['callbacks']['checkpoint']['save_freq'] = test_config.get('save_freq', 5000)
    config['callbacks']['eval']['eval_freq'] = test_config.get('eval_freq', 5000)
    config['callbacks']['eval']['n_eval_episodes'] = test_config.get('n_eval_episodes', 3)
    config['ppo']['n_steps'] = test_config.get('n_steps', 512)
    config['ppo']['batch_size'] = test_config.get('batch_size', 32)

    logger.info("🧪 測試模式已啟用")
    logger.info(f"  - 訓練步數: {config['training']['total_timesteps']:,}")
    logger.info(f"  - Checkpoint 頻率: {config['callbacks']['checkpoint']['save_freq']:,}")
    logger.info(f"  - 評估頻率: {config['callbacks']['eval']['eval_freq']:,}")


def show_next_steps(config: dict):
    """顯示訓練完成後的下一步建議"""
    if not config.get('output', {}).get('show_next_steps', True):
        return

    output_config = config.get('output', {})
    tensorboard_log = config.get('training', {}).get('tensorboard_log', 'logs/sb3_deeplob')
    best_model_path = output_config.get('best_model_path', 'checkpoints/sb3/ppo_deeplob/best_model')

    logger.info("\n" + "=" * 60)
    logger.info("🎉 訓練流程完成")
    logger.info("=" * 60)
    logger.info(f"✅ 日誌目錄: {tensorboard_log}")
    logger.info("\n下一步:")
    logger.info(f"  1. 查看訓練日誌: tensorboard --logdir {tensorboard_log}")
    logger.info(f"  2. 評估最佳模型: python scripts/evaluate_sb3.py --model {best_model_path}")
    logger.info(f"  3. 開始超參數優化（階段三）")


def main():
    parser = argparse.ArgumentParser(description='PPO + DeepLOB 完整訓練')
    parser.add_argument('--config', type=str, default='configs/sb3_deeplob_config.yaml',
                      help='配置文件路徑')
    parser.add_argument('--deeplob-checkpoint', type=str, default=None,
                      help='DeepLOB 檢查點路徑（覆蓋配置文件）')
    parser.add_argument('--timesteps', type=int, default=None,
                      help='訓練步數（覆蓋配置文件）')
    parser.add_argument('--device', type=str, default=None,
                      help='設備: cuda / cpu（覆蓋配置文件）')
    parser.add_argument('--test', action='store_true',
                      help='測試模式（快速驗證流程）')
    parser.add_argument('--n-envs', type=int, default=None,
                      help='並行環境數量（覆蓋配置文件）')
    parser.add_argument('--vec-type', type=str, default=None,
                      help='向量化類型: dummy / subproc（覆蓋配置文件）')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 PPO + DeepLOB 完整訓練腳本 v2.0")
    logger.info("=" * 60)

    try:
        # 1. 載入配置
        logger.info(f"\n📄 載入配置: {args.config}")
        config = load_config(args.config)

        # 顯示專案信息
        project_info = config.get('project', {})
        if project_info:
            logger.info(f"  - 專案: {project_info.get('name', 'N/A')}")
            logger.info(f"  - 版本: {project_info.get('version', 'N/A')}")
            logger.info(f"  - 描述: {project_info.get('description', 'N/A')}")

        # 2. 應用命令行參數覆蓋
        if args.test:
            apply_test_mode(config)

        if args.timesteps is not None:
            config['training']['total_timesteps'] = args.timesteps
            logger.info(f"⚙️  覆蓋訓練步數: {args.timesteps:,}")

        # 3. 確定 DeepLOB 檢查點路徑
        deeplob_checkpoint = args.deeplob_checkpoint or config['env_config'].get('deeplob_checkpoint')
        if not deeplob_checkpoint:
            raise ValueError("未指定 DeepLOB 檢查點路徑")

        # 4. 確定設備
        device = args.device
        if device is None:
            device = config.get('device', {}).get('default', 'cuda')

        # 檢查 CUDA
        auto_fallback = config.get('device', {}).get('auto_fallback', True)
        if device == 'cuda' and not torch.cuda.is_available():
            if auto_fallback:
                logger.warning("⚠️  CUDA 不可用，自動回退到 CPU")
                device = 'cpu'
            else:
                raise RuntimeError("CUDA 不可用且未啟用自動回退")

        logger.info(f"✅ 使用設備: {device}")
        if device == 'cuda':
            logger.info(f"  - GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  - 總顯存: {total_memory:.2f} GB")

        # 5. 驗證 DeepLOB 檢查點
        logger.info(f"\n🔍 驗證 DeepLOB 檢查點")
        verify_deeplob_checkpoint(deeplob_checkpoint, config)

        # 6. 創建訓練環境
        logger.info("\n🏗️  創建訓練環境")
        env = create_vec_env(config, n_envs=args.n_envs, vec_type=args.vec_type)

        # 7. 創建評估環境
        logger.info("\n🏗️  創建評估環境")
        eval_env = create_eval_env(config)

        # 8. 創建回調
        logger.info("\n🔔 設置訓練回調")
        callbacks = create_callbacks(config, eval_env)

        # 9. 創建 PPO + DeepLOB 模型
        logger.info("\n🤖 創建 PPO + DeepLOB 模型")
        model = create_ppo_deeplob_model(
            env,
            config,
            deeplob_checkpoint=deeplob_checkpoint,
            device=device
        )

        # 10. 開始訓練
        model = train_model(model, config, callbacks=callbacks)

        # 11. 保存最終模型
        logger.info("\n💾 保存最終模型")
        save_path = save_final_model(model, config)

        # 12. 顯示下一步建議
        show_next_steps(config)

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
