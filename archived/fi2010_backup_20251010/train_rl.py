"""RLlib PPO 強化學習訓練腳本

此腳本負責訓練基於 PPO 算法的高頻交易策略。
整合 DeepLOB 價格預測模型和 LSTM Policy Network，
在 LOB 交易環境中學習最優交易決策。

訓練架構:
    輸入: LOB 特徵 (40維) + DeepLOB 預測 (3維) + 交易狀態 (5維)
    模型: RecurrentPPO + LSTM (256 units, 2 layers)
    環境: LOBTradingEnv (Gymnasium 標準)
    輸出: 交易動作 {Hold, Buy, Sell}

訓練目標:
    - Sharpe Ratio > 2.0
    - 勝率 > 55%
    - GPU 利用率 > 85%
    - 穩定收斂

硬體需求:
    - NVIDIA RTX 5090 (32GB VRAM)
    - CUDA 12.9
    - 16+ CPU cores

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-09
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.lob_trading_env import LOBTradingEnv
from src.models.trading_lstm_model import TradingLSTMModel, register_trading_lstm_model
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

# 設定日誌
logger = setup_logger(__name__)


def register_environment():
    """註冊自訂交易環境到 RLlib

    此函數將 LOBTradingEnv 註冊到 Ray Tune 的環境註冊表，
    使其可以在 RLlib 配置中通過名稱引用。

    註冊名稱: "LOBTradingEnv"

    使用方式:
        在 PPO 配置中使用 .environment("LOBTradingEnv")
    """
    from ray.tune.registry import register_env

    def env_creator(env_config):
        """環境創建函數

        此函數作為工廠模式創建環境實例。
        RLlib 會在每個 worker 上調用此函數創建獨立的環境。

        參數:
            env_config: 環境配置字典，來自 PPOConfig

        返回:
            LOBTradingEnv: 配置好的環境實例
        """
        return LOBTradingEnv(env_config)

    register_env("LOBTradingEnv", env_creator)
    logger.info("✓ 已註冊 LOBTradingEnv 環境")


def create_ppo_config(args, env_config: dict, model_config: dict) -> PPOConfig:
    """創建 PPO 算法配置

    此函數構建完整的 RecurrentPPO 配置，針對高頻交易任務優化。

    參數:
        args: 命令行參數
            包含學習率、訓練批次大小等超參數

        env_config: 環境配置字典
            包含 max_steps, transaction_cost_rate 等環境參數

        model_config: 模型配置字典
            包含 lstm_cell_size, num_lstm_layers 等模型參數

    返回:
        PPOConfig: 配置完整的 PPO 配置對象

    配置重點:
        1. 啟用 LSTM（RecurrentPPO）
        2. GPU 資源分配
        3. 訓練批次大小優化
        4. Actor-Critic 分離
        5. 探索策略

    參考:
        CLAUDE.md 中的 PPO Config 未啟用 LSTM 問題修正
    """
    config = (
        PPOConfig()
        # ========== 環境配置 ==========
        .environment(
            env="LOBTradingEnv",
            env_config=env_config,
        )
        # ========== 框架配置 ==========
        .framework("torch")
        # ========== 訓練超參數 ==========
        .training(
            # 學習率
            lr=args.lr,
            # 折扣因子（越接近1越重視長期回報）
            gamma=args.gamma,
            # GAE Lambda（優勢函數估計）
            lambda_=0.95,
            # PPO 裁剪參數（防止策略更新過大）
            clip_param=0.2,
            # 熵係數（鼓勵探索）
            entropy_coeff=args.entropy_coeff,
            # KL 散度目標（自適應 lr）
            kl_coeff=0.2,
            kl_target=0.01,
            # 訓練批次大小（充分利用 RTX 5090）
            train_batch_size=args.train_batch_size,
            # SGD 小批次大小
            sgd_minibatch_size=args.sgd_minibatch_size,
            # SGD 迭代次數
            num_sgd_iter=10,
            # 價值函數損失係數
            vf_loss_coeff=0.5,
            # 梯度裁剪（防止梯度爆炸）
            grad_clip=0.5,
        )
        # ========== Rollout 配置 ==========
        .rollouts(
            # Worker 數量（CPU 並行採樣）
            num_rollout_workers=args.num_workers,
            # 每個 worker 的 rollout 片段長度
            # 需要與 episode 長度協調
            rollout_fragment_length=200,
            # 批次模式（complete_episodes 確保 LSTM 狀態正確）
            batch_mode="complete_episodes",
        )
        # ========== 資源配置 ==========
        .resources(
            # Learner GPU（訓練用）
            num_gpus=1.0 if args.num_gpus > 0 else 0.0,
            # 每個 worker 的 CPU 數量
            num_cpus_per_worker=2,
            # 每個 worker 的 GPU 數量（可選，用於環境推理）
            num_gpus_per_worker=0,
        )
        # ========== 模型配置（關鍵！）==========
        .model({
            # 使用自訂 LSTM 模型
            "custom_model": "TradingLSTMModel",
            # 自訂模型配置
            "custom_model_config": model_config,
            # 啟用 LSTM（RecurrentPPO）
            "use_lstm": True,
            # LSTM 隱藏層大小
            "lstm_cell_size": model_config.get("lstm_cell_size", 256),
            # LSTM 最大序列長度
            "max_seq_len": model_config.get("max_seq_len", 100),
            # LSTM 使用前一動作作為輸入
            "lstm_use_prev_action": True,
            # LSTM 使用前一獎勵作為輸入
            "lstm_use_prev_reward": True,
            # Actor 和 Critic 不共享層
            "vf_share_layers": False,
        })
        # ========== 探索配置 ==========
        .exploration(
            # 初始探索率（高探索）
            explore=True,
        )
        # ========== 評估配置 ==========
        .evaluation(
            # 評估間隔（每 N 次訓練迭代評估一次）
            evaluation_interval=10,
            # 評估 episode 數量
            evaluation_num_workers=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        # ========== 調試配置 ==========
        .debugging(
            # 日誌級別
            log_level="INFO",
        )
    )

    return config


def train(args):
    """執行 RLlib PPO 訓練

    此函數是訓練的主要流程，包含：
        1. Ray 初始化
        2. 環境和模型註冊
        3. PPO 配置創建或檢查點載入（支援續訓）
        4. 訓練循環
        5. 檢查點保存
        6. 最佳模型追蹤

    參數:
        args: 命令行參數對象
            包含所有訓練超參數和配置

    訓練流程:
        1. 載入配置文件
        2. 初始化 Ray 集群
        3. 註冊環境和模型
        4. 創建或載入 PPO Trainer（默認續訓）
           - 如果 args.resume=True (默認)，自動尋找最新檢查點
           - 檢查點搜尋順序: final_model > best_model > 最新 checkpoint_*
           - 可通過 args.resume_from 指定特定檢查點
        5. 訓練循環（num_iterations 次，從上次中斷處繼續）
        6. 定期保存檢查點
        7. 追蹤最佳模型（同時保存 best_reward.txt）
        8. 訓練完成後保存最終模型

    輸出:
        - checkpoints/rl/: 定期檢查點
        - checkpoints/rl/best_model/: 最佳模型
        - checkpoints/rl/best_reward.txt: 最佳回報值（用於續訓）
        - checkpoints/rl/final_model/: 最終模型
        - logs/rl/: TensorBoard 日誌

    監控指標:
        - episode_reward_mean: 平均 episode 回報
        - policy_loss: 策略損失
        - vf_loss: 價值函數損失
        - entropy: 策略熵（探索度）
        - kl: KL 散度（策略變化幅度）

    續訓功能:
        - 默認開啟 (args.resume=True)
        - 使用 --no-resume 禁用續訓
        - 使用 --resume-from <path> 指定檢查點
        - 自動載入迭代次數和最佳回報值
    """
    # ========== 步驟 1: 載入配置 ==========
    logger.info("========== RLlib PPO 訓練開始 ==========")
    logger.info(f"配置文件: {args.config}")

    # 載入 RL 配置
    rl_config = load_config(args.config)
    env_config = rl_config.get("env_config", {})
    model_config = rl_config.get("model_config", {})

    # 覆蓋配置（命令行參數優先）
    if args.deeplob_checkpoint:
        env_config["deeplob_checkpoint"] = args.deeplob_checkpoint

    logger.info(f"環境配置: {env_config}")
    logger.info(f"模型配置: {model_config}")

    # ========== 步驟 2: 初始化 Ray ==========
    logger.info("初始化 Ray 集群...")

    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        ignore_reinit_error=True,
    )

    logger.info(f"✓ Ray 已初始化")
    logger.info(f"  - CPUs: {args.num_cpus}")
    logger.info(f"  - GPUs: {args.num_gpus}")
    logger.info(f"  - Dashboard: http://localhost:8265")

    # ========== 步驟 3: 註冊環境和模型 ==========
    register_environment()
    register_trading_lstm_model()

    # ========== 步驟 4: 創建或載入 PPO Trainer ==========
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 檢查是否有檢查點可以續訓
    resume_checkpoint = None
    start_iteration = 1
    best_reward = float('-inf')

    if args.resume:
        # 優先使用指定的檢查點路徑
        if args.resume_from:
            resume_path = Path(args.resume_from)
            if resume_path.exists():
                resume_checkpoint = str(resume_path)
                logger.info(f"✓ 找到指定檢查點: {resume_checkpoint}")
            else:
                logger.warning(f"⚠️ 指定檢查點不存在: {args.resume_from}")

        # 如果沒有指定路徑，自動尋找最新檢查點
        if resume_checkpoint is None:
            # 搜尋順序: 1. final_model 2. best_model 3. 最新 checkpoint_*
            candidates = [
                checkpoint_dir / "final_model",
                checkpoint_dir / "best_model",
            ]

            # 添加所有 checkpoint_* 目錄 (按時間排序)
            checkpoint_pattern = list(checkpoint_dir.glob("checkpoint_*"))
            checkpoint_pattern.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidates.extend(checkpoint_pattern)

            for candidate in candidates:
                if candidate.exists() and candidate.is_dir():
                    # 驗證檢查點是否有效 (包含必要文件)
                    if (candidate / "algorithm_state.pkl").exists():
                        resume_checkpoint = str(candidate)
                        logger.info(f"✓ 自動找到檢查點: {resume_checkpoint}")
                        break

            if resume_checkpoint is None:
                logger.info("ℹ️ 未找到可用檢查點，從頭開始訓練")

    # 創建或載入 Trainer
    if resume_checkpoint:
        logger.info(f"📂 從檢查點載入 Trainer: {resume_checkpoint}")
        try:
            from ray.rllib.algorithms.ppo import PPO
            trainer = PPO.from_checkpoint(resume_checkpoint)

            # 嘗試從檢查點名稱提取迭代次數
            checkpoint_name = Path(resume_checkpoint).name
            if checkpoint_name.startswith("checkpoint_"):
                try:
                    start_iteration = int(checkpoint_name.split("_")[-1]) + 1
                    logger.info(f"✓ 從迭代 {start_iteration} 繼續訓練")
                except ValueError:
                    logger.info("✓ Trainer 已載入 (無法提取迭代次數，從 1 開始計數)")
            else:
                logger.info(f"✓ 從 {checkpoint_name} 繼續訓練")

            # 載入 best_reward (如果有保存)
            best_reward_file = checkpoint_dir / "best_reward.txt"
            if best_reward_file.exists():
                try:
                    with open(best_reward_file, 'r') as f:
                        best_reward = float(f.read().strip())
                    logger.info(f"✓ 載入最佳回報: {best_reward:.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ 無法載入最佳回報: {e}")

        except Exception as e:
            logger.error(f"❌ 載入檢查點失敗: {e}")
            logger.info("從頭開始訓練...")
            ppo_config = create_ppo_config(args, env_config, model_config)
            trainer = ppo_config.build()
    else:
        logger.info("創建新的 PPO Trainer...")
        ppo_config = create_ppo_config(args, env_config, model_config)
        trainer = ppo_config.build()
        logger.info("✓ PPO Trainer 已創建")

    # ========== 步驟 5: 訓練循環 ==========
    logger.info(f"開始訓練 {args.num_iterations} 次迭代 (從第 {start_iteration} 次開始)...")

    for iteration in range(start_iteration, start_iteration + args.num_iterations):
        # 執行一次訓練迭代
        result = trainer.train()

        # ===== 記錄訓練指標 =====
        episode_reward_mean = result.get("episode_reward_mean", 0)
        episode_len_mean = result.get("episode_len_mean", 0)
        policy_loss = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("policy_loss", 0)

        logger.info(
            f"迭代 {iteration}/{args.num_iterations} | "
            f"回報: {episode_reward_mean:.2f} | "
            f"Episode長度: {episode_len_mean:.1f} | "
            f"策略損失: {policy_loss:.4f}"
        )

        # ===== 定期保存檢查點 =====
        if iteration % args.checkpoint_interval == 0:
            checkpoint_path = trainer.save(checkpoint_dir)
            logger.info(f"✓ 已保存檢查點: {checkpoint_path}")

        # ===== 追蹤最佳模型 =====
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            best_model_dir = checkpoint_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint = trainer.save(best_model_dir)
            logger.info(f"🏆 新的最佳模型！回報: {best_reward:.2f}")
            logger.info(f"   保存至: {best_checkpoint}")

            # 保存最佳回報值 (用於續訓)
            best_reward_file = checkpoint_dir / "best_reward.txt"
            with open(best_reward_file, 'w') as f:
                f.write(f"{best_reward:.4f}")

    # ========== 步驟 6: 訓練完成 ==========
    logger.info("========== 訓練完成 ==========")
    logger.info(f"最佳回報: {best_reward:.2f}")
    logger.info(f"總訓練迭代: {iteration}")

    # 保存最終模型
    final_checkpoint = trainer.save(checkpoint_dir / "final_model")
    logger.info(f"✓ 最終模型已保存: {final_checkpoint}")

    # 關閉 Trainer 和 Ray
    trainer.stop()
    ray.shutdown()

    logger.info("✓ Ray 已關閉")


def main():
    """主函數 - 解析命令行參數並啟動訓練

    此函數負責：
        1. 解析命令行參數
        2. 設定訓練環境
        3. 調用訓練函數

    命令行參數:
        --config: 配置文件路徑
        --deeplob-checkpoint: DeepLOB 預訓練模型路徑
        --num-iterations: 訓練迭代次數
        --lr: 學習率
        --gamma: 折扣因子
        --train-batch-size: 訓練批次大小
        --num-workers: Rollout workers 數量
        --num-gpus: GPU 數量
        --checkpoint-dir: 檢查點保存目錄
    """
    parser = argparse.ArgumentParser(
        description="RLlib PPO 強化學習訓練 - 高頻交易策略"
    )

    # ===== 配置文件 =====
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_config.yaml",
        help="RL 配置文件路徑"
    )

    # ===== DeepLOB 模型 =====
    parser.add_argument(
        "--deeplob-checkpoint",
        type=str,
        default=None,
        help="DeepLOB 預訓練模型檢查點路徑"
    )

    # ===== 續訓參數 =====
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="是否續訓 (默認: True，自動尋找最新檢查點)"
    )

    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="禁用續訓，從頭開始訓練"
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="指定續訓檢查點路徑 (默認: 自動尋找最新)"
    )

    # ===== 訓練參數 =====
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="訓練迭代次數"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="學習率"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子（Gamma）"
    )

    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="熵係數（鼓勵探索）"
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4096,
        help="訓練批次大小"
    )

    parser.add_argument(
        "--sgd-minibatch-size",
        type=int,
        default=512,
        help="SGD 小批次大小"
    )

    # ===== 資源配置 =====
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Rollout workers 數量"
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="GPU 數量"
    )

    parser.add_argument(
        "--num-cpus",
        type=int,
        default=16,
        help="CPU 數量"
    )

    # ===== 檢查點 =====
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/rl",
        help="檢查點保存目錄"
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="檢查點保存間隔（迭代次數）"
    )

    # ===== 解析參數 =====
    args = parser.parse_args()

    # ===== 啟動訓練 =====
    try:
        train(args)
    except KeyboardInterrupt:
        logger.info("\n訓練被用戶中斷")
        ray.shutdown()
    except Exception as e:
        logger.error(f"訓練過程發生錯誤: {e}", exc_info=True)
        ray.shutdown()
        raise


if __name__ == "__main__":
    main()
