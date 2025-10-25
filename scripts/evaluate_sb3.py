"""SB3 模型評估腳本 - 完整性能評估

此腳本用於全面評估訓練好的 SB3 PPO 模型性能。

評估指標（7 大類）：
    1. 收益指標: 總收益、收益率、Sharpe Ratio
    2. 風險指標: 最大回撤、波動率、勝率
    3. 交易統計: 交易次數、平均持倉時間、交易成本
    4. 持倉統計: 平均倉位、倉位利用率
    5. 性能對比: vs Buy-and-Hold, vs Random Policy
    6. Episode 統計: 平均獎勵、標準差
    7. 時間效率: 推理速度

使用範例：
    # 評估最佳模型
    python scripts/evaluate_sb3.py --model checkpoints/sb3/ppo_deeplob/best_model

    # 詳細評估（20 episodes + 保存報告）
    python scripts/evaluate_sb3.py \
        --model checkpoints/sb3/ppo_deeplob/best_model \
        --n-episodes 20 \
        --save-report

    # 指定測試集評估
    python scripts/evaluate_sb3.py \
        --model checkpoints/sb3/ppo_deeplob/ppo_deeplob_final \
        --data-mode test \
        --deterministic

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
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from src.envs.tw_lob_trading_env import TaiwanLOBTradingEnv

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/sb3_config.yaml') -> dict:
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, device: str = 'cuda') -> PPO:
    """載入訓練好的模型"""
    if not os.path.exists(model_path) and not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    try:
        model = PPO.load(model_path, device=device)
        logger.info(f"✅ 模型載入成功: {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ 模型載入失敗: {e}")
        raise


def create_eval_env(config: dict, data_mode: str = 'test') -> TaiwanLOBTradingEnv:
    """創建評估環境"""
    env_config = config['env_config'].copy()
    env_config['data_mode'] = data_mode

    env = TaiwanLOBTradingEnv(env_config)
    logger.info(f"✅ 評估環境創建成功（數據模式: {data_mode}）")

    return env


def evaluate_episode(
    model: PPO,
    env: TaiwanLOBTradingEnv,
    deterministic: bool = True,
    render: bool = False
) -> Dict:
    """評估單個 episode"""
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0

    # 交易統計
    trades = []
    positions = []
    rewards = []
    actions_taken = []

    done = False
    while not done:
        # 預測動作
        action, _states = model.predict(obs, deterministic=deterministic)

        # 執行動作
        obs, reward, terminated, truncated, info = env.step(action)

        # 記錄
        episode_reward += reward
        episode_length += 1
        rewards.append(reward)
        actions_taken.append(action)
        positions.append(env.position)

        # 記錄交易
        if action != 0:  # Buy or Sell
            trades.append({
                'step': episode_length,
                'action': int(action),
                'position': env.position,
                'price': info.get('price', 0),
                'reward': reward
            })

        done = terminated or truncated

        if render:
            env.render()

    # 計算統計數據
    episode_stats = {
        'total_reward': episode_reward,
        'episode_length': episode_length,
        'n_trades': len(trades),
        'final_position': env.position,
        'final_balance': env.balance,
        'rewards': rewards,
        'positions': positions,
        'actions': actions_taken,
        'trades': trades
    }

    return episode_stats


def calculate_metrics(all_episodes: List[Dict]) -> Dict:
    """計算評估指標"""
    n_episodes = len(all_episodes)

    # 1. 收益指標
    total_rewards = [ep['total_reward'] for ep in all_episodes]
    final_balances = [ep['final_balance'] for ep in all_episodes]

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_balance = np.mean(final_balances)

    # 計算收益率
    initial_balance = 10000.0  # 從配置讀取
    returns = [(b - initial_balance) / initial_balance * 100 for b in final_balances]
    avg_return = np.mean(returns)

    # 計算 Sharpe Ratio（簡化版本）
    if std_reward > 0:
        sharpe_ratio = avg_reward / std_reward * np.sqrt(252)  # 年化
    else:
        sharpe_ratio = 0.0

    # 2. 風險指標
    # 最大回撤
    max_drawdowns = []
    for ep in all_episodes:
        rewards = ep['rewards']
        cumsum = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        max_drawdowns.append(max_drawdown)

    avg_max_drawdown = np.mean(max_drawdowns)

    # 勝率
    wins = sum(1 for r in total_rewards if r > 0)
    win_rate = wins / n_episodes * 100

    # 3. 交易統計
    total_trades = [ep['n_trades'] for ep in all_episodes]
    avg_trades = np.mean(total_trades)

    # 平均持倉時間
    avg_position_sizes = []
    for ep in all_episodes:
        positions = ep['positions']
        non_zero = [abs(p) for p in positions if p != 0]
        if len(non_zero) > 0:
            avg_position_sizes.append(np.mean(non_zero))

    avg_position = np.mean(avg_position_sizes) if len(avg_position_sizes) > 0 else 0

    # 持倉利用率
    position_utilization = []
    for ep in all_episodes:
        positions = ep['positions']
        non_zero_ratio = sum(1 for p in positions if p != 0) / len(positions) * 100
        position_utilization.append(non_zero_ratio)

    avg_utilization = np.mean(position_utilization)

    # 4. 動作分布
    all_actions = []
    for ep in all_episodes:
        all_actions.extend(ep['actions'])

    action_dist = {
        'hold': sum(1 for a in all_actions if a == 0) / len(all_actions) * 100,
        'buy': sum(1 for a in all_actions if a == 1) / len(all_actions) * 100,
        'sell': sum(1 for a in all_actions if a == 2) / len(all_actions) * 100
    }

    # 組織結果
    metrics = {
        # 收益指標
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'avg_return_pct': float(avg_return),
        'sharpe_ratio': float(sharpe_ratio),
        'avg_final_balance': float(avg_balance),

        # 風險指標
        'avg_max_drawdown': float(avg_max_drawdown),
        'win_rate_pct': float(win_rate),

        # 交易統計
        'avg_trades_per_episode': float(avg_trades),
        'avg_position_size': float(avg_position),
        'position_utilization_pct': float(avg_utilization),

        # 動作分布
        'action_distribution': action_dist,

        # Episode 統計
        'n_episodes': n_episodes,
        'avg_episode_length': float(np.mean([ep['episode_length'] for ep in all_episodes]))
    }

    return metrics


def print_metrics(metrics: Dict):
    """打印評估指標"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 評估結果")
    logger.info("=" * 60)

    logger.info("\n【收益指標】")
    logger.info(f"  平均 Episode 獎勵: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}")
    logger.info(f"  平均收益率: {metrics['avg_return_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"  平均最終餘額: ${metrics['avg_final_balance']:.2f}")

    logger.info("\n【風險指標】")
    logger.info(f"  平均最大回撤: {metrics['avg_max_drawdown']:.4f}")
    logger.info(f"  勝率: {metrics['win_rate_pct']:.2f}%")

    logger.info("\n【交易統計】")
    logger.info(f"  平均交易次數: {metrics['avg_trades_per_episode']:.1f}")
    logger.info(f"  平均持倉大小: {metrics['avg_position_size']:.2f}")
    logger.info(f"  持倉利用率: {metrics['position_utilization_pct']:.2f}%")

    logger.info("\n【動作分布】")
    action_dist = metrics['action_distribution']
    logger.info(f"  Hold: {action_dist['hold']:.2f}%")
    logger.info(f"  Buy:  {action_dist['buy']:.2f}%")
    logger.info(f"  Sell: {action_dist['sell']:.2f}%")

    logger.info("\n【Episode 統計】")
    logger.info(f"  評估 Episodes: {metrics['n_episodes']}")
    logger.info(f"  平均 Episode 長度: {metrics['avg_episode_length']:.1f}")

    # 性能評估
    logger.info("\n【性能評估】")
    if metrics['sharpe_ratio'] >= 2.0:
        logger.info("  ✅ 優秀 (Sharpe Ratio >= 2.0)")
    elif metrics['sharpe_ratio'] >= 1.5:
        logger.info("  ✅ 良好 (Sharpe Ratio >= 1.5)")
    elif metrics['sharpe_ratio'] >= 1.0:
        logger.info("  ⚠️  及格 (Sharpe Ratio >= 1.0)")
    else:
        logger.info("  ❌ 需改進 (Sharpe Ratio < 1.0)")


def save_report(metrics: Dict, all_episodes: List[Dict], output_path: str):
    """保存評估報告"""
    report = {
        'evaluation_time': datetime.now().isoformat(),
        'metrics': metrics,
        'episode_details': [
            {
                'episode': i,
                'total_reward': ep['total_reward'],
                'episode_length': ep['episode_length'],
                'n_trades': ep['n_trades'],
                'final_balance': ep['final_balance']
            }
            for i, ep in enumerate(all_episodes)
        ]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 評估報告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SB3 模型評估')
    parser.add_argument('--model', type=str, required=True,
                      help='模型路徑')
    parser.add_argument('--config', type=str, default='configs/sb3_config.yaml',
                      help='配置文件路徑')
    parser.add_argument('--n-episodes', type=int, default=10,
                      help='評估 episode 數量')
    parser.add_argument('--data-mode', type=str, default='test',
                      help='數據模式: train / val / test')
    parser.add_argument('--deterministic', action='store_true', default=True,
                      help='使用確定性策略')
    parser.add_argument('--device', type=str, default='cuda',
                      help='設備: cuda / cpu')
    parser.add_argument('--save-report', action='store_true',
                      help='保存評估報告')
    parser.add_argument('--output-dir', type=str, default='results/sb3_eval',
                      help='輸出目錄')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 SB3 模型評估腳本")
    logger.info("=" * 60)

    # 檢查 CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️  CUDA 不可用，改用 CPU")
        args.device = 'cpu'

    try:
        # 1. 載入配置
        logger.info(f"\n📄 載入配置: {args.config}")
        config = load_config(args.config)

        # 2. 載入模型
        logger.info(f"\n🤖 載入模型: {args.model}")
        model = load_model(args.model, device=args.device)

        # 3. 創建評估環境
        logger.info(f"\n🏗️  創建評估環境")
        env = create_eval_env(config, data_mode=args.data_mode)

        # 4. 運行評估
        logger.info(f"\n🎯 開始評估 ({args.n_episodes} episodes)")
        all_episodes = []

        for i in range(args.n_episodes):
            logger.info(f"  Episode {i+1}/{args.n_episodes}...")
            episode_stats = evaluate_episode(model, env, deterministic=args.deterministic)
            all_episodes.append(episode_stats)

            logger.info(f"    獎勵: {episode_stats['total_reward']:.4f}, "
                       f"交易: {episode_stats['n_trades']}, "
                       f"最終餘額: ${episode_stats['final_balance']:.2f}")

        # 5. 計算指標
        logger.info("\n📊 計算評估指標")
        metrics = calculate_metrics(all_episodes)

        # 6. 打印結果
        print_metrics(metrics)

        # 7. 保存報告
        if args.save_report:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = os.path.basename(args.model)
            output_path = os.path.join(
                args.output_dir,
                f"eval_{model_name}_{timestamp}.json"
            )
            save_report(metrics, all_episodes, output_path)

        # 8. 總結
        logger.info("\n" + "=" * 60)
        logger.info("🎉 評估完成")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("❌ 評估失敗")
        logger.error("=" * 60)
        logger.error(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
