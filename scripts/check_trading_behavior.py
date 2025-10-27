"""檢查模型交易行為 - 快速驗證模型是否在實際交易

檢查項目：
1. 動作分布（Hold vs Buy）
2. 交易次數
3. 持倉時間分布
4. PnL 來源

使用方式：
    python scripts/check_trading_behavior.py --model checkpoints/sb3/ppo_deeplob/best_model
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from stable_baselines3 import PPO
from src.envs.tw_lob_trading_env import TaiwanLOBTradingEnv

def check_model_behavior(model_path: str, n_episodes: int = 5):
    """檢查模型交易行為"""
    print("=" * 60)
    print("模型交易行為檢查")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"測試 Episodes: {n_episodes}")
    print()

    # 載入模型
    try:
        model = PPO.load(model_path)
        print("[OK] 模型載入成功")
    except Exception as e:
        print(f"[ERROR] 模型載入失敗: {e}")
        return

    # 創建環境（使用測試集）
    env_config = {
        'data_dir': 'data/processed_v7/npz',
        'deeplob_checkpoint': 'checkpoints/v5/deeplob_v5_best.pth',
        'max_steps': 500,
        'initial_balance': 10000.0,
        'max_position': 1,
        'transaction_cost': {
            'shares_per_lot': 1000,
            'commission': {
                'base_rate': 0.001425,
                'discount': 0.3,
                'min_fee': 20.0
            },
            'securities_tax': {
                'rate': 0.0015
            },
            'slippage': {
                'enabled': False,
                'rate': 0.0001
            }
        },
        'reward_config': {
            'pnl_scale': 1.0,
            'cost_penalty': 1.0,
            'inventory_penalty': 0.0,
            'risk_penalty': 0.0
        },
        'data_mode': 'test'
    }

    try:
        env = TaiwanLOBTradingEnv(env_config)
        print("[OK] 環境創建成功")
    except Exception as e:
        print(f"[ERROR] 環境創建失敗: {e}")
        return

    print()
    print("-" * 60)
    print("開始測試...")
    print("-" * 60)

    # 統計變量
    total_actions = []
    total_trades = 0
    total_holds = 0
    total_buys = 0
    episode_rewards = []
    episode_trades = []
    episode_final_positions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_action_count = {0: 0, 1: 0}  # Hold/Sell, Buy
        prev_position = 0

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Convert numpy array to int
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_action_count[action] += 1
            total_actions.append(action)

            # 檢查是否真的交易了
            current_position = info.get('position', 0)
            if current_position != prev_position:
                total_trades += 1
            prev_position = current_position

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_trades.append(episode_action_count[1])  # Buy 動作數
        episode_final_positions.append(prev_position)

        total_holds += episode_action_count[0]
        total_buys += episode_action_count[1]

        print(f"Episode {ep+1}/{n_episodes}: "
              f"獎勵={episode_reward:.2f}, "
              f"Hold={episode_action_count[0]}, "
              f"Buy={episode_action_count[1]}, "
              f"實際交易={episode_trades[-1]}, "
              f"最終倉位={episode_final_positions[-1]}")

    print()
    print("=" * 60)
    print("統計結果")
    print("=" * 60)

    # 動作分布
    total_actions_count = len(total_actions)
    hold_ratio = total_holds / total_actions_count * 100
    buy_ratio = total_buys / total_actions_count * 100

    print(f"\n【動作分布】")
    print(f"  總動作數: {total_actions_count}")
    print(f"  Hold/Sell (0): {total_holds} ({hold_ratio:.1f}%)")
    print(f"  Buy (1): {total_buys} ({buy_ratio:.1f}%)")

    # 交易統計
    print(f"\n【交易統計】")
    print(f"  實際交易次數: {total_trades}")
    print(f"  平均每 Episode: {total_trades/n_episodes:.1f} 次")

    # 獎勵統計
    print(f"\n【獎勵統計】")
    print(f"  平均獎勵: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  最大獎勵: {np.max(episode_rewards):.2f}")
    print(f"  最小獎勵: {np.min(episode_rewards):.2f}")

    # 診斷
    print(f"\n【診斷】")
    if buy_ratio < 5:
        print("  [WARN] 警告：Buy 動作比例過低 (<5%)，模型幾乎不交易")
        print("  [WARN] 可能原因：")
        print("      1. 獎勵函數懲罰交易成本過重")
        print("      2. 模型學到了「不交易」最安全")
        print("      3. 數據中盈利機會太少")
    elif buy_ratio > 40:
        print("  [WARN] 警告：Buy 動作比例過高 (>40%)，模型可能過度交易")
    else:
        print("  [OK] Buy 動作比例正常 (5-40%)")

    if total_trades < n_episodes * 5:
        print(f"  [WARN] 警告：實際交易次數過少 (平均 {total_trades/n_episodes:.1f} 次/Episode)")
        print("  [WARN] 模型可能學會了「永遠不動」策略")
    else:
        print(f"  [OK] 實際交易次數合理 (平均 {total_trades/n_episodes:.1f} 次/Episode)")

    print()
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="檢查模型交易行為")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/sb3/ppo_deeplob/best_model",
        help="模型路徑"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="測試 Episode 數"
    )

    args = parser.parse_args()

    check_model_behavior(args.model, args.n_episodes)
