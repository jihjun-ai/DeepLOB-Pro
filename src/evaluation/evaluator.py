"""策略評估器 - 計算交易績效指標

此模組提供完整的交易策略評估功能，計算各種績效指標並與基準策略對比。

主要功能:
    1. 離線評估 RL 策略
    2. 計算績效指標 (Sharpe, MDD, Win Rate, etc.)
    3. 生成詳細評估報告
    4. 與基準策略對比

績效指標:
    - Sharpe Ratio: 風險調整後收益
    - Max Drawdown (MDD): 最大回撤
    - Calmar Ratio: 收益/最大回撤
    - Win Rate: 勝率
    - Profit Factor: 盈虧比
    - Total Return: 總收益率
    - Avg Return per Trade: 平均每筆交易收益

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-10
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# 績效指標計算函數
# ============================================================================

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """計算 Sharpe Ratio (夏普比率)

    Sharpe Ratio 衡量每單位風險的超額收益，是最常用的風險調整績效指標。

    公式: SR = (E[R] - Rf) / σ[R] * √T

    參數:
        returns: 收益率序列 (每個 episode 或每日收益)
        risk_free_rate: 無風險利率 (年化)，預設 0.0
        periods_per_year: 年化係數，預設 252 (交易日)

    返回:
        sharpe_ratio: 年化 Sharpe Ratio

    解釋:
        - SR > 2.0: 優秀
        - SR > 1.0: 良好
        - SR > 0.5: 可接受
        - SR < 0: 負收益

    注意:
        - 假設收益率服從正態分佈
        - 對極端值敏感
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # 年化 Sharpe Ratio
    sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """計算最大回撤 (Maximum Drawdown)

    最大回撤是投資組合從峰值到谷值的最大跌幅，衡量下行風險。

    公式: MDD = max((Peak - Trough) / Peak)

    參數:
        returns: 收益率序列或累積收益序列

    返回:
        mdd: 最大回撤比例 (0-1，如 0.2 表示 20% 回撤)
        start_idx: 回撤開始位置 (峰值)
        end_idx: 回撤結束位置 (谷值)

    解釋:
        - MDD < 10%: 風險很低
        - MDD < 20%: 可接受
        - MDD > 30%: 風險較高
        - MDD > 50%: 風險很高

    注意:
        - 回撤期間可能很長
        - 不考慮回撤發生的頻率
    """
    if len(returns) == 0:
        return 0.0, 0, 0

    # 計算累積收益曲線
    cumulative = np.cumsum(returns)

    # 計算歷史最高點
    running_max = np.maximum.accumulate(cumulative)

    # 計算回撤序列
    drawdown = (cumulative - running_max)

    # 找到最大回撤
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]

    # 找到對應的峰值位置
    peak_idx = np.argmax(cumulative[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # 標準化為比例
    if running_max[peak_idx] != 0:
        mdd_ratio = abs(max_dd / running_max[peak_idx])
    else:
        mdd_ratio = 0.0

    return float(mdd_ratio), int(peak_idx), int(max_dd_idx)


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """計算 Calmar Ratio (卡瑪比率)

    Calmar Ratio 是年化收益率與最大回撤的比值，衡量每單位回撤的收益。

    公式: Calmar = 年化收益率 / MDD

    參數:
        total_return: 總收益率 (如 0.5 表示 50% 收益)
        max_drawdown: 最大回撤比例 (如 0.2 表示 20% 回撤)

    返回:
        calmar_ratio: Calmar Ratio

    解釋:
        - Calmar > 3.0: 優秀
        - Calmar > 1.0: 良好
        - Calmar > 0.5: 可接受
        - Calmar < 0: 負收益

    注意:
        - 比 Sharpe Ratio 更關注極端風險
        - 適合趨勢策略評估
    """
    if max_drawdown == 0:
        return float('inf') if total_return > 0 else 0.0

    return float(total_return / max_drawdown)


def calculate_win_rate(returns: np.ndarray) -> float:
    """計算勝率 (Win Rate)

    勝率是盈利交易次數佔總交易次數的比例。

    公式: Win Rate = 盈利次數 / 總次數

    參數:
        returns: 收益序列 (每筆交易或每日收益)

    返回:
        win_rate: 勝率 (0-1，如 0.55 表示 55%)

    解釋:
        - WR > 60%: 優秀
        - WR > 50%: 良好
        - WR = 50%: 隨機
        - WR < 50%: 需改進

    注意:
        - 不考慮盈虧大小
        - 高勝率不一定代表高收益
    """
    if len(returns) == 0:
        return 0.0

    winning_trades = np.sum(returns > 0)
    total_trades = len(returns)

    return float(winning_trades / total_trades)


def calculate_profit_factor(returns: np.ndarray) -> float:
    """計算盈虧比 (Profit Factor)

    盈虧比是總盈利與總虧損的比值，衡量盈利能力。

    公式: PF = 總盈利 / 總虧損

    參數:
        returns: 收益序列

    返回:
        profit_factor: 盈虧比

    解釋:
        - PF > 2.0: 優秀
        - PF > 1.5: 良好
        - PF > 1.0: 盈利
        - PF = 1.0: 打平
        - PF < 1.0: 虧損

    注意:
        - 考慮了盈虧大小
        - 與勝率互補
    """
    if len(returns) == 0:
        return 0.0

    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if profits > 0 else 0.0

    return float(profits / losses)


# ============================================================================
# 策略評估器類別
# ============================================================================

class RLStrategyEvaluator:
    """RL 策略評估器

    此類別提供完整的策略評估功能，包括：
    - 在測試環境中運行策略
    - 收集交易記錄
    - 計算績效指標
    - 生成評估報告
    - 與基準策略對比

    使用範例:
        >>> evaluator = RLStrategyEvaluator(test_env, policy)
        >>> results = evaluator.evaluate(num_episodes=100)
        >>> evaluator.print_report(results)
        >>> evaluator.save_report(results, 'results/evaluation/')
    """

    def __init__(self, env, policy, baseline_strategies: Optional[Dict] = None):
        """初始化評估器

        參數:
            env: 測試環境 (LOBTradingEnv 實例)
            policy: 訓練好的策略 (RLlib Policy 或 Algorithm)
            baseline_strategies: 基準策略字典 (可選)
        """
        self.env = env
        self.policy = policy
        self.baseline_strategies = baseline_strategies or {}

        logger.info("✅ RLStrategyEvaluator 已初始化")

    def evaluate(self, num_episodes: int = 100,
                deterministic: bool = True) -> Dict[str, Any]:
        """評估策略性能

        在測試集上運行多個 episode，收集統計數據。

        參數:
            num_episodes: 評估 episode 數量
            deterministic: 是否使用確定性策略 (不探索)

        返回:
            results: 評估結果字典
        """
        logger.info(f"開始評估策略 (episodes={num_episodes})")

        episode_returns = []
        episode_lengths = []
        all_trades = []
        all_actions = []
        all_positions = []

        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_return = 0.0
            episode_length = 0

            while not (done or truncated):
                # 選擇動作
                if hasattr(self.policy, 'compute_single_action'):
                    result = self.policy.compute_single_action(
                        obs,
                        explore=not deterministic
                    )
                    # 處理不同的返回格式
                    if isinstance(result, tuple):
                        action = result[0]
                    else:
                        action = result
                else:
                    action = self.policy.compute_actions([obs])[0][0]

                # 執行動作
                obs, reward, done, truncated, info = self.env.step(action)

                episode_return += reward
                episode_length += 1

                # 記錄交易信息
                all_actions.append(action)
                if 'position' in info:
                    all_positions.append(info['position'])

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

            if (ep + 1) % 20 == 0:
                logger.info(f"  完成 {ep + 1}/{num_episodes} episodes")

        # 計算績效指標
        returns_array = np.array(episode_returns)

        total_return = np.sum(returns_array)
        avg_return = np.mean(returns_array)
        sharpe = calculate_sharpe_ratio(returns_array)
        mdd, peak_idx, trough_idx = calculate_max_drawdown(returns_array)
        calmar = calculate_calmar_ratio(total_return, mdd) if mdd > 0 else 0.0
        win_rate = calculate_win_rate(returns_array)
        profit_factor = calculate_profit_factor(returns_array)

        results = {
            'num_episodes': num_episodes,
            'total_return': float(total_return),
            'avg_return_per_episode': float(avg_return),
            'avg_episode_length': float(np.mean(episode_lengths)),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(mdd),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'returns_std': float(np.std(returns_array)),
            'returns_min': float(np.min(returns_array)),
            'returns_max': float(np.max(returns_array)),
            'action_distribution': {
                'hold': float(np.mean(np.array(all_actions) == 0)),
                'buy': float(np.mean(np.array(all_actions) == 1)),
                'sell': float(np.mean(np.array(all_actions) == 2)),
            }
        }

        logger.info("✅ 策略評估完成")
        return results

    def print_report(self, results: Dict):
        """打印評估報告

        參數:
            results: evaluate() 返回的結果字典
        """
        print("\n" + "="*80)
        print("RL 策略評估報告")
        print("="*80)
        print(f"\n📊 基礎統計:")
        print(f"  - Episodes: {results['num_episodes']}")
        print(f"  - 平均 Episode 長度: {results['avg_episode_length']:.1f}")
        print(f"\n💰 收益指標:")
        print(f"  - 總收益: {results['total_return']:.4f}")
        print(f"  - 平均 Episode 收益: {results['avg_return_per_episode']:.4f}")
        print(f"  - 收益標準差: {results['returns_std']:.4f}")
        print(f"  - 最小收益: {results['returns_min']:.4f}")
        print(f"  - 最大收益: {results['returns_max']:.4f}")
        print(f"\n📈 風險調整指標:")
        print(f"  - Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"  - 最大回撤: {results['max_drawdown']*100:.2f}%")
        print(f"  - Calmar Ratio: {results['calmar_ratio']:.4f}")
        print(f"\n🎯 交易統計:")
        print(f"  - 勝率: {results['win_rate']*100:.2f}%")
        print(f"  - 盈虧比: {results['profit_factor']:.4f}")
        print(f"\n🎬 動作分佈:")
        print(f"  - Hold: {results['action_distribution']['hold']*100:.1f}%")
        print(f"  - Buy: {results['action_distribution']['buy']*100:.1f}%")
        print(f"  - Sell: {results['action_distribution']['sell']*100:.1f}%")
        print("="*80 + "\n")

    def save_report(self, results: Dict, output_dir: str):
        """保存評估報告到文件

        參數:
            results: evaluate() 返回的結果字典
            output_dir: 輸出目錄
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        json_path = output_path / 'rl_evaluation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 評估報告已保存: {json_path}")
