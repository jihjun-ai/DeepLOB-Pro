"""評估模組 - 策略性能評估與基準對比

此模組提供完整的交易策略評估功能，包括：
- 離線評估（測試集）
- 績效指標計算（Sharpe, MDD, Win Rate）
- 基準策略對比（Random, Buy-Hold, DeepLOB-Only）
- 詳細交易分析

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-10
"""

from .evaluator import (
    RLStrategyEvaluator,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
)

from .baseline_strategies import (
    RandomStrategy,
    BuyAndHoldStrategy,
    DeepLOBOnlyStrategy,
    AlwaysHoldStrategy,
)

__all__ = [
    # 評估器
    'RLStrategyEvaluator',
    # 指標計算
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    # 基準策略
    'RandomStrategy',
    'BuyAndHoldStrategy',
    'DeepLOBOnlyStrategy',
    'AlwaysHoldStrategy',
]
