"""基準策略 - 用於性能對比

此模組實作多種基準策略，用於評估 RL 策略的相對性能。

基準策略:
    1. RandomStrategy: 隨機選擇動作
    2. AlwaysHoldStrategy: 總是持有不交易
    3. BuyAndHoldStrategy: 買入並持有
    4. DeepLOBOnlyStrategy: 僅基於 DeepLOB 預測交易

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-10
"""

import numpy as np
from typing import Any
import torch


class BaseStrategy:
    """基準策略基類

    所有基準策略的父類，定義統一接口。
    """

    def __init__(self, name: str):
        self.name = name

    def compute_action(self, obs: np.ndarray, **kwargs) -> int:
        """計算動作

        參數:
            obs: 觀測 (48,) 或 (100, 40)
            **kwargs: 額外參數

        返回:
            action: 動作 {0: Hold, 1: Buy, 2: Sell}
        """
        raise NotImplementedError

    def reset(self):
        """重置策略狀態"""
        pass


class RandomStrategy(BaseStrategy):
    """隨機策略

    隨機選擇動作，作為最簡單的基準。
    預期性能: Sharpe ≈ 0, Win Rate ≈ 33%

    用途:
        - 驗證環境正確性
        - 最低性能基準
    """

    def __init__(self):
        super().__init__("Random")

    def compute_action(self, obs: np.ndarray, **kwargs) -> int:
        return np.random.randint(0, 3)


class AlwaysHoldStrategy(BaseStrategy):
    """總是持有策略

    總是選擇 Hold 動作，不進行任何交易。
    預期性能: Return = 0, No Trades

    用途:
        - 零交易成本基準
        - 驗證交易成本影響
    """

    def __init__(self):
        super().__init__("AlwaysHold")

    def compute_action(self, obs: np.ndarray, **kwargs) -> int:
        return 0  # Always Hold


class BuyAndHoldStrategy(BaseStrategy):
    """買入並持有策略

    第一步買入，之後一直持有。
    預期性能: 取決於市場趨勢

    用途:
        - 趨勢市場基準
        - 被動投資對比
    """

    def __init__(self):
        super().__init__("BuyAndHold")
        self.first_step = True

    def compute_action(self, obs: np.ndarray, **kwargs) -> int:
        if self.first_step:
            self.first_step = False
            return 1  # Buy
        return 0  # Hold

    def reset(self):
        self.first_step = True


class DeepLOBOnlyStrategy(BaseStrategy):
    """僅基於 DeepLOB 預測的策略

    根據 DeepLOB 的價格預測選擇動作：
    - 預測上漲 → Buy (1)
    - 預測下跌 → Sell (2)
    - 預測持平 → Hold (0)

    假設:
        - 觀測格式: (48,) = LOB(40) + DeepLOB(3) + State(5)
        - DeepLOB 輸出: [下跌概率, 持平概率, 上漲概率]

    用途:
        - 驗證 DeepLOB 預測質量
        - 評估 RL 的增量價值
    """

    def __init__(self, confidence_threshold: float = 0.4):
        """初始化策略

        參數:
            confidence_threshold: 置信度閾值
                只有當預測概率超過此閾值時才交易
        """
        super().__init__("DeepLOBOnly")
        self.confidence_threshold = confidence_threshold

    def compute_action(self, obs: np.ndarray, **kwargs) -> int:
        """根據 DeepLOB 預測選擇動作

        邏輯:
            1. 提取 DeepLOB 預測 (obs[40:43])
            2. 找到最大概率的類別
            3. 如果概率 > 閾值，執行對應動作
            4. 否則選擇 Hold
        """
        # 提取 DeepLOB 預測概率 (假設觀測格式正確)
        if len(obs.shape) == 1 and len(obs) >= 43:
            # 向量模式: obs[40:43] 是 DeepLOB 預測
            deeplob_probs = obs[40:43]
        else:
            # 序列模式或格式不符，使用隨機
            return np.random.randint(0, 3)

        # 找到最大概率的類別
        predicted_class = np.argmax(deeplob_probs)
        max_prob = deeplob_probs[predicted_class]

        # 檢查置信度
        if max_prob < self.confidence_threshold:
            return 0  # Hold (不確定時不交易)

        # 映射預測到動作
        # DeepLOB 輸出: [下跌(0), 持平(1), 上漲(2)]
        # 動作: [Hold(0), Buy(1), Sell(2)]
        if predicted_class == 2:  # 預測上漲
            return 1  # Buy
        elif predicted_class == 0:  # 預測下跌
            return 2  # Sell
        else:  # 預測持平
            return 0  # Hold
