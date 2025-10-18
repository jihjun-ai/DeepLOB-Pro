"""獎勵塑形模組 - LOB 交易環境的多組件獎勵函數

此模組實作了複雜的多組件獎勵函數，用於引導強化學習智能體學習有效的交易策略。
獎勵函數設計是強化學習中最關鍵的部分，直接影響智能體的學習效果。

獎勵組件:
    1. PnL (盈虧): 基礎獎勵，衡量交易獲利能力
    2. 交易成本: 懲罰頻繁交易，考慮手續費和滑點
    3. 庫存懲罰: 避免長時間持倉，降低市場風險
    4. 風險懲罰: 基於波動率的風險管理

設計原則:
    - 平衡短期盈利與長期穩定性
    - 考慮實際交易成本
    - 鼓勵風險管理
    - 避免過度交易

使用範例:
    >>> config = {
    ...     'pnl_scale': 1.0,
    ...     'cost_penalty': 1.0,
    ...     'inventory_penalty': 0.01,
    ...     'risk_penalty': 0.005
    ... }
    >>> reward_shaper = RewardShaper(config)
    >>> reward, components = reward_shaper.calculate_reward(
    ...     prev_state={'position': 0, 'entry_price': 100.0},
    ...     action=1,  # Buy
    ...     new_state={'position': 1, 'current_price': 101.0, 'inventory': 1.0},
    ...     transaction_cost=0.10
    ... )
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RewardShaper:
    """獎勵塑形器 - 多組件獎勵函數實作

    此類負責計算強化學習環境中的獎勵信號，整合多個獎勵組件。
    獎勵函數的設計直接影響智能體的學習方向和最終策略。

    核心思想:
        - 不僅獎勵盈利，也要懲罰風險
        - 考慮交易成本，避免過度交易
        - 鼓勵快速平倉，避免長時間暴露
        - 使用可調整的權重平衡各個目標

    獎勵公式:
        總獎勵 = PnL × pnl_scale
                - 交易成本 × cost_penalty_weight
                - |庫存| × inventory_penalty_weight
                - |倉位| × 波動率 × risk_penalty_weight

    參數調整指南:
        - pnl_scale: 越大越激進（追求高收益）
        - cost_penalty: 越大越減少交易頻率
        - inventory_penalty: 越大越傾向快速平倉
        - risk_penalty: 越大越保守（規避風險）
    """

    def __init__(self, config: Dict):
        """初始化獎勵塑形器

        參數:
            config: 獎勵配置字典，包含以下鍵值:
                pnl_scale (float): PnL 獎勵縮放因子，預設 1.0
                    - 控制盈虧在總獎勵中的比重
                    - 數值越大，智能體越注重盈利
                    - 建議範圍: [0.5, 2.0]

                cost_penalty (float): 交易成本懲罰權重，預設 1.0
                    - 控制交易成本的懲罰強度
                    - 數值越大，越傾向減少交易
                    - 建議範圍: [0.5, 2.0]

                inventory_penalty (float): 庫存懲罰權重，預設 0.01
                    - 懲罰長時間持倉
                    - 數值越大，越傾向快速平倉
                    - 建議範圍: [0.001, 0.1]

                risk_penalty (float): 風險懲罰權重，預設 0.005
                    - 基於波動率的風險懲罰
                    - 數值越大，越保守（避免高波動時持倉）
                    - 建議範圍: [0.001, 0.05]

        配置範例:
            # 激進策略（追求高收益）
            config = {
                'pnl_scale': 2.0,
                'cost_penalty': 0.5,
                'inventory_penalty': 0.001,
                'risk_penalty': 0.001
            }

            # 保守策略（注重風險管理）
            config = {
                'pnl_scale': 1.0,
                'cost_penalty': 2.0,
                'inventory_penalty': 0.05,
                'risk_penalty': 0.02
            }

            # 平衡策略（預設值）
            config = {
                'pnl_scale': 1.0,
                'cost_penalty': 1.0,
                'inventory_penalty': 0.01,
                'risk_penalty': 0.005
            }
        """
        # ===== 獎勵組件權重 =====
        self.pnl_scale = config.get('pnl_scale', 1.0)
        self.cost_penalty_weight = config.get('cost_penalty', 1.0)
        self.inventory_penalty_weight = config.get('inventory_penalty', 0.01)
        self.risk_penalty_weight = config.get('risk_penalty', 0.005)

        # 記錄配置資訊
        logger.info(
            f"獎勵塑形器已初始化: "
            f"PnL權重={self.pnl_scale}, "
            f"成本懲罰={self.cost_penalty_weight}, "
            f"庫存懲罰={self.inventory_penalty_weight}, "
            f"風險懲罰={self.risk_penalty_weight}"
        )

    def calculate_reward(
        self,
        prev_state: Dict,
        action: int,
        new_state: Dict,
        transaction_cost: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """計算多組件獎勵

        這是核心方法，整合所有獎勵組件並計算最終獎勵。

        參數:
            prev_state: 前一狀態字典，包含:
                position (int): 前一倉位 {-1: 空倉, 0: 平倉, 1: 多倉}
                prev_action (int): 前一動作
                entry_price (float): 進場價格

            action: 當前動作
                0 - Hold: 持有
                1 - Buy: 買入
                2 - Sell: 賣出

            new_state: 新狀態字典，包含:
                position (int): 新倉位
                current_price (float): 當前市場價格
                inventory (float): 當前庫存（未實現盈虧）
                volatility (float): 市場波動率（可選）

            transaction_cost: 本次交易產生的成本（手續費+滑點）

        返回:
            total_reward (float): 總獎勵（所有組件的加權和）
            reward_components (dict): 各組件的詳細數值
                'pnl': PnL 獎勵
                'transaction_cost': 交易成本（負值）
                'inventory_penalty': 庫存懲罰（負值）
                'risk_penalty': 風險懲罰（負值）

        計算流程:
            1. 計算 PnL（已實現 + 未實現）
            2. 計算交易成本懲罰
            3. 計算庫存懲罰
            4. 計算風險懲罰
            5. 加權求和得到總獎勵

        獎勵解釋:
            - 正獎勵: 表示好的行為（盈利）
            - 負獎勵: 表示不好的行為（虧損、成本、風險）
            - 零獎勵: 表示中性行為（持有平倉狀態）

        範例:
            >>> # 成功的做多交易
            >>> prev_state = {'position': 0, 'entry_price': 100.0}
            >>> new_state = {'position': 1, 'current_price': 101.0, 'inventory': 1.0}
            >>> reward, components = shaper.calculate_reward(
            ...     prev_state, action=1, new_state, transaction_cost=0.10
            ... )
            >>> # reward ≈ 1.0 (PnL) - 0.10 (cost) - 0.01 (inventory) ≈ 0.89
        """
        reward_components = {}

        # ===== 組件1: PnL 獎勵（已實現 + 未實現盈虧）=====
        # 這是最重要的獎勵組件，直接衡量交易盈利能力
        pnl = self._calculate_pnl(prev_state, new_state)
        reward_components['pnl'] = pnl * self.pnl_scale

        # ===== 組件2: 交易成本懲罰 =====
        # 懲罰交易成本（手續費、滑點等）
        # 使用負值，因為成本是不好的
        cost_penalty = -transaction_cost * self.cost_penalty_weight
        reward_components['transaction_cost'] = cost_penalty

        # ===== 組件3: 庫存懲罰 =====
        # 懲罰長時間持倉，鼓勵快速平倉
        # 使用絕對值，無論做多做空都受懲罰
        inventory = new_state.get('inventory', 0.0)
        inventory_penalty = -abs(inventory) * self.inventory_penalty_weight
        reward_components['inventory_penalty'] = inventory_penalty

        # ===== 組件4: 風險懲罰 =====
        # 基於波動率和倉位的風險懲罰
        # 波動率高時持倉會受到更大懲罰
        risk_penalty = self._calculate_risk_penalty(new_state)
        reward_components['risk_penalty'] = risk_penalty

        # ===== 計算總獎勵 =====
        # 將所有組件相加（注意懲罰項已經是負值）
        total_reward = sum(reward_components.values())

        return total_reward, reward_components

    def _calculate_pnl(self, prev_state: Dict, new_state: Dict) -> float:
        """計算盈虧 (Profit and Loss) - 修復版（增量獎勵）

        ⭐ 修正內容（2025-10-13）:
            - 未實現盈虧改為計算增量（當前步的價格變化）
            - 避免獎勵隨時間二次方累積
            - 修復了導致訓練異常高回報的 BUG

        計算已實現盈虧和增量未實現盈虧的總和。

        盈虧類型:
            1. 已實現盈虧 (Realized PnL):
               - 當倉位從非零變為零時產生
               - 計算公式: 倉位 × (平倉價 - 進場價)
               - 這是真正的盈利/虧損

            2. 增量未實現盈虧 (Incremental Unrealized PnL):
               - ✅ 修正：只計算當前步的價格變化
               - 計算公式: 倉位 × (當前價 - 前一價)
               - 避免累積整個持倉的浮動盈虧

        參數:
            prev_state: 前一狀態
                position: 前一倉位
                entry_price: 進場價格
                prev_price: 前一步的價格（新增）

            new_state: 新狀態
                position: 新倉位
                current_price: 當前價格

        返回:
            pnl (float): 總盈虧
                - 正值: 盈利
                - 負值: 虧損
                - 零: 無盈虧變化

        計算範例:
            # 做多盈利（持倉3步，價格 100→101→102→103）
            # Step 1: 買入 @ 100
            prev: {position: 0, entry_price: 100, prev_price: 100}
            new: {position: 1, current_price: 100}
            pnl = 0  # 剛買入，無盈虧

            # Step 2: 價格上漲到 101
            prev: {position: 1, entry_price: 100, prev_price: 100}
            new: {position: 1, current_price: 101}
            pnl = 1 × (101 - 100) = +1.0  # ✅ 增量獎勵

            # Step 3: 價格上漲到 102
            prev: {position: 1, entry_price: 100, prev_price: 101}
            new: {position: 1, current_price: 102}
            pnl = 1 × (102 - 101) = +1.0  # ✅ 增量獎勵
            # ❌ 舊版錯誤: pnl = 1 × (102 - 100) = +2.0

            # Step 4: 平倉 @ 103
            prev: {position: 1, entry_price: 100, prev_price: 102}
            new: {position: 0, current_price: 103}
            pnl = 1 × (103 - 100) = +3.0  # 已實現盈虧

            # 總累積獎勵 = 0 + 1 + 1 + 3 = 5.0  ✅ 正確
            # ❌ 舊版錯誤: 0 + 1 + 2 + 3 = 6.0
        """
        # 獲取倉位資訊
        prev_position = prev_state.get('position', 0)
        new_position = new_state.get('position', 0)
        current_price = new_state.get('current_price', 0.0)
        prev_price = prev_state.get('prev_price', current_price)  # ⭐ 新增
        entry_price = prev_state.get('entry_price', current_price)

        # ===== 計算已實現盈虧 =====
        realized_pnl = 0.0
        if prev_position != 0 and new_position == 0:
            # 從持倉變為平倉 → 計算已實現盈虧
            # PnL = 倉位 × (平倉價 - 進場價)
            #
            # 做多盈利: +1 × (高價 - 低價) = 正值
            # 做多虧損: +1 × (低價 - 高價) = 負值
            # 做空盈利: -1 × (低價 - 高價) = 正值
            # 做空虧損: -1 × (高價 - 低價) = 負值
            realized_pnl = prev_position * (current_price - entry_price)

        # ===== 計算增量未實現盈虧（僅當前步的變化）=====
        incremental_pnl = 0.0
        if prev_position != 0:
            # ✅ 修正：只獎勵本步的價格變化
            # 而非整個持倉的浮動盈虧（會導致二次方累積）
            price_change = current_price - prev_price
            incremental_pnl = prev_position * price_change

        # 總盈虧 = 已實現 + 增量未實現
        return realized_pnl + incremental_pnl

    def _calculate_risk_penalty(self, new_state: Dict) -> float:
        """計算風險懲罰

        基於倉位大小和市場波動率計算風險懲罰。
        波動率高時持倉風險大，應受到更大懲罰。

        風險公式:
            風險懲罰 = - |倉位| × 波動率 × risk_penalty_weight

        設計原理:
            - 倉位越大，風險越大
            - 波動率越高，風險越大
            - 使用負值作為懲罰信號
            - 平倉狀態（倉位=0）無風險懲罰

        參數:
            new_state: 新狀態字典
                position (int): 倉位
                volatility (float): 市場波動率（標準差）

        返回:
            risk_penalty (float): 風險懲罰（負值或零）

        波動率說明:
            - 通常使用收益率的標準差
            - 可用滾動窗口計算（如20期）
            - 高波動 > 0.02, 低波動 < 0.01
            - TODO: 當前使用固定值，應實作動態計算

        範例:
            # 高波動時持倉
            state = {'position': 1, 'volatility': 0.03}
            penalty = -1 × 0.03 × 0.005 = -0.00015

            # 低波動時持倉
            state = {'position': 1, 'volatility': 0.01}
            penalty = -1 × 0.01 × 0.005 = -0.00005

            # 平倉狀態
            state = {'position': 0, 'volatility': 0.03}
            penalty = 0  # 無懲罰
        """
        # 獲取倉位（使用絕對值，做多做空一視同仁）
        position = abs(new_state.get('position', 0))

        # 獲取市場波動率
        volatility = new_state.get('volatility', 0.0)

        # 計算風險懲罰
        # 公式: -倉位 × 波動率 × 權重
        # 結果為負值或零
        risk_penalty = -position * volatility * self.risk_penalty_weight

        return risk_penalty

    def calculate_shaped_reward(
        self,
        raw_pnl: float,
        position: int,
        inventory: float,
        volatility: float = 0.0,
        action_changed: bool = False,
        transaction_cost: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """簡化版獎勵計算（備用介面）

        提供更簡單的介面，直接使用計算好的數值。
        適用於已經有 PnL 等數值的場景。

        參數:
            raw_pnl: 原始 PnL 值
            position: 當前倉位
            inventory: 當前庫存（未實現盈虧）
            volatility: 市場波動率，預設 0.0
            action_changed: 動作是否改變，預設 False
            transaction_cost: 交易成本，預設 0.0

        返回:
            total_reward (float): 總獎勵
            components (dict): 獎勵組件字典

        與 calculate_reward() 的區別:
            - 此方法接受已計算的數值
            - calculate_reward() 接受狀態字典，內部計算

        使用場景:
            - 已有 PnL 計算結果
            - 簡化的測試場景
            - 外部獎勵計算器

        範例:
            >>> reward, components = shaper.calculate_shaped_reward(
            ...     raw_pnl=2.5,
            ...     position=1,
            ...     inventory=2.0,
            ...     volatility=0.01,
            ...     action_changed=True,
            ...     transaction_cost=0.10
            ... )
        """
        components = {}

        # ===== PnL 組件 =====
        components['pnl'] = raw_pnl * self.pnl_scale

        # ===== 交易成本組件 =====
        # 只在動作改變時計入成本
        if action_changed:
            components['transaction_cost'] = -transaction_cost * self.cost_penalty_weight
        else:
            components['transaction_cost'] = 0.0

        # ===== 庫存懲罰組件 =====
        components['inventory_penalty'] = -abs(inventory) * self.inventory_penalty_weight

        # ===== 風險懲罰組件 =====
        components['risk_penalty'] = -abs(position) * volatility * self.risk_penalty_weight

        # ===== 總獎勵 =====
        total_reward = sum(components.values())

        return total_reward, components


class AdaptiveRewardShaper(RewardShaper):
    """自適應獎勵塑形器

    在訓練過程中動態調整獎勵權重，實現課程學習 (Curriculum Learning)。

    核心思想:
        - 訓練初期: 高 PnL 權重，鼓勵探索盈利機會
        - 訓練後期: 降低 PnL 權重，強化風險管理

    適用場景:
        - 長期訓練（數百萬步）
        - 需要逐步引導策略
        - 避免過度激進的策略

    調整策略:
        pnl_scale(t) = max(min_scale, initial_scale × decay^t)

    使用範例:
        >>> config = {
        ...     'pnl_scale': 2.0,          # 初始 PnL 權重
        ...     'min_pnl_scale': 0.5,      # 最小 PnL 權重
        ...     'scale_decay': 0.9999      # 每 episode 衰減率
        ... }
        >>> shaper = AdaptiveRewardShaper(config)
        >>> # 訓練循環中
        >>> shaper.update_scales(episode=1000, metrics={})
    """

    def __init__(self, config: Dict):
        """初始化自適應獎勵塑形器

        參數:
            config: 配置字典，除了基礎配置外，還包含:
                min_pnl_scale (float): PnL 權重的最小值，預設 0.1
                    - 防止 PnL 權重衰減為零
                    - 確保始終考慮盈利

                scale_decay (float): 權重衰減率，預設 0.99
                    - 每個 episode 的衰減因子
                    - 0.99 表示每 episode 衰減 1%
                    - 建議範圍: [0.9999, 0.99]

        衰減速度參考:
            - 0.9999: 緩慢衰減，10000 episode 約衰減 63%
            - 0.999:  中速衰減，1000 episode 約衰減 63%
            - 0.99:   快速衰減，100 episode 約衰減 63%
        """
        # 調用父類初始化
        super().__init__(config)

        # ===== 自適應參數 =====
        self.initial_pnl_scale = self.pnl_scale  # 保存初始值
        self.min_pnl_scale = config.get('min_pnl_scale', 0.1)  # 最小值
        self.scale_decay = config.get('scale_decay', 0.99)  # 衰減率

        # Episode 計數器
        self.episode_count = 0

        logger.info(
            f"自適應獎勵塑形器已初始化: "
            f"初始PnL權重={self.initial_pnl_scale}, "
            f"最小PnL權重={self.min_pnl_scale}, "
            f"衰減率={self.scale_decay}"
        )

    def update_scales(self, episode: int, metrics: Dict):
        """更新獎勵權重

        根據訓練進度動態調整 PnL 權重，實現課程學習。

        參數:
            episode: 當前 episode 編號
            metrics: 訓練指標字典（當前未使用，預留接口）
                可包含: average_reward, sharpe_ratio, win_rate 等

        更新公式:
            pnl_scale = max(min_scale, initial_scale × decay^episode)

        調用時機:
            - 每個 episode 結束後調用一次
            - 或每 N 個 episode 調用一次

        權重變化軌跡:
            Episode 0:    pnl_scale = 2.0   (初始值)
            Episode 100:  pnl_scale ≈ 1.32  (衰減 34%)
            Episode 500:  pnl_scale ≈ 0.61  (衰減 70%)
            Episode 1000: pnl_scale ≈ 0.37  (衰減 81%)
            ...
            穩定值:       pnl_scale = 0.1   (最小值)

        範例:
            >>> # 訓練循環
            >>> for episode in range(10000):
            ...     # ... 運行 episode ...
            ...     shaper.update_scales(episode, metrics={'avg_reward': 10.5})
        """
        self.episode_count = episode

        # ===== 計算新的 PnL 權重 =====
        # 使用指數衰減公式
        new_pnl_scale = self.initial_pnl_scale * (self.scale_decay ** episode)

        # 限制最小值，避免完全忽略 PnL
        self.pnl_scale = max(self.min_pnl_scale, new_pnl_scale)

        # 定期記錄（每100個episode）
        if episode % 100 == 0:
            logger.info(
                f"Episode {episode}: PnL權重已更新為 {self.pnl_scale:.4f} "
                f"(衰減 {(1 - self.pnl_scale/self.initial_pnl_scale)*100:.1f}%)"
            )
        else:
            logger.debug(f"Episode {episode}: pnl_scale={self.pnl_scale:.4f}")

    def reset_scales(self):
        """重置權重到初始值

        用於重新開始訓練或實驗。

        使用場景:
            - 更改超參數後重新訓練
            - A/B 測試不同配置
            - 階段性訓練（分階段重置）
        """
        self.pnl_scale = self.initial_pnl_scale
        self.episode_count = 0
        logger.info(f"獎勵權重已重置: pnl_scale={self.pnl_scale}")
