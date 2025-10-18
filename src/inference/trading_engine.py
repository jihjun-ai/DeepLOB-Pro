"""即時交易引擎

此模組負責將訓練好的 RL 模型部署到實際交易環境。

核心功能:
  1. 載入訓練好的模型（DeepLOB + RL Policy）
  2. 接收實時 LOB 數據
  3. 生成交易信號
  4. 執行風險控制
  5. 送出交易委託

使用範例:
    >>> engine = TradingEngine(
    ...     rl_checkpoint="checkpoints/rl/best_model",
    ...     deeplob_checkpoint="model/deeplob/deeplob_generic_best.pth"
    ... )
    >>>
    >>> # 接收實時 LOB 數據
    >>> lob_data = get_live_lob_data()  # 從券商 API 獲取
    >>>
    >>> # 生成交易決策
    >>> action, confidence = engine.predict(lob_data)
    >>>
    >>> # 執行交易
    >>> if action == 1:  # Buy
    ...     engine.execute_trade("buy", quantity=1)

作者: RLlib-DeepLOB 專案團隊
更新: 2025-10-12
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from collections import deque

from ray.rllib.algorithms.ppo import PPO
from ..models.deeplob import DeepLOB

logger = logging.getLogger(__name__)


class TradingEngine:
    """即時交易引擎

    整合 DeepLOB 預測模型和 RL 策略，提供端到端的交易決策功能。
    """

    def __init__(
        self,
        rl_checkpoint: str,
        deeplob_checkpoint: str,
        device: str = "cuda",
        max_position: int = 1,
        buy_cost_rate: float = 0.00071,
        sell_cost_rate: float = 0.00371,
        long_only: bool = False,
    ):
        """初始化交易引擎

        參數:
            rl_checkpoint: RL 模型檢查點路徑
            deeplob_checkpoint: DeepLOB 模型檢查點路徑
            device: 運算設備 ("cuda" 或 "cpu")
            max_position: 最大持倉 {-1, 0, 1}（long_only=True 時僅支援 0~1）
            buy_cost_rate: 買入成本率（台股預設 0.071% = 0.00071，券商手續費 50% 折扣）
            sell_cost_rate: 賣出成本率（台股預設 0.371% = 0.00371，手續費 + 0.3% 證交稅）
            long_only: 是否只做多（True: 禁止做空，False: 允許多空）

        注意:
            - 台股實際成本: 買 0.071% + 賣 0.371% = 來回 0.442%
            - 訓練模型使用的成本: 0.1% (configs/rl_config.yaml: transaction_cost_rate=0.001)
            - 實際交易會因較高成本而降低交易頻率
        """
        self.device = device
        self.max_position = max_position
        self.buy_cost_rate = buy_cost_rate
        self.sell_cost_rate = sell_cost_rate
        self.long_only = long_only

        # 載入模型
        logger.info("載入交易引擎...")
        self._load_deeplob_model(deeplob_checkpoint)
        self._load_rl_model(rl_checkpoint)

        # 交易狀態
        self.position = 0  # 當前持倉 {-1: 做空, 0: 空倉, 1: 做多}
        self.entry_price = 0.0
        self.balance = 0.0
        self.total_pnl = 0.0

        # LOB 歷史（用於 DeepLOB 預測）
        self.lob_history = deque(maxlen=100)  # 保留最近 100 時間步

        # LSTM 隱藏狀態（用於 RL 策略）
        self.lstm_state = None

        logger.info("✅ 交易引擎已就緒")

    def _load_deeplob_model(self, checkpoint_path: str):
        """載入 DeepLOB 模型"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 DeepLOB 模型: {checkpoint_path}")

        logger.info(f"載入 DeepLOB 模型: {checkpoint_path}")

        # 載入檢查點
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 解析模型配置
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'config' in checkpoint and 'model' in checkpoint['config']:
                model_config = checkpoint['config']['model']
            else:
                model_config = checkpoint.get('model_config', {})
        else:
            state_dict = checkpoint
            model_config = {}

        # 創建模型
        deeplob_params = {
            'input_shape': tuple(model_config.get('input_shape', [100, 20])),
            'num_classes': model_config.get('num_classes', 3),
            'conv1_filters': model_config.get('conv1_filters', 32),
            'conv2_filters': model_config.get('conv2_filters', 32),
            'conv3_filters': model_config.get('conv3_filters', 32),
            'lstm_hidden_size': model_config.get('lstm_hidden_size', 64),
            'fc_hidden_size': model_config.get('fc_hidden_size', 64),
            'dropout': model_config.get('dropout', 0.2),
        }

        self.deeplob_model = DeepLOB(**deeplob_params)
        self.deeplob_model.load_state_dict(state_dict)
        self.deeplob_model.to(self.device)
        self.deeplob_model.eval()

        logger.info("✅ DeepLOB 模型載入成功")

    def _load_rl_model(self, checkpoint_path: str):
        """載入 RL 策略模型"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 RL 模型: {checkpoint_path}")

        logger.info(f"載入 RL 策略: {checkpoint_path}")

        # 使用 RLlib 載入檢查點
        self.rl_agent = PPO.from_checkpoint(str(checkpoint_path.absolute()))

        logger.info("✅ RL 策略載入成功")

    def preprocess_lob(self, lob_data: np.ndarray) -> np.ndarray:
        """預處理 LOB 數據

        參數:
            lob_data: 原始 LOB 數據 (20,) 或 (5, 4)
                      [bid_price_1, bid_vol_1, ..., ask_price_1, ask_vol_1, ...]

        返回:
            標準化後的 LOB 數據 (20,)
        """
        # 確保形狀正確
        lob_data = np.array(lob_data).flatten()
        if lob_data.shape[0] != 20:
            raise ValueError(f"LOB 數據維度錯誤: 期望 20 維，實際 {lob_data.shape[0]} 維")

        # Z-score 標準化（使用訓練時的統計量）
        # TODO: 從訓練數據載入 mean/std
        # mean = np.load("data/processed/lob_mean.npy")
        # std = np.load("data/processed/lob_std.npy")
        # lob_normalized = (lob_data - mean) / (std + 1e-8)

        # 暫時使用簡單標準化
        lob_normalized = (lob_data - np.mean(lob_data)) / (np.std(lob_data) + 1e-8)

        return lob_normalized

    def get_deeplob_prediction(self, lob_sequence: np.ndarray) -> np.ndarray:
        """使用 DeepLOB 預測價格變動

        參數:
            lob_sequence: LOB 序列 (100, 20)

        返回:
            價格預測概率 (3,) [下跌, 持平, 上漲]
        """
        # 轉換為 PyTorch 張量
        lob_tensor = torch.FloatTensor(lob_sequence).unsqueeze(0).to(self.device)  # (1, 100, 20)

        # 預測
        with torch.no_grad():
            logits = self.deeplob_model(lob_tensor)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()[0]  # (3,)

    def get_observation(self, lob_data: np.ndarray, current_price: float) -> np.ndarray:
        """構建 RL 策略的觀測

        參數:
            lob_data: 當前 LOB 數據 (20,)
            current_price: 當前價格

        返回:
            觀測向量 (28,) = LOB(20) + DeepLOB(3) + State(5)
        """
        # 1. LOB 特徵 (20 維)
        lob_features = self.preprocess_lob(lob_data)

        # 2. DeepLOB 預測 (3 維)
        self.lob_history.append(lob_features)

        if len(self.lob_history) < 100:
            # 如果歷史不足 100，用零填充
            lob_sequence = np.zeros((100, 20), dtype=np.float32)
            lob_sequence[-len(self.lob_history):] = list(self.lob_history)
        else:
            lob_sequence = np.array(self.lob_history)

        deeplob_pred = self.get_deeplob_prediction(lob_sequence)

        # 3. 交易狀態 (5 維)
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position

        state_features = np.array([
            float(self.position),  # 持倉 {-1, 0, 1}
            unrealized_pnl / 1000.0,  # 未實現損益（標準化）
            float(self.position),  # 庫存（簡化版）
            self.entry_price / 100.0 if self.entry_price > 0 else 0.0,  # 成本（標準化）
            0.5,  # 時間進度（假設中間）
        ], dtype=np.float32)

        # 組合觀測
        observation = np.concatenate([lob_features, deeplob_pred, state_features])

        return observation

    def predict(
        self,
        lob_data: np.ndarray,
        current_price: float
    ) -> Tuple[int, float, Dict[str, Any]]:
        """生成交易決策

        參數:
            lob_data: 當前 LOB 數據 (20,)
            current_price: 當前價格

        返回:
            (action, confidence, info):
                - action: 交易動作 {0: Hold, 1: Buy, 2: Sell}
                - confidence: 信心度 [0, 1]
                - info: 詳細信息字典
        """
        # 構建觀測
        obs = self.get_observation(lob_data, current_price)

        # RL 策略預測
        action_dict = self.rl_agent.compute_single_action(
            obs,
            state=self.lstm_state,
            explore=False  # 不探索，使用確定性策略
        )

        if isinstance(action_dict, tuple):
            action, self.lstm_state, info_dict = action_dict
        else:
            action = action_dict
            info_dict = {}

        # 獲取動作概率（信心度）
        policy = self.rl_agent.get_policy()
        if hasattr(policy, 'compute_log_likelihoods'):
            # 獲取動作概率分佈
            action_probs = info_dict.get('action_prob', [0.33, 0.33, 0.34])
            confidence = float(max(action_probs))
        else:
            confidence = 0.5  # 默認信心度

        # ========== 只做多模式：過濾做空動作 ==========
        original_action = action
        if self.long_only:
            # 如果模型預測做空（action=2），且當前有持倉，則改為賣出平倉
            # 如果沒有持倉，則改為 Hold
            if action == 2:  # Sell（做空）
                if self.position > 0:
                    # 有持倉 → 允許賣出（平倉）
                    logger.info(f"[Long Only] 允許賣出平倉: position={self.position}")
                    pass  # 保持 action=2
                else:
                    # 沒有持倉 → 禁止做空，改為 Hold
                    action = 0
                    logger.info(f"[Long Only] 禁止做空，改為 Hold (原動作: Sell)")

        # 構建返回信息
        info = {
            'deeplob_prediction': self.lob_history[-1] if self.lob_history else None,
            'position': self.position,
            'unrealized_pnl': (current_price - self.entry_price) * self.position if self.entry_price > 0 else 0.0,
            'confidence': confidence,
            'original_action': original_action,  # 保留原始預測
            'filtered': original_action != action,  # 是否被過濾
        }

        return int(action), confidence, info

    def execute_trade(
        self,
        action: int,
        current_price: float,
        quantity: int = 1
    ) -> Dict[str, Any]:
        """執行交易

        參數:
            action: 交易動作 {0: Hold, 1: Buy, 2: Sell}
            current_price: 當前價格
            quantity: 交易數量

        返回:
            交易結果字典
        """
        result = {
            'action': action,
            'action_name': ['Hold', 'Buy', 'Sell'][action],
            'executed': False,
            'price': current_price,
            'quantity': 0,
            'cost': 0.0,
            'pnl': 0.0,
            'new_position': self.position,
        }

        # 0: Hold - 不交易
        if action == 0:
            logger.info(f"Hold | Position: {self.position}")
            return result

        # 1: Buy - 買入
        if action == 1:
            if self.position < self.max_position:
                # 執行買入（台股成本: 0.071% 券商手續費）
                cost = current_price * quantity * (1 + self.buy_cost_rate)

                # 更新持倉
                old_position = self.position
                self.position += quantity

                # 更新成本
                if old_position <= 0:
                    self.entry_price = current_price
                else:
                    # 加倉，更新平均成本
                    total_cost = self.entry_price * old_position + current_price * quantity
                    self.entry_price = total_cost / self.position

                result['executed'] = True
                result['quantity'] = quantity
                result['cost'] = cost
                result['new_position'] = self.position

                logger.info(f"Buy {quantity} @ {current_price:.2f} | Position: {old_position} → {self.position}")

        # 2: Sell - 賣出
        elif action == 2:
            if self.position > -self.max_position:
                # 執行賣出（台股成本: 0.071% 手續費 + 0.3% 證交稅 = 0.371%）
                revenue = current_price * quantity * (1 - self.sell_cost_rate)

                # 計算已實現損益
                if self.position > 0 and self.entry_price > 0:
                    pnl = (current_price - self.entry_price) * min(quantity, self.position)
                    self.total_pnl += pnl
                    result['pnl'] = pnl

                # 更新持倉
                old_position = self.position
                self.position -= quantity

                # 更新成本
                if self.position <= 0:
                    self.entry_price = current_price if self.position < 0 else 0.0

                result['executed'] = True
                result['quantity'] = -quantity
                result['cost'] = -revenue
                result['new_position'] = self.position

                logger.info(f"Sell {quantity} @ {current_price:.2f} | Position: {old_position} → {self.position} | PnL: {pnl:.2f}")

        return result

    def reset(self):
        """重置交易狀態"""
        self.position = 0
        self.entry_price = 0.0
        self.balance = 0.0
        self.total_pnl = 0.0
        self.lob_history.clear()
        self.lstm_state = None

        logger.info("交易狀態已重置")

    def get_status(self) -> Dict[str, Any]:
        """獲取當前狀態"""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'total_pnl': self.total_pnl,
            'balance': self.balance,
        }


if __name__ == "__main__":
    # 測試範例
    logging.basicConfig(level=logging.INFO)

    # 創建交易引擎
    engine = TradingEngine(
        rl_checkpoint="checkpoints/rl/best_model",
        deeplob_checkpoint="model/deeplob/deeplob_generic_best.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 模擬 LOB 數據
    fake_lob = np.random.randn(20).astype(np.float32)
    fake_price = 100.0

    # 預測
    action, confidence, info = engine.predict(fake_lob, fake_price)

    print(f"\n預測結果:")
    print(f"  - 動作: {['Hold', 'Buy', 'Sell'][action]}")
    print(f"  - 信心度: {confidence:.2%}")
    print(f"  - 持倉: {info['position']}")

    # 執行交易
    result = engine.execute_trade(action, fake_price)

    print(f"\n交易結果:")
    print(f"  - 已執行: {result['executed']}")
    print(f"  - 新持倉: {result['new_position']}")
