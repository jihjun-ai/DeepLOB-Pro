"""台股 LOB 交易環境 - 基於預處理台股數據的強化學習環境

此環境專為台股 5檔 LOB 數據（20維特徵）設計，與標準 LOBTradingEnv 的差異：
  1. LOB 維度: 20維（5檔）而非 40維（10檔）
  2. 數據來源: stock_embedding_*.npz 而非 FI-2010
  3. 觀測空間: 28維（LOB 20 + DeepLOB 3 + 狀態 5）

環境規格:
    觀測空間: (28,) = LOB(20) + DeepLOB預測(3) + 交易狀態(5)
    動作空間: Discrete(3) - {Hold, Buy, Sell}
    獎勵: PnL - 交易成本 - 庫存懲罰 - 風險懲罰

作者: RLlib-DeepLOB 專案團隊
更新: 2025-10-12
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from typing import Dict, Tuple, Optional, Any
import logging

from ..models.deeplob import DeepLOB
from .reward_shaper import RewardShaper
from .tw_stock_data_provider import TaiwanStockDataProvider

logger = logging.getLogger(__name__)


class TaiwanLOBTradingEnv(gym.Env):
    """台股 LOB 交易環境

    專為台股 5檔 LOB 數據設計的交易環境。
    整合 DeepLOB 模型提供價格預測信號。
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict):
        """初始化台股交易環境

        參數:
            config: 環境配置字典
                data_dir (str): 數據目錄，預設 'data/processed'
                max_steps (int): Episode 最大步數，預設 500
                initial_balance (float): 初始資金，預設 10000.0
                transaction_cost_rate (float): 交易成本率，預設 0.001
                max_position (int): 最大持倉，預設 1
                deeplob_checkpoint (str): DeepLOB 模型路徑（可選）
                reward_config (dict): 獎勵配置
                data_mode (str): 數據模式 'train'/'val'/'test'
        """
        super().__init__()

        self.config = config

        # 數據配置
        self.data_dir = config.get('data_dir', 'data/processed')
        self.max_steps = config.get('max_steps', 500)
        self.initial_balance = config.get('initial_balance', 10000.0)

        # 交易參數
        self.transaction_cost_rate = config.get('transaction_cost_rate', 0.001)
        self.max_position = config.get('max_position', 1)

        # 觀測空間: LOB(20) + DeepLOB(3) + State(5) = 28維
        self.lob_dim = 20  # 5檔 LOB
        obs_dim = 20 + 3 + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 動作空間: {0: Hold, 1: Buy, 2: Sell}
        self.action_space = spaces.Discrete(3)

        # 載入 DeepLOB 模型
        self.deeplob_model = None
        deeplob_checkpoint = config.get('deeplob_checkpoint')
        if deeplob_checkpoint:
            self._load_deeplob_model(deeplob_checkpoint)

        # 初始化獎勵塑形器
        self.reward_shaper = RewardShaper(config.get('reward_config', {}))

        # 數據模式
        self.data_mode = config.get('data_mode', 'train')

        # 載入台股數據
        self._load_data()

        # 狀態變數
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.inventory = 0.0
        self.total_cost = 0.0
        self.prev_action = 0

        # 歷史記錄
        self.lob_history = []
        self.trade_history = []

        logger.info(f"台股交易環境已初始化: 觀測空間={self.observation_space.shape}")

    def _load_deeplob_model(self, checkpoint_path: str):
        """載入 DeepLOB 模型"""
        try:
            # PyTorch 2.8+ 需要設定 weights_only=False 以載入包含非權重物件的舊模型
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # 從檢查點中提取模型配置和權重
            if 'model_state_dict' in checkpoint:
                # 完整檢查點格式 (包含 epoch, optimizer, config 等)
                # 優先使用 config['model']，因為它包含完整的參數（如 conv filters）
                if 'config' in checkpoint and 'model' in checkpoint['config']:
                    model_config = checkpoint['config']['model']
                else:
                    # 降級到 model_config (可能缺少某些參數)
                    model_config = checkpoint.get('model_config', {})

                # 從配置中提取 DeepLOB 初始化參數
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
                self.deeplob_model.load_state_dict(checkpoint['model_state_dict'])

                logger.info(f"✅ DeepLOB 模型載入成功: {checkpoint_path}")
                logger.info(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
                logger.info(f"   - 驗證準確率: {checkpoint.get('val_acc', 0):.4f}")
                logger.info(f"   - Conv Filters: {deeplob_params['conv1_filters']}/{deeplob_params['conv2_filters']}/{deeplob_params['conv3_filters']}")
                logger.info(f"   - LSTM Hidden: {deeplob_params['lstm_hidden_size']}, Dropout: {deeplob_params['dropout']}")
            else:
                # 純權重格式 (只有 state_dict)
                self.deeplob_model = DeepLOB()
                self.deeplob_model.load_state_dict(checkpoint)
                logger.info(f"✅ DeepLOB 模型載入成功 (純權重): {checkpoint_path}")

            self.deeplob_model.eval()
        except Exception as e:
            logger.error(f"❌ DeepLOB 模型載入失敗: {e}")
            logger.warning("⚠️ 將使用隨機預測（僅用於測試）")
            self.deeplob_model = None

    def _load_data(self):
        """載入台股 LOB 數據"""
        try:
            # 創建台股數據提供者
            # 使用數據採樣（預設 10%）以減少記憶體使用
            sample_ratio = self.config.get('data_sample_ratio', 0.1)
            logger.info(f"載入台股數據，目錄: {self.data_dir}，採樣比例: {sample_ratio:.1%}")
            data_provider = TaiwanStockDataProvider(
                data_dir=self.data_dir,
                sample_ratio=sample_ratio
            )

            # 根據模式載入數據
            if self.data_mode == 'train':
                self.lob_data, self.labels = data_provider.get_train_data()
                logger.info("✅ 載入訓練集數據")
            elif self.data_mode == 'val':
                self.lob_data, self.labels = data_provider.get_val_data()
                logger.info("✅ 載入驗證集數據")
            elif self.data_mode == 'test':
                self.lob_data, self.labels = data_provider.get_test_data()
                logger.info("✅ 載入測試集數據")
            else:
                raise ValueError(f"未知數據模式: {self.data_mode}")

            # 驗證數據形狀
            assert self.lob_data.ndim == 3, f"數據維度錯誤: {self.lob_data.ndim}"
            assert self.lob_data.shape[1] == 100, f"時間步錯誤: {self.lob_data.shape[1]}"
            assert self.lob_data.shape[2] == 20, f"特徵維度錯誤: {self.lob_data.shape[2]}"

            self.data_length = len(self.lob_data)

            # 載入真實價格數據（如果有）
            # 注意：不能從標籤計算價格，那會造成數據洩漏！
            self.prices = data_provider.get_prices(self.data_mode)

            if self.prices is None:
                logger.warning("⚠️  數據中沒有真實價格，從 LOB 計算中價 (Mid Price)")
                # 從 LOB 數據計算中價作為真實價格
                # LOB 格式: 20 維
                try:
                    # 取每個樣本的最後一個時間步（t=99）的 LOB 數據
                    last_lob = self.lob_data[:, -1, :]  # (N, 20)

                    # 嘗試提取最佳買價和賣價
                    # 假設格式為交錯式（每檔價量交替）
                    best_bid_price = last_lob[:, 0]   # 第0維：最佳買價
                    best_ask_price = last_lob[:, 10]  # 第10維：最佳賣價

                    # 檢查價格是否合理（買價應 < 賣價）
                    if np.mean(best_bid_price < best_ask_price) > 0.9:
                        mid_prices = (best_bid_price + best_ask_price) / 2
                        logger.info(f"✅ 使用 LOB 中價 (格式: 交錯式，90%+ 樣本買價<賣價)")
                    else:
                        # 可能是其他格式，嘗試其他索引
                        logger.warning("⚠️  LOB 格式不符預期，嘗試其他索引")
                        # 嘗試區塊式格式
                        best_bid_price = last_lob[:, 0]
                        best_ask_price = last_lob[:, 10]
                        mid_prices = (np.abs(best_bid_price) + np.abs(best_ask_price)) / 2
                        logger.info(f"✅ 使用 LOB 絕對值中價 (備用方案)")

                    self.prices = mid_prices
                    logger.info(f"✅ 從 LOB 計算中價: 範圍=[{self.prices.min():.2f}, {self.prices.max():.2f}]")

                except Exception as e:
                    logger.error(f"❌ 從 LOB 計算中價失敗: {e}")
                    logger.warning("⚠️  使用隨機遊走模擬價格（僅用於測試）")
                    # 備用方案：使用隨機遊走（不依賴標籤）
                    random_returns = np.random.randn(self.data_length) * 0.01
                    self.prices = 100.0 + np.cumsum(random_returns)

            logger.info(
                f"✅ 成功載入台股數據: "
                f"{self.data_length:,} 個樣本, "
                f"LOB形狀={self.lob_data.shape}, "
                f"價格範圍=[{self.prices.min():.2f}, {self.prices.max():.2f}]"
            )

        except Exception as e:
            logger.error(f"❌ 數據載入失敗: {e}")
            logger.error("❌ 訓練無法繼續，請檢查記憶體配置或降低 data_sample_ratio")
            raise RuntimeError(f"數據載入失敗，訓練終止: {e}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置環境"""
        super().reset(seed=seed)

        # 重置狀態
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.inventory = 0.0
        self.total_cost = 0.0
        self.prev_action = 0

        # 清空歷史
        self.lob_history = []
        self.trade_history = []

        # 隨機選擇起始樣本
        max_start = max(1, self.data_length - self.max_steps)
        self.current_data_idx = np.random.randint(0, max_start)

        # 初始化 LOB 歷史（使用第一個樣本的 100 時間步）
        self.lob_history = self.lob_data[self.current_data_idx].tolist()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行交易動作"""
        # 獲取當前價格
        current_price = self.prices[self.current_data_idx]

        # 保存前一狀態
        prev_position = self.position
        prev_state = {
            'position': self.position,
            'entry_price': self.entry_price,
            'prev_action': self.prev_action,
            'prev_price': current_price  # ⭐ 新增：當前價格作為下一步的 prev_price
        }

        # 執行交易
        transaction_cost = 0.0

        if action != prev_position + 1:
            position_change = abs(action - (prev_position + 1))
            transaction_cost = position_change * current_price * self.transaction_cost_rate
            self.total_cost += transaction_cost

            # 更新倉位
            if action == 0:  # Sell
                new_position = self.position - 1
                self.position = max(-self.max_position, new_position)
            elif action == 2:  # Buy
                new_position = self.position + 1
                self.position = min(self.max_position, new_position)

            # 記錄交易
            if self.position != prev_position:
                self.trade_history.append({
                    'step': self.current_step,
                    'action': action,
                    'price': current_price,
                    'position': self.position,
                    'cost': transaction_cost
                })

            # 更新進場價
            if prev_position == 0 and self.position != 0:
                self.entry_price = current_price

        # 更新庫存
        if self.position != 0:
            self.inventory = self.position * (current_price - self.entry_price)
        else:
            self.inventory = 0.0

        # 推進到下一步
        self.current_step += 1
        self.current_data_idx = min(self.current_data_idx + 1, self.data_length - 1)

        # 更新 LOB 歷史
        self.lob_history = self.lob_data[self.current_data_idx].tolist()

        # 獲取新價格
        next_price = self.prices[self.current_data_idx]
        price_change = next_price - current_price

        # 構建新狀態
        new_state = {
            'position': self.position,
            'current_price': next_price,
            'inventory': self.inventory,
            'volatility': 0.01
        }

        # 計算獎勵
        reward, reward_info = self.reward_shaper.calculate_reward(
            prev_state=prev_state,
            action=action,
            new_state=new_state,
            transaction_cost=transaction_cost
        )

        # 檢查終止條件
        terminated = False
        truncated = self.current_step >= self.max_steps

        # 生成觀測
        obs = self._get_observation()
        info = self._get_info()
        info.update(reward_info)
        info['action'] = action
        info['transaction_cost'] = transaction_cost
        info['price'] = next_price
        info['price_change'] = price_change

        self.prev_action = action

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """獲取觀測（28維）"""
        # 當前 LOB 特徵（取最後一個時間步）
        current_lob = np.array(self.lob_history[-1], dtype=np.float32)

        # DeepLOB 預測
        if self.deeplob_model is not None:
            with torch.no_grad():
                lob_seq = torch.FloatTensor(self.lob_history).unsqueeze(0)  # (1, 100, 20)
                deeplob_probs = self.deeplob_model.predict_proba(lob_seq)[0].numpy()
        else:
            deeplob_probs = np.random.rand(3).astype(np.float32)
            deeplob_probs /= deeplob_probs.sum()

        # 交易狀態
        state_features = np.array([
            self.position / self.max_position if self.max_position > 0 else 0.0,
            self.inventory / self.initial_balance,
            self.total_cost / self.initial_balance,
            self.current_step / self.max_steps,
            self.prev_action / 2.0
        ], dtype=np.float32)

        # 串接所有特徵: LOB(20) + DeepLOB(3) + State(5) = 28維
        obs = np.concatenate([current_lob, deeplob_probs, state_features])

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """獲取資訊字典"""
        return {
            'step': self.current_step,
            'position': self.position,
            'balance': self.balance,
            'inventory': self.inventory,
            'total_cost': self.total_cost,
            'num_trades': len(self.trade_history)
        }

    def render(self):
        """渲染環境"""
        if self.current_step % 100 == 0:
            print(
                f"[Step {self.current_step}] "
                f"倉位: {self.position:+d}, "
                f"庫存: {self.inventory:+.2f}, "
                f"交易次數: {len(self.trade_history)}"
            )

    def close(self):
        """清理資源"""
        pass


def make_tw_lob_trading_env(config: Dict) -> TaiwanLOBTradingEnv:
    """創建台股交易環境的工廠函數"""
    return TaiwanLOBTradingEnv(config)
