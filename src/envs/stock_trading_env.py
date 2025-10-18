"""股票交易環境 - 基於技術指標 + DeepLOB 預測的強化學習訓練環境

此模組實作了基於股票技術指標（CSV 數據）+ DeepLOB 價格預測的交易環境，
符合 Gymnasium 標準介面。用於訓練強化學習交易策略，支援每日多股票批次訓練場景。

主要功能:
    1. 載入股票技術指標 CSV 數據（支援多股票）
    2. 可擴充特徵空間（預留 128 維，支援未來擴充）
    3. 整合 DeepLOB_StockEmbedding 價格預測（可選）
    4. 支援每日訓練、續訓場景
    5. Hold/Buy/Sell 三種交易動作
    6. 多組件獎勵函數（PnL、成本、庫存、風險）

環境規格:
    觀測空間（根據 use_deeplob 配置）:
        - 不使用 DeepLOB: (max_features + 1,) = 技術指標(128) + 持倉數(1) = 129 維
        - 使用 DeepLOB: (max_features + 4,) = 技術指標(128) + DeepLOB預測(3) + 持倉數(1) = 132 維

    動作空間:
        - Discrete(3): {0: Hold, 1: Buy, 2: Sell}

    獎勵:
        - 多組件: PnL - 交易成本 - 庫存懲罰 - 風險懲罰

數據格式:
    CSV 文件命名: {股票編號}_{日期}_indicators.csv
    範例: 6742_20250902_indicators.csv

    必要欄位: Date, Open, High, Low, Close, Volume
    可選欄位: 所有技術指標（EMA, MACD, RSI, etc.）

使用範例:
    >>> env = StockTradingEnv({
    ...     'data_dir': 'data/processed',
    ...     'max_features': 128,
    ...     'max_steps': 500,
    ...     'transaction_cost_rate': 0.001
    ... })
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import logging
from pathlib import Path
import torch

from ..data.stock_indicators_loader import StockIndicatorsLoader
from .reward_shaper import RewardShaper
from ..models.dual_mode_manager import DeepLOBModelManager

logger = logging.getLogger(__name__)


class StockTradingEnv(gym.Env):
    """股票技術指標交易環境

    這是一個 Gymnasium 標準的強化學習環境，模擬基於股票技術指標的交易。
    支援每日訓練、多股票批次處理、特徵擴充等功能。

    環境特點:
        - 自動載入目錄下所有 *_indicators.csv 文件
        - 特徵空間可擴充（預留到 128 維）
        - 支援每日增量訓練和續訓
        - 考慮交易成本和市場影響
        - 多維度狀態追蹤

    狀態變數:
        current_step: 當前時間步（0 到 max_steps）
        position: 當前持倉數量（可為負值代表空倉）
        entry_price: 進場價格
        balance: 帳戶餘額
        total_pnl: 總盈虧
        total_cost: 累計交易成本
        prev_action: 上一個動作
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict):
        """初始化股票交易環境

        參數:
            config: 環境配置字典，包含以下鍵值:
                data_dir (str): CSV 數據目錄，預設 'data/processed'
                max_features (int): 最大特徵維度（預留擴充），預設 128
                max_steps (int): 每個 episode 的最大步數，預設 500
                initial_balance (float): 初始資金，預設 10000.0
                transaction_cost_rate (float): 交易成本率，預設 0.001 (0.1%)
                max_position (int): 最大持倉量，預設 10
                shares_per_trade (int): 每次交易股數，預設 1
                reward_config (dict): 獎勵函數配置
                normalize (bool): 是否標準化數據，預設 True
                stock_id (str): 指定股票編號（可選，None=隨機選取）
                date (str): 指定日期（可選，None=使用所有日期）
                use_deeplob (bool): 是否使用 DeepLOB 預測，預設 False
                deeplob_stock_checkpoint (str): 個股特化模型檢查點路徑（可選）
                deeplob_generic_checkpoint (str): 通用模型檢查點路徑（可選）
                deeplob_seq_len (int): DeepLOB 輸入序列長度，預設 100
                lob_features (int): LOB 特徵維度，預設 20 (台股 5 檔 × 4)
                lob_data_path (str): LOB 數據路徑（可選，若使用 DeepLOB）

        環境配置範例:
            config = {
                'data_dir': 'data/processed',
                'max_features': 128,
                'max_steps': 500,
                'initial_balance': 10000.0,
                'transaction_cost_rate': 0.001,
                'max_position': 10,
                'shares_per_trade': 1,
                'use_deeplob': True,
                'deeplob_stock_checkpoint': 'checkpoints/deeplob_stock_embedding_best.pth',
                'deeplob_generic_checkpoint': 'checkpoints/deeplob_generic_best.pth',
                'deeplob_seq_len': 100,
                'lob_features': 20,
                'lob_data_path': 'data/lob_data',
                'reward_config': {
                    'pnl_scale': 1.0,
                    'cost_penalty': 1.0,
                    'inventory_penalty': 0.01
                }
            }
        """
        super().__init__()

        self.config = config

        # ===== 數據配置 =====
        self.data_dir = config.get('data_dir', 'data/processed')
        self.max_features = config.get('max_features', 128)
        self.max_steps = config.get('max_steps', 500)
        self.initial_balance = config.get('initial_balance', 10000.0)

        # ===== 交易參數 =====
        self.transaction_cost_rate = config.get('transaction_cost_rate', 0.001)
        self.max_position = config.get('max_position', 10)
        self.shares_per_trade = config.get('shares_per_trade', 1)

        # ===== 數據選擇 =====
        self.stock_id = config.get('stock_id', None)  # None = 隨機選取
        self.date = config.get('date', None)  # None = 所有日期

        # ===== DeepLOB 配置 =====
        self.use_deeplob = config.get('use_deeplob', False)
        self.deeplob_seq_len = config.get('deeplob_seq_len', 100)
        self.lob_features = config.get('lob_features', 20)
        self.lob_data_path = config.get('lob_data_path', None)
        self.deeplob_manager = None
        self.lob_sequences = None  # LOB 數據序列（若使用 DeepLOB）
        self.stock_id_mapping = None  # 股票 ID 映射

        # ===== 初始化 DeepLOB 雙模型管理器 =====
        if self.use_deeplob:
            logger.info("初始化 DeepLOB 雙模型管理器...")
            stock_checkpoint = config.get('deeplob_stock_checkpoint', None)
            generic_checkpoint = config.get('deeplob_generic_checkpoint', None)

            if stock_checkpoint or generic_checkpoint:
                self.deeplob_manager = DeepLOBModelManager(
                    stock_embedding_checkpoint=stock_checkpoint,
                    generic_checkpoint=generic_checkpoint
                )
                logger.info("✓ DeepLOB 雙模型管理器初始化成功")
            else:
                logger.warning("未提供 DeepLOB 檢查點，DeepLOB 功能將被禁用")
                self.use_deeplob = False

        # ===== 定義觀測空間 =====
        # 根據是否使用 DeepLOB 決定觀測空間維度
        if self.use_deeplob and self.deeplob_manager:
            # 技術指標(max_features) + DeepLOB預測(3) + 持倉數(1) = max_features + 4
            obs_dim = self.max_features + 4
        else:
            # 技術指標(max_features) + 持倉數(1) = max_features + 1
            obs_dim = self.max_features + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ===== 定義動作空間 =====
        # Discrete(3): 三個離散動作
        #   0: Hold  - 持有當前倉位（不交易）
        #   1: Buy   - 買入 shares_per_trade 股
        #   2: Sell  - 賣出 shares_per_trade 股
        self.action_space = spaces.Discrete(3)

        # ===== 初始化獎勵塑形器 =====
        self.reward_shaper = RewardShaper(config.get('reward_config', {}))

        # ===== 初始化數據載入器 =====
        self.data_loader = StockIndicatorsLoader(
            data_dir=self.data_dir,
            max_features=self.max_features,
            normalize=config.get('normalize', True),
            fill_missing=True,
            cache=True
        )

        # ===== 載入交易數據 =====
        self._load_data()

        # ===== 初始化狀態變數 =====
        self.current_step = 0
        self.position = 0  # 持倉數量（正=多倉，負=空倉，0=平倉）
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.total_cost = 0.0
        self.prev_action = 0

        # ===== 歷史記錄 =====
        self.trade_history = []

        logger.info(f"股票交易環境已初始化:")
        logger.info(f"  - 觀測空間: {self.observation_space.shape}")
        logger.info(f"  - 動作空間: {self.action_space.n}")
        logger.info(f"  - 數據長度: {self.data_length}")
        logger.info(f"  - 特徵維度: {self.feature_count} (max: {self.max_features})")

    def _load_data(self):
        """載入股票交易數據

        根據配置載入 CSV 數據:
            - 如果指定 stock_id 和 date，載入單一股票
            - 如果只指定 date，載入該日期所有股票並串接
            - 如果都不指定，載入所有股票並串接

        數據格式:
            - features: (timesteps, max_features) 技術指標特徵（已填充）
            - prices: (timesteps,) Close 價格序列
            - metadata: 股票元數據（編號、日期、檔案路徑等）
        """
        try:
            # ===== 根據配置選擇載入方式 =====
            if self.stock_id and self.date:
                # 載入指定股票和日期
                logger.info(f"載入股票: {self.stock_id}, 日期: {self.date}")
                features, prices, metadata = self.data_loader.get_stock_by_id(
                    stock_id=self.stock_id,
                    date=self.date
                )
                if features is None:
                    raise ValueError(f"未找到股票數據: {self.stock_id}, {self.date}")

                self.features = features
                self.prices = prices
                self.metadata = [metadata] if metadata else []

            elif self.date:
                # 載入指定日期的所有股票
                logger.info(f"載入日期: {self.date} 的所有股票")
                all_features, all_prices, all_metadata = self.data_loader.get_stocks_by_date(
                    date=self.date
                )
                if not all_features:
                    raise ValueError(f"未找到日期 {self.date} 的股票數據")

                # 串接所有股票數據
                self.features = np.vstack(all_features)
                self.prices = np.concatenate(all_prices)
                self.metadata = all_metadata

            else:
                # 載入所有股票
                logger.info("載入目錄下所有股票數據")
                self.features, self.prices, self.metadata = self.data_loader.concatenate_all()

                if len(self.features) == 0:
                    raise ValueError("未找到任何股票數據")

            # ===== 驗證數據 =====
            assert self.features.shape[1] == self.max_features, \
                f"特徵維度錯誤: 期望 {self.max_features}，實際 {self.features.shape[1]}"
            assert len(self.features) == len(self.prices), \
                f"數據長度不匹配: Features={len(self.features)}, Prices={len(self.prices)}"

            self.data_length = len(self.features)
            self.feature_count = self.data_loader.feature_count

            logger.info(f"✅ 成功載入數據:")
            logger.info(f"  - 總時間步數: {self.data_length:,}")
            logger.info(f"  - 特徵維度: {self.features.shape[1]}")
            logger.info(f"  - 實際特徵數: {self.feature_count}")
            logger.info(f"  - 股票數量: {len(self.metadata)}")
            logger.info(f"  - 價格範圍: [{self.prices.min():.2f}, {self.prices.max():.2f}]")

        except Exception as e:
            logger.error(f"❌ 數據載入失敗: {e}")
            raise

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置環境到初始狀態

        每個 episode 開始時調用此方法，將所有狀態變數重置。

        參數:
            seed: 隨機種子（用於可復現性）
            options: 額外選項（如指定起始位置）

        返回:
            observation: 初始觀測 (max_features + 1,)
            info: 額外信息字典
        """
        super().reset(seed=seed)

        # ===== 重置狀態 =====
        # 隨機選擇起始位置（確保有足夠步數完成 episode）
        max_start = max(0, self.data_length - self.max_steps - 1)
        self.current_step = self.np_random.integers(0, max_start + 1) if max_start > 0 else 0

        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.total_cost = 0.0
        self.prev_action = 0

        # 重置交易歷史
        self.trade_history = []

        # 獲取初始觀測
        observation = self._get_observation()

        # 構建 info
        info = {
            'start_step': self.current_step,
            'stock_count': len(self.metadata),
            'initial_price': float(self.prices[self.current_step])
        }

        return observation, info

    def _get_observation(self) -> np.ndarray:
        """構建當前觀測

        觀測包含:
            1. 技術指標特徵 (max_features 維)
            2. DeepLOB 預測（若啟用）(3 維)
            3. 當前持倉數量 (1 維，歸一化到 [-1, 1])

        返回:
            observation: (max_features + 1,) 或 (max_features + 4,) numpy 數組
        """
        # 當前時刻的技術指標
        features = self.features[self.current_step]  # (max_features,)

        # 歸一化持倉數量到 [-1, 1]
        normalized_position = np.clip(
            self.position / self.max_position,
            -1.0,
            1.0
        )

        # ===== 構建觀測 =====
        components = [features]

        # 添加 DeepLOB 預測（若啟用）
        if self.use_deeplob and self.deeplob_manager and self.lob_sequences is not None:
            deeplob_probs = self._get_deeplob_prediction()
            components.append(deeplob_probs)

        # 添加持倉
        components.append([normalized_position])

        # 串接所有組件
        observation = np.concatenate(components).astype(np.float32)

        return observation

    def _get_deeplob_prediction(self) -> np.ndarray:
        """獲取 DeepLOB 預測

        返回:
            probs: 類別機率分佈 (3,) [下跌, 穩定, 上漲]
        """
        try:
            # 檢查是否有足夠的歷史數據
            if self.current_step < self.deeplob_seq_len:
                # 不足序列長度，返回均勻分佈
                return np.array([1/3, 1/3, 1/3], dtype=np.float32)

            # 提取當前時刻的 LOB 序列
            start_idx = self.current_step - self.deeplob_seq_len
            end_idx = self.current_step
            lob_seq = self.lob_sequences[start_idx:end_idx]  # (100, 20)

            # 轉換為張量
            lob_tensor = torch.FloatTensor(lob_seq).unsqueeze(0)  # (1, 100, 20)

            # 獲取當前股票 ID（若有）
            stock_id_tensor = None
            if self.stock_id_mapping is not None and self.current_step < len(self.stock_id_mapping):
                stock_id = self.stock_id_mapping[self.current_step]
                stock_id_tensor = torch.LongTensor([stock_id])

            # 使用雙模型管理器進行預測
            with torch.no_grad():
                probs = self.deeplob_manager.predict_proba(lob_tensor, stock_id_tensor)
                probs = probs.cpu().numpy()[0]  # (3,)

            return probs.astype(np.float32)

        except Exception as e:
            logger.warning(f"DeepLOB 預測失敗: {e}，返回均勻分佈")
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行一個時間步

        參數:
            action: 動作 {0: Hold, 1: Buy, 2: Sell}

        返回:
            observation: 下一個觀測
            reward: 獎勵值
            terminated: 是否達到終止條件
            truncated: 是否達到截斷條件
            info: 額外信息
        """
        # ===== 步驟 1: 獲取當前價格 =====
        current_price = float(self.prices[self.current_step])

        # ===== 步驟 2: 執行動作 =====
        trade_cost = 0.0
        trade_info = None

        if action == 1:  # Buy
            # 檢查持倉限制
            if self.position < self.max_position:
                # 計算交易成本
                trade_value = current_price * self.shares_per_trade
                trade_cost = trade_value * self.transaction_cost_rate

                # 檢查餘額是否足夠
                if self.balance >= trade_value + trade_cost:
                    # 執行買入
                    self.position += self.shares_per_trade
                    self.balance -= (trade_value + trade_cost)
                    self.total_cost += trade_cost

                    # 更新進場價格（加權平均）
                    if self.position > 0:
                        prev_value = self.entry_price * (self.position - self.shares_per_trade)
                        new_value = current_price * self.shares_per_trade
                        self.entry_price = (prev_value + new_value) / self.position

                    trade_info = {
                        'action': 'BUY',
                        'shares': self.shares_per_trade,
                        'price': current_price,
                        'cost': trade_cost
                    }

        elif action == 2:  # Sell
            # 檢查持倉限制
            if self.position > -self.max_position:
                # 計算交易成本
                trade_value = current_price * self.shares_per_trade
                trade_cost = trade_value * self.transaction_cost_rate

                # 執行賣出
                self.position -= self.shares_per_trade
                self.balance += (trade_value - trade_cost)
                self.total_cost += trade_cost

                # 更新進場價格（加權平均）
                if self.position < 0:
                    prev_value = self.entry_price * (abs(self.position) - self.shares_per_trade)
                    new_value = current_price * self.shares_per_trade
                    self.entry_price = (prev_value + new_value) / abs(self.position)

                trade_info = {
                    'action': 'SELL',
                    'shares': self.shares_per_trade,
                    'price': current_price,
                    'cost': trade_cost
                }

        # 記錄交易
        if trade_info:
            trade_info['step'] = self.current_step
            trade_info['position'] = self.position
            trade_info['balance'] = self.balance
            self.trade_history.append(trade_info)

        # ===== 步驟 3: 計算獎勵 =====
        # 計算未實現盈虧
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
        else:
            unrealized_pnl = 0.0

        # 使用 RewardShaper 計算獎勵
        prev_state = {
            'position': self.position,
            'prev_action': self.prev_action,
            'inventory': abs(self.position)
        }

        new_state = {
            'position': self.position,
            'inventory': abs(self.position),
            'unrealized_pnl': unrealized_pnl
        }

        price_change = 0.0
        if self.current_step > 0:
            prev_price = float(self.prices[self.current_step - 1])
            price_change = current_price - prev_price

        reward, reward_info = self.reward_shaper.calculate_reward(
            prev_state=prev_state,
            action=action,
            new_state=new_state,
            price_change=price_change
        )

        # ===== 步驟 4: 更新狀態 =====
        self.current_step += 1
        self.prev_action = action

        # ===== 步驟 5: 檢查終止條件 =====
        # Episode 長度達到上限
        truncated = self.current_step >= min(self.max_steps, self.data_length - 1)

        # 帳戶爆倉
        total_value = self.balance + unrealized_pnl
        terminated = total_value <= 0

        # ===== 步驟 6: 獲取下一個觀測 =====
        if not (terminated or truncated):
            observation = self._get_observation()
        else:
            # Episode 結束，返回零觀測
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        # ===== 步驟 7: 構建 info =====
        info = {
            'step': self.current_step,
            'price': current_price,
            'position': self.position,
            'balance': self.balance,
            'unrealized_pnl': unrealized_pnl,
            'total_value': total_value,
            'total_cost': self.total_cost,
            'trade_count': len(self.trade_history),
            'reward_components': reward_info
        }

        if trade_info:
            info['trade'] = trade_info

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """渲染環境狀態（可選）

        參數:
            mode: 渲染模式，目前只支持 'human'
        """
        current_price = float(self.prices[self.current_step])
        unrealized_pnl = (current_price - self.entry_price) * self.position if self.position != 0 else 0.0
        total_value = self.balance + unrealized_pnl

        print(f"\n===== Step {self.current_step} =====")
        print(f"價格: {current_price:.2f}")
        print(f"持倉: {self.position}")
        print(f"餘額: {self.balance:.2f}")
        print(f"未實現盈虧: {unrealized_pnl:.2f}")
        print(f"總價值: {total_value:.2f}")
        print(f"交易次數: {len(self.trade_history)}")


def make_stock_trading_env(config: Dict) -> StockTradingEnv:
    """工廠函數: 創建股票交易環境

    參數:
        config: 環境配置字典

    返回:
        StockTradingEnv 實例

    範例:
        >>> config = {
        ...     'data_dir': 'data/processed',
        ...     'max_features': 128,
        ...     'max_steps': 500
        ... }
        >>> env = make_stock_trading_env(config)
        >>> obs, info = env.reset()
    """
    return StockTradingEnv(config)
