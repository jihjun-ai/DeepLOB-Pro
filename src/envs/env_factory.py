"""環境工廠模組 - 整合真實 FI-2010 數據到交易環境

此模組負責創建配置好真實數據的交易環境實例。
它整合 FI-2010 數據載入器和預處理器，提供完整的數據管線。

主要功能:
    1. 載入 FI-2010 原始數據
    2. 預處理和標準化
    3. 創建時間序列窗口
    4. 分割訓練/驗證/測試集
    5. 提供給環境的數據接口

設計模式:
    工廠模式 - 統一創建環境的入口點
    單例模式 - 數據只載入一次，多個環境共享

數據流:
    FI-2010 原始數據 → 載入器 → 預處理器 → 序列創建 → 環境

作者: RLlib-DeepLOB 專案團隊
更新: 2025-01-09
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

from ..data.fi2010_loader import FI2010Loader
from ..data.preprocessor import LOBPreprocessor

logger = logging.getLogger(__name__)


class EnvDataProvider:
    """環境數據提供者

    此類別負責為交易環境提供預處理好的 LOB 數據。
    實現數據的延遲載入和快取機制，避免重複載入。

    設計特點:
        - 延遲載入：只在第一次使用時載入數據
        - 數據快取：載入後的數據存儲在記憶體中
        - 多環境共享：不同環境實例可共享同一數據
        - 分割管理：自動處理訓練/驗證/測試分割

    使用方式:
        provider = EnvDataProvider(data_dir="data/raw")
        train_data, train_prices = provider.get_train_data()
        val_data, val_prices = provider.get_val_data()
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        normalization_method: str = 'z-score',
        sequence_length: int = 100,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ):
        """初始化數據提供者

        參數:
            data_dir: FI-2010 數據目錄
            normalization_method: 標準化方法 ('z-score', 'min-max', 'robust')
            sequence_length: 序列長度（時間步數）
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            test_ratio: 測試集比例
        """
        self.data_dir = Path(data_dir)
        self.normalization_method = normalization_method
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 數據快取
        self._train_data = None
        self._train_prices = None
        self._val_data = None
        self._val_prices = None
        self._test_data = None
        self._test_prices = None

        # 組件
        self.loader = None
        self.preprocessor = None

        # 載入狀態
        self._is_loaded = False

        logger.info(
            f"數據提供者已初始化: "
            f"序列長度={sequence_length}, "
            f"標準化={normalization_method}, "
            f"分割比例={train_ratio}/{val_ratio}/{test_ratio}"
        )

    def _load_and_preprocess(self):
        """載入並預處理數據（內部方法）

        此方法執行完整的數據管線：
            1. 載入 FI-2010 原始數據
            2. 標準化處理
            3. 創建時間序列窗口
            4. 分割數據集
            5. 快取到記憶體

        注意:
            - 此方法只會被調用一次（延遲載入）
            - 數據載入後會設定 _is_loaded 標記
            - 後續調用會直接返回快取數據
        """
        if self._is_loaded:
            return

        logger.info("開始載入和預處理 FI-2010 數據...")

        # ===== 步驟 1: 初始化組件 =====
        self.loader = FI2010Loader(self.data_dir)
        self.preprocessor = LOBPreprocessor(
            normalization_method=self.normalization_method,
            sequence_length=self.sequence_length
        )

        # ===== 步驟 2: 載入原始數據 =====
        logger.info("載入原始 LOB 數據...")
        features, labels = self.loader.load_raw_data()
        logger.info(f"  原始數據形狀: features={features.shape}, labels={labels.shape}")

        # ===== 步驟 3: 處理缺失值 =====
        logger.info("處理缺失值...")
        features = self.preprocessor.handle_missing_values(features, method='forward_fill')

        # ===== 步驟 4: 標準化 =====
        logger.info(f"標準化數據（方法: {self.normalization_method}）...")
        features = self.preprocessor.normalize(features, fit=True, clip_outliers=True)

        # ===== 步驟 5: 創建時間序列窗口 =====
        logger.info(f"創建時間序列窗口（長度: {self.sequence_length}）...")
        # 使用第一個預測時間範圍的標籤（索引 0 = k=10）
        sequences, seq_labels = self.preprocessor.create_sequences(
            features, labels[:, 0], stride=1
        )
        logger.info(f"  序列形狀: {sequences.shape}, 標籤形狀: {seq_labels.shape}")

        # ===== 步驟 6: 生成價格序列（用於環境）=====
        # 從 LOB 數據計算中間價作為價格序列
        lob_parsed = self.loader.parse_lob_snapshot(features)
        mid_prices = (lob_parsed['ask_prices'][:, 0] + lob_parsed['bid_prices'][:, 0]) / 2

        # 為每個序列關聯價格序列
        prices_sequences = []
        for i in range(len(sequences)):
            start_idx = i
            end_idx = start_idx + self.sequence_length
            prices_sequences.append(mid_prices[start_idx:end_idx + 1])  # +1 包含下一時刻價格

        prices_sequences = np.array(prices_sequences, dtype=np.float32)

        # ===== 步驟 7: 分割數據集 =====
        logger.info("分割數據集...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self.preprocessor.train_val_test_split(
                sequences, seq_labels,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
            )

        # 分割價格序列
        train_end = int(len(prices_sequences) * self.train_ratio)
        val_end = int(len(prices_sequences) * (self.train_ratio + self.val_ratio))

        train_prices = prices_sequences[:train_end]
        val_prices = prices_sequences[train_end:val_end]
        test_prices = prices_sequences[val_end:]

        # ===== 步驟 8: 快取到記憶體 =====
        self._train_data = X_train
        self._train_prices = train_prices
        self._val_data = X_val
        self._val_prices = val_prices
        self._test_data = X_test
        self._test_prices = test_prices

        self._is_loaded = True

        logger.info("✓ 數據載入和預處理完成")
        logger.info(
            f"  訓練集: {len(X_train)} 序列"
            f"  驗證集: {len(X_val)} 序列"
            f"  測試集: {len(X_test)} 序列"
        )

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取訓練集數據

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - lob_sequences: LOB 序列數據 (N, 100, 40)
                - prices: 對應的價格序列 (N, 101)
        """
        if not self._is_loaded:
            self._load_and_preprocess()
        return self._train_data, self._train_prices

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取驗證集數據

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - lob_sequences: LOB 序列數據
                - prices: 對應的價格序列
        """
        if not self._is_loaded:
            self._load_and_preprocess()
        return self._val_data, self._val_prices

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取測試集數據

        返回:
            Tuple[np.ndarray, np.ndarray]:
                - lob_sequences: LOB 序列數據
                - prices: 對應的價格序列
        """
        if not self._is_loaded:
            self._load_and_preprocess()
        return self._test_data, self._test_prices

    def get_data_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """根據分割名稱獲取數據

        參數:
            split: 分割名稱 ('train', 'val', 'test')

        返回:
            對應分割的數據和價格

        異常:
            ValueError: 若分割名稱無效
        """
        if split == 'train':
            return self.get_train_data()
        elif split in ('val', 'validation'):
            return self.get_val_data()
        elif split == 'test':
            return self.get_test_data()
        else:
            raise ValueError(f"無效的分割名稱: {split}。應為 'train', 'val', 或 'test'")


class EnvFactory:
    """環境工廠 - 創建配置好數據的交易環境

    此類別作為工廠，統一創建和配置交易環境。
    它管理數據提供者實例，並將數據注入環境。

    使用方式:
        factory = EnvFactory(data_dir="data/raw")
        train_env = factory.create_env(split='train')
        val_env = factory.create_env(split='val')
    """

    # 類變數：共享數據提供者（單例模式）
    _shared_provider: Optional[EnvDataProvider] = None

    @classmethod
    def get_or_create_provider(
        cls,
        data_dir: str = "data/raw",
        **kwargs
    ) -> EnvDataProvider:
        """獲取或創建數據提供者（單例模式）

        此方法確保所有環境共享同一個數據提供者實例，
        避免重複載入數據浪費記憶體。

        參數:
            data_dir: 數據目錄
            **kwargs: 傳遞給 EnvDataProvider 的其他參數

        返回:
            EnvDataProvider: 數據提供者實例（共享）
        """
        if cls._shared_provider is None:
            logger.info("創建新的數據提供者實例...")
            cls._shared_provider = EnvDataProvider(data_dir=data_dir, **kwargs)
        return cls._shared_provider

    @classmethod
    def create_env(
        cls,
        env_config: Dict,
        split: str = 'train',
        data_dir: str = "data/raw",
    ):
        """創建配置好數據的交易環境

        此方法是主要的環境創建入口點。

        參數:
            env_config: 環境配置字典
                必須包含環境的所有參數（max_steps, transaction_cost_rate 等）

            split: 數據分割 ('train', 'val', 'test')
                決定環境使用哪個數據集

            data_dir: FI-2010 數據目錄

        返回:
            LOBTradingEnv: 配置好真實數據的環境實例

        環境配置:
            環境會被注入真實的 LOB 數據和價格序列
            data 欄位會被自動填充，無需手動提供
        """
        from .lob_trading_env import LOBTradingEnv

        # 獲取數據提供者
        provider = cls.get_or_create_provider(data_dir=data_dir)

        # 載入對應分割的數據
        lob_data, prices = provider.get_data_split(split)

        # 創建環境配置副本，避免修改原始配置
        env_config_copy = env_config.copy()

        # 注入真實數據
        env_config_copy['lob_data'] = lob_data
        env_config_copy['prices'] = prices
        env_config_copy['data_split'] = split

        logger.info(
            f"創建環境: split={split}, "
            f"數據量={len(lob_data)}, "
            f"序列形狀={lob_data.shape}"
        )

        # 創建環境實例
        env = LOBTradingEnv(env_config_copy)

        return env


def create_train_env(env_config: Dict):
    """創建訓練環境的便利函數

    參數:
        env_config: 環境配置字典

    返回:
        LOBTradingEnv: 訓練環境
    """
    return EnvFactory.create_env(env_config, split='train')


def create_val_env(env_config: Dict):
    """創建驗證環境的便利函數

    參數:
        env_config: 環境配置字典

    返回:
        LOBTradingEnv: 驗證環境
    """
    return EnvFactory.create_env(env_config, split='val')


def create_test_env(env_config: Dict):
    """創建測試環境的便利函數

    參數:
        env_config: 環境配置字典

    返回:
        LOBTradingEnv: 測試環境
    """
    return EnvFactory.create_env(env_config, split='test')
