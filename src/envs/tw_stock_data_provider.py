"""台股數據提供者 - 載入預處理好的台股 LOB embedding 數據

此模組負責載入已預處理的台股 LOB 數據（stock_embedding_*.npz），
並提供給交易環境使用。與 EnvDataProvider 不同，這個提供者：
  1. 載入 .npz 格式的預處理數據（不是原始 CSV）
  2. 支援 5檔 LOB（20維特徵）而非 10檔（40維）
  3. 直接使用 embedding 數據，無需額外預處理

數據格式:
    X: (N, 100, 20) - N個樣本，每個100時間步 × 20維LOB特徵
    y: (N,) - 價格變動標籤 {0: 下跌, 1: 持平, 2: 上漲}
    stock_ids: (N,) - 股票編號

使用範例:
    >>> provider = TaiwanStockDataProvider(data_dir="data/processed")
    >>> train_data, train_labels = provider.get_train_data()
    >>> print(train_data.shape)  # (5584553, 100, 20)

作者: RLlib-DeepLOB 專案團隊
更新: 2025-10-12
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# 全域數據快取（跨 Worker 共享）
_GLOBAL_DATA_CACHE = {
    'train': None,
    'val': None,
    'test': None,
}


class TaiwanStockDataProvider:
    """台股數據提供者

    此類別負責為交易環境提供預處理好的台股 LOB 數據。
    與 FI-2010 數據不同，台股數據已經過完整預處理並保存為 .npz 格式。

    設計特點:
        - 載入預處理數據：直接讀取 stock_embedding_*.npz
        - 5檔 LOB：20維特徵（買賣各5檔價量）
        - 時序窗口：100時間步
        - 數據快取：避免重複載入
        - 分割管理：train/val/test 三個分割

    數據維度:
        - LOB 特徵: (N, 100, 20)
          * N: 樣本數
          * 100: 時間步數
          * 20: 5檔 LOB 特徵（bid_price×5 + bid_vol×5 + ask_price×5 + ask_vol×5）
        - 標籤: (N,) - {0: 下跌, 1: 持平, 2: 上漲}
        - 股票ID: (N,) - 股票編號
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        use_embedding: bool = True,
        sample_ratio: float = 1.0,
    ):
        """初始化台股數據提供者

        參數:
            data_dir: 數據目錄（包含 stock_embedding_*.npz 文件）
            use_embedding: 是否使用 embedding 數據（預設 True）
            sample_ratio: 數據採樣比例（0.0-1.0），預設 1.0（全部數據）
                         設為 0.1 可只使用 10% 數據，大幅減少記憶體使用
        """
        self.data_dir = Path(data_dir)
        self.use_embedding = use_embedding
        self.sample_ratio = max(0.01, min(1.0, sample_ratio))  # 限制在 1%-100%

        # 數據快取
        self._train_data = None
        self._train_labels = None
        self._val_data = None
        self._val_labels = None
        self._test_data = None
        self._test_labels = None

        # 載入狀態
        self._is_loaded = False

        logger.info(
            f"台股數據提供者已初始化: "
            f"數據目錄={data_dir}, "
            f"使用 embedding={use_embedding}, "
            f"採樣比例={self.sample_ratio:.1%}"
        )

    def _load_npz(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """載入 .npz 數據文件（優化記憶體使用）

        【修正】使用分塊載入避免記憶體溢出
        問題：X_full[indices] 會強制 NumPy 先載入整個陣列（41.6 GB）
        解法：使用 np.take 分批載入，或直接用連續索引切片

        參數:
            filename: 數據文件名（例如 "stock_embedding_train.npz"）

        返回:
            (data, labels):
                - data: (N, 100, 20) LOB 序列
                - labels: (N,) 價格變動標籤
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"找不到數據文件: {filepath}")

        logger.info(f"載入數據文件: {filepath}")

        # 1. 先載入標籤（很小，幾 MB）
        logger.info(f"⏳ 步驟 1/2: 載入標籤...")
        with np.load(filepath) as npz_data:
            if 'X' not in npz_data or 'y' not in npz_data:
                raise ValueError(f"數據文件格式錯誤，缺少 'X' 或 'y' 鍵: {filepath}")

            # 只載入 y（標籤）到記憶體
            y_full = npz_data['y'][:]  # 使用 [:] 確保複製
            n_samples = len(y_full)
            logger.info(f"✅ 標籤載入完成: {n_samples:,} 個樣本")

        # 2. 計算採樣索引
        if self.sample_ratio < 1.0:
            n_sampled = int(n_samples * self.sample_ratio)
            np.random.seed(42)  # 固定種子
            indices = np.random.choice(n_samples, size=n_sampled, replace=False)
            indices = np.sort(indices)  # 保持時序性
            logger.info(
                f"📉 數據採樣: {n_samples:,} → {n_sampled:,} 樣本 ({self.sample_ratio:.1%})"
            )
        else:
            indices = None
            n_sampled = n_samples

        # 3. 載入 X（大數據，使用分塊載入避免記憶體爆炸）
        logger.info(f"⏳ 步驟 2/2: 載入 LOB 數據...")

        if indices is not None:
            # 【關鍵修正】採樣模式：使用分塊載入避免一次性載入全部數據
            # 問題：X_full[indices] 會觸發 NumPy 載入整個 41.6 GB 陣列
            # 解法：分批讀取小塊數據

            chunk_size = 50000  # 每次讀取 5萬個樣本 (~190 MB)
            X_list = []
            y = y_full[indices]  # 先採樣標籤

            logger.info(f"📦 使用分塊載入 (chunk_size={chunk_size:,})...")

            with np.load(filepath, mmap_mode='r') as npz_data:
                X_mmap = npz_data['X']

                # 將索引分組，每組從連續範圍讀取
                n_chunks = (len(indices) + chunk_size - 1) // chunk_size
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(indices))
                    chunk_indices = indices[start_idx:end_idx]

                    # 讀取這一塊的數據
                    X_chunk = X_mmap[chunk_indices]
                    X_list.append(X_chunk.copy())  # 複製到記憶體

                    if (i + 1) % 10 == 0 or i == n_chunks - 1:
                        logger.info(f"  進度: {i+1}/{n_chunks} chunks ({(i+1)/n_chunks*100:.1f}%)")

            # 合併所有塊
            X = np.concatenate(X_list, axis=0)

            # 驗證形狀
            assert X.ndim == 3, f"X 維度錯誤: 期望3維，實際{X.ndim}維"
            assert X.shape[1] == 100, f"時間步錯誤: 期望100，實際{X.shape[1]}"
            assert X.shape[2] == 20, f"特徵維度錯誤: 期望20，實際{X.shape[2]}"

            saved_memory = (n_samples - n_sampled) * 100 * 20 * 4 / 1e9
            logger.info(f"✅ 採樣數據載入完成，節省記憶體: {saved_memory:.2f} GB")
        else:
            # 全量模式：載入全部數據
            with np.load(filepath) as npz_data:
                X = npz_data['X'][:]
            y = y_full
            logger.info(f"✅ 全量數據載入完成: {len(X):,} 個樣本")

        logger.info(
            f"✅ 數據載入成功: 形狀={X.shape}, dtype={X.dtype}, "
            f"記憶體使用: {X.nbytes / 1e9:.2f} GB"
        )

        return X, y

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取訓練集數據（使用全域快取避免重複載入）

        返回:
            (train_data, train_labels):
                - train_data: (N, 100, 20) LOB 序列
                - train_labels: (N,) 標籤
        """
        global _GLOBAL_DATA_CACHE

        # 優先使用全域快取（跨 Worker 共享）
        if _GLOBAL_DATA_CACHE['train'] is not None:
            logger.info("✅ 使用全域快取的訓練集數據（避免重複載入）")
            return _GLOBAL_DATA_CACHE['train']

        # 如果全域快取不存在，使用實例快取
        if self._train_data is None:
            logger.info("首次載入訓練集數據...")
            self._train_data, self._train_labels = self._load_npz("stock_embedding_train.npz")
            # 更新全域快取
            _GLOBAL_DATA_CACHE['train'] = (self._train_data, self._train_labels)

        return self._train_data, self._train_labels

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取驗證集數據（使用全域快取）

        返回:
            (val_data, val_labels):
                - val_data: (N, 100, 20) LOB 序列
                - val_labels: (N,) 標籤
        """
        global _GLOBAL_DATA_CACHE

        if _GLOBAL_DATA_CACHE['val'] is not None:
            logger.info("✅ 使用全域快取的驗證集數據")
            return _GLOBAL_DATA_CACHE['val']

        if self._val_data is None:
            logger.info("首次載入驗證集數據...")
            self._val_data, self._val_labels = self._load_npz("stock_embedding_val.npz")
            _GLOBAL_DATA_CACHE['val'] = (self._val_data, self._val_labels)

        return self._val_data, self._val_labels

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取測試集數據（使用全域快取）

        返回:
            (test_data, test_labels):
                - test_data: (N, 100, 20) LOB 序列
                - test_labels: (N,) 標籤
        """
        global _GLOBAL_DATA_CACHE

        if _GLOBAL_DATA_CACHE['test'] is not None:
            logger.info("✅ 使用全域快取的測試集數據")
            return _GLOBAL_DATA_CACHE['test']

        if self._test_data is None:
            logger.info("首次載入測試集數據...")
            self._test_data, self._test_labels = self._load_npz("stock_embedding_test.npz")
            _GLOBAL_DATA_CACHE['test'] = (self._test_data, self._test_labels)

        return self._test_data, self._test_labels

    def get_prices(self, mode: str = 'train') -> Optional[np.ndarray]:
        """獲取真實價格數據（如果有）

        參數:
            mode: 數據模式 'train'/'val'/'test'

        返回:
            prices: (N,) 價格序列，如果數據中沒有價格則返回 None
        """
        filename_map = {
            'train': 'stock_embedding_train.npz',
            'val': 'stock_embedding_val.npz',
            'test': 'stock_embedding_test.npz'
        }

        filepath = self.data_dir / filename_map[mode]

        if not filepath.exists():
            logger.warning(f"找不到數據文件: {filepath}")
            return None

        try:
            with np.load(filepath) as data:
                # V7 數據使用 'prices' (複數), 向後兼容 'price' (單數)
                if 'prices' in data.keys():
                    prices_data = data['prices'][:]
                    # V7 格式: (N, 100) - 每個樣本 100 個時間步的價格
                    # 我們取最後一個時間步的價格 (用於交易決策)
                    if prices_data.ndim == 2:
                        logger.info(f"✅ 載入真實價格數據 (prices): {mode} 集, 形狀 {prices_data.shape}, 取最後時間步")
                        return prices_data[:, -1]  # 取每個樣本的最後一個價格
                    else:
                        logger.info(f"✅ 載入真實價格數據 (prices): {mode} 集, 形狀 {prices_data.shape}")
                        return prices_data
                elif 'price' in data.keys():
                    price_data = data['price'][:]
                    logger.info(f"✅ 載入真實價格數據 (price): {mode} 集, 形狀 {price_data.shape}")
                    # 舊格式應該是 (N,) 一維
                    if price_data.ndim == 2:
                        return price_data[:, -1]
                    return price_data
                else:
                    logger.warning(f"⚠️  數據文件中沒有 'price' 或 'prices' 字段: {filepath}")
                    return None
        except Exception as e:
            logger.error(f"❌ 載入價格數據失敗: {e}")
            return None

    def get_metadata(self) -> Dict:
        """獲取數據集元數據

        返回:
            metadata: 包含數據集統計信息的字典
        """
        import json
        meta_path = self.data_dir / "meta.json"

        if not meta_path.exists():
            logger.warning(f"找不到元數據文件: {meta_path}")
            return {}

        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata
