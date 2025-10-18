"""股票技術指標數據載入器

此模組負責載入多股票的技術指標 CSV 數據，支援每日訓練場景。
自動掃描指定目錄下的所有 *_indicators.csv 文件，並進行批次處理。

文件命名格式: {股票編號}_{日期}_indicators.csv
範例: 6742_20250902_indicators.csv

主要功能:
    1. 自動掃描目錄下所有 indicators.csv 文件
    2. 解析文件名提取股票編號和日期
    3. 載入並標準化技術指標數據
    4. 支援特徵擴充（預留空間）
    5. 批次處理多股票數據

使用範例:
    >>> loader = StockIndicatorsLoader(
    ...     data_dir='data/processed',
    ...     max_features=128  # 預留擴充空間
    ... )
    >>> data, metadata = loader.load_all_stocks()
    >>> print(f"載入 {len(metadata)} 檔股票")
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StockIndicatorsLoader:
    """股票技術指標數據載入器

    自動掃描目錄下所有符合格式的 CSV 文件，並進行批次載入和處理。
    支援特徵擴充，確保未來可以無縫添加新特徵而不影響已訓練模型。

    參數:
        data_dir: 數據目錄路徑
        max_features: 最大特徵數量（預留擴充空間），默認 128
        normalize: 是否進行 Z-score 標準化，默認 True
        fill_missing: 是否填充缺失值，默認 True
        cache: 是否啟用緩存，默認 True

    屬性:
        stock_files: 已發現的股票文件列表
        feature_names: 當前使用的特徵名稱
        feature_count: 實際特徵數量
        metadata: 股票元數據（編號、日期、檔案路徑）
    """

    def __init__(
        self,
        data_dir: str = 'data/processed',
        max_features: int = 128,
        normalize: bool = True,
        fill_missing: bool = True,
        cache: bool = True
    ):
        """初始化數據載入器

        參數:
            data_dir: 數據目錄路徑
            max_features: 最大特徵維度（預留擴充）
            normalize: 是否標準化
            fill_missing: 是否填充缺失值
            cache: 是否啟用緩存
        """
        self.data_dir = Path(data_dir)
        self.max_features = max_features
        self.normalize = normalize
        self.fill_missing = fill_missing
        self.cache = cache

        # 內部狀態
        self.stock_files: List[Path] = []
        self.feature_names: List[str] = []
        self.feature_count: int = 0
        self.metadata: List[Dict] = []

        # 緩存
        self._cache_data: Optional[np.ndarray] = None
        self._cache_prices: Optional[np.ndarray] = None

        # 文件名正則表達式: {編號}_{日期}_indicators.csv
        self.filename_pattern = re.compile(r'^(\w+)_(\d{8})_indicators\.csv$')

        logger.info(f"初始化 StockIndicatorsLoader:")
        logger.info(f"  - 數據目錄: {self.data_dir}")
        logger.info(f"  - 最大特徵數: {self.max_features}")
        logger.info(f"  - 標準化: {self.normalize}")
        logger.info(f"  - 填充缺失: {self.fill_missing}")

    def scan_directory(self) -> int:
        """掃描目錄下所有 indicators.csv 文件

        返回:
            發現的文件數量

        文件格式: {股票編號}_{日期}_indicators.csv
        範例: 6742_20250902_indicators.csv, 2330_20250902_indicators.csv
        """
        if not self.data_dir.exists():
            logger.error(f"數據目錄不存在: {self.data_dir}")
            return 0

        # 搜尋所有 *_indicators.csv 文件
        pattern = str(self.data_dir / "*_indicators.csv")
        files = glob.glob(pattern)

        self.stock_files = []
        self.metadata = []

        for file_path in files:
            file_name = Path(file_path).name
            match = self.filename_pattern.match(file_name)

            if match:
                stock_id = match.group(1)
                date_str = match.group(2)

                # 驗證日期格式
                try:
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    logger.warning(f"⚠️ 日期格式錯誤，跳過: {file_name}")
                    continue

                self.stock_files.append(Path(file_path))
                self.metadata.append({
                    'stock_id': stock_id,
                    'date': date_str,
                    'date_obj': date_obj,
                    'file_path': file_path,
                    'file_name': file_name
                })
            else:
                logger.debug(f"文件名格式不符，跳過: {file_name}")

        # 按日期和股票編號排序
        sorted_indices = sorted(
            range(len(self.metadata)),
            key=lambda i: (self.metadata[i]['date'], self.metadata[i]['stock_id'])
        )

        self.stock_files = [self.stock_files[i] for i in sorted_indices]
        self.metadata = [self.metadata[i] for i in sorted_indices]

        logger.info(f"✓ 掃描完成，發現 {len(self.stock_files)} 個股票數據文件")

        if self.metadata:
            dates = sorted(set(m['date'] for m in self.metadata))
            stocks_per_date = {}
            for m in self.metadata:
                stocks_per_date[m['date']] = stocks_per_date.get(m['date'], 0) + 1

            logger.info(f"  - 日期範圍: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)")
            logger.info(f"  - 每日平均股票數: {np.mean(list(stocks_per_date.values())):.1f}")

        return len(self.stock_files)

    def load_single_stock(
        self,
        file_path: Path,
        return_metadata: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
        """載入單一股票數據

        參數:
            file_path: CSV 文件路徑
            return_metadata: 是否返回元數據

        返回:
            features: (timesteps, max_features) 特徵數組（填充到 max_features）
            prices: (timesteps,) 價格數組（Close 價格）
            metadata: 元數據字典（如果 return_metadata=True）
        """
        try:
            # 讀取 CSV
            df = pd.read_csv(file_path)

            if len(df) == 0:
                logger.warning(f"⚠️ 空文件，跳過: {file_path.name}")
                return None, None, None

            # 提取價格（用於 reward 計算和記錄）
            if 'Close' not in df.columns:
                logger.error(f"❌ 缺少 Close 欄位: {file_path.name}")
                return None, None, None

            prices = df['Close'].values.astype(np.float32)

            # 排除非特徵欄位
            exclude_cols = ['Date', 'Datetime']  # 時間欄位
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # 提取特徵
            features = df[feature_cols].values.astype(np.float32)

            # 填充缺失值（使用前向填充 + 後向填充）
            if self.fill_missing:
                df_features = pd.DataFrame(features, columns=feature_cols)
                df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
                features = df_features.values.astype(np.float32)

            # 記錄特徵名稱（第一次載入時）
            if not self.feature_names:
                self.feature_names = feature_cols
                self.feature_count = len(feature_cols)
                logger.info(f"✓ 特徵欄位 ({self.feature_count} 個): {', '.join(feature_cols[:5])}...")

            # 驗證特徵數量一致性
            if features.shape[1] != self.feature_count:
                logger.warning(
                    f"⚠️ 特徵數量不一致: {file_path.name} "
                    f"({features.shape[1]} vs {self.feature_count})，嘗試對齊"
                )
                # 嘗試對齊特徵
                aligned_features = np.zeros((features.shape[0], self.feature_count), dtype=np.float32)
                min_cols = min(features.shape[1], self.feature_count)
                aligned_features[:, :min_cols] = features[:, :min_cols]
                features = aligned_features

            # Z-score 標準化（每個特徵獨立標準化）
            if self.normalize:
                mean = np.mean(features, axis=0, keepdims=True)
                std = np.std(features, axis=0, keepdims=True) + 1e-8
                features = (features - mean) / std

            # 填充到 max_features（預留擴充空間）
            if features.shape[1] < self.max_features:
                padded_features = np.zeros((features.shape[0], self.max_features), dtype=np.float32)
                padded_features[:, :features.shape[1]] = features
                features = padded_features
            elif features.shape[1] > self.max_features:
                logger.warning(
                    f"⚠️ 特徵數量 ({features.shape[1]}) 超過 max_features ({self.max_features})，截斷"
                )
                features = features[:, :self.max_features]

            # 構建元數據
            metadata = None
            if return_metadata:
                metadata = {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'timesteps': len(features),
                    'feature_count': self.feature_count,
                    'max_features': self.max_features,
                    'price_range': (float(prices.min()), float(prices.max())),
                    'feature_names': self.feature_names
                }

            return features, prices, metadata

        except Exception as e:
            logger.error(f"❌ 載入失敗: {file_path.name}, 錯誤: {e}")
            return None, None, None

    def load_all_stocks(
        self,
        max_stocks: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """載入所有股票數據

        參數:
            max_stocks: 最大載入股票數量（None 表示全部）

        返回:
            all_features: 特徵數組列表 [(timesteps, max_features), ...]
            all_prices: 價格數組列表 [(timesteps,), ...]
            all_metadata: 元數據列表
        """
        if not self.stock_files:
            logger.info("尚未掃描目錄，開始自動掃描...")
            self.scan_directory()

        if not self.stock_files:
            logger.error("❌ 未找到任何股票數據文件")
            return [], [], []

        all_features = []
        all_prices = []
        all_metadata = []

        load_count = min(len(self.stock_files), max_stocks) if max_stocks else len(self.stock_files)

        logger.info(f"開始載入 {load_count} 個股票數據...")

        for i, file_path in enumerate(self.stock_files[:load_count]):
            if (i + 1) % 50 == 0:
                logger.info(f"  進度: {i+1}/{load_count}")

            features, prices, _ = self.load_single_stock(file_path, return_metadata=False)

            if features is not None:
                all_features.append(features)
                all_prices.append(prices)
                all_metadata.append(self.metadata[i])

        logger.info(f"✓ 成功載入 {len(all_features)}/{load_count} 個股票數據")

        return all_features, all_prices, all_metadata

    def concatenate_all(
        self,
        max_stocks: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """載入並串接所有股票數據為單一數組

        參數:
            max_stocks: 最大載入股票數量

        返回:
            features: (total_timesteps, max_features) 串接後的特徵
            prices: (total_timesteps,) 串接後的價格
            metadata: 元數據列表（包含每個股票的起始索引）
        """
        # 檢查緩存
        if self.cache and self._cache_data is not None:
            logger.info("✓ 使用緩存數據")
            return self._cache_data, self._cache_prices, self.metadata

        all_features, all_prices, all_metadata = self.load_all_stocks(max_stocks)

        if not all_features:
            return np.array([]), np.array([]), []

        # 串接所有數據
        features = np.vstack(all_features)
        prices = np.concatenate(all_prices)

        # 記錄每個股票的起始索引
        cumsum = 0
        for i, feat in enumerate(all_features):
            all_metadata[i]['start_idx'] = cumsum
            all_metadata[i]['end_idx'] = cumsum + len(feat)
            cumsum += len(feat)

        logger.info(f"✓ 數據串接完成:")
        logger.info(f"  - 總時間步數: {len(features)}")
        logger.info(f"  - 特徵維度: {features.shape[1]}")
        logger.info(f"  - 股票數量: {len(all_metadata)}")

        # 緩存
        if self.cache:
            self._cache_data = features
            self._cache_prices = prices

        return features, prices, all_metadata

    def get_stock_by_id(
        self,
        stock_id: str,
        date: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """根據股票編號（和日期）載入數據

        參數:
            stock_id: 股票編號 (如 '6742')
            date: 日期字串 (如 '20250902')，None 表示載入該股票的所有日期

        返回:
            features, prices, metadata
        """
        if not self.metadata:
            self.scan_directory()

        for i, meta in enumerate(self.metadata):
            if meta['stock_id'] == stock_id:
                if date is None or meta['date'] == date:
                    file_path = self.stock_files[i]
                    return self.load_single_stock(file_path, return_metadata=True)

        logger.warning(f"⚠️ 未找到股票: {stock_id}" + (f" (日期: {date})" if date else ""))
        return None, None, None

    def get_stocks_by_date(self, date: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """載入指定日期的所有股票數據

        參數:
            date: 日期字串 (如 '20250902')

        返回:
            all_features, all_prices, all_metadata
        """
        if not self.metadata:
            self.scan_directory()

        all_features = []
        all_prices = []
        all_metadata = []

        for i, meta in enumerate(self.metadata):
            if meta['date'] == date:
                file_path = self.stock_files[i]
                features, prices, _ = self.load_single_stock(file_path, return_metadata=False)

                if features is not None:
                    all_features.append(features)
                    all_prices.append(prices)
                    all_metadata.append(meta)

        logger.info(f"✓ 載入日期 {date} 的 {len(all_features)} 檔股票")

        return all_features, all_prices, all_metadata

    def get_feature_info(self) -> Dict:
        """獲取特徵信息

        返回:
            特徵信息字典
        """
        return {
            'feature_names': self.feature_names,
            'feature_count': self.feature_count,
            'max_features': self.max_features,
            'padding': self.max_features - self.feature_count,
            'stock_count': len(self.stock_files),
            'dates': sorted(set(m['date'] for m in self.metadata)) if self.metadata else []
        }


if __name__ == '__main__':
    # 測試代碼
    logging.basicConfig(level=logging.INFO)

    loader = StockIndicatorsLoader(
        data_dir='data/processed',
        max_features=128,
        normalize=True
    )

    # 掃描目錄
    count = loader.scan_directory()
    print(f"\n發現 {count} 個文件")

    # 載入單一股票
    if loader.stock_files:
        features, prices, meta = loader.load_single_stock(
            loader.stock_files[0],
            return_metadata=True
        )
        print(f"\n單一股票:")
        print(f"  - 文件: {meta['file_name']}")
        print(f"  - 時間步數: {meta['timesteps']}")
        print(f"  - 特徵維度: {features.shape}")
        print(f"  - 價格範圍: {meta['price_range']}")

    # 獲取特徵信息
    info = loader.get_feature_info()
    print(f"\n特徵信息:")
    print(f"  - 特徵數量: {info['feature_count']}")
    print(f"  - 最大特徵: {info['max_features']}")
    print(f"  - 填充維度: {info['padding']}")
    print(f"  - 股票數量: {info['stock_count']}")
