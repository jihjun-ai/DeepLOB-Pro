"""
台股即時報價特徵提取器
支援從原始 tick 數據生成 DeepLOB 訓練格式

文件格式: YYYYMMDD.txt (如 20251007.txt)
數據格式: || 分隔的即時報價 (包含5檔價量)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime


class TWTickDataExtractor:
    """台股即時報價特徵提取器"""

    def __init__(self, filter_trial_match: bool = True):
        """
        Args:
            filter_trial_match: 是否過濾試撮數據 (建議 True)
        """
        self.filter_trial_match = filter_trial_match
        self.scaler = None

    def parse_line(self, line: str) -> Optional[Dict]:
        """
        解析單行數據

        數據格式 (|| 分隔):
        [0]  QType
        [1]  Symbol (股票代號)
        [2]  Name (股票名稱)
        [3]  ReferencePrice (參考價)
        [4]  UpperPrice (漲停價)
        [5]  LowerPrice (跌停價)
        [6]  OpenPrice (開盤價)
        [7]  HighPrice (最高價)
        [8]  LowPrice (最低價)
        [9]  LastPrice (成交價)
        [10] LastVolume (成交量)
        [11] TotalVolume (累計成交量)
        [12-21] Bid1-5 Price/Volume (買1-5價量)
        [22-31] Ask1-5 Price/Volume (賣1-5價量)
        [32] MatchTime (撮合時間 HHMMSS)
        [33] IsTrialMatch (是否試撮)

        Returns:
            {
                'symbol': str,
                'name': str,
                'timestamp': str (HHMMSS),
                'features': np.array([20,]),  # 台股 LOB 格式（5 檔）
                'mid_price': float,
                'spread': float,
                'spread_pct': float,
                'last_price': float,
                'last_volume': int,
                'total_volume': int,
                'is_trial': bool
            }
        """
        fields = line.strip().split('||')

        if len(fields) < 34:
            return None

        try:
            # 基本信息
            symbol = fields[1].strip()
            name = fields[2].strip()
            timestamp = fields[32].strip()
            is_trial = fields[33].strip() == '1'

            # 過濾試撮
            if self.filter_trial_match and is_trial:
                return None

            # 成交信息 (读取但不用于过滤 - mid_price 不需要成交就存在)
            last_volume = int(fields[10])

            # ⭐ 注意：不过滤 LastVolume=0 的数据
            # 因为标签生成基于 mid_price (买卖一价)，不需要实际成交
            # LOB 数据在无成交时依然包含有价值的委托单信息

            # 5檔買賣價量
            bid_prices = [
                float(fields[12]),  # Bid1
                float(fields[14]),  # Bid2
                float(fields[16]),  # Bid3
                float(fields[18]),  # Bid4
                float(fields[20])   # Bid5
            ]

            bid_volumes = [
                int(fields[13]),
                int(fields[15]),
                int(fields[17]),
                int(fields[19]),
                int(fields[21])
            ]

            ask_prices = [
                float(fields[22]),  # Ask1
                float(fields[24]),  # Ask2
                float(fields[26]),  # Ask3
                float(fields[28]),  # Ask4
                float(fields[30])   # Ask5
            ]

            ask_volumes = [
                int(fields[23]),
                int(fields[25]),
                int(fields[27]),
                int(fields[29]),
                int(fields[31])
            ]

            # 檢查異常值 (價格為0或負數)
            if ask_prices[0] <= 0 or bid_prices[0] <= 0:
                return None

            # 台股 LOB 格式: [Ask Price 1-5, Ask Vol 1-5, Bid Price 1-5, Bid Vol 1-5]
            # 總共 20 維（5 檔買賣價量）
            features = np.array(
                ask_prices + ask_volumes + bid_prices + bid_volumes,
                dtype=np.float32
            )

            # 計算衍生特徵
            mid_price = (ask_prices[0] + bid_prices[0]) / 2.0
            spread = ask_prices[0] - bid_prices[0]
            spread_pct = spread / mid_price * 100  # 價差百分比

            # 成交信息
            last_price = float(fields[9])
            # last_volume 已在上面讀取過了
            total_volume = int(fields[11])

            return {
                'symbol': symbol,
                'name': name,
                'timestamp': timestamp,
                'features': features,
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'last_price': last_price,
                'last_volume': last_volume,
                'total_volume': total_volume,
                'is_trial': is_trial
            }

        except (ValueError, IndexError, ZeroDivisionError) as e:
            # 靜默跳過錯誤數據
            return None

    def process_file(
        self,
        filepath: str,
        symbols: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        處理單個文件

        Args:
            filepath: 數據文件路徑 (如 data/raw/20251007.txt)
            symbols: 指定股票列表 (None = 全部)
            verbose: 是否顯示處理進度

        Returns:
            {
                '2330': DataFrame(columns=['timestamp', 'features', 'mid_price', ...]),
                '0050': DataFrame(...),
                ...
            }
        """
        data_by_symbol = defaultdict(list)

        line_count = 0
        parsed_count = 0
        error_count = 0

        if verbose:
            print(f"[PROCESSING] 處理文件: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1

                    tick = self.parse_line(line)

                    if tick is None:
                        error_count += 1
                        continue

                    # 過濾指定標的
                    if symbols and tick['symbol'] not in symbols:
                        continue

                    data_by_symbol[tick['symbol']].append(tick)
                    parsed_count += 1

        except FileNotFoundError:
            print(f"[ERROR] 文件不存在: {filepath}")
            return {}
        except Exception as e:
            print(f"[ERROR] 處理文件時發生錯誤: {e}")
            return {}

        if verbose:
            print(f"[OK] 處理完成")
            print(f"   總行數: {line_count:,}")
            print(f"   有效數據: {parsed_count:,}")
            print(f"   跳過數據: {error_count:,} (試撮/異常值)")
            print(f"   標的數量: {len(data_by_symbol)}")

        # 轉換為 DataFrame
        result = {}
        for symbol, ticks in data_by_symbol.items():
            df = pd.DataFrame(ticks)

            # 按時間排序
            df = df.sort_values('timestamp').reset_index(drop=True)

            # ⭐ 去除重複資料 (基於時間戳和五檔價格)
            # 保留第一筆,刪除後續重複的
            original_count = len(df)
            df = df.drop_duplicates(
                subset=['timestamp', 'mid_price', 'total_volume'],
                keep='first'
            ).reset_index(drop=True)

            duplicates_removed = original_count - len(df)

            result[symbol] = df

            if verbose and symbols:  # 只在指定標的時顯示詳情
                msg = f"   [DATA] {symbol} ({df.iloc[0]['name']}): {len(df):,} ticks"
                if duplicates_removed > 0:
                    msg += f" (去重: -{duplicates_removed:,})"
                print(msg)

        return result

    def process_multiple_files(
        self,
        filepaths: List[str],
        symbols: List[str],
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        處理多個文件並合併

        Args:
            filepaths: 文件路徑列表 (如 ['20251007.txt', '20251008.txt'])
            symbols: 指定股票列表
            verbose: 是否顯示處理進度

        Returns:
            合併後的數據字典
        """
        all_data = defaultdict(list)

        for i, filepath in enumerate(filepaths, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"處理文件 {i}/{len(filepaths)}")
                print(f"{'='*60}")

            file_data = self.process_file(filepath, symbols, verbose=verbose)

            for symbol, df in file_data.items():
                all_data[symbol].append(df)

        # 合併每個標的的數據
        if verbose:
            print(f"\n{'='*60}")
            print("合併數據")
            print(f"{'='*60}")

        result = {}
        for symbol, df_list in all_data.items():
            merged = pd.concat(df_list, ignore_index=True)
            merged = merged.sort_values('timestamp').reset_index(drop=True)
            result[symbol] = merged

            if verbose:
                print(f"[OK] {symbol}: 合併後 {len(merged):,} ticks")

        return result

    def generate_labels(
        self,
        mid_prices: np.ndarray,
        k: int = 50,
        alpha: float = 0.005
    ) -> Tuple[np.ndarray, Dict]:
        """
        生成價格變動標籤

        Args:
            mid_prices: 中間價序列 (N,)
            k: 未來 k 個 tick (建議 30-100)
            alpha: 閾值 (0.5% = 0.005, 1% = 0.01)

        Returns:
            labels: (N-k,) [0=上漲, 1=穩定, 2=下跌]
            stats: 標籤統計信息
        """
        labels = []

        for i in range(len(mid_prices) - k):
            current = mid_prices[i]
            future = mid_prices[i + k]

            change = (future - current) / current

            if change > alpha:
                label = 0  # 上漲
            elif change < -alpha:
                label = 2  # 下跌
            else:
                label = 1  # 穩定

            labels.append(label)

        labels = np.array(labels, dtype=np.int64)

        # 統計
        stats = {
            'total': len(labels),
            'up': int(np.sum(labels == 0)),
            'stable': int(np.sum(labels == 1)),
            'down': int(np.sum(labels == 2)),
            'up_ratio': float(np.mean(labels == 0)),
            'stable_ratio': float(np.mean(labels == 1)),
            'down_ratio': float(np.mean(labels == 2))
        }

        return labels, stats

    def generate_labels_fi2010_style(
        self,
        mid_prices: np.ndarray,
        k: int = 50,
        alpha: float = 0.005,
        smoothing: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        生成 FI-2010 風格的價格變動標籤

        Args:
            mid_prices: 中間價序列 (N,)
            k: 預測窗口步數（用於前後平均）
            alpha: 閾值（百分比，如 0.005 = 0.5%）
            smoothing: 是否使用前後 k 視窗平均（FI-2010 標準）

        Returns:
            labels: (N-2k,) [0=上漲, 1=穩定, 2=下跌]
            stats: 標籤統計信息

        標籤定義（FI-2010 論文）:
            - 計算「前 k 視窗平均」: m_prev = mean(mid_prices[i-k:i])
            - 計算「後 k 視窗平均」: m_next = mean(mid_prices[i+1:i+k+1])
            - 相對變化: r = (m_next - m_prev) / m_prev
            - if r > alpha: label = 0 (上漲)
            - elif r < -alpha: label = 2 (下跌)
            - else: label = 1 (穩定)

        注意:
            - 輸出標籤數量比輸入少 2k（前 k + 後 k）
            - 若 smoothing=False，則退化為原 generate_labels()
        """
        if not smoothing:
            # 退化為原標籤生成邏輯
            return self.generate_labels(mid_prices, k, alpha)

        N = len(mid_prices)
        if N < 2 * k + 1:
            return np.array([]), {'error': 'insufficient data'}

        labels = []

        # 從 i=k 到 i=N-k-1
        for i in range(k, N - k):
            # 前 k 視窗平均
            m_prev = mid_prices[i-k:i].mean()

            # 後 k 視窗平均
            m_next = mid_prices[i+1:i+k+1].mean()

            # 相對變化
            r = (m_next - m_prev) / (m_prev + 1e-12)

            # 標籤分類
            if r > alpha:
                label = 0  # 上漲
            elif r < -alpha:
                label = 2  # 下跌
            else:
                label = 1  # 穩定

            labels.append(label)

        labels = np.array(labels, dtype=np.int64)

        # 統計
        stats = {
            'total': len(labels),
            'up': int(np.sum(labels == 0)),
            'stable': int(np.sum(labels == 1)),
            'down': int(np.sum(labels == 2)),
            'up_ratio': float(np.mean(labels == 0)),
            'stable_ratio': float(np.mean(labels == 1)),
            'down_ratio': float(np.mean(labels == 2)),
            'method': 'FI-2010 k-window average'
        }

        return labels, stats

    def create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        創建時間序列樣本

        Args:
            features: (N, 20) LOB 特徵
            labels: (N-k,) 標籤
            seq_len: 序列長度 (通常 100)

        Returns:
            X: (num_samples, 100, 20)
            y: (num_samples,)
        """
        # 確保特徵和標籤對齊
        features = features[:len(labels)]

        if len(features) < seq_len:
            print(f"[WARNING] 數據量 ({len(features)}) 不足以生成序列 (需要 >= {seq_len})")
            return np.array([]), np.array([])

        X, y = [], []

        for i in range(len(labels) - seq_len + 1):
            X.append(features[i:i+seq_len])
            y.append(labels[i+seq_len-1])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def create_sequences_no_cross_day(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        day_indices: np.ndarray,
        seq_len: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        創建時間序列樣本（不跨交易日）

        Args:
            features: (N, 20) LOB 特徵
            labels: (N,) 標籤
            day_indices: (N,) 每個 tick 對應的交易日索引（0-based）
            seq_len: 序列長度（通常 100）

        Returns:
            X: (num_samples, seq_len, 20)
            y: (num_samples,)
            day_ids: (num_samples,) 每個樣本對應的交易日索引

        實作邏輯:
            1. 確保 features 和 labels 對齊
            2. 遍歷每個交易日
            3. 在每個交易日內生成滑動視窗（不跨越日期邊界）
        """
        if len(features) != len(labels) or len(features) != len(day_indices):
            raise ValueError("features, labels, day_indices 長度必須一致")

        X, y, day_ids = [], [], []

        # 按交易日分組
        unique_days = np.unique(day_indices)

        for day_idx in unique_days:
            # 取出當日數據
            mask = (day_indices == day_idx)
            day_features = features[mask]
            day_labels = labels[mask]

            # 檢查數據量
            if len(day_features) < seq_len:
                continue

            # 在當日內生成滑動視窗
            for i in range(len(day_labels) - seq_len + 1):
                X.append(day_features[i:i+seq_len])
                y.append(day_labels[i+seq_len-1])
                day_ids.append(day_idx)

        if len(X) == 0:
            return np.array([]), np.array([]), np.array([])

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            np.array(day_ids, dtype=np.int32)
        )

    def fit_scaler(self, train_features: np.ndarray):
        """
        訓練標準化器

        Args:
            train_features: (N, 20) 訓練集特徵
        """
        self.scaler = StandardScaler()
        self.scaler.fit(train_features)
        print(f"[OK] Scaler 訓練完成")
        print(f"   Mean (前5維): {self.scaler.mean_[:5]}")
        print(f"   Std (前5維): {self.scaler.scale_[:5]}")

    def normalize_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        標準化序列數據

        Args:
            X: (num_samples, 100, 20)

        Returns:
            X_norm: (num_samples, 100, 20)
        """
        if self.scaler is None:
            raise ValueError("請先調用 fit_scaler() 訓練標準化器")

        num_samples, seq_len, num_features = X.shape

        # Reshape
        X_reshaped = X.reshape(-1, num_features)

        # 標準化
        X_normalized = self.scaler.transform(X_reshaped)

        # Reshape back
        return X_normalized.reshape(num_samples, seq_len, num_features).astype(np.float32)

    def save_scaler(self, filepath: str):
        """保存標準化器"""
        if self.scaler is None:
            raise ValueError("Scaler 尚未訓練")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[OK] Scaler 已保存: {filepath}")

    def load_scaler(self, filepath: str):
        """載入標準化器"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"[OK] Scaler 已載入: {filepath}")


def extract_date_from_filename(filename: str) -> str:
    """
    從檔案名提取日期

    Args:
        filename: 如 '20251007.txt' 或 'data/raw/20251007.txt'

    Returns:
        '20251007'
    """
    import os
    basename = os.path.basename(filename)
    return basename.split('.')[0].split('-')[0]  # 處理 20251007-1.txt 格式


# ============ 使用範例 ============

if __name__ == "__main__":
    import sys

    # 簡單測試
    print("=" * 60)
    print("台股數據特徵提取器測試")
    print("=" * 60)

    extractor = TWTickDataExtractor(filter_trial_match=True)

    # 測試樣本文件
    test_file = 'data/raw/20251007-1.txt'

    if Path(test_file).exists():
        print(f"\n[OK] 找到測試文件: {test_file}")

        # 處理文件
        data_dict = extractor.process_file(
            test_file,
            symbols=['2330', '0050', '0056'],  # 測試3個標的
            verbose=True
        )

        # 顯示樣本數據
        if '2330' in data_dict:
            df = data_dict['2330']
            print(f"\n{'='*60}")
            print("2330 (台積電) 數據樣本:")
            print(f"{'='*60}")
            print(df[['timestamp', 'mid_price', 'spread', 'last_price', 'total_volume']].head(10))
            print(f"\n特徵形狀: {df.iloc[0]['features'].shape}")
            print(f"特徵值 (前5維): {df.iloc[0]['features'][:5]}")
    else:
        print(f"[ERROR] 測試文件不存在: {test_file}")
        print("請將測試數據放在 data/raw/ 目錄下")
