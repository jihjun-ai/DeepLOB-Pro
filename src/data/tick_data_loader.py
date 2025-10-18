"""券商 Tick 報價數據載入器

此模組負責載入券商實時報價數據，並轉換為：
1. LOB 5檔深度數據 (用於 DeepLOB)
2. 分K數據 + 技術指標 (用於 PPO)

數據格式範例:
0||0050||元大台灣50||60.05||66.05||54.05||0||0||0||60.85||861||0||60.85||22||60.8||131||60.75||59||60.7||313||60.65||320||60.9||35||60.95||22||61||142||61.05||130||61.1||59||084459||1||

欄位說明:
- 欄位 10-11: 最佳買價、最佳買量
- 欄位 13-22: 買1~買5 (價格、量)
- 欄位 23-32: 賣1~賣5 (價格、量)
- 欄位 33: 時間戳 (HHMMSS)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TickDataLoader:
    """券商 Tick 數據載入器

    功能:
        1. 解析券商 Tick 報價數據
        2. 提取 LOB 5檔深度 (10維: 5檔買+5檔賣)
        3. 聚合為分K數據
        4. 計算技術指標
    """

    def __init__(self, delimiter='||'):
        """初始化

        Args:
            delimiter: 數據分隔符，默認 '||'
        """
        self.delimiter = delimiter

    def parse_tick_line(self, line: str) -> Dict:
        """解析單筆 Tick 數據

        Args:
            line: 原始數據行

        Returns:
            解析後的字典，包含 LOB 深度和元數據
        """
        fields = line.strip().split(self.delimiter)

        if len(fields) < 34:
            raise ValueError(f"數據欄位不足: {len(fields)} < 34")

        # 提取基本信息
        symbol = fields[1]  # 股票代號
        name = fields[2]    # 股票名稱
        timestamp = fields[33]  # 時間戳 HHMMSS

        # 提取 LOB 5檔深度
        # 買盤: 欄位 13-22 (買1~買5價量)
        bid_prices = [float(fields[i]) for i in range(13, 23, 2)]
        bid_volumes = [float(fields[i]) for i in range(14, 23, 2)]

        # 賣盤: 欄位 23-32 (賣1~賣5價量)
        ask_prices = [float(fields[i]) for i in range(23, 33, 2)]
        ask_volumes = [float(fields[i]) for i in range(24, 33, 2)]

        # 最佳買賣價量
        best_bid_price = float(fields[10])
        best_bid_volume = float(fields[11])
        best_ask_price = float(fields[23])
        best_ask_volume = float(fields[24])

        # 中間價
        mid_price = (best_bid_price + best_ask_price) / 2.0

        return {
            'symbol': symbol,
            'name': name,
            'timestamp': timestamp,
            'mid_price': mid_price,
            'bid_prices': bid_prices,
            'bid_volumes': bid_volumes,
            'ask_prices': ask_prices,
            'ask_volumes': ask_volumes,
            'best_bid_price': best_bid_price,
            'best_bid_volume': best_bid_volume,
            'best_ask_price': best_ask_price,
            'best_ask_volume': best_ask_volume,
        }

    def tick_to_lob_features(self, tick_data: Dict) -> np.ndarray:
        """將 Tick 數據轉換為 LOB 特徵 (FI-2010 格式)

        FI-2010 格式 (40維):
            [ask_price_1, ask_vol_1, bid_price_1, bid_vol_1,
             ask_price_2, ask_vol_2, bid_price_2, bid_vol_2,
             ...,
             ask_price_5, ask_vol_5, bid_price_5, bid_vol_5]

        但我們只有5檔，所以是 20維:
            [ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...,
             ask_price_5, ask_vol_5, bid_price_5, bid_vol_5]

        Args:
            tick_data: parse_tick_line 返回的字典

        Returns:
            (20,) LOB 特徵數組
        """
        features = []

        for i in range(5):
            features.extend([
                tick_data['ask_prices'][i],
                tick_data['ask_volumes'][i],
                tick_data['bid_prices'][i],
                tick_data['bid_volumes'][i]
            ])

        return np.array(features, dtype=np.float32)

    def load_tick_file(
        self,
        file_path: str,
        stock_filter: str = None
    ) -> pd.DataFrame:
        """載入 Tick 數據文件

        Args:
            file_path: Tick 數據文件路徑
            stock_filter: 股票代號過濾 (如 '0050')，None = 全部

        Returns:
            DataFrame，包含解析後的 Tick 數據
        """
        tick_list = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    tick = self.parse_tick_line(line)

                    # 過濾股票
                    if stock_filter and tick['symbol'] != stock_filter:
                        continue

                    tick_list.append(tick)

                except Exception as e:
                    logger.warning(f"解析失敗 (行{line_num}): {e}")
                    continue

        if not tick_list:
            raise ValueError("未載入任何有效數據")

        df = pd.DataFrame(tick_list)

        # 轉換時間戳為 datetime
        df['time'] = pd.to_datetime(df['timestamp'], format='%H%M%S')

        logger.info(f"✓ 載入 {len(df)} 筆 Tick 數據")

        return df

    def resample_to_minute_bars(
        self,
        tick_df: pd.DataFrame,
        freq='1T'
    ) -> pd.DataFrame:
        """將 Tick 數據聚合為分K

        Args:
            tick_df: Tick DataFrame
            freq: 聚合頻率 ('1T'=1分鐘, '5T'=5分鐘)

        Returns:
            分K DataFrame (OHLCV格式)
        """
        tick_df = tick_df.set_index('time')

        # OHLC 聚合
        ohlc = tick_df['mid_price'].resample(freq).ohlc()
        volume = tick_df['best_bid_volume'].resample(freq).sum()

        # 合併
        minute_bars = pd.DataFrame({
            'Open': ohlc['open'],
            'High': ohlc['high'],
            'Low': ohlc['low'],
            'Close': ohlc['close'],
            'Volume': volume
        }).dropna()

        logger.info(f"✓ 聚合為 {len(minute_bars)} 根分K")

        return minute_bars

    def calculate_technical_indicators(
        self,
        ohlcv_df: pd.DataFrame,
        max_features: int = 128
    ) -> np.ndarray:
        """計算技術指標並擴展至指定維度

        Args:
            ohlcv_df: OHLCV DataFrame (包含 Open, High, Low, Close, Volume)
            max_features: 目標特徵維度 (默認 128)

        Returns:
            (num_bars, max_features) 技術指標特徵矩陣
        """
        df = ohlcv_df.copy()

        # 基礎價格特徵
        df['price_change'] = df['Close'].pct_change()
        df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_range'] = (df['Close'] - df['Open']) / df['Open']

        # 移動平均線 (EMA)
        for period in [5, 10, 20, 60]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df[f'EMA_{period}_diff'] = (df['Close'] - df[f'EMA_{period}']) / df['Close']

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # RSI (14期)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 布林通道
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)

        # 成交量指標
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_ma5'] + 1e-8)

        # ATR (平均真實波幅)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        df['ATR_ratio'] = df['ATR'] / df['Close']

        # 選擇特徵欄位（排除原始 OHLCV）
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

        # 提取特徵矩陣
        features = df[feature_cols].values.astype(np.float32)

        # 處理缺失值 (前向填充 → 後向填充 → 0)
        features_df = pd.DataFrame(features)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features = features_df.values

        # Z-score 標準化
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        # 擴展至 max_features 維度
        if features.shape[1] < max_features:
            padded_features = np.zeros((features.shape[0], max_features), dtype=np.float32)
            padded_features[:, :features.shape[1]] = features
            features = padded_features
        elif features.shape[1] > max_features:
            logger.warning(f"特徵數 {features.shape[1]} 超過 max_features {max_features}，截斷至前 {max_features} 維")
            features = features[:, :max_features]

        logger.info(f"✓ 計算技術指標: {features.shape}")

        return features

    def load_and_process_for_training(
        self,
        tick_file_path: str,
        stock_filter: str = None,
        freq: str = '1T',
        max_features: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """完整數據處理流程：Tick → LOB特徵 + 技術指標

        Args:
            tick_file_path: Tick 數據文件路徑
            stock_filter: 股票代號過濾
            freq: K線聚合頻率
            max_features: 技術指標特徵維度

        Returns:
            lob_features: (num_ticks, 20) LOB 特徵矩陣 (用於 DeepLOB)
            indicator_features: (num_bars, max_features) 技術指標矩陣 (用於 PPO)
        """
        # 1. 載入 Tick 數據
        tick_df = self.load_tick_file(tick_file_path, stock_filter)

        # 2. 提取 LOB 特徵 (用於 DeepLOB)
        lob_features_list = []
        for _, tick in tick_df.iterrows():
            lob_feat = self.tick_to_lob_features(tick)
            lob_features_list.append(lob_feat)

        lob_features = np.array(lob_features_list, dtype=np.float32)

        # 3. 聚合為分K
        minute_bars = self.resample_to_minute_bars(tick_df, freq)

        # 4. 計算技術指標 (用於 PPO)
        indicator_features = self.calculate_technical_indicators(minute_bars, max_features)

        logger.info(f"✓ 處理完成: LOB {lob_features.shape}, Indicators {indicator_features.shape}")

        return lob_features, indicator_features


if __name__ == '__main__':
    # 測試代碼
    logging.basicConfig(level=logging.INFO)

    # 測試解析單筆 Tick
    test_line = "0||0050||元大台灣50||60.05||66.05||54.05||0||0||0||60.85||861||0||60.85||22||60.8||131||60.75||59||60.7||313||60.65||320||60.9||35||60.95||22||61||142||61.05||130||61.1||59||084459||1||"

    loader = TickDataLoader()
    tick = loader.parse_tick_line(test_line)

    print(f"股票: {tick['symbol']} {tick['name']}")
    print(f"時間: {tick['timestamp']}")
    print(f"中間價: {tick['mid_price']:.2f}")
    print(f"最佳買價: {tick['best_bid_price']:.2f} ({tick['best_bid_volume']:.0f})")
    print(f"最佳賣價: {tick['best_ask_price']:.2f} ({tick['best_ask_volume']:.0f})")

    # 轉換為 LOB 特徵
    lob_features = loader.tick_to_lob_features(tick)
    print(f"\nLOB 特徵 (20維): {lob_features.shape}")
    print(f"前8維: {lob_features[:8]}")
