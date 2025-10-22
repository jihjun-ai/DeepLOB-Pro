# -*- coding: utf-8 -*-
"""
extract_tw_stock_data_v6.py - V6 雙階段資料流水線（階段2：讀取預處理數據）
=============================================================================
【更新日期】2025-10-21
【版本說明】v6.0 - 讀取預處理 NPZ，生成訓練數據

改進重點：
  1. 從預處理的 NPZ 檔案讀取數據（而非原始 TXT）
  2. 跳過清洗、聚合步驟（已在階段1完成）
  3. 支援快速調整 Triple-Barrier 參數
  4. 保持與 V5 相同的輸出格式

使用方式：
  # 先執行階段1預處理
  python scripts/batch_preprocess.bat

  # 再執行階段2生成訓練數據
  python scripts/extract_tw_stock_data_v6.py \
      --preprocessed-dir ./data/preprocessed_v5 \
      --output-dir ./data/processed_v6 \
      --config configs/config_pro_v5_ml_optimal.yaml

輸出：
  - ./data/processed_v6/npz/stock_embedding_train.npz
  - ./data/processed_v6/npz/stock_embedding_val.npz
  - ./data/processed_v6/npz/stock_embedding_test.npz
  - ./data/processed_v6/npz/normalization_meta.json

版本：v6.0
更新：2025-10-21
"""

import os
import json
import argparse
import glob
import logging
import warnings
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加項目根目錄
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.yaml_manager import YAMLManager

# 設定版本號
VERSION = "6.0.0"

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 固定常數
SEQ_LEN = 100

# 全域統計
global_stats = {
    "loaded_npz_files": 0,
    "symbols_passed_filter": 0,
    "symbols_filtered_out": 0,
    "data_quality_errors": 0,  # 新增：數據質量錯誤計數
    "valid_windows": 0,
    "tb_success": 0
}


# ============================================================
# 繼承自 V5 的波動率與標籤函數
# ============================================================

def ewma_vol(close: pd.Series, halflife: int = 60) -> pd.Series:
    """
    EWMA 波動率估計（修復：無未來資訊洩漏）

    修改說明：
    - 移除 bfill()（使用未來資訊）
    - 改用前向填充或固定初始值

    ⚠️ 注意：此函數假設輸入數據已清洗（無 0 值、無 NaN）
    如果輸入有異常值，會拋出異常，而不是靜默修補。
    """
    # 數據質量檢查（不修補，只報錯）
    if (close == 0).any():
        zero_count = (close == 0).sum()
        zero_indices = np.where(close == 0)[0][:5]  # 前 5 個
        raise ValueError(
            f"❌ 數據質量錯誤：發現 {zero_count} 個 mids=0 的點\n"
            f"   前 5 個索引: {zero_indices.tolist()}\n"
            f"   → 這應該在預處理階段 (preprocess_single_day.py) 就被移除！"
        )

    if close.isna().any():
        nan_count = close.isna().sum()
        raise ValueError(
            f"❌ 數據質量錯誤：發現 {nan_count} 個 NaN 價格\n"
            f"   → 這應該在預處理階段就被移除！"
        )

    if (close < 0).any():
        neg_count = (close < 0).sum()
        raise ValueError(
            f"❌ 數據質量錯誤：發現 {neg_count} 個負價格\n"
            f"   → 這應該在預處理階段就被移除！"
        )

    ret = np.log(close).diff()
    var = ret.ewm(halflife=halflife, adjust=False).var()
    vol = np.sqrt(var)

    # 修復未來資訊洩漏：前 halflife 個點用保守估計
    if vol.isna().any():
        # 使用序列開頭非 NaN 值的平均作為初始值
        valid_vols = vol.dropna()
        if len(valid_vols) > 0:
            initial_vol = valid_vols.iloc[:min(100, len(valid_vols))].mean()
        else:
            initial_vol = 0.01  # 默認值
        vol = vol.fillna(initial_vol)

    return vol


def tb_labels(close: pd.Series,
              vol: pd.Series,
              pt_mult: float = 2.0,
              sl_mult: float = 2.0,
              max_holding: int = 200,
              min_return: float = 0.0001,
              day_end_idx: Optional[int] = None) -> pd.DataFrame:
    """Triple-Barrier 標籤生成"""
    try:
        n = len(close)
        results = []

        for i in range(n - 1):
            entry_price = close.iloc[i]
            entry_vol = vol.iloc[i]

            up_barrier = entry_price * (1 + pt_mult * entry_vol)
            dn_barrier = entry_price * (1 - sl_mult * entry_vol)

            if day_end_idx is not None:
                end_idx = min(i + max_holding, day_end_idx + 1, n)
            else:
                end_idx = min(i + max_holding, n)

            triggered = False
            trigger_idx = end_idx - 1
            trigger_why = 'time'

            for j in range(i + 1, end_idx):
                future_price = close.iloc[j]

                if future_price >= up_barrier:
                    trigger_idx = j
                    trigger_why = 'up'
                    triggered = True
                    break

                if future_price <= dn_barrier:
                    trigger_idx = j
                    trigger_why = 'down'
                    triggered = True
                    break

            exit_price = close.iloc[trigger_idx]
            ret = (exit_price - entry_price) / entry_price

            # 數據質量檢查（不修補，只報錯）
            if np.isnan(ret) or np.isinf(ret):
                raise ValueError(
                    f"❌ Triple-Barrier 計算錯誤：收益率為 NaN 或 inf\n"
                    f"   entry_price={entry_price:.4f}, exit_price={exit_price:.4f}\n"
                    f"   entry_idx={i}, trigger_idx={trigger_idx}\n"
                    f"   → 這應該在預處理階段就被避免！"
                )

            if trigger_why == 'time':
                if np.abs(ret) < min_return:
                    label = 0
                else:
                    label = int(np.sign(ret))
            else:
                label = int(np.sign(ret))

            results.append({
                'ret': ret,
                'y': label,
                'tt': trigger_idx - i,
                'why': trigger_why,
                'up_p': up_barrier,
                'dn_p': dn_barrier
            })

        if n > 0:
            results.append({
                'ret': 0.0,
                'y': 0,
                'tt': 0,
                'why': 'time',
                'up_p': close.iloc[-1],
                'dn_p': close.iloc[-1]
            })

        out = pd.DataFrame(results, index=close.index)
        global_stats["tb_success"] += 1

        return out.dropna()

    except Exception as e:
        logging.error(f"Triple-Barrier 失敗: {e}")
        raise


def make_sample_weight(ret: pd.Series,
                      tt: pd.Series,
                      y: pd.Series,
                      tau: float = 100.0,
                      scale: float = 10.0,
                      balance: bool = True,
                      use_log_scale: bool = True) -> pd.Series:
    """樣本權重計算"""
    from sklearn.utils.class_weight import compute_class_weight

    ret_array = np.array(ret.values, dtype=np.float64)
    tt_array = np.array(tt.values, dtype=np.float64)

    if use_log_scale:
        ret_weight = np.log1p(np.abs(ret_array) * 1000) * scale
        ret_weight = np.maximum(ret_weight, 0.1)
    else:
        ret_weight = np.abs(ret_array) * scale

    time_decay = np.exp(-tt_array / float(tau))
    # C. 時間衰減標準化：避免晚期樣本整體被壓低
    time_decay = time_decay / (time_decay.mean() + 1e-12)

    base = ret_weight * time_decay
    base = np.clip(base, 0.05, None)

    if balance:
        classes = np.array(sorted(y.unique()))
        y_array = np.array(y.values, dtype=np.int64)
        cls_w = compute_class_weight('balanced', classes=classes, y=y_array)

        cls_w = np.clip(cls_w, 0.5, 3.0)
        cls_w = cls_w / cls_w.mean()

        w_map = dict(zip(classes, cls_w))
        cw = np.array(y.map(w_map).values, dtype=np.float64)
        w = base * cw
    else:
        w = base

    w = w / np.mean(w)
    w = np.clip(w, 0.1, 5.0)

    # 裁剪後重新歸一化，確保均值=1.0
    w = w / np.mean(w)

    return pd.Series(w, index=y.index)


def zscore_fit(X: np.ndarray, method: str = 'global', window: int = 100, min_periods: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算 Z-Score 參數（支持多種標準化方法）

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        輸入特徵
    method : str
        標準化方法：
        - 'global': 全局統計量（舊方法，可能導致分布漂移）
        - 'rolling_zscore': 滾動窗口 Z-Score（推薦，適應市場變化）
    window : int
        滾動窗口大小（僅 rolling_zscore 使用）
    min_periods : int
        最小有效樣本數（僅 rolling_zscore 使用）

    Returns:
    --------
    mu, sd : 均值和標準差（global 方法）或 None, None（rolling 方法）

    Notes:
    ------
    - 全局方法：使用整個訓練集統計量（快速但可能漂移）
    - 滾動方法：使用最近 N 個樣本統計量（穩定但稍慢）
    """
    if method == 'global':
        # 原有的全局標準化
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-8, 1.0, sd)

        if np.any(np.abs(mu) > 1e6):
            logging.warning(f"偵測到異常大的均值: max|μ|={np.max(np.abs(mu)):.2f}")

        return mu, sd

    elif method == 'rolling_zscore':
        # 滾動窗口標準化不需要預先計算統計量
        # 統計量在 zscore_apply 中實時計算
        logging.info(f"使用滾動窗口標準化: window={window}, min_periods={min_periods}")
        return None, None

    else:
        raise ValueError(f"未知的標準化方法: {method}")


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                 method: str = 'global', window: int = 100, min_periods: int = 20) -> np.ndarray:
    """
    應用 Z-Score 正規化

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        輸入特徵
    mu : np.ndarray or None
        均值（global 方法）或 None（rolling 方法）
    sd : np.ndarray or None
        標準差（global 方法）或 None（rolling 方法）
    method : str
        標準化方法
    window : int
        滾動窗口大小
    min_periods : int
        最小有效樣本數

    Returns:
    --------
    normalized : np.ndarray
        標準化後的特徵
    """
    if method == 'global':
        # 全局標準化
        return (X - mu.reshape(1, -1)) / sd.reshape(1, -1)

    elif method == 'rolling_zscore':
        # 滾動窗口標準化
        n_samples, n_features = X.shape
        normalized = np.zeros_like(X)

        for i in range(n_samples):
            # 確定滾動窗口範圍
            start_idx = max(0, i - window + 1)
            window_data = X[start_idx:i+1, :]

            # 計算當前窗口的統計量
            if len(window_data) >= min_periods:
                mu_rolling = window_data.mean(axis=0)
                sd_rolling = window_data.std(axis=0, ddof=0)
                sd_rolling = np.where(sd_rolling < 1e-8, 1.0, sd_rolling)
            else:
                # warm-up 期：使用 expanding window
                mu_rolling = window_data.mean(axis=0)
                sd_rolling = window_data.std(axis=0, ddof=0)
                sd_rolling = np.where(sd_rolling < 1e-8, 1.0, sd_rolling)

            # 標準化當前樣本
            normalized[i, :] = (X[i, :] - mu_rolling) / sd_rolling

        return normalized

    else:
        raise ValueError(f"未知的標準化方法: {method}")


# ============================================================
# V6 核心：載入預處理數據
# ============================================================

def validate_preprocessed_data(features: np.ndarray, mids: np.ndarray, meta: Dict, npz_path: str) -> bool:
    """
    驗證預處理數據質量（不修補，只檢查）

    Returns:
        True if valid, False if invalid (會記錄警告)
    """
    symbol = meta.get('symbol', 'unknown')
    date = meta.get('date', 'unknown')

    # 檢查 1: mids 不能為 0
    if (mids == 0).any():
        zero_count = (mids == 0).sum()
        zero_pct = zero_count / len(mids) * 100
        logging.error(
            f"❌ 數據質量錯誤 [{symbol} @ {date}]\n"
            f"   發現 {zero_count} 個 mids=0 ({zero_pct:.1f}%)\n"
            f"   檔案: {npz_path}\n"
            f"   → 預處理階段 (preprocess_single_day.py) 應該移除這些點！"
        )
        return False

    # 檢查 2: mids 不能為 NaN
    if np.isnan(mids).any():
        nan_count = np.isnan(mids).sum()
        nan_pct = nan_count / len(mids) * 100
        logging.error(
            f"❌ 數據質量錯誤 [{symbol} @ {date}]\n"
            f"   發現 {nan_count} 個 NaN ({nan_pct:.1f}%)\n"
            f"   檔案: {npz_path}"
        )
        return False

    # 檢查 3: mids 不能為負數
    if (mids < 0).any():
        neg_count = (mids < 0).sum()
        logging.error(
            f"❌ 數據質量錯誤 [{symbol} @ {date}]\n"
            f"   發現 {neg_count} 個負價格\n"
            f"   檔案: {npz_path}"
        )
        return False

    # 檢查 4: features 不能為 NaN
    if np.isnan(features).any():
        nan_count = np.isnan(features).sum()
        logging.error(
            f"❌ 數據質量錯誤 [{symbol} @ {date}]\n"
            f"   發現 {nan_count} 個 NaN 特徵值\n"
            f"   檔案: {npz_path}"
        )
        return False

    # 檢查 5: 形狀匹配
    if features.shape[0] != len(mids):
        logging.error(
            f"❌ 數據質量錯誤 [{symbol} @ {date}]\n"
            f"   features 和 mids 長度不匹配: {features.shape[0]} vs {len(mids)}\n"
            f"   檔案: {npz_path}"
        )
        return False

    return True


def load_preprocessed_npz(npz_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """
    載入預處理後的 NPZ 檔案（含數據質量驗證）

    Returns:
        (features, mids, bucket_mask, metadata) or None if filtered/invalid
    """
    try:
        data = np.load(npz_path, allow_pickle=True)

        features = data['features']
        mids = data['mids']
        bucket_mask = data.get('bucket_mask', np.zeros(len(mids), dtype=np.int32))  # A.2: 載入 mask
        meta = json.loads(str(data['metadata']))

        global_stats["loaded_npz_files"] += 1

        # 檢查過濾狀態
        if not meta['pass_filter']:
            global_stats["symbols_filtered_out"] += 1
            return None

        # 數據質量驗證（新增）
        if not validate_preprocessed_data(features, mids, meta, npz_path):
            logging.warning(f"⚠️ 跳過有問題的數據: {npz_path}")
            global_stats["data_quality_errors"] += 1
            global_stats["symbols_filtered_out"] += 1
            return None

        global_stats["symbols_passed_filter"] += 1

        return features, mids, bucket_mask, meta

    except Exception as e:
        logging.warning(f"無法載入 {npz_path}: {e}")
        return None


def load_all_preprocessed_data(preprocessed_dir: str) -> List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """
    載入所有預處理數據

    Returns:
        List[(date, symbol, features, mids, bucket_mask, metadata)]
    """
    daily_dir = os.path.join(preprocessed_dir, "daily")

    if not os.path.exists(daily_dir):
        logging.error(f"預處理目錄不存在: {daily_dir}")
        return []

    all_data = []

    # 掃描所有 NPZ 檔案
    for npz_file in sorted(glob.glob(os.path.join(daily_dir, "*", "*.npz"))):
        result = load_preprocessed_npz(npz_file)

        if result is None:
            continue

        features, mids, bucket_mask, meta = result  # A.2: 包含 bucket_mask
        date = meta['date']
        symbol = meta['symbol']

        all_data.append((date, symbol, features, mids, bucket_mask, meta))

    # G.1: 空數據報告增強
    if len(all_data) == 0:
        logging.error(f"\n❌ 沒有可用的預處理數據！")
        logging.error(f"可能原因:")
        logging.error(f"  1. preprocessed_dir 路徑錯誤: {preprocessed_dir}")
        logging.error(f"  2. 所有股票被過濾（pass_filter=false）")
        logging.error(f"  3. 數據質量錯誤（mids=0, NaN 等）")
        logging.error(f"  4. 預處理尚未執行")
        logging.error(f"\n建議檢查:")
        logging.error(f"  - 確認預處理目錄存在: {os.path.exists(daily_dir)}")
        logging.error(f"  - 檢查 NPZ 檔案數量: {len(list(glob.glob(os.path.join(daily_dir, '*', '*.npz'))))}")
        logging.error(f"  - 查看預處理日誌或 summary.json")
    else:
        logging.info(f"載入了 {len(all_data)} 個 symbol-day 組合（通過過濾）")
        logging.info(f"過濾掉: {global_stats['symbols_filtered_out']} 個")
        if global_stats['data_quality_errors'] > 0:
            logging.warning(f"數據質量錯誤: {global_stats['data_quality_errors']} 個（已跳過）")

    return all_data


# ============================================================
# V6 滑窗流程（簡化版，數據已清洗）
# ============================================================

def sliding_windows_v6(
    preprocessed_data: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, Dict]],
    out_dir: str,
    config: Dict[str, Any]
):
    """
    V6 滑窗流程（基於預處理數據 + 品質過濾）

    差異：
    - 輸入已是乾淨的 features 和 mids（無需解析、聚合）
    - 包含 bucket_mask（用於品質過濾）
    - 直接進行 Z-Score → 波動率 → TB → 滑窗（含品質檢查）
    """
    global global_stats

    if not preprocessed_data:
        logging.warning("沒有資料可供產生 .npz 檔案")
        return

    respect_day_boundary = config.get('respect_day_boundary', True)
    ffill_quality_threshold = config.get('ffill_quality_threshold', 0.5)  # A.2: 新增品質閾值

    logging.info(f"\n{'='*60}")
    logging.info(f"V6 滑窗流程開始，共 {len(preprocessed_data)} 個 symbol-day 組合")
    logging.info(f"日界線保護: {'啟用' if respect_day_boundary else '禁用'}")
    logging.info(f"滑窗品質過濾: ffill 占比 > {ffill_quality_threshold*100:.0f}% 將被跳過")
    logging.info(f"{'='*60}")

    # 步驟 1: 重組為股票序列
    stock_data = defaultdict(lambda: {'day_data': []})

    for date, sym, features, mids, bucket_mask, meta in preprocessed_data:
        stock_data[sym]['day_data'].append((date, features, mids, bucket_mask))

    logging.info(f"共 {len(stock_data)} 個股票有資料")

    # 步驟 2: 排序並過濾短序列
    stock_sequences = []

    for sym, data in stock_data.items():
        day_data_sorted = sorted(data['day_data'], key=lambda x: x[0])
        total_points = sum(features.shape[0] for _, features, _, _ in day_data_sorted)

        stock_sequences.append((sym, total_points, day_data_sorted))

    stock_sequences_sorted = sorted(stock_sequences, key=lambda x: x[1], reverse=True)

    MIN_POINTS = SEQ_LEN + 50

    valid_stocks = [s for s in stock_sequences if s[1] >= MIN_POINTS]
    filtered_stocks = len(stock_sequences) - len(valid_stocks)

    if filtered_stocks > 0:
        logging.warning(f"過濾 {filtered_stocks} 檔序列太短的股票（< {MIN_POINTS} 個點）")

    if not valid_stocks:
        logging.error("沒有股票有足夠的時間點！")
        return

    logging.info(f"有效股票: {len(valid_stocks)} 檔")

    # 步驟 3: 按日期區間切分 70/15/15（時序數據最佳實踐）
    # 收集所有日期
    all_dates = set()
    for sym, n_points, day_data_sorted in valid_stocks:
        for date, features, mids, bucket_mask in day_data_sorted:
            all_dates.add(date)

    all_dates_sorted = sorted(list(all_dates))
    n_dates = len(all_dates_sorted)

    if n_dates < 3:
        logging.error(f"日期數量不足（僅 {n_dates} 天），無法切分 train/val/test")
        return

    # 按時間順序切分日期
    train_ratio = config['split']['train_ratio']
    val_ratio = config['split']['val_ratio']

    train_end_idx = max(1, int(n_dates * train_ratio))
    val_end_idx = min(n_dates - 1, train_end_idx + max(1, int(n_dates * val_ratio)))

    train_dates = set(all_dates_sorted[:train_end_idx])
    val_dates = set(all_dates_sorted[train_end_idx:val_end_idx])
    test_dates = set(all_dates_sorted[val_end_idx:])

    logging.info("\n資料切分（按日期區間）:")
    logging.info(f"  總日期數: {n_dates} 天")
    logging.info(f"  Train: {all_dates_sorted[0]} ~ {all_dates_sorted[train_end_idx-1]} ({len(train_dates)} 天)")
    logging.info(f"  Val:   {all_dates_sorted[train_end_idx]} ~ {all_dates_sorted[val_end_idx-1]} ({len(val_dates)} 天)")
    logging.info(f"  Test:  {all_dates_sorted[val_end_idx]} ~ {all_dates_sorted[-1]} ({len(test_dates)} 天)")

    # 將股票-日期組合分配到對應集合
    def assign_to_split(stock_sequences, split_dates):
        """將符合日期範圍的股票數據分配到對應集合"""
        split_stocks = []
        for sym, n_points, day_data_sorted in stock_sequences:
            # 過濾該股票中屬於 split_dates 的日期
            filtered_day_data = [(date, features, mids, bucket_mask)
                                for date, features, mids, bucket_mask in day_data_sorted
                                if date in split_dates]

            if filtered_day_data:
                # 重新計算該股票在此集合中的總點數
                total_points = sum(features.shape[0] for _, features, _, _ in filtered_day_data)
                split_stocks.append((sym, total_points, filtered_day_data))

        return split_stocks

    train_stocks = assign_to_split(valid_stocks, train_dates)
    val_stocks = assign_to_split(valid_stocks, val_dates)
    test_stocks = assign_to_split(valid_stocks, test_dates)

    logging.info("\n各集合股票覆蓋:")
    logging.info(f"  Train: {len(train_stocks)} 檔股票（在訓練期間有數據）")
    logging.info(f"  Val:   {len(val_stocks)} 檔股票（在驗證期間有數據）")
    logging.info(f"  Test:  {len(test_stocks)} 檔股票（在測試期間有數據）")

    if len(train_stocks) == 0 or len(val_stocks) == 0 or len(test_stocks) == 0:
        logging.error("❌ 日期切分後某個集合為空！")
        logging.error(f"   Train: {len(train_stocks)}, Val: {len(val_stocks)}, Test: {len(test_stocks)}")
        logging.error("   可能原因:")
        logging.error(f"   1. 數據時間跨度太短（僅 {n_dates} 天）")
        logging.error("   2. 部分股票僅在特定時期有數據")
        logging.error("   建議: 增加數據收集時間範圍或調整切分比例")
        return

    splits = {
        'train': train_stocks,
        'val': val_stocks,
        'test': test_stocks
    }

    # 步驟 4: 計算 Z-Score 參數（訓練集）
    logging.info(f"\n{'='*60}")
    logging.info("計算 Z-Score 參數（基於訓練集）")
    logging.info(f"{'='*60}")

    # 讀取標準化配置
    norm_config = config.get('normalization', {})
    norm_method = norm_config.get('method', 'global')
    norm_window = norm_config.get('window', 100)
    norm_min_periods = norm_config.get('min_periods', 20)

    logging.info(f"標準化方法: {norm_method}")
    if norm_method == 'rolling_zscore':
        logging.info(f"  - 滾動窗口: {norm_window}")
        logging.info(f"  - 最小樣本: {norm_min_periods}")

    train_X_list = []
    for sym, n_points, day_data_sorted in train_stocks:
        for date, features, mids, bucket_mask in day_data_sorted:
            train_X_list.append(features)

    Xtr = np.concatenate(train_X_list, axis=0) if train_X_list else np.zeros((0, 20))

    if Xtr.size == 0:
        mu = np.zeros((20,), dtype=np.float64)
        sd = np.ones((20,), dtype=np.float64)
        logging.warning("訓練集為空，使用預設參數")
    else:
        mu, sd = zscore_fit(Xtr, method=norm_method, window=norm_window, min_periods=norm_min_periods)
        logging.info(f"訓練集大小: {Xtr.shape[0]:,} 個時間點")

    # 步驟 5: 產生滑窗樣本
    def build_split_v6(split_name):
        """V6 版本：直接從 features/mids 生成樣本"""
        stock_list = splits[split_name]

        logging.info(f"\n{'='*60}")
        logging.info(f"處理 {split_name.upper()} 集，共 {len(stock_list)} 檔股票")
        logging.info(f"{'='*60}")

        X_windows = []
        y_labels = []
        weights = []
        stock_ids = []

        total_windows = 0
        total_days = 0
        tb_stats = {"up": 0, "down": 0, "time": 0}

        for sym, n_points, day_data_sorted in stock_list:
            stock_windows = 0

            for date, features, mids, bucket_mask in day_data_sorted:  # A.2: 加入 bucket_mask
                total_days += 1

                # 1. Z-Score 正規化
                Xn = zscore_apply(features, mu, sd, method=norm_method, window=norm_window, min_periods=norm_min_periods)

                # 2. 波動率估計
                close = pd.Series(mids, name='close')
                vol_method = config['volatility']['method']

                try:
                    if vol_method == 'ewma':
                        vol = ewma_vol(close, halflife=config['volatility']['halflife'])
                    else:
                        raise ValueError(f"不支援的方法: {vol_method}")
                except Exception as e:
                    logging.warning(f"  {sym} @ {date}: 波動率計算失敗 ({e})")
                    continue

                # 3. Triple-Barrier 標籤（優化版：降採樣計算）
                tb_cfg = config['triple_barrier']
                # 強制開啟日界保護，避免跨日洩漏
                day_end_idx = len(close) - 1  # 修改：始終啟用

                # ⚡ 優化：每 10 個點計算一次 TB（避免重複計算）
                # 原因：滑窗會重疊，不需要對每個點都算 TB
                tb_stride = config.get('tb_stride', 10)  # 可配置，預設 10

                try:
                    # 只對採樣點計算 TB
                    sample_indices = list(range(0, len(close), tb_stride))
                    if sample_indices[-1] != len(close) - 1:
                        sample_indices.append(len(close) - 1)

                    close_sampled = close.iloc[sample_indices]
                    vol_sampled = vol.iloc[sample_indices]

                    tb_df_sampled = tb_labels(
                        close=close_sampled,
                        vol=vol_sampled,
                        pt_mult=tb_cfg['pt_multiplier'],
                        sl_mult=tb_cfg['sl_multiplier'],
                        max_holding=tb_cfg['max_holding'],
                        min_return=tb_cfg['min_return'],
                        day_end_idx=len(close_sampled) - 1
                    )

                    # 重新索引到原始時間軸（前向填充）
                    tb_df = pd.DataFrame(index=close.index)
                    for col in tb_df_sampled.columns:
                        tb_df[col] = tb_df_sampled[col].reindex(close.index, method='ffill')

                except Exception as e:
                    logging.warning(f"  {sym} @ {date}: TB 失敗 ({e})")
                    continue

                # 4. 轉換標籤
                y_tb = tb_df["y"].map({-1: 0, 0: 1, 1: 2})

                # E.1: 標籤邊界檢查（Hard Assertion）
                if y_tb.isna().any():
                    nan_count = y_tb.isna().sum()
                    raise ValueError(
                        f"❌ 標籤映射失敗 [{sym} @ {date}]\n"
                        f"   發現 {nan_count} 個 NaN 標籤\n"
                        f"   原始標籤可能不在 {{-1, 0, 1}} 範圍內\n"
                        f"   → 請檢查 Triple-Barrier 輸出"
                    )

                if not y_tb.isin([0, 1, 2]).all():
                    invalid = y_tb[~y_tb.isin([0, 1, 2])].unique()
                    raise ValueError(
                        f"❌ 標籤值異常 [{sym} @ {date}]\n"
                        f"   發現非 {{0,1,2}} 的值: {invalid}\n"
                        f"   → 標籤映射後應只包含 0/1/2"
                    )

                # 5. 樣本權重
                if config['sample_weights']['enabled']:
                    w = make_sample_weight(
                        ret=tb_df["ret"],
                        tt=tb_df["tt"],
                        y=y_tb,
                        tau=config['sample_weights']['tau'],
                        scale=config['sample_weights']['return_scaling'],
                        balance=config['sample_weights']['balance_classes'],
                        use_log_scale=config['sample_weights'].get('use_log_scale', True)
                    )
                else:
                    w = pd.Series(np.ones(len(y_tb)), index=y_tb.index)

                # E.2: 權重邊界檢查（Hard Assertion）
                if not np.isfinite(w).all():
                    inf_count = np.sum(~np.isfinite(w))
                    raise ValueError(
                        f"❌ 權重包含 NaN/inf [{sym} @ {date}]\n"
                        f"   異常數量: {inf_count}\n"
                        f"   → 請檢查 make_sample_weight() 輸出"
                    )

                w_mean = w.mean()
                # 修正：裁剪後重新歸一化，均值應接近 1.0
                # 注意：單一股票單日樣本少時，可能因極端分布略微偏離 1.0
                if not (0.85 < w_mean < 1.15):
                    logging.warning(
                        f"⚠️ 權重均值異常 [{sym} @ {date}]\n"
                        f"   均值: {w_mean:.3f} (預期 ≈ 1.0)\n"
                        f"   樣本數: {len(w)}\n"
                        f"   → 單股單日樣本少可能導致偏離，訓練時會在 batch 層級平衡"
                    )

                # 6. 統計觸發原因
                for reason in tb_df["why"].value_counts().items():
                    if reason[0] in tb_stats:
                        tb_stats[reason[0]] += reason[1]

                # 7. 滑窗生成
                T = Xn.shape[0]
                max_t = min(T, len(y_tb))

                if max_t < SEQ_LEN:
                    continue

                day_windows = 0

                for t in range(SEQ_LEN - 1, max_t):
                    window_start = t - SEQ_LEN + 1

                    if respect_day_boundary and window_start < 0:
                        continue

                    window = Xn[window_start:t + 1, :]

                    if window.shape[0] != SEQ_LEN:
                        continue

                    # A.2: 滑窗品質過濾（檢查 ffill 占比）
                    window_mask = bucket_mask[window_start:t + 1]
                    ffill_ratio = (window_mask == 1).sum() / len(window_mask)

                    if ffill_ratio > ffill_quality_threshold:
                        # 跳過品質不佳的窗口（過多 ffill）
                        continue

                    label = int(y_tb.iloc[t])
                    weight = float(w.iloc[t])

                    if label not in [0, 1, 2]:
                        continue

                    X_windows.append(window.astype(np.float32))
                    y_labels.append(label)
                    weights.append(weight)
                    stock_ids.append(sym)
                    day_windows += 1

                stock_windows += day_windows

            if stock_windows > 0:
                logging.info(f"  {sym}: {stock_windows:,} 個樣本")

        logging.info(f"\n{split_name.upper()} 總計: {len(y_labels):,} 個樣本")
        logging.info(f"觸發原因: {tb_stats}")

        global_stats["valid_windows"] += len(y_labels)

        if X_windows:
            X_array = np.stack(X_windows, axis=0)
            y_array = np.array(y_labels, dtype=np.int64)
            w_array = np.array(weights, dtype=np.float32)
            sid_array = np.array(stock_ids, dtype=object)

            label_dist = np.bincount(y_array, minlength=3)
            label_pct = label_dist / label_dist.sum() * 100 if label_dist.sum() > 0 else np.zeros(3)

            logging.info(f"  形狀: X={X_array.shape}, y={y_array.shape}")
            logging.info(f"  標籤分布: Down={label_dist[0]} ({label_pct[0]:.1f}%), "
                        f"Neutral={label_dist[1]} ({label_pct[1]:.1f}%), "
                        f"Up={label_dist[2]} ({label_pct[2]:.1f}%)")

            # E.3: Neutral 比例檢查（數據質量關鍵指標）
            neutral_pct = label_pct[1]
            if neutral_pct < 15.0:
                logging.warning(
                    f"\n⚠️ {split_name.upper()} 集 Neutral 比例過低！\n"
                    f"   當前: {neutral_pct:.1f}%，目標: 20-45%\n"
                    f"   → 建議調整 Triple-Barrier 參數:\n"
                    f"      - 增加 min_return 閾值（當前: {config['triple_barrier']['min_return']}）\n"
                    f"      - 或放寬過濾條件（使用 P25 而非 P50）\n"
                    f"   風險: 模型可能學不到「不交易」的情況"
                )
            elif neutral_pct > 60.0:
                logging.warning(
                    f"\n⚠️ {split_name.upper()} 集 Neutral 比例過高！\n"
                    f"   當前: {neutral_pct:.1f}%，目標: 20-45%\n"
                    f"   → 建議調整 Triple-Barrier 參數:\n"
                    f"      - 減少 min_return 閾值（當前: {config['triple_barrier']['min_return']}）\n"
                    f"      - 或減少 max_holding 時間\n"
                    f"   風險: 模型可能學習到過多「不動作」"
                )
            else:
                logging.info(f"  ✅ Neutral 比例健康: {neutral_pct:.1f}% (目標: 20-45%)")

            return X_array, y_array, w_array, sid_array
        else:
            logging.warning(f"{split_name} 集沒有樣本！")
            return (
                np.zeros((0, SEQ_LEN, 20), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.array([], dtype=object)
            )

    # 步驟 6: 保存 NPZ
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for split in ["train", "val", "test"]:
        X, y, w, sid = build_split_v6(split)

        npz_path = os.path.join(out_dir, f"stock_embedding_{split}.npz")
        np.savez_compressed(npz_path, X=X, y=y, weights=w, stock_ids=sid)

        logging.info(f"\n✅ 已保存: {npz_path}")

        results[split] = {
            "samples": len(y),
            "label_dist": np.bincount(y, minlength=3).tolist() if len(y) > 0 else [0, 0, 0],
            "weight_stats": {
                "mean": float(w.mean()) if len(w) > 0 else 0.0,
                "std": float(w.std()) if len(w) > 0 else 0.0,
                "max": float(w.max()) if len(w) > 0 else 0.0
            }
        }

    # 步驟 7: Metadata
    meta = {
        "format": "deeplob_v6",
        "version": VERSION,
        "creation_date": datetime.now().isoformat(),
        "seq_len": SEQ_LEN,
        "feature_dim": 20,

        "volatility": {
            "method": config['volatility']['method'],
            "halflife": config['volatility'].get('halflife', 60)
        },

        "triple_barrier": config['triple_barrier'],

        "sample_weights": {
            "enabled": config['sample_weights']['enabled'],
            "tau": config['sample_weights']['tau'],
            "return_scaling": config['sample_weights']['return_scaling'],
            "balance_classes": config['sample_weights']['balance_classes'],
            "use_log_scale": config['sample_weights'].get('use_log_scale', False)
        },

        "normalization": {
            "method": norm_method,
            "window": norm_window if norm_method == 'rolling_zscore' else None,
            "min_periods": norm_min_periods if norm_method == 'rolling_zscore' else None,
            "computed_on": "train_set" if norm_method == 'global' else "rolling_window",
            "feature_means": mu.tolist() if mu is not None else None,
            "feature_stds": sd.tolist() if sd is not None else None,
            "note": "滾動窗口標準化不保存全局統計量" if norm_method == 'rolling_zscore' else None
        },

        "data_quality": global_stats,

        "data_split": {
            "method": "by_date_range",  # 改為按日期區間切分
            "train_date_range": f"{all_dates_sorted[0]} ~ {all_dates_sorted[train_end_idx-1]}" if train_end_idx > 0 else "N/A",
            "val_date_range": f"{all_dates_sorted[train_end_idx]} ~ {all_dates_sorted[val_end_idx-1]}" if val_end_idx > train_end_idx else "N/A",
            "test_date_range": f"{all_dates_sorted[val_end_idx]} ~ {all_dates_sorted[-1]}" if val_end_idx < n_dates else "N/A",
            "train_dates": len(train_dates),
            "val_dates": len(val_dates),
            "test_dates": len(test_dates),
            "train_stocks": len(train_stocks),
            "val_stocks": len(val_stocks),
            "test_stocks": len(test_stocks),
            "total_stocks": len(valid_stocks),
            "total_dates": n_dates,
            "filtered_stocks": filtered_stocks,
            "results": results
        },

        "note": "V6: 基於預處理 NPZ，動態過濾閾值。Labels: {0:下跌, 1:持平, 2:上漲}",
    }

    meta_path = os.path.join(out_dir, "normalization_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ Metadata 已保存: {meta_path}")
    logging.info(f"\n{'='*60}")
    logging.info("V6 訓練數據生成完成！")
    logging.info(f"{'='*60}")


# ============================================================
# 主程式
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("extract_tw_stock_data_v6", description="V6 雙階段資料流水線（階段2）")
    p.add_argument("--preprocessed-dir", default="./data/preprocessed_v5", help="預處理結果目錄")
    p.add_argument("--output-dir", default="./data/processed_v6", help="輸出目錄")
    p.add_argument("--config", default="./configs/config_pro_v5_ml_optimal.yaml", help="配置文件")
    return p.parse_args()


def main():
    try:
        args = parse_args()

        # 載入配置
        if not os.path.exists(args.config):
            logging.error(f"配置文件不存在: {args.config}")
            return 1

        yaml_manager = YAMLManager(args.config)
        config = yaml_manager.as_dict()

        # 驗證目錄
        if not os.path.exists(args.preprocessed_dir):
            logging.error(f"預處理目錄不存在: {args.preprocessed_dir}")
            return 1

        os.makedirs(args.output_dir, exist_ok=True)

        logging.info(f"{'='*60}")
        logging.info(f"V6 資料流水線啟動（階段2：讀取預處理數據）")
        logging.info(f"{'='*60}")
        logging.info(f"預處理目錄: {args.preprocessed_dir}")
        logging.info(f"輸出目錄: {args.output_dir}")
        logging.info(f"配置版本: {config['version']}")
        logging.info(f"{'='*60}\n")

        # 載入預處理數據
        preprocessed_data = load_all_preprocessed_data(args.preprocessed_dir)

        if not preprocessed_data:
            logging.error("沒有可用的預處理數據！")
            return 1

        # 生成訓練數據
        sliding_windows_v6(
            preprocessed_data,
            os.path.join(args.output_dir, "npz"),
            config
        )

        logging.info(f"\n{'='*60}")
        logging.info(f"[完成] V6 轉換成功，輸出資料夾: {args.output_dir}")
        logging.info(f"{'='*60}")
        logging.info(f"統計資料:")
        logging.info(f"  載入 NPZ: {global_stats['loaded_npz_files']:,}")
        logging.info(f"  通過過濾: {global_stats['symbols_passed_filter']:,}")
        logging.info(f"  被過濾: {global_stats['symbols_filtered_out']:,}")

        # 數據質量報告（重要！）
        if global_stats['data_quality_errors'] > 0:
            logging.warning(f"  ⚠️ 數據質量錯誤: {global_stats['data_quality_errors']:,} 個檔案")
            logging.warning(f"     → 請檢查上方的錯誤訊息，並重新運行預處理！")
        else:
            logging.info(f"  ✅ 數據質量檢查: 全部通過")

        logging.info(f"  有效窗口: {global_stats['valid_windows']:,}")
        logging.info(f"  TB 成功: {global_stats['tb_success']:,}")
        logging.info(f"{'='*60}\n")

        # 如果有數據質量錯誤，返回警告狀態碼
        if global_stats['data_quality_errors'] > 0:
            logging.warning(f"\n⚠️ 警告：發現 {global_stats['data_quality_errors']} 個數據質量問題")
            logging.warning(f"建議重新運行預處理腳本以修復問題")
            return 2  # 返回 2 表示有警告

        return 0

    except Exception as e:
        logging.error(f"程式執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
