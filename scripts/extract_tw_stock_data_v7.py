# -*- coding: utf-8 -*-
"""
extract_tw_stock_data_v7.py - V7 簡化版資料流水線（專注數據組織）
=============================================================================
【更新日期】2025-10-23
【版本說明】v7.0.0-simplified - 預處理已完成所有計算，V7 只做數據組織

核心理念：
  "預處理已完成，V7 只做數據組織"

V7 簡化流程：
  預處理階段（preprocess_single_day.py）:
    ✅ 數據清洗與聚合
    ✅ Z-Score 標準化
    ✅ 標籤計算（Triple-Barrier / Trend）
    ✅ 權重策略計算（11 種）
    ✅ 統計信息記錄

  V7 階段（extract_tw_stock_data_v7.py）:
    ✅ 讀取預處理 NPZ（直接使用 features, labels）
    ✅ 數據選擇（dataset_selection.json 或配置過濾）
    ✅ 滑動窗口生成（100 timesteps）
    ✅ 按股票劃分（train/val/test = 70/15/15）
    ✅ 輸出 NPZ（與 V6 格式兼容）

    ❌ 不重新計算標籤
    ❌ 不重新計算波動率
    ❌ 不重新計算權重
    ❌ 不重新標準化

使用方式：
  # 選項 1: 使用 dataset_selection.json（推薦）
  python scripts/analyze_label_distribution.py       --preprocessed-dir data/preprocessed_v5       --mode smart_recommend       --target-dist "0.30,0.40,0.30"       --output results/dataset_selection_auto.json

  python scripts/extract_tw_stock_data_v7.py       --preprocessed-dir ./data/preprocessed_v5       --output-dir ./data/processed_v7       --config configs/config_pro_v7_optimal.yaml

  # 選項 2: 使用配置過濾
  python scripts/extract_tw_stock_data_v7.py       --preprocessed-dir ./data/preprocessed_v5       --output-dir ./data/processed_v7       --config configs/config_v7_test.yaml

輸出：
  - ./data/processed_v7/npz/stock_embedding_train.npz
  - ./data/processed_v7/npz/stock_embedding_val.npz
  - ./data/processed_v7/npz/stock_embedding_test.npz
  - ./data/processed_v7/npz/normalization_meta.json

版本：v7.0.0-simplified
更新：2025-10-23
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
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path

# 添加項目根目錄
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.yaml_manager import YAMLManager

# 【新增】導入專業金融工程函數庫（2025-10-23）
from src.utils.financial_engineering import trend_labels_adaptive

# 設定版本號
VERSION = "7.0.0-simplified"

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
    "data_quality_errors": 0,
    "valid_windows": 0,
    "labels_from_npz": 0,  # V7: 從 NPZ 讀取標籤
    "weights_from_metadata": 0,  # V7: 從 metadata 讀取權重
}


# ============================================================
# ============================================================
# V7 新增：JSON 讀取與數據選擇
# ============================================================

def read_dataset_selection_json(json_path: str) -> Optional[Dict]:
    """
    讀取 dataset_selection.json

    Args:
        json_path: JSON 文件路徑

    Returns:
        Dict with 'metadata' and 'file_list', or None if error
    """
    try:
        if not os.path.exists(json_path):
            logging.error(f"❌ JSON 文件不存在: {json_path}")
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 驗證必要字段
        if 'file_list' not in data:
            logging.error(f"❌ JSON 缺少 'file_list' 字段")
            return None

        if not isinstance(data['file_list'], list):
            logging.error(f"❌ 'file_list' 必須是列表")
            return None

        logging.info(f"✅ 成功讀取 JSON: {len(data['file_list'])} 個文件")
        if 'metadata' in data:
            logging.info(f"   總樣本數: {data['metadata'].get('total_samples', 'N/A')}")

        return data

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON 格式錯誤: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ 讀取 JSON 失敗: {e}")
        return None


def filter_data_by_selection(
    all_data: List[Tuple[str, str, np.ndarray, np.ndarray, Dict]],
    config: Dict,
    json_file_override: Optional[str] = None
) -> List[Tuple[str, str, np.ndarray, np.ndarray, Dict]]:
    """
    根據配置過濾數據

    優先級:
      1. 使用 --json 命令參數（最高優先級）
      2. 使用配置文件中的 dataset_selection.json
      3. 使用配置過濾（start_date, num_days, symbols）

    Args:
        all_data: [(date, symbol, features, labels, meta), ...]
        config: 配置字典
        json_file_override: 命令參數指定的 JSON 檔案（優先於配置文件）

    Returns:
        過濾後的數據列表
    """
    data_selection = config.get('data_selection', {})

    # 優先級 1: 使用命令參數指定的 JSON 文件
    json_file = json_file_override or data_selection.get('json_file')
    if json_file:
        logging.info(f"📋 使用 dataset_selection.json: {json_file}")
        json_data = read_dataset_selection_json(json_file)

        if json_data is None:
            logging.warning(f"⚠️ JSON 讀取失敗，回退到配置過濾")
        else:
            # 從 JSON 提取需要的文件
            file_list = json_data['file_list']
            selected_files = {(item['date'], item['symbol']) for item in file_list}

            filtered_data = [
                item for item in all_data
                if (item[0], item[1]) in selected_files
            ]

            logging.info(f"✅ JSON 過濾: {len(all_data)} → {len(filtered_data)} 個文件")
            return filtered_data

    # 優先級 2: 使用配置過濾
    logging.info(f"📋 使用配置過濾")

    # 1. 日期過濾
    start_date = data_selection.get('start_date')
    end_date = data_selection.get('end_date')
    num_days = data_selection.get('num_days')

    if start_date or end_date or num_days:
        # 提取所有日期並排序
        all_dates = sorted(set(item[0] for item in all_data))

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        if num_days and num_days > 0:
            all_dates = all_dates[:num_days]

        selected_dates = set(all_dates)
        all_data = [item for item in all_data if item[0] in selected_dates]

        logging.info(f"   日期過濾: 保留 {len(selected_dates)} 天")

    # 2. 股票過濾
    symbols = data_selection.get('symbols')
    if symbols:
        all_data = [item for item in all_data if item[1] in symbols]
        logging.info(f"   股票過濾: 保留 {len(symbols)} 檔")

    # 3. 隨機採樣
    sample_ratio = data_selection.get('sample_ratio', 1.0)
    if sample_ratio < 1.0:
        np.random.seed(data_selection.get('random_seed', 42))
        n_samples = int(len(all_data) * sample_ratio)
        indices = np.random.choice(len(all_data), n_samples, replace=False)
        all_data = [all_data[i] for i in sorted(indices)]
        logging.info(f"   隨機採樣: {sample_ratio:.1%} → {n_samples} 個文件")

    logging.info(f"✅ 配置過濾完成: {len(all_data)} 個文件")
    return all_data


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


def load_preprocessed_npz(npz_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    載入預處理後的 NPZ 檔案（V7 版本：返回 labels）

    Returns:
        (features, labels, metadata) or None if filtered/invalid

    V7 改動:
        - 返回 labels 而非 mids/bucket_mask
        - 強制要求 NPZ v2.0+（必須有 labels 字段）
    """
    try:
        data = np.load(npz_path, allow_pickle=True)

        # V7 版本檢查：必須有 labels 字段
        if 'labels' not in data:
            logging.error(
                f"❌ NPZ 版本過舊（v1.0）: {npz_path}\n"
                f"   V7 要求 v2.0+ NPZ（含 labels 字段）\n"
                f"   解決方法:\n"
                f"   1. 運行: scripts\\batch_preprocess.bat\n"
                f"   2. 確保 NPZ 含有 'labels' 字段"
            )
            return None

        features = data['features']  # (T, 20)
        labels = data['labels']      # (T,) with values {-1, 0, 1} or {0, 1}
        mids = data.get('mids', np.zeros(len(labels)))  # 僅用於驗證
        meta = json.loads(str(data['metadata']))

        global_stats["loaded_npz_files"] += 1
        global_stats["labels_from_npz"] += 1  # V7 統計

        # 檢查過濾狀態
        if not meta['pass_filter']:
            global_stats["symbols_filtered_out"] += 1
            return None

        # 數據質量驗證（V7 簡化版）
        if not validate_preprocessed_data(features, mids, meta, npz_path):
            logging.warning(f"⚠️ 跳過有問題的數據: {npz_path}")
            global_stats["data_quality_errors"] += 1
            global_stats["symbols_filtered_out"] += 1
            return None

        # V7 額外驗證：檢查 labels
        if len(labels) != len(features):
            logging.error(f"❌ labels 長度不匹配: {npz_path}")
            return None

        unique_labels = np.unique(labels)
        if not all(label in [-1, 0, 1, 0.0, 1.0] for label in unique_labels):
            logging.error(f"❌ labels 包含異常值 {unique_labels}: {npz_path}")
            return None

        global_stats["symbols_passed_filter"] += 1

        return features, labels, meta

    except Exception as e:
        logging.warning(f"無法載入 {npz_path}: {e}")
        return None


def load_all_preprocessed_data(preprocessed_dir: str) -> List[Tuple[str, str, np.ndarray, np.ndarray, Dict]]:
    """
    載入所有預處理數據（V7 版本：返回 labels）

    Returns:
        List[(date, symbol, features, labels, metadata)]

    V7 改動:
        - 返回 labels 而非 mids/bucket_mask
    """
    daily_dir = os.path.join(preprocessed_dir, "daily")

    if not os.path.exists(daily_dir):
        logging.error(f"預處理目錄不存在: {daily_dir}")
        return []

    all_data = []

    # 掃描所有 NPZ 檔案
    npz_files = sorted(glob.glob(os.path.join(daily_dir, "*", "*.npz")))
    logging.info(f"開始載入 {len(npz_files)} 個 NPZ 檔案...")

    for npz_file in tqdm(npz_files, desc="載入 NPZ", unit="檔"):
        result = load_preprocessed_npz(npz_file)

        if result is None:
            continue

        features, labels, meta = result  # V7: 只返回 features, labels, meta
        date = meta['date']
        symbol = meta['symbol']

        all_data.append((date, symbol, features, labels, meta))

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

def sliding_windows_v7(
    preprocessed_data: List[Tuple[str, str, np.ndarray, np.ndarray, Dict]],
    out_dir: str,
    config: Dict[str, Any],
    json_file: Optional[str] = None
):
    """
    V7 簡化版滑窗流程（專注數據組織，不重複計算）

    V7 簡化改動：
    - 輸入: (date, symbol, features, labels, meta)
    - 不重新計算標籤（直接使用 NPZ 的 labels）
    - 不重新計算波動率（不需要）
    - 不重新計算權重（從 metadata 讀取）
    - 專注於滑動窗口生成和數據劃分

    Args:
        json_file: 命令參數指定的 JSON 檔案路徑（優先於配置文件）
    """
    global global_stats
    
    logging.info("=" * 80)
    logging.info("V7 簡化版滑窗流程開始")
    logging.info("=" * 80)

    if not preprocessed_data:
        logging.warning("沒有資料可供產生 .npz 檔案")
        return

    # 步驟 0: 應用數據選擇過濾
    preprocessed_data = filter_data_by_selection(preprocessed_data, config, json_file)
    
    if not preprocessed_data:
        logging.error("❌ 過濾後無數據")
        return

    logging.info(f"過濾後數據: {len(preprocessed_data)} 個 symbol-day")

    # 步驟 1: 按股票分組（保存 metadata 用於權重提取）
    stock_data = defaultdict(list)
    stock_metadata = {}  # 每個股票的第一個 metadata（用於權重策略）
    for date, sym, features, labels, meta in preprocessed_data:
        stock_data[sym].append((date, features, labels))
        if sym not in stock_metadata:
            stock_metadata[sym] = meta  # 保存第一個 metadata

    logging.info(f"共 {len(stock_data)} 個股票")

    # 步驟 2: 按股票劃分 train/val/test (70/15/15)
    symbols = sorted(stock_data.keys())
    n_symbols = len(symbols)
    
    n_train = int(n_symbols * 0.70)
    n_val = int(n_symbols * 0.15)
    
    train_symbols = set(symbols[:n_train])
    val_symbols = set(symbols[n_train:n_train + n_val])
    test_symbols = set(symbols[n_train + n_val:])
    
    logging.info(f"劃分: train={len(train_symbols)}, val={len(val_symbols)}, test={len(test_symbols)}")

    # 步驟 3: 提取權重策略配置
    weight_strategy_name = config.get('sample_weights', {}).get('strategy', 'uniform')
    weight_enabled = config.get('sample_weights', {}).get('enabled', True)
    logging.info(f"樣本權重: {'enabled' if weight_enabled else 'disabled'}, strategy='{weight_strategy_name}'")

    # 步驟 4: 生成滑動窗口（包含 weights 和 stock_ids）
    train_X, train_y, train_weights, train_stock_ids = [], [], [], []
    val_X, val_y, val_weights, val_stock_ids = [], [], [], []
    test_X, test_y, test_weights, test_stock_ids = [], [], [], []

    logging.info(f"開始生成滑動窗口（{len(symbols)} 檔股票）...")
    for sym in tqdm(symbols, desc="生成滑窗", unit="股"):
        # 提取該股票的權重策略
        meta = stock_metadata.get(sym, {})
        weight_strategies = meta.get('weight_strategies', {})

        # 獲取指定策略的權重
        if weight_enabled and weight_strategy_name in weight_strategies:
            class_weights = weight_strategies[weight_strategy_name].get('class_weights', {})
            weight_down = class_weights.get('-1', 1.0)
            weight_neutral = class_weights.get('0', 1.0)
            weight_up = class_weights.get('1', 1.0)
        else:
            # 默認無權重
            weight_down = weight_neutral = weight_up = 1.0

        # 合併該股票所有天的數據
        all_features = []
        all_labels = []

        for date, features, labels in sorted(stock_data[sym], key=lambda x: x[0]):
            all_features.append(features)
            all_labels.append(labels)

        # 拼接
        if not all_features:
            continue

        concat_features = np.vstack(all_features)  # (T_total, 20)
        concat_labels = np.hstack(all_labels)      # (T_total,)

        # 生成滑動窗口 (100 timesteps)
        T = len(concat_features)
        if T < SEQ_LEN:
            logging.warning(f"⚠️ {sym}: 數據不足 {T} < {SEQ_LEN}，跳過")
            continue

        for i in range(T - SEQ_LEN):
            X_window = concat_features[i:i+SEQ_LEN]  # (100, 20)
            y_label = concat_labels[i+SEQ_LEN-1]     # 最後一個時間步的標籤

            # V7: 標籤轉換 {-1, 0, 1} → {0, 1, 2}，並分配權重
            if y_label == -1:
                y_label = 0
                sample_weight = weight_down
            elif y_label == 0:
                y_label = 1
                sample_weight = weight_neutral
            elif y_label == 1:
                y_label = 2
                sample_weight = weight_up
            else:
                # 處理 {0.0, 1.0} 格式（某些預處理版本）
                if y_label == 0.0:
                    y_label = 1  # 視為 neutral
                    sample_weight = weight_neutral
                elif y_label == 1.0:
                    y_label = 2  # 視為 up
                    sample_weight = weight_up
                else:
                    logging.warning(f"⚠️ 異常標籤值: {y_label}")
                    continue

            # 分配到對應集合（包含 weights 和 stock_ids）
            if sym in train_symbols:
                train_X.append(X_window)
                train_y.append(y_label)
                train_weights.append(sample_weight)
                train_stock_ids.append(sym)
            elif sym in val_symbols:
                val_X.append(X_window)
                val_y.append(y_label)
                val_weights.append(sample_weight)
                val_stock_ids.append(sym)
            else:
                test_X.append(X_window)
                test_y.append(y_label)
                test_weights.append(sample_weight)
                test_stock_ids.append(sym)

        global_stats["valid_windows"] += (T - SEQ_LEN)

    # 步驟 5: 轉換為 numpy 陣列（包含 weights 和 stock_ids）
    train_X = np.array(train_X, dtype=np.float32)  # (N_train, 100, 20)
    train_y = np.array(train_y, dtype=np.int32)
    train_weights = np.array(train_weights, dtype=np.float32)
    train_stock_ids = np.array(train_stock_ids, dtype='<U10')  # Unicode string

    val_X = np.array(val_X, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.int32)
    val_weights = np.array(val_weights, dtype=np.float32)
    val_stock_ids = np.array(val_stock_ids, dtype='<U10')

    test_X = np.array(test_X, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)
    test_weights = np.array(test_weights, dtype=np.float32)
    test_stock_ids = np.array(test_stock_ids, dtype='<U10')

    logging.info(f"生成樣本:")
    logging.info(f"  Train: {len(train_X)} 樣本")
    logging.info(f"  Val:   {len(val_X)} 樣本")
    logging.info(f"  Test:  {len(test_X)} 樣本")

    # 標籤分布
    for name, y in [("Train", train_y), ("Val", val_y), ("Test", test_y)]:
        dist = np.bincount(y, minlength=3)
        pct = dist / len(y) * 100 if len(y) > 0 else [0, 0, 0]
        logging.info(f"  {name} 分布: Down={pct[0]:.1f}%, Neutral={pct[1]:.1f}%, Up={pct[2]:.1f}%")

    # 步驟 6: 保存 NPZ（包含 weights 和 stock_ids）
    out_npz_dir = os.path.join(out_dir, 'npz')
    os.makedirs(out_npz_dir, exist_ok=True)

    train_path = os.path.join(out_npz_dir, 'stock_embedding_train.npz')
    val_path = os.path.join(out_npz_dir, 'stock_embedding_val.npz')
    test_path = os.path.join(out_npz_dir, 'stock_embedding_test.npz')

    logging.info("開始保存 NPZ 檔案（包含 weights 和 stock_ids）...")
    datasets = [
        ("train", train_path, train_X, train_y, train_weights, train_stock_ids),
        ("val", val_path, val_X, val_y, val_weights, val_stock_ids),
        ("test", test_path, test_X, test_y, test_weights, test_stock_ids)
    ]

    for name, path, X, y, weights, stock_ids in tqdm(datasets, desc="保存 NPZ", unit="檔"):
        np.savez_compressed(path, X=X, y=y, weights=weights, stock_ids=stock_ids)

    logging.info(f"✅ 保存完成:")
    logging.info(f"   {train_path}")
    logging.info(f"   {val_path}")
    logging.info(f"   {test_path}")
    logging.info("   格式: X, y, weights, stock_ids")

    # 步驟 7: 聚合所有股票的權重策略（用於 metadata）
    logging.info("聚合權重策略...")
    aggregated_weight_strategies = {}

    # 收集所有權重策略名稱
    all_strategy_names = set()
    for meta in stock_metadata.values():
        ws = meta.get('weight_strategies', {})
        all_strategy_names.update(ws.keys())

    # 對每個策略計算平均權重
    for strategy_name in all_strategy_names:
        weights_by_class = {'-1': [], '0': [], '1': []}
        strategy_descriptions = []

        for meta in stock_metadata.values():
            ws = meta.get('weight_strategies', {})
            if strategy_name in ws:
                strategy = ws[strategy_name]
                class_weights = strategy.get('class_weights', {})
                weights_by_class['-1'].append(class_weights.get('-1', 1.0))
                weights_by_class['0'].append(class_weights.get('0', 1.0))
                weights_by_class['1'].append(class_weights.get('1', 1.0))
                if 'description' in strategy and strategy['description'] not in strategy_descriptions:
                    strategy_descriptions.append(strategy['description'])

        # 計算平均權重
        if weights_by_class['-1']:  # 確保有數據
            avg_weights = {
                '-1': float(np.mean(weights_by_class['-1'])),
                '0': float(np.mean(weights_by_class['0'])),
                '1': float(np.mean(weights_by_class['1']))
            }

            aggregated_weight_strategies[strategy_name] = {
                'class_weights': avg_weights,
                'type': 'class_weight',
                'description': strategy_descriptions[0] if strategy_descriptions else f'{strategy_name} (aggregated)',
                'aggregation_method': 'mean',
                'n_stocks': len(weights_by_class['-1'])
            }

    logging.info(f"✅ 聚合 {len(aggregated_weight_strategies)} 種權重策略")

    # 步驟 8: 保存 metadata（包含權重策略和 normalization 信息）
    meta_path = os.path.join(out_npz_dir, 'normalization_meta.json')

    # V7: 數據來自預處理，已經是原始價格（未標準化）
    # label_viewer 需要 normalization 欄位來反標準化，但 V7 數據不需要反標準化
    # 因此提供一個 identity transformation (mean=0, std=1)
    normalization_info = {
        "method": "none",
        "note": "V7 數據來自預處理 NPZ，features 為原始價格（未標準化）",
        "feature_means": [0.0] * 20,  # Identity: mean = 0
        "feature_stds": [1.0] * 20    # Identity: std = 1
    }

    metadata = {
        "format": "deeplob_v7_simplified",
        "version": VERSION,
        "creation_date": datetime.now().isoformat(),
        "normalization": normalization_info,  # 添加 normalization 欄位
        "data_source": {
            "preprocessed_files": len(preprocessed_data),
            "symbols_count": len(symbols)
        },
        "data_split": {
            "method": "by_symbol",
            "train": {
                "symbols": len(train_symbols),
                "samples": len(train_X),
                "label_dist": train_y.tolist() if len(train_y) > 0 else []
            },
            "val": {
                "symbols": len(val_symbols),
                "samples": len(val_X),
                "label_dist": val_y.tolist() if len(val_y) > 0 else []
            },
            "test": {
                "symbols": len(test_symbols),
                "samples": len(test_X),
                "label_dist": test_y.tolist() if len(test_y) > 0 else []
            }
        },
        "label_source": {
            "method": "preprocessed",
            "note": "直接使用預處理 NPZ 的 labels 字段，未重新計算"
        },
        "sample_weights": {
            "enabled": weight_enabled,
            "strategy_used": weight_strategy_name,
            "available_strategies": list(aggregated_weight_strategies.keys()),
            "note": "權重策略從預處理數據聚合而來（平均值）"
        },
        "weight_strategies": aggregated_weight_strategies
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Metadata 保存: {meta_path}")
    logging.info("=" * 80)


def parse_args():
    p = argparse.ArgumentParser("extract_tw_stock_data_v6", description="V7 簡化版資料流水線（階段2）")
    p.add_argument("--preprocessed-dir", default="./data/preprocessed_v5", help="預處理結果目錄")
    p.add_argument("--output-dir", default="./data/processed_v6", help="輸出目錄")
    p.add_argument("--config", default="./configs/config_pro_v5_ml_optimal.yaml", help="配置文件")
    p.add_argument("--json", default=None, help="dataset_selection.json 檔案路徑（優先於配置文件中的設定）")
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
        sliding_windows_v7(
            preprocessed_data,
            args.output_dir,  # 修正：直接傳入 output_dir，函數內部會加上 npz
            config,
            args.json  # 傳入命令參數指定的 JSON 檔案
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
        logging.info(f"  標籤來源: NPZ 預處理數據（{global_stats['labels_from_npz']:,} 個檔案）")
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
