# -*- coding: utf-8 -*-
"""
preprocess_single_day.py - 單檔逐日預處理腳本（動態過濾版）
=============================================================================
【更新日期】2025-10-21
【版本說明】v1.0 - 單檔自適應過濾預處理

功能：
  1. 讀取單一天的 TXT 檔案
  2. 解析、清洗、聚合（繼承 V5 邏輯）
  3. 計算每個 symbol 的日內統計
  4. 【核心】動態決定當天的過濾閾值（基於目標標籤分布）
  5. 應用過濾並保存為中間格式（NPZ）
  6. 生成當天摘要報告

輸出：
  - data/preprocessed_v5/daily/{date}/{symbol}.npz
  - data/preprocessed_v5/daily/{date}/summary.json

使用方式：
  python scripts/preprocess_single_day.py \
      --input ./data/temp/20250901.txt \
      --output-dir ./data/preprocessed_v5 \
      --config configs/config_pro_v5_ml_optimal.yaml

批次處理：
  bash scripts/batch_preprocess.sh

版本：v1.0
更新：2025-10-21
"""

import os
import re
import json
import argparse
import logging
import warnings
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.yaml_manager import YAMLManager

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 固定常數（繼承自 V5）
AGG_FACTOR = 10
SEQ_LEN = 100
TRADING_START = 90000
TRADING_END = 133000

# 欄位索引
IDX_REF = 3
IDX_UPPER = 4
IDX_LOWER = 5
IDX_LASTPRICE = 9
IDX_LASTVOL = 10
IDX_TV = 11
IDX_TIME = 32
IDX_TRIAL = 33

# 五檔價量索引
BID_P_IDX = [12, 14, 16, 18, 20]
BID_Q_IDX = [13, 15, 17, 19, 21]
ASK_P_IDX = [22, 24, 26, 28, 30]
ASK_Q_IDX = [23, 25, 27, 29, 31]

# 統計變數
stats = {
    "total_raw_events": 0,
    "cleaned_events": 0,
    "aggregated_points": 0,
    "symbols_processed": 0,
    "symbols_passed_filter": 0,
    "symbols_filtered_out": 0
}


# ============================================================
# 繼承自 V5 的工具函數
# ============================================================

def hhmmss_to_int(s: str) -> int:
    s = s.strip()
    if not s.isdigit():
        return -1
    return int(s)


def to_float(x: str, default=0.0) -> float:
    try:
        return float(x)
    except:
        return default


def is_in_trading_window(t: int) -> bool:
    return TRADING_START <= t <= TRADING_END


def spread_ok(bid1: float, ask1: float) -> bool:
    if bid1 <= 0 or ask1 <= 0:
        return False
    if bid1 >= ask1:
        return False
    mid = 0.5 * (bid1 + ask1)
    return (ask1 - bid1) / max(mid, 1e-12) <= 0.05


def within_limits(px: float, lo: float, hi: float) -> bool:
    if lo > 0 and px < lo - 1e-12:
        return False
    if hi > 0 and px > hi + 1e-12:
        return False
    return True


def extract_date_from_filename(fp: str) -> str:
    """從檔名抓取日期"""
    name = os.path.basename(fp)
    m = re.search(r"(20\d{6})", name)
    if m:
        return m.group(1)
    return name


def parse_line(raw: str) -> Tuple[str, int, Optional[Dict[str, Any]]]:
    """解析單行數據（繼承自 V5）"""
    global stats
    stats["total_raw_events"] += 1

    parts = raw.strip().split("||")
    if len(parts) < 34:
        return ("", -1, None)

    sym = parts[1].strip()

    try:
        t = hhmmss_to_int(parts[IDX_TIME])
    except:
        t = -1

    # 試撮移除／時間窗檢查
    if parts[IDX_TRIAL].strip() == "1":
        return (sym, t, None)
    if not is_in_trading_window(t):
        return (sym, t, None)

    # 取五檔價量
    bids_p = [to_float(parts[i], 0.0) for i in BID_P_IDX]
    bids_q = [to_float(parts[i], 0.0) for i in BID_Q_IDX]
    asks_p = [to_float(parts[i], 0.0) for i in ASK_P_IDX]
    asks_q = [to_float(parts[i], 0.0) for i in ASK_Q_IDX]

    bid1, ask1 = bids_p[0], asks_p[0]
    if not spread_ok(bid1, ask1):
        return (sym, t, None)

    # 零值處理
    for p, q in zip(bids_p + asks_p, bids_q + asks_q):
        if p == 0.0 and q != 0.0:
            return (sym, t, None)

    ref = to_float(parts[IDX_REF], 0.0)
    upper = to_float(parts[IDX_UPPER], 0.0)
    lower = to_float(parts[IDX_LOWER], 0.0)
    last_px = to_float(parts[IDX_LASTPRICE], 0.0)
    tv = max(0, int(to_float(parts[IDX_TV], 0.0)))

    # 價格限制檢查
    prices_to_check = [p for p in bids_p + asks_p if p > 0]
    if not all(within_limits(p, lower, upper) for p in prices_to_check):
        return (sym, t, None)

    # 組 20 維特徵
    feat = np.array(bids_p + asks_p + bids_q + asks_q, dtype=np.float64)
    mid = 0.5 * (bid1 + ask1)

    rec = {
        "feat": feat,
        "mid": mid,
        "ref": ref,
        "upper": upper,
        "lower": lower,
        "last_px": last_px,
        "tv": tv,
        "raw": raw.strip()
    }

    stats["cleaned_events"] += 1
    return (sym, t, rec)


def dedup_by_timestamp_keep_last(rows: List[Tuple[int, Dict[str,Any]]]) -> List[Tuple[int, Dict[str,Any]]]:
    """時間戳去重，保留最後一筆"""
    if not rows:
        return rows

    dedup_dict = {}
    for idx, (t, r) in enumerate(rows):
        tv = r.get("tv", 0)
        key = (t, tv)
        dedup_dict[key] = (idx, r)

    result_with_idx = sorted(dedup_dict.values(), key=lambda x: x[0])
    result = [(rows[idx][0], r) for idx, r in result_with_idx]

    return result


def aggregate_to_1hz(
    seq: List[Tuple[int, Dict[str,Any]]],
    reducer: str = 'last',
    ffill_limit: int = 120
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1Hz 時間聚合（秒級）

    Args:
        seq: [(timestamp_int, record)] 已排序的事件序列
        reducer: 多事件聚合策略 {'last', 'median', 'vwap-mid'}
        ffill_limit: 前值填補最大間隔（秒）

    Returns:
        features: (T, 20) LOB 特徵
        mids: (T,) 中間價
        bucket_event_count: (T,) 每秒事件數
        bucket_mask: (T,) 標記 {0: 單事件, 1: ffill, 2: 缺失, 3: 多事件聚合}
    """
    global stats

    if not seq:
        return (np.zeros((0, 20), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32))

    # 轉換為秒索引（從交易開始算起）
    # 例如 90000 (09:00:00) → idx 0, 90001 → idx 1
    def time_to_sec_idx(t):
        """HHMMSS 轉秒索引"""
        hh = t // 10000
        mm = (t // 100) % 100
        ss = t % 100
        total_sec = hh * 3600 + mm * 60 + ss
        start_total_sec = 9 * 3600  # 09:00:00
        return total_sec - start_total_sec

    # 計算總秒數
    n_seconds = (13 * 3600 + 30 * 60) - (9 * 3600) + 1  # 09:00:00 ~ 13:30:00

    # 初始化桶（每秒一個）
    buckets = [[] for _ in range(n_seconds)]

    # 將事件分配到桶
    for t, rec in seq:
        sec_idx = time_to_sec_idx(t)
        if 0 <= sec_idx < n_seconds:
            buckets[sec_idx].append(rec)

    # 聚合每個桶
    features_list = []
    mids_list = []
    event_counts = []
    masks = []

    last_valid_feat = None
    last_valid_mid = None
    last_valid_idx = -1

    for sec_idx, bucket in enumerate(buckets):
        event_count = len(bucket)
        event_counts.append(event_count)

        if event_count == 0:
            # 無事件：檢查是否可以 ffill
            gap = sec_idx - last_valid_idx if last_valid_idx >= 0 else ffill_limit + 1

            if gap <= ffill_limit and last_valid_feat is not None:
                # ffill
                features_list.append(last_valid_feat)
                mids_list.append(last_valid_mid)
                masks.append(1)  # ffill
            else:
                # 缺失
                features_list.append(np.zeros(20, dtype=np.float64))
                mids_list.append(0.0)
                masks.append(2)  # missing

        elif event_count == 1:
            # 單事件：直接取
            rec = bucket[0]
            feat = rec['feat']
            mid = rec['mid']

            features_list.append(feat)
            mids_list.append(mid)
            masks.append(0)  # native-single

            last_valid_feat = feat
            last_valid_mid = mid
            last_valid_idx = sec_idx

        else:
            # 多事件：根據 reducer 聚合
            if reducer == 'last':
                rec = bucket[-1]  # 取最後一筆
                feat = rec['feat']
                mid = rec['mid']

            elif reducer == 'median':
                # 逐欄中位數
                feats_array = np.array([r['feat'] for r in bucket])
                mids_array = np.array([r['mid'] for r in bucket])
                feat = np.median(feats_array, axis=0)
                mid = float(np.median(mids_array))

            elif reducer == 'vwap-mid':
                # VWAP（若有成交量）
                has_volume = any(r.get('tv', 0) > 0 for r in bucket)

                if has_volume:
                    total_vol = sum(r.get('tv', 0) for r in bucket)
                    if total_vol > 0:
                        mid = sum(r['mid'] * r.get('tv', 0) for r in bucket) / total_vol
                        # 特徵仍用 last（VWAP 僅用於 mid）
                        feat = bucket[-1]['feat']
                    else:
                        # 回退到 last
                        rec = bucket[-1]
                        feat = rec['feat']
                        mid = rec['mid']
                else:
                    # 無成交量，回退到 last
                    rec = bucket[-1]
                    feat = rec['feat']
                    mid = rec['mid']
            else:
                raise ValueError(f"Unknown reducer: {reducer}")

            features_list.append(feat)
            mids_list.append(mid)
            masks.append(3)  # multi-event aggregated

            last_valid_feat = feat
            last_valid_mid = mid
            last_valid_idx = sec_idx

    # 移除首尾連續缺失 + 移除中間的缺失桶（修復 mids=0 問題）
    # ⚠️ 策略變更：不保留 mask=2 的缺失桶，避免產生 mids=0
    # 原因：mids=0 會導致 log(0) = -inf，進而導致 Triple-Barrier 失敗

    first_valid = -1
    last_valid = -1

    for i, mask in enumerate(masks):
        if mask != 2:  # 非缺失
            if first_valid == -1:
                first_valid = i
            last_valid = i

    if first_valid == -1:
        # 全部缺失
        return (np.zeros((0, 20), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32))

    # 截取有效範圍（首尾）
    features_list = features_list[first_valid:last_valid+1]
    mids_list = mids_list[first_valid:last_valid+1]
    event_counts = event_counts[first_valid:last_valid+1]
    masks = masks[first_valid:last_valid+1]

    # 進一步移除中間的缺失桶（mask=2，mids=0）
    # 保留 mask=0 (單事件), mask=1 (ffill), mask=3 (多事件)
    valid_indices = [i for i, m in enumerate(masks) if m != 2]

    if len(valid_indices) == 0:
        return (np.zeros((0, 20), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32))

    features_list = [features_list[i] for i in valid_indices]
    mids_list = [mids_list[i] for i in valid_indices]
    event_counts = [event_counts[i] for i in valid_indices]
    masks = [masks[i] for i in valid_indices]

    features = np.stack(features_list, axis=0)
    mids = np.array(mids_list, dtype=np.float64)
    bucket_event_count = np.array(event_counts, dtype=np.int32)
    bucket_mask = np.array(masks, dtype=np.int32)

    # 驗證：確保沒有 mids=0（除非真實價格就是 0，但這不太可能）
    if (mids == 0).any():
        logging.warning(f"警告：發現 {(mids == 0).sum()} 個 mids=0 的點（可能是數據問題）")
        # 移除 mids=0 的點
        valid_mids = mids > 0
        if valid_mids.sum() == 0:
            return (np.zeros((0, 20), dtype=np.float64),
                    np.zeros((0,), dtype=np.float64),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32))

        features = features[valid_mids]
        mids = mids[valid_mids]
        bucket_event_count = bucket_event_count[valid_mids]
        bucket_mask = bucket_mask[valid_mids]

    stats["aggregated_points"] += len(mids)

    return features, mids, bucket_event_count, bucket_mask


def calculate_intraday_volatility(mids: np.ndarray, date: str, symbol: str) -> Optional[Dict[str, Any]]:
    """計算當日震盪統計"""
    if mids.size == 0:
        return None

    open_price = mids[0]
    close_price = mids[-1]
    high_price = mids.max()
    low_price = mids.min()

    if open_price <= 0:
        return None

    range_pct = (high_price - low_price) / open_price
    return_pct = (close_price - open_price) / open_price

    return {
        "date": date,
        "symbol": symbol,
        "range_pct": float(range_pct),
        "return_pct": float(return_pct),
        "high": float(high_price),
        "low": float(low_price),
        "open": float(open_price),
        "close": float(close_price),
        "n_points": len(mids)
    }


# ============================================================
# 核心功能：動態閾值決策
# ============================================================

def estimate_label_distribution(stats_list: List[Dict], tb_config: Dict) -> Dict[str, float]:
    """
    估計標籤分布（簡化版）

    基於啟發式規則：
    - 若 |return| < min_return → neutral
    - 若 return > 0 → up
    - 若 return < 0 → down

    注意：這是簡化估計，實際標籤由 Triple-Barrier 決定
    """
    down_count = 0
    neutral_count = 0
    up_count = 0

    min_return = tb_config.get('min_return', 0.0015)

    for s in stats_list:
        return_pct = s['return_pct']

        if abs(return_pct) < min_return:
            neutral_count += 1
        elif return_pct > 0:
            up_count += 1
        else:
            down_count += 1

    total = down_count + neutral_count + up_count

    if total == 0:
        return {'down': 0.33, 'neutral': 0.34, 'up': 0.33}

    return {
        'down': down_count / total,
        'neutral': neutral_count / total,
        'up': up_count / total
    }


def calculate_distribution_distance(pred: Dict[str, float], target: Dict[str, float]) -> float:
    """計算分布距離（平方差）"""
    return sum((pred[k] - target[k])**2 for k in target.keys())


def determine_adaptive_threshold(
    daily_stats: List[Dict],
    config: Dict,
    target_label_dist: Optional[Dict[str, float]] = None
) -> Tuple[float, str, Dict]:
    """
    動態決定當天的過濾閾值

    ⚠️ 警告：此函數存在「後見之明洩漏」(Hindsight Bias)
    ---------------------------------------------------------
    問題：使用當日**收盤後**的完整統計量來決定過濾閾值
    影響：實盤時無法在盤中複製此決策（需等收盤才知道分布）

    實盤替代方案：
    1. 使用前 N 天的滾動統計量（如前 5 日 P50）
    2. 固定閾值策略（如始終使用 P50 = 1.005）
    3. 使用盤中前 1 小時的統計量（可在盤中獲得）

    當前僅用於**離線回測**，需改進後才可用於實盤。
    ---------------------------------------------------------

    策略：
    1. 計算當天波動率分位數（P10, P25, P50, P75）
    2. 對每個候選閾值，模擬過濾後的標籤分布
    3. 選擇最接近目標分布的閾值

    Args:
        daily_stats: 當天所有 symbol 的統計
        config: 配置參數
        target_label_dist: 目標標籤分布，預設 {'down': 0.30, 'neutral': 0.40, 'up': 0.30}

    Returns:
        (threshold, method_name, predicted_dist)
    """
    if target_label_dist is None:
        target_label_dist = {'down': 0.30, 'neutral': 0.40, 'up': 0.30}

    range_values = [s['range_pct'] for s in daily_stats]

    if len(range_values) == 0:
        logging.warning("當天無有效數據，使用預設閾值 0.005")
        return 0.005, "default", target_label_dist

    # 候選閾值（基於分位數 + 絕對值）
    # 修改：增加更多細粒度候選，平衡數據保留與標籤質量
    candidates = {
        'P10': np.percentile(range_values, 10),
        'P15': np.percentile(range_values, 15),
        'P20': np.percentile(range_values, 20),
        'P25': np.percentile(range_values, 25),
        'P30': np.percentile(range_values, 30),
        'P50': np.percentile(range_values, 50),
        # 絕對值候選（適用於所有日期）
        'fixed_0.5%': 0.005,
        'fixed_1.0%': 0.010,
        'fixed_1.5%': 0.015,
    }

    tb_config = config.get('triple_barrier', {})

    best_threshold = None
    best_method = None
    best_score = float('inf')
    best_predicted = None

    for name, threshold in candidates.items():
        # 模擬過濾
        if threshold > 0:
            filtered_stats = [s for s in daily_stats if s['range_pct'] >= threshold]
        else:
            filtered_stats = daily_stats

        if len(filtered_stats) == 0:
            continue

        # 估計標籤分布
        predicted_dist = estimate_label_distribution(filtered_stats, tb_config)

        # 計算距離
        score = calculate_distribution_distance(predicted_dist, target_label_dist)

        if score < best_score:
            best_score = score
            best_threshold = threshold
            best_method = name
            best_predicted = predicted_dist

    logging.info(f"選擇閾值: {best_method} = {best_threshold:.4f} (分數: {best_score:.4f})")
    logging.info(f"預測標籤分布: Down={best_predicted['down']:.1%}, "
                f"Neutral={best_predicted['neutral']:.1%}, Up={best_predicted['up']:.1%}")

    return best_threshold, best_method, best_predicted


# ============================================================
# 保存與報告
# ============================================================

def save_preprocessed_npz(
    output_dir: str,
    date: str,
    symbol: str,
    features: np.ndarray,
    mids: np.ndarray,
    bucket_event_count: np.ndarray,
    bucket_mask: np.ndarray,
    vol_stats: Dict,
    pass_filter: bool,
    filter_threshold: float,
    filter_method: str
):
    """保存預處理後的 NPZ 檔案（1Hz 版本）"""
    day_dir = os.path.join(output_dir, "daily", date)
    os.makedirs(day_dir, exist_ok=True)

    # 準備 metadata
    metadata = {
        "symbol": symbol,
        "date": date,
        "n_points": int(features.shape[0]),

        # 日內統計
        "range_pct": float(vol_stats['range_pct']),
        "return_pct": float(vol_stats['return_pct']),
        "high": float(vol_stats['high']),
        "low": float(vol_stats['low']),
        "open": float(vol_stats['open']),
        "close": float(vol_stats['close']),

        # 過濾資訊
        "pass_filter": bool(pass_filter),
        "filter_threshold": float(filter_threshold),
        "filter_method": filter_method,
        "filter_reason": None if pass_filter else "range_too_low",

        # 處理資訊
        "processed_at": datetime.now().isoformat(),
        "raw_events": vol_stats.get('raw_events', 0),
        "aggregated_points": int(features.shape[0]),

        # 1Hz 聚合資訊
        "sampling_mode": "time",
        "bucket_seconds": 1,
        "ffill_limit": 120,
        "agg_reducer": "last",
        "n_seconds": vol_stats.get('n_seconds', 0),
        "ffill_ratio": vol_stats.get('ffill_ratio', 0.0),
        "missing_ratio": vol_stats.get('missing_ratio', 0.0),
        "multi_event_ratio": vol_stats.get('multi_event_ratio', 0.0),
        "max_gap_sec": vol_stats.get('max_gap_sec', 0)
    }

    # 保存
    npz_path = os.path.join(day_dir, f"{symbol}.npz")
    np.savez_compressed(
        npz_path,
        features=features.astype(np.float32),
        mids=mids.astype(np.float64),
        bucket_event_count=bucket_event_count.astype(np.int32),
        bucket_mask=bucket_mask.astype(np.int32),
        metadata=json.dumps(metadata, ensure_ascii=False)
    )

    return npz_path


def generate_daily_summary(
    output_dir: str,
    date: str,
    daily_stats: List[Dict],
    filter_threshold: float,
    filter_method: str,
    predicted_dist: Dict,
    symbols_passed: int,
    symbols_filtered: int
):
    """生成當天摘要報告"""
    day_dir = os.path.join(output_dir, "daily", date)

    range_values = [s['range_pct'] for s in daily_stats]

    summary = {
        "date": date,
        "total_symbols": len(daily_stats),
        "passed_filter": symbols_passed,
        "filtered_out": symbols_filtered,
        "filter_threshold": float(filter_threshold),
        "filter_method": filter_method,

        "volatility_distribution": {
            "min": float(np.min(range_values)) if range_values else 0.0,
            "max": float(np.max(range_values)) if range_values else 0.0,
            "mean": float(np.mean(range_values)) if range_values else 0.0,
            "P10": float(np.percentile(range_values, 10)) if range_values else 0.0,
            "P25": float(np.percentile(range_values, 25)) if range_values else 0.0,
            "P50": float(np.percentile(range_values, 50)) if range_values else 0.0,
            "P75": float(np.percentile(range_values, 75)) if range_values else 0.0,
        },

        "predicted_label_dist": predicted_dist,

        "top_volatile": sorted(
            [{"symbol": s['symbol'], "range_pct": s['range_pct']} for s in daily_stats],
            key=lambda x: x['range_pct'],
            reverse=True
        )[:10]
    }

    summary_path = os.path.join(day_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ 當天摘要已保存: {summary_path}")

    return summary_path


# ============================================================
# 主處理流程
# ============================================================

def process_single_day(txt_file: str, output_dir: str, config: Dict) -> Dict:
    """
    處理單一天的 TXT 檔案

    Returns:
        處理統計字典
    """
    global stats

    # 重置統計
    stats = {
        "total_raw_events": 0,
        "cleaned_events": 0,
        "aggregated_points": 0,
        "symbols_processed": 0,
        "symbols_passed_filter": 0,
        "symbols_filtered_out": 0
    }

    date = extract_date_from_filename(txt_file)

    logging.info(f"\n{'='*60}")
    logging.info(f"處理日期: {date}")
    logging.info(f"輸入檔案: {txt_file}")
    logging.info(f"{'='*60}\n")

    # Step 1: 讀取並解析
    per_symbol_raw: Dict[str, List[Tuple[int, Dict[str,Any]]]] = defaultdict(list)

    try:
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                sym, t, rec = parse_line(raw)
                if rec is None or sym == "" or t < 0:
                    continue
                per_symbol_raw[sym].append((t, rec))
    except Exception as e:
        logging.error(f"讀取檔案失敗: {e}")
        return stats

    logging.info(f"清洗後: {stats['cleaned_events']:,} 事件，共 {len(per_symbol_raw)} 個股票")

    # Step 2: 聚合並計算統計
    symbol_data = {}  # {symbol: (features, mids, vol_stats)}
    daily_stats = []

    for sym, rows in per_symbol_raw.items():
        if not rows:
            continue

        # 排序、去重
        rows.sort(key=lambda x: x[0])
        rows = dedup_by_timestamp_keep_last(rows)

        # 1Hz 聚合（新版）
        features, mids, bucket_event_count, bucket_mask = aggregate_to_1hz(
            rows,
            reducer='last',  # 可配置：'last', 'median', 'vwap-mid'
            ffill_limit=60  # A.1: 降低至 60 秒（避免長期 ffill 造成假訊號）
        )
        if features.shape[0] == 0:
            continue

        # 計算統計
        vol_stats = calculate_intraday_volatility(mids, date, sym)
        if vol_stats is None:
            continue

        vol_stats['raw_events'] = len(rows)

        # 新增：1Hz 聚合統計
        vol_stats['n_seconds'] = len(mids)
        vol_stats['ffill_ratio'] = float((bucket_mask == 1).sum() / len(bucket_mask))
        vol_stats['missing_ratio'] = float((bucket_mask == 2).sum() / len(bucket_mask))
        vol_stats['multi_event_ratio'] = float((bucket_mask == 3).sum() / len(bucket_mask))
        vol_stats['max_gap_sec'] = int(np.max(np.diff(np.where(bucket_mask != 2)[0]))) if (bucket_mask != 2).sum() > 1 else 0

        symbol_data[sym] = (features, mids, bucket_event_count, bucket_mask, vol_stats)
        daily_stats.append(vol_stats)
        stats["symbols_processed"] += 1

    logging.info(f"聚合後: {stats['aggregated_points']:,} 時間點，{stats['symbols_processed']} 個股票")

    if not daily_stats:
        logging.warning("當天無有效數據！")
        return stats

    # Step 3: 動態決定過濾閾值
    filter_threshold, filter_method, predicted_dist = determine_adaptive_threshold(
        daily_stats,
        config
    )

    # Step 4: 應用過濾並保存
    for sym, (features, mids, bucket_event_count, bucket_mask, vol_stats) in symbol_data.items():
        pass_filter = vol_stats['range_pct'] >= filter_threshold

        if pass_filter:
            stats["symbols_passed_filter"] += 1
        else:
            stats["symbols_filtered_out"] += 1

        # 保存（無論是否通過過濾，都保存，但標記狀態）
        save_preprocessed_npz(
            output_dir=output_dir,
            date=date,
            symbol=sym,
            features=features,
            mids=mids,
            bucket_event_count=bucket_event_count,
            bucket_mask=bucket_mask,
            vol_stats=vol_stats,
            pass_filter=pass_filter,
            filter_threshold=filter_threshold,
            filter_method=filter_method
        )

    # Step 5: 生成摘要
    generate_daily_summary(
        output_dir=output_dir,
        date=date,
        daily_stats=daily_stats,
        filter_threshold=filter_threshold,
        filter_method=filter_method,
        predicted_dist=predicted_dist,
        symbols_passed=stats["symbols_passed_filter"],
        symbols_filtered=stats["symbols_filtered_out"]
    )

    # 輸出統計
    logging.info(f"\n{'='*60}")
    logging.info(f"處理完成: {date}")
    logging.info(f"{'='*60}")
    logging.info(f"總事件數: {stats['total_raw_events']:,}")
    logging.info(f"清洗後: {stats['cleaned_events']:,}")
    logging.info(f"聚合時間點: {stats['aggregated_points']:,}")
    logging.info(f"處理股票: {stats['symbols_processed']}")
    logging.info(f"通過過濾: {stats['symbols_passed_filter']} ({stats['symbols_passed_filter']/stats['symbols_processed']*100:.1f}%)")
    logging.info(f"被過濾: {stats['symbols_filtered_out']} ({stats['symbols_filtered_out']/stats['symbols_processed']*100:.1f}%)")
    logging.info(f"{'='*60}\n")

    return stats


def parse_args():
    p = argparse.ArgumentParser("preprocess_single_day", description="單檔逐日預處理腳本")
    p.add_argument("--input", required=True, help="輸入 TXT 檔案路徑")
    p.add_argument("--output-dir", default="./data/preprocessed_v5", help="輸出目錄")
    p.add_argument("--config", default="./configs/config_pro_v5_ml_optimal.yaml", help="配置文件")
    return p.parse_args()


def main():
    args = parse_args()

    # 載入配置
    if not os.path.exists(args.config):
        logging.error(f"配置文件不存在: {args.config}")
        return 1

    yaml_manager = YAMLManager(args.config)
    config = yaml_manager.as_dict()

    # 驗證輸入檔案
    if not os.path.exists(args.input):
        logging.error(f"輸入檔案不存在: {args.input}")
        return 1

    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 處理
    process_single_day(args.input, args.output_dir, config)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
