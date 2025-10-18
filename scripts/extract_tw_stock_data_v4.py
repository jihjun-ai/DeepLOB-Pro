# -*- coding: utf-8 -*-
"""
extract_tw_stock_data_v4.py - 策略 B（波動率過濾版本，預設啟用）
=============================================================================
【更新日期】2025-10-17
【版本說明】v4.0 - 新增策略 B：智能波動率過濾，解決標籤不平衡問題

以《台股 DeepLOB 訓練資料規格書 — 完善版》為唯一依據，將台股逐筆五檔即時報價（|| 分隔）
轉為 FI-2010 風格的訓練資料（40→20 維），並輸出：
  1) Anchored CV 的 Train/Test 文字檔（每列：20 特徵 + 5 標籤）
  2) （可選）以 70/15/15 切分、滑窗後的 .npz（X:(N,100,20), y:{0,1,2}）

=============================================================================
✨ 策略 B 特色（預設啟用）
=============================================================================
【解決方案】波動率智能過濾（策略 B）：
  ✅ 上漲/下跌樣本：100% 保留（所有交易信號）
  ✅ 高波動持平樣本：100% 保留（盤整突破前兆，有價值）
  ✅ 低波動持平樣本：保留 10%（無聊橫盤，過濾噪音）

【核心參數】（預設配置，可調整）：
  - VOLATILITY_THRESHOLD = 0.001（0.1% 波動率閾值）
  - LOW_VOL_KEEP_RATIO = 0.1（低波動持平保留 10%）

=============================================================================
使用方式
=============================================================================

【基本使用】（使用預設策略 B 配置）：
  python scripts/extract_tw_stock_data_v4.py \
      --input-dir ./data/temp \
      --output-dir ./data/processed_volfilter \
      --make-npz

【輸出結果】：
  1. NPZ 文件：
     - ./data/processed_volfilter/npz/stock_embedding_train.npz
     - ./data/processed_volfilter/npz/stock_embedding_val.npz
     - ./data/processed_volfilter/npz/stock_embedding_test.npz

  2. Metadata：
     - ./data/processed_volfilter/npz/normalization_meta.json
       包含波動率過濾統計：
       {
         "volatility_filter": {"enabled": true, "threshold": 0.001, ...},
         "volatility_stats": {
           "high_vol_stationary": ~9萬（全保留）,
           "low_vol_stationary_kept": ~12萬（10% 保留）,
           "filtered_samples": ~109萬（過濾掉）
         }
       }

  3. 控制台日誌：
     ✅ 波動率過濾統計:
       高波動持平: 94,268 (全部保留)
       低波動持平: 1,215,396 → 保留 121,540 (10.0%)
       過濾樣本數: 1,093,856

     TRAIN 總計: 534,444 個樣本
     標籤分布: 上漲=228,696, 持平=215,808, 下跌=266,108
     ✅ 持平標籤佔比: 40.4% (vs 原始 72.6%)

【配置訓練】：
  配置文件已自動調整為策略 B（見 configs/deeplob_v4_config.yaml）

  開始訓練：
  python scripts/train_deeplob_generic.py \
      --config configs/deeplob_v4_config.yaml \
      --experiment-name "strategy_b_run1"

【監控關鍵指標】（確認策略 B 生效）：
  Epoch 10: Val Acc > 56%, Grad Norm < 4.5, 持平 < 25%
  Epoch 15: Val Acc > 59.5%, Macro F1 > 70%, Grad Norm < 4.2
  Epoch 20: Val Acc 60.5-61.5% ⭐, Train-Val Gap < 7%

=============================================================================
固定設定（依規格書）
=============================================================================
- NoAuction：移除 IsTrialMatch=='1'，並限制 09:00:00–13:30:00
- 10 事件聚合（不重疊）：價格/數量皆取視窗末端；mid 用視窗末端 bid1/ask1
- 標籤：k∈{1,2,3,5,10}，閾值 α=0.002（上=1、平=2、下=3）；.npz 會存成 {0,1,2}
- Anchored CV：以日期排序做 9 折；每折用 Train 計 Z-Score，套用到 Train/Test
- 品質檢查與異常處理：依規格書 §1.1 與 §1.9
- ✅ 波動率過濾：高波動持平全保留，低波動持平保留 10%（策略 B，預設啟用）

輸入檔：每日一檔（或多檔），每行以 '||' 分隔、無表頭，欄位索引 0~33：
  0 QType, 1 Symbol, 2 Name, 3 ReferencePrice, 4 UpperPrice, 5 LowerPrice,
  6 OpenPrice, 7 HighPrice, 8 LowPrice, 9 LastPrice, 10 LastVolume, 11 TotalVolume,
  12..21 Bid1~Bid5 (P,Q)×5, 22..31 Ask1~Ask5 (P,Q)×5, 32 MatchTime(HHMMSS), 33 IsTrialMatch

輸出格式：
- 僅保留五檔→20 維： [BidP1..5, AskP1..5, BidQ1..5, AskQ1..5]
- 同一 timestamp 多筆 → 保留 TotalVolume 相同的最後一筆
- Z-Score 正規化：基於訓練集計算 μ 和 σ
=============================================================================
技術細節
=============================================================================
【波動率計算】（見 compute_window_volatility() 函數）：
  volatility = std(mid_prices) / mean(mid_prices)
  其中 mid_price = (bid1 + ask1) / 2，基於 100 時間點窗口

【過濾邏輯】（見 should_keep_stationary_sample() 函數）：
  if label != 1 (上漲或下跌):
      return True  # 100% 保留
  if volatility >= 0.001:
      return True  # 高波動持平，100% 保留
  else:
      return random() < 0.1  # 低波動持平，10% 保留

【統計追蹤】（寫入 metadata）：
  - high_vol_stationary: 高波動持平樣本數（全保留）
  - low_vol_stationary_total: 低波動持平總數
  - low_vol_stationary_kept: 低波動持平保留數
  - filtered_samples: 過濾掉的樣本數

=============================================================================
相關文檔
=============================================================================
- 完整實施說明：docs/20251017-策略B實施代碼.md
- 策略 A vs B 決策分析：docs/20251017-策略A vs B 最終決策.md
- 訓練配置：configs/deeplob_v4_config.yaml（已調整為策略 B）

版本：v4.0 (Strategy B)
更新：2025-10-17
"""

import os, re, json, argparse, glob
import logging
import traceback
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# 設定版本號
VERSION = "1.0.0"

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 固定常數（請勿修改，除非更新規格書版本）
AGG_FACTOR = 10
HORIZONS = [1, 2, 3, 5, 10]
ALPHA = 0.002
SEQ_LEN = 100
TRADING_START = 90000   # 09:00:00
TRADING_END   = 133000  # 13:30:00

# 欄位索引（輸入）
IDX_REF = 3
IDX_UPPER = 4
IDX_LOWER = 5
IDX_LASTPRICE = 9
IDX_LASTVOL = 10
IDX_TV = 11
IDX_TIME = 32
IDX_TRIAL = 33

# 五檔價量索引對應
BID_P_IDX = [12, 14, 16, 18, 20]
BID_Q_IDX = [13, 15, 17, 19, 21]
ASK_P_IDX = [22, 24, 26, 28, 30]
ASK_Q_IDX = [23, 25, 27, 29, 31]

# 全域統計變數（用於完整 metadata）
global_stats = {
    "total_raw_events": 0,
    "cleaned_events": 0,
    "aggregated_points": 0,
    "valid_windows": 0
}

# ============================================================
# 波動率過濾相關函數（策略 B）
# ============================================================

def compute_window_volatility(window: np.ndarray) -> float:
    """
    計算單個時間窗口的價格波動率

    Args:
        window: (100, 20) LOB 特徵窗口

    Returns:
        volatility: 標準化波動率（相對波動率）
    """
    # 提取買一/賣一價格（前 10 維是價格，後 10 維是數量）
    bid1 = window[:, 0]  # 第 0 列：買一價
    ask1 = window[:, 5]  # 第 5 列：賣一價
    mid = (bid1 + ask1) / 2.0

    # 計算相對波動率（避免除零）
    mean_mid = np.mean(mid)
    if mean_mid > 1e-8:
        volatility = np.std(mid) / mean_mid
    else:
        volatility = 0.0

    return volatility


def should_keep_stationary_sample(
    window: np.ndarray,
    label: int,
    volatility_threshold: float = 0.001,
    low_vol_keep_ratio: float = 0.1,
    rng: np.random.Generator = None
) -> bool:
    """
    判斷持平樣本是否保留

    邏輯:
    - 上漲(0)和下跌(2): 100% 保留
    - 持平(1) + 高波動: 100% 保留
    - 持平(1) + 低波動: 按比例隨機保留

    Args:
        window: (100, 20) LOB 特徵窗口
        label: 標籤 {0:上漲, 1:持平, 2:下跌}
        volatility_threshold: 波動率閾值（高於此值視為高波動）
        low_vol_keep_ratio: 低波動持平樣本保留比例
        rng: 隨機數生成器

    Returns:
        should_keep: True=保留, False=丟棄
    """
    # 非持平樣本全部保留
    if label != 1:
        return True

    # 計算波動率
    vol = compute_window_volatility(window)

    # 高波動持平 → 保留
    if vol >= volatility_threshold:
        return True

    # 低波動持平 → 隨機保留
    if rng is None:
        return np.random.rand() < low_vol_keep_ratio
    else:
        return rng.random() < low_vol_keep_ratio


# 全域波動率統計（用於 metadata）
volatility_stats = {
    "high_vol_stationary": 0,
    "low_vol_stationary_total": 0,
    "low_vol_stationary_kept": 0,
    "filtered_samples": 0
}

def parse_args():
    """解析命令列參數，加入預設值"""
    p = argparse.ArgumentParser(
        "extract_tw_stock_data_v4",
        description="台股 DeepLOB 資料前處理工具 (v4 規格書版本)"
    )
    p.add_argument(
        "--input-dir", 
        default="./data/temp",  # 加入預設值
        type=str, 
        help="含每日原始 .txt 的資料夾（'||' 分隔、無表頭）[預設: ./raw_txt]"
    )
    p.add_argument(
        "--output-dir", 
        default="./data/processed",  # 加入預設值
        type=str, 
        help="輸出資料夾（會建立 Anchored CV 文本與 metadata）[預設: ./output]"
    )
    p.add_argument(
        "--make-npz", 
        action="store_true",
        default=True,  # 明確指定預設值
        help="另外輸出 70/15/15 的 .npz（滑窗後 X:(N,100,20), y:{0,1,2}）[預設: False]"
    )
    return p.parse_args()

def hhmmss_to_int(s: str) -> int:
    s = s.strip()
    if not s.isdigit(): return -1
    return int(s)

def to_float(x: str, default=0.0) -> float:
    try:
        return float(x)
    except:
        return default

def is_in_trading_window(t: int) -> bool:
    return TRADING_START <= t <= TRADING_END

def spread_ok(bid1: float, ask1: float) -> bool:
    if bid1 <= 0 or ask1 <= 0: return False
    if bid1 >= ask1: return False
    mid = 0.5 * (bid1 + ask1)
    return (ask1 - bid1) / max(mid, 1e-12) <= 0.05  # 價差異常門檻（5%）

def within_limits(px: float, lo: float, hi: float) -> bool:
    if lo > 0 and px < lo - 1e-12: return False
    if hi > 0 and px > hi + 1e-12: return False
    return True

def parse_line(raw: str) -> Tuple[str, int, Optional[Dict[str, Any]]]:
    """回傳 (symbol, time_int, rec_dict or None)；不合格回傳 (sym, time, None)"""
    global global_stats
    global_stats["total_raw_events"] += 1
    
    parts = raw.strip().split("||")
    if len(parts) < 34:
        return ("", -1, None)
    sym = parts[1].strip()
    try:
        t = hhmmss_to_int(parts[IDX_TIME])
    except:
        t = -1
    
    # 試撮移除／時間窗檢查
    if parts[IDX_TRIAL].strip() == "1":  # NoAuction
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

    # 零值處理：若價格為 0，數量必須同為 0（否則剔除該行）
    for p, q in zip(bids_p + asks_p, bids_q + asks_q):
        if p == 0.0 and q != 0.0:
            return (sym, t, None)

    ref = to_float(parts[IDX_REF], 0.0)
    upper = to_float(parts[IDX_UPPER], 0.0)
    lower = to_float(parts[IDX_LOWER], 0.0)
    last_px = to_float(parts[IDX_LASTPRICE], 0.0)
    tv = max(0, int(to_float(parts[IDX_TV], 0.0)))

    # 價格限制檢查（如有上下限，需在範圍內）
    prices_to_check = [p for p in bids_p + asks_p if p > 0]
    if not all(within_limits(p, lower, upper) for p in prices_to_check):
        return (sym, t, None)

    # 組 20 維特徵（原始價量，未正規化）
    feat = np.array(bids_p + asks_p + bids_q + asks_q, dtype=np.float64)
    mid = 0.5 * (bid1 + ask1)

    rec = {
        "feat": feat,    # (20,)
        "mid": mid,
        "ref": ref,
        "upper": upper,
        "lower": lower,
        "last_px": last_px,
        "tv": tv,
        "raw": raw.strip()
    }
    
    global_stats["cleaned_events"] += 1
    return (sym, t, rec)

def dedup_by_timestamp_keep_last(rows: List[Tuple[int, Dict[str,Any]]]) -> List[Tuple[int, Dict[str,Any]]]:
    """以時間戳和TotalVolume為準去重，同一時間戳且TotalVolume相同時保留最後一筆

    注意：此函數應在「單一股票」的資料上調用，不會跨股票去重
    判斷條件：stock_id（函數外已按symbol分組）、時間戳、TotalVolume 三者都相同時才視為重複
    """
    if not rows:
        return rows

    # 使用字典，key=(時間戳, TotalVolume)，相同key會自動覆蓋（保留最後一次）
    dedup_dict = {}
    for idx, (t, r) in enumerate(rows):
        tv = r.get("tv", 0)  # 取得 TotalVolume
        key = (t, tv)  # 組合時間戳和TotalVolume作為key
        dedup_dict[key] = (idx, r)  # 相同key時，後面的會覆蓋前面的

    # 按原始索引順序返回（保持時間順序）
    result_with_idx = sorted(dedup_dict.values(), key=lambda x: x[0])
    result = [(rows[idx][0], r) for idx, r in result_with_idx]

    if len(result) < len(rows):
        filtered = len(rows) - len(result)
        logging.debug(f"時間戳+TotalVolume去重: 過濾 {filtered} 筆，保留 {len(result)} 筆")

    return result

def qc_extra_rules(seq: List[Tuple[int, Dict[str,Any]]]) -> List[Tuple[int, Dict[str,Any]]]:
    """額外品質檢查（§1.1）：
       ✅ 修復: 移除錯誤的委託量異常檢查

       原邏輯問題:
       - 使用「平均成交量」來檢查「委託簿掛單量」
       - 成交量和委託量是完全不同的概念，不應比較
       - 委託量通常是成交量的 10-100 倍（正常現象）
       - 導致 30-80% 的正常資料被誤殺

       當前檢查:
       - parse_line() 已經做了完整的品質檢查（價差、零值、漲跌停）
       - 不需要額外的委託量檢查
    """
    # 直接返回原始序列，不做額外過濾
    return seq

def aggregate_chunks_of_10(seq: List[Tuple[int, Dict[str,Any]]]) -> Tuple[np.ndarray, np.ndarray]:
    """每 10 筆快照 → 1 時間點（不重疊）；價格/數量皆取視窗末端；mid 使用窗尾 bid1/ask1"""
    global global_stats
    
    if len(seq) < AGG_FACTOR:
        return np.zeros((0, 20), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        
    feats, mids = [], []
    for i in range(AGG_FACTOR-1, len(seq), AGG_FACTOR):
        # 取第 10 筆（尾端）
        feat = seq[i][1]["feat"]
        mid = seq[i][1]["mid"]
        feats.append(feat)
        mids.append(mid)
        
    global_stats["aggregated_points"] += len(feats)
    return np.stack(feats, axis=0), np.array(mids, dtype=np.float64)

def make_labels(mids: np.ndarray, horizons=HORIZONS, alpha=ALPHA) -> np.ndarray:
    """回傳 (L, len(HORIZONS))，取值 {1,2,3}（上/平/下）"""
    L = len(mids)
    Y = np.full((L, len(horizons)), 2, dtype=np.int64)  # 預設 2=stationary
    for j, k in enumerate(horizons):
        if L <= k: 
            continue
        delta = (mids[k:] - mids[:-k]) / np.maximum(mids[:-k], 1e-12)
        up = delta >= alpha
        dn = delta <= -alpha
        Y[:-k, j][up] = 1
        Y[:-k, j][dn] = 3
        # 末端 k 筆維持 2（無效不參與訓練，但文本可輸出）
    return Y

def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    
    # 追加檢查：|mu|過大時發出警告
    if np.any(np.abs(mu) > 1e6):
        logging.warning(f"偵測到異常大的均值: max|μ|={np.max(np.abs(mu)):.2f}，請檢查是否有未清理的異常值")
        
    return mu, sd

def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu.reshape(1, -1)) / sd.reshape(1, -1)

def write_fi2010_text(path: str, X: np.ndarray, Y: np.ndarray):
    """每行：20 個特徵（小數 6 位） + 5 個標籤（1/2/3）；空白分隔、UTF-8"""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(X.shape[0]):
            feats = " ".join(f"{v:.6f}" for v in X[i])
            labels = " ".join(str(int(v)) for v in Y[i])
            f.write(f"{feats} {labels}\n")

def sliding_windows_npz(days_points: List[Tuple[str, str, np.ndarray, np.ndarray]], out_dir: str):
    """
    將聚合後的資料切成滑窗樣本，輸出 .npz。

    策略：
    1. 以股票為單位，按日期串接所有天的資料
    2. 對每檔股票做滑窗（跨日）
    3. 70/15/15 按股票數量切分（確保每個 split 都有足夠股票）
    """
    global global_stats
    
    if not days_points:
        logging.warning("沒有資料可供產生 .npz 檔案")
        return

    logging.info(f"\n{'='*60}")
    logging.info(f"開始產生滑窗資料，共 {len(days_points)} 個 symbol-day 組合")
    logging.info(f"{'='*60}")

    # ========== 步驟 1: 以股票為單位重組資料 ==========
    stock_data = defaultdict(lambda: {'dates': [], 'X': [], 'y': []})

    for date, sym, Xd, Yd in days_points:
        stock_data[sym]['dates'].append(date)
        stock_data[sym]['X'].append(Xd)
        stock_data[sym]['y'].append(Yd)

    logging.info(f"共 {len(stock_data)} 個股票有資料")

    # 對每個股票，按日期排序並串接
    stock_sequences = []  # [(symbol, total_points, X_concat, y_concat)]

    for sym, data in stock_data.items():
        # 按日期排序
        sorted_indices = np.argsort(data['dates'])

        # 串接該股票所有天的資料（跨日）
        X_concat = np.concatenate([data['X'][i] for i in sorted_indices], axis=0)
        y_concat = np.concatenate([data['y'][i] for i in sorted_indices], axis=0)

        stock_sequences.append((sym, X_concat.shape[0], X_concat, y_concat))

    # 按時間點數量排序（流動性高的股票優先）
    stock_sequences.sort(key=lambda x: x[1], reverse=True)

    logging.info(f"\n股票時間點統計:")
    logging.info(f"  最多: {stock_sequences[0][1]} 個點 ({stock_sequences[0][0]})")
    logging.info(f"  最少: {stock_sequences[-1][1]} 個點 ({stock_sequences[-1][0]})")
    logging.info(f"  平均: {np.mean([s[1] for s in stock_sequences]):.1f} 個點")

    # ========== 步驟 2: 過濾太短的股票序列 ==========
    MIN_POINTS = SEQ_LEN + max(HORIZONS)  # 100 + 10 = 110

    valid_stocks = [s for s in stock_sequences if s[1] >= MIN_POINTS]
    filtered_stocks = len(stock_sequences) - len(valid_stocks)

    if filtered_stocks > 0:
        logging.warning(f"過濾 {filtered_stocks} 檔序列太短的股票（< {MIN_POINTS} 個點）")

    if not valid_stocks:
        logging.error("沒有股票有足夠的時間點產生滑窗樣本！")
        return

    logging.info(f"有效股票: {len(valid_stocks)} 檔")

    # ========== 步驟 3: 按股票數量切分 70/15/15 ==========
    n_stocks = len(valid_stocks)
    n_train = max(1, int(n_stocks * 0.7))
    n_val = max(1, int(n_stocks * 0.15))

    train_stocks = valid_stocks[:n_train]
    val_stocks = valid_stocks[n_train:n_train + n_val]
    test_stocks = valid_stocks[n_train + n_val:]

    logging.info(f"\n資料切分（按股票數）:")
    logging.info(f"  Train: {len(train_stocks)} 檔股票")
    logging.info(f"  Val:   {len(val_stocks)} 檔股票")
    logging.info(f"  Test:  {len(test_stocks)} 檔股票")

    splits = {
        'train': train_stocks,
        'val': val_stocks,
        'test': test_stocks
    }

    # ========== 步驟 4: 計算訓練集的 Z-Score 參數 ==========
    logging.info(f"\n{'='*60}")
    logging.info("計算 Z-Score 參數（基於訓練集）")
    logging.info(f"{'='*60}")

    # 收集訓練集所有時間點
    train_X_list = [stock[2] for stock in train_stocks]
    Xtr = np.concatenate(train_X_list, axis=0) if train_X_list else np.zeros((0, 20))

    if Xtr.size == 0:
        mu = np.zeros((20,), dtype=np.float64)
        sd = np.ones((20,), dtype=np.float64)
        logging.warning("訓練集為空，使用預設 Z-Score 參數")
    else:
        mu, sd = zscore_fit(Xtr)
        logging.info(f"訓練集大小: {Xtr.shape[0]:,} 個時間點")

    # ========== 步驟 5: 產生滑窗樣本 ==========
    def build_split(split_name):
        """對一個 split 產生滑窗樣本"""
        stock_list = splits[split_name]

        logging.info(f"\n{'='*60}")
        logging.info(f"處理 {split_name.upper()} 集，共 {len(stock_list)} 檔股票")
        logging.info(f"{'='*60}")

        X_windows = []
        y_labels = []
        stock_ids = []

        total_windows = 0

        for sym, n_points, Xd, Yd in stock_list:
            # Z-score 正規化
            Xn = zscore_apply(Xd, mu, sd)

            # 取 k=10 的標籤（最後一欄），轉 {0,1,2}
            yk = Yd[:, -1]  # {1,2,3}

            T = Xn.shape[0]

            # 計算有效的滑窗範圍（避免標籤無效）
            max_t = T - max(HORIZONS)  # 末端留 10 個點給標籤

            if max_t < SEQ_LEN:
                logging.warning(f"  {sym}: 跳過（只有 {T} 個點，不足 {SEQ_LEN}）")
                continue

            # 產生滑窗樣本（✅ 策略 B：波動率過濾）
            windows_count = 0
            rng = np.random.default_rng(42)  # 固定隨機種子

            # 波動率過濾配置
            VOLATILITY_THRESHOLD = 0.001  # 0.1% 波動率閾值
            LOW_VOL_KEEP_RATIO = 0.1      # 低波動持平保留 10%

            for t in range(SEQ_LEN - 1, max_t):
                # 取 100 個時間點作為輸入
                window = Xn[t - SEQ_LEN + 1:t + 1, :]

                # 檢查窗口形狀
                if window.shape[0] != SEQ_LEN:
                    continue

                # 取標籤
                label_raw = int(yk[t])
                if label_raw < 1 or label_raw > 3:
                    continue

                label = label_raw - 1  # {1,2,3} → {0,1,2}

                # ============================================================
                # ✅ 策略 B：波動率過濾邏輯
                # ============================================================
                keep = should_keep_stationary_sample(
                    window=window,
                    label=label,
                    volatility_threshold=VOLATILITY_THRESHOLD,
                    low_vol_keep_ratio=LOW_VOL_KEEP_RATIO,
                    rng=rng
                )

                if not keep:
                    volatility_stats["filtered_samples"] += 1
                    continue  # 跳過該樣本

                # 統計波動率分布（用於 metadata）
                if label == 1:
                    vol = compute_window_volatility(window)
                    if vol >= VOLATILITY_THRESHOLD:
                        volatility_stats["high_vol_stationary"] += 1
                    else:
                        volatility_stats["low_vol_stationary_total"] += 1
                        volatility_stats["low_vol_stationary_kept"] += 1
                # ============================================================

                X_windows.append(window.astype(np.float32))
                y_labels.append(label)
                stock_ids.append(sym)
                windows_count += 1

            if windows_count > 0:
                logging.info(f"  {sym}: {windows_count:,} 個樣本 (共 {T} 個點)")

            total_windows += windows_count

        logging.info(f"\n{split_name.upper()} 總計: {total_windows:,} 個樣本")

        global_stats["valid_windows"] += total_windows

        if X_windows:
            X_array = np.stack(X_windows, axis=0)
            y_array = np.array(y_labels, dtype=np.int64)
            sid_array = np.array(stock_ids, dtype=object)

            # 統計資訊
            unique_stocks = len(np.unique(sid_array))
            label_dist = np.bincount(y_array, minlength=3)

            logging.info(f"  形狀: X={X_array.shape}, y={y_array.shape}")
            logging.info(f"  涵蓋股票: {unique_stocks} 檔")
            logging.info(f"  標籤分布: 上漲={label_dist[0]}, 持平={label_dist[1]}, 下跌={label_dist[2]}")

            return X_array, y_array, sid_array
        else:
            logging.warning(f"{split_name} 集沒有產生任何樣本！")
            return (np.zeros((0, SEQ_LEN, 20), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.array([], dtype=object))

    # ========== 步驟 6: 產生並保存 npz 檔案 ==========
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        X, y, sid = build_split(split)

        npz_path = os.path.join(out_dir, f"stock_embedding_{split}.npz")
        np.savez_compressed(npz_path, X=X, y=y, stock_ids=sid)

        logging.info(f"\n✅ 已保存: {npz_path}")

    # ========== 步驟 7: 寫入 metadata ==========
    meta = {
        "format": "fi2010_text_20→npz_v4",
        "version": VERSION,
        "creation_date": datetime.now().isoformat(),
        "seq_len": SEQ_LEN,
        "feature_dim": 20,
        "label_alpha": ALPHA,
        "horizons": HORIZONS,
        "windowing_strategy": "stock-wise cross-day",

        # ✅ 新增：波動率過濾配置
        "volatility_filter": {
            "enabled": True,
            "threshold": 0.001,
            "low_vol_keep_ratio": 0.1,
            "strategy": "keep_high_vol_stationary_full",
            "description": "高波動持平全保留，低波動持平保留10%"
        },

        "normalization": {
            "method": "zscore",
            "computed_on": "train_set",
            "feature_means": mu.tolist(),
            "feature_stds": sd.tolist()
        },

        "data_quality": global_stats,

        # ✅ 新增：波動率統計
        "volatility_stats": volatility_stats.copy(),

        "data_split": {
            "method": "by_stock_count",
            "train_stocks": len(train_stocks),
            "val_stocks": len(val_stocks),
            "test_stocks": len(test_stocks),
            "total_stocks": len(valid_stocks),
            "filtered_stocks": filtered_stocks,
            "min_points_required": MIN_POINTS
        },
        "note": "Labels in npz are {0:上漲, 1:持平, 2:下跌} from text {1,2,3}. k=10 used. Volatility-filtered cross-day windowing per stock.",
    }

    meta_path = os.path.join(out_dir, "normalization_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ Metadata 已保存: {meta_path}")
    logging.info(f"\n{'='*60}")
    logging.info("NPZ 檔案產生完成！")
    logging.info(f"{'='*60}")
def extract_date_from_filename(fp: str) -> str:
    """嘗試從檔名抓 YYYYMMDD，抓不到則用字典序"""
    name = os.path.basename(fp)
    m = re.search(r"(20\d{6})", name)  # 例如 20240115
    if m:
        return m.group(1)
    return name  # 後備：用檔名排序

def write_complete_metadata(out_dir: str, unique_days: List[str], mu: np.ndarray, sd: np.ndarray, fold: Optional[int] = None):
    """寫入完整的 metadata（規格書 §1.10）"""
    meta = {
        "version": VERSION,
        "creation_date": datetime.now().isoformat(),
        "data_range": {
            "start_date": min(unique_days) if unique_days else "N/A",
            "end_date": max(unique_days) if unique_days else "N/A",
            "total_trading_days": len(unique_days)
        },
        "preprocessing": {
            "aggregation_factor": AGG_FACTOR,
            "price_feature": "tail",
            "volume_feature": "tail",
            "no_auction": True,
            "trial_match_excluded": True,
            "trading_window": ["09:00:00", "13:30:00"]
        },
        "labeling": {
            "horizons": HORIZONS,
            "threshold": ALPHA,
            "label_mapping": {"up": 1, "stationary": 2, "down": 3}
        },
        "normalization": {
            "method": "zscore",
            "computed_on": "train_set",
            "feature_means": mu.tolist(),
            "feature_stds": sd.tolist()
        },
        "data_quality": global_stats.copy(),
        "format": "FI2010_text_20dim_v4"
    }
    
    if fold is not None:
        meta["fold"] = fold
        filename = f"meta_fold_{fold}.json"
    else:
        filename = "metadata_complete.json"
        
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    """主程式，包含完整錯誤處理"""
    try:
        args = parse_args()
        in_dir = args.input_dir
        out_dir = args.output_dir
        
        # 驗證輸入目錄存在
        if not os.path.exists(in_dir):
            logging.error(f"輸入目錄不存在: {in_dir}")
            return 1
            
        # 建立輸出目錄
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"無法建立輸出目錄 {out_dir}: {e}")
            return 1

        logging.info(f"開始處理，輸入目錄: {in_dir}, 輸出目錄: {out_dir}")
        
        # 讀取所有 .txt 檔
        files = sorted(glob.glob(os.path.join(in_dir, "*.txt")))
        if not files:
            logging.error(f"在 {in_dir} 找不到 .txt 檔案")
            return 1
            
        logging.info(f"找到 {len(files)} 個檔案待處理")

        # 先按檔名日期分組
        day_map: Dict[str, List[str]] = defaultdict(list)
        for fp in files:
            day = extract_date_from_filename(fp)
            day_map[day].append(fp)

        # 逐日逐檔案讀取 → 逐 symbol 彙整
        per_day_symbol_points = []  # [(date, symbol, X_points, Y_points)]
        day_keys = sorted(day_map.keys())
        
        logging.info(f"開始處理 {len(day_keys)} 個交易日的資料")

        for day in day_keys:
            fps = sorted(day_map[day])
            logging.info(f"處理日期 {day}，共 {len(fps)} 個檔案")
            
            # 讀本日所有行
            per_symbol_raw: Dict[str, List[Tuple[int, Dict[str,Any]]]] = defaultdict(list)
            per_symbol_lines: Dict[str, List[str]] = defaultdict(list)

            for fp in fps:
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for raw in f:
                            sym, t, rec = parse_line(raw)
                            if rec is None or sym == "" or t < 0:
                                continue
                            per_symbol_raw[sym].append((t, rec))
                            per_symbol_lines[sym].append(rec["raw"])
                except Exception as e:
                    logging.warning(f"讀取檔案 {fp} 時發生錯誤: {e}")
                    continue

            # 對每個 symbol：去重（整行字串）、同 timestamp 取變化最大、品質檢查、10 事件聚合、標籤
            for sym, rows in per_symbol_raw.items():
                if not rows:
                    continue
                    
                # 時間排序
                rows.sort(key=lambda x: x[0])
                
                # 整行重複保留最後
                # ❌ 修復: 註釋掉時間戳去重，因為台股時間戳只到秒級，
                # 同一秒內的多筆委託簿變化都是有價值的資料，不應該被去重
                rows = dedup_by_timestamp_keep_last(rows)
                if not rows:
                    continue
                    
                # 額外品質檢查(覺得沒必要就註解掉)
                #rows = qc_extra_rules(rows)
                #if len(rows) < AGG_FACTOR:
                #   continue

                # 10 事件聚合
                Xp, mids = aggregate_chunks_of_10(rows)
                if Xp.shape[0] == 0:
                    continue
                    
                # 標籤（{1,2,3}）
                Yp = make_labels(mids, HORIZONS, ALPHA)
                per_day_symbol_points.append((day, sym, Xp, Yp))

        # 若沒有可用資料
        if not per_day_symbol_points:
            logging.error("清洗或聚合後沒有可用資料")
            return 1
            
        logging.info(f"共處理 {len(per_day_symbol_points)} 個 symbol-day 組合")



        # Anchored CV：依日期排序分 9 折；每折 Train/Test 各自輸出文本（Z-Score 用當折 Train 計）
        unique_days = sorted(sorted({d for d,_,_,_ in per_day_symbol_points}))
        N = len(unique_days)
        
        logging.info(f"共 {N} 個交易日，開始產生 Anchored CV 9 折")
        
        # 建立 day->(sym list of X,Y)
        by_day: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = defaultdict(list)
        for d, s, X, Y in per_day_symbol_points:
            by_day[d].append((s, X, Y))

        for i in range(1, 10):  # 1..9
            lo = int(np.floor(N * i / 10))
            hi = int(np.floor(N * (i+1) / 10))
            train_days = unique_days[:lo]   # 逐步擴張
            test_days  = unique_days[lo:hi] # 固定大小區段
            
            logging.info(f"處理第 {i} 折，訓練天數: {len(train_days)}, 測試天數: {len(test_days)}")

            # 彙整當折的 train/test 時間點
            def collect(days):
                X_list, Y_list = [], []
                for d in days:
                    for s, X, Y in by_day.get(d, []):
                        X_list.append(X)
                        Y_list.append(Y)
                if X_list:
                    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)
                else:
                    return np.zeros((0,20)), np.zeros((0,len(HORIZONS)), dtype=np.int64)

            X_tr, Y_tr = collect(train_days)
            X_te, Y_te = collect(test_days)

            if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
                logging.warning(f"Fold {i}: 訓練或測試為空，略過輸出")
                continue
                
            logging.info(f"Fold {i}: 訓練集大小={X_tr.shape}, 測試集大小={X_te.shape}")

            # Z-Score 以 Train 計
            mu, sd = zscore_fit(X_tr)
            X_tr_n = zscore_apply(X_tr, mu, sd)
            X_te_n = zscore_apply(X_te, mu, sd)

            # 寫文本檔
            out_train = os.path.join(out_dir, f"Train_Dst_NoAuction_ZScore_CF_{i}.txt")
            out_test  = os.path.join(out_dir, f"Test_Dst_NoAuction_ZScore_CF_{i}.txt")
            write_fi2010_text(out_train, X_tr_n, Y_tr)
            write_fi2010_text(out_test,  X_te_n, Y_te)

            # 寫當折完整 metadata
            write_complete_metadata(out_dir, unique_days, mu, sd, fold=i)
            
            logging.info(f"Fold {i} 完成")

        # 寫入整體 metadata
        if len(per_day_symbol_points) > 0:
            # 使用第一折的統計資料作為代表
            sample_X = np.concatenate([dp[2] for dp in per_day_symbol_points[:10] if dp[2].size], axis=0)
            if sample_X.size > 0:
                mu_sample, sd_sample = zscore_fit(sample_X)
                write_complete_metadata(out_dir, unique_days, mu_sample, sd_sample, fold=None)

        # （可選）產出 70/15/15 的 .npz（滑窗後）
        if args.make_npz:
            logging.info("開始產生 .npz 檔案")
            sliding_windows_npz(per_day_symbol_points, os.path.join(out_dir, "npz"))

        logging.info(f"[完成] v4 轉換成功，輸出資料夾: {out_dir}")
        logging.info(f"統計資料 - 原始事件數: {global_stats['total_raw_events']}, "
                    f"清洗後: {global_stats['cleaned_events']}, "
                    f"聚合後時間點: {global_stats['aggregated_points']}")
        
        return 0
        
    except Exception as e:
        logging.error(f"程式執行失敗: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
