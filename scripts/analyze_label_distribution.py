# -*- coding: utf-8 -*-
"""
analyze_label_distribution.py - 智能標籤分布分析與數據集選取工具 v3.0
=============================================================================
【核心功能】
  1. 自動從起始日期開始，逐日遞增掃描所有預處理 NPZ 數據
  2. 基於標籤分布，智能組合出最適合學習的日期+個股組合
  3. 自動計算所需數量，確保達到目標標籤分布（可完整學習）
  4. 互動式選擇界面（顯示多個候選方案，讓使用者選擇）
  5. 🆕 最小化推薦模式（逐檔累積，達標即停，Neff 最優化）
  6. 選取後生成詳細報告（日期列表、個股ID、數值比例）

【使用方式】
  # 基礎分析模式（掃描所有數據）
  python scripts/analyze_label_distribution.py \
      --preprocessed-dir data/preprocessed_v5 \
      --mode analyze

  # 智能推薦模式（自動組合最佳數據集）
  python scripts/analyze_label_distribution.py \
      --preprocessed-dir data/preprocessed_v5 \
      --mode smart_recommend \
      --start-date 20250901 \
      --target-dist "0.30,0.40,0.30" \
      --min-samples 100000 \
      --output dataset_selection.json

  # 互動模式（顯示候選方案讓使用者選擇）
  python scripts/analyze_label_distribution.py \
      --preprocessed-dir data/preprocessed_v5 \
      --mode interactive \
      --start-date 20250901 \
      --target-dist "0.30,0.40,0.30"

  # 🆕 最小化推薦模式（逐檔累積，達標即停，控制數據量）
  python scripts/analyze_label_distribution.py \
      --preprocessed-dir data/preprocessed_v5 \
      --mode minimal \
      --min-samples 1000000 \
      --days 10 \
      --output dataset_selection_minimal.json

  # 🆕 最小化模式 + 指定目標分布
  python scripts/analyze_label_distribution.py \
      --preprocessed-dir data/preprocessed_v5 \
      --mode minimal \
      --min-samples 1000000 \
      --target-dist "0.30,0.40,0.30" \
      --days 15 \
      --output dataset_selection_minimal.json

【輸出範例】
  候選方案 1:
    - 日期範圍: 20250901-20250910 (10 天)
    - 個股數量: 145 檔
    - 總樣本數: 1,234,567
    - 標籤分布: Down 30.2% | Neutral 39.8% | Up 30.0%
    - 偏差度: 0.012 (與目標 30/40/30 的差異)

  選取方案 1 後：
    ✅ 已選取 145 檔股票，10 天數據
    📋 日期列表: [20250901, 20250902, ..., 20250910]
    🏢 個股列表: [2330, 2317, 2454, ..., 6488]
    📊 最終分布: Down 372,567 (30.2%) | Neutral 490,123 (39.8%) | Up 371,877 (30.0%)

【演算法說明】
  1. 日期排序：按日期升序排列（從起始日期開始）
  2. 逐日累積：逐日加入股票，計算累積標籤分布
  3. 偏差評估：計算與目標分布的 KL 散度或 L2 距離
  4. 多方案生成：
     - 保守方案（偏差 < 0.01，可能樣本較少）
     - 平衡方案（偏差 < 0.02，樣本數適中）
     - 積極方案（偏差 < 0.03，樣本數較多）
  5. 使用者選擇：顯示 3-5 個候選方案，讓使用者決定

【🆕 最小化推薦演算法】(mode=minimal)
  核心理念：在不爆量的前提下，組出達標且 Neff 最優的訓練集

  演算法流程：
    1. 天數限制：只使用最近 --days 天的數據（預設 10 天）
    2. 按股票分組：每檔股票包含所有可用日期的數據
    3. 逐檔累積：按固定順序（股票代碼）逐檔加入
    4. 安全檢查：
       - 任一類別占比不得 > 60%
       - 任一類別樣本數不得為 0
    5. 達標即停：total_samples >= min_samples 且安全時停止
    6. 最優選擇：
       - 未指定 target_dist：選 Neff 最大 + 均勻分布偏差最小
       - 指定 target_dist：選 Neff 最大 + 目標分布偏差最小

  參數說明：
    --min-samples (必填): 總樣本下限（例：1,000,000）
    --target-dist (選填): 目標分布（例：0.30,0.40,0.30），未提供時用均勻分布
    --days (選填): 最大天數上限（預設 10），避免處理量過大

  Neff 計算公式：
    Neff = N / max(w_i)
    其中 w_i = 1 / p_i（類別 i 的權重）
    Neff 越大表示類別越平衡

【版本】v3.0
【更新】2025-10-24
【作者】DeepLOB-Pro Team
"""

import os
import json
import glob
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================
# 資料載入與前處理
# ============================================================

def load_all_stocks_metadata(preprocessed_dir: str, start_date: Optional[str] = None) -> List[Dict]:
    """
    載入所有股票的 metadata（支援日期過濾）

    Args:
        preprocessed_dir: 預處理數據目錄
        start_date: 起始日期（格式: YYYYMMDD），None 表示載入全部

    Returns:
        股票 metadata 列表，按 (date, symbol) 升序排列
    """
    all_metadata = []

    # 找到所有 NPZ 檔案
    npz_pattern = os.path.join(preprocessed_dir, "daily", "*", "*.npz")
    npz_files = glob.glob(npz_pattern)

    logging.info(f"掃描到 {len(npz_files)} 個 NPZ 檔案")

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            metadata = json.loads(str(data['metadata']))

            # 只保留通過過濾且有標籤預覽的股票
            if not metadata.get('pass_filter', False):
                continue
            if metadata.get('label_preview') is None:
                continue

            # 日期過濾
            date_str = metadata['date']
            if start_date and date_str < start_date:
                continue

            lp = metadata['label_preview']
            all_metadata.append({
                'symbol': metadata['symbol'],
                'date': date_str,
                'file_path': npz_file,
                'range_pct': metadata['range_pct'],
                'n_points': metadata['n_points'],
                'total_labels': lp['total_labels'],
                'down_count': lp['down_count'],
                'neutral_count': lp['neutral_count'],
                'up_count': lp['up_count'],
                'down_pct': lp['down_pct'],
                'neutral_pct': lp['neutral_pct'],
                'up_pct': lp['up_pct']
            })
        except Exception as e:
            logging.debug(f"讀取 {npz_file} 失敗: {e}")
            continue

    # 按日期和股票代碼排序
    all_metadata.sort(key=lambda x: (x['date'], x['symbol']))

    logging.info(f"載入 {len(all_metadata)} 檔有效股票（日期 >= {start_date or '全部'}）")
    return all_metadata


def group_by_date(stocks: List[Dict]) -> Dict[str, List[Dict]]:
    """按日期分組股票"""
    grouped = defaultdict(list)
    for stock in stocks:
        grouped[stock['date']].append(stock)
    return dict(sorted(grouped.items()))


# ============================================================
# 標籤分布計算與評估
# ============================================================

def calculate_balance_score(stock: Dict) -> float:
    """
    計算單一股票的標籤平衡度分數（越高越平衡）

    使用香農熵 (Shannon Entropy) 來衡量分布均勻度
    完全均勻分布時熵最大 (log(3) ≈ 1.099)
    完全不平衡時熵為 0

    Args:
        stock: 單一股票的 metadata

    Returns:
        平衡度分數 (0~1，1 表示完全平衡)
    """
    # 提取三類占比
    down_pct = stock.get('down_pct', 0.0)
    neutral_pct = stock.get('neutral_pct', 0.0)
    up_pct = stock.get('up_pct', 0.0)

    # 避免 log(0)
    probs = [max(p, 1e-10) for p in [down_pct, neutral_pct, up_pct]]

    # 計算香農熵
    entropy = -sum(p * np.log(p) for p in probs)

    # 正規化到 [0, 1]（最大熵為 log(3)）
    max_entropy = np.log(3)
    balance_score = entropy / max_entropy

    return float(balance_score)


def calculate_stock_group_balance(stocks: List[Dict]) -> float:
    """
    計算一組股票（同一檔股票的多天數據）的平均平衡度

    Args:
        stocks: 同一檔股票的所有日期數據

    Returns:
        平均平衡度分數
    """
    if not stocks:
        return 0.0

    scores = [calculate_balance_score(s) for s in stocks]
    return float(np.mean(scores))


def calculate_neff(stocks: List[Dict]) -> float:
    """
    計算有效樣本數 (Neff) - 使用類別權重倒數加權

    公式: Neff = N / max(w_i)
    其中 w_i 是各類別的權重 (1 / p_i)

    Args:
        stocks: 股票 metadata 列表

    Returns:
        有效樣本數 Neff (越大越好)
    """
    dist = calculate_distribution(stocks)

    # 避免除以零
    down_pct = max(dist['down_pct'], 1e-10)
    neutral_pct = max(dist['neutral_pct'], 1e-10)
    up_pct = max(dist['up_pct'], 1e-10)

    # 計算各類別權重
    w_down = 1.0 / down_pct
    w_neutral = 1.0 / neutral_pct
    w_up = 1.0 / up_pct

    # Neff = N / max(w_i)
    max_weight = max(w_down, w_neutral, w_up)
    neff = dist['total_samples'] / max_weight

    return float(neff)


def calculate_distribution(stocks: List[Dict]) -> Dict[str, Any]:
    """
    計算給定股票列表的標籤分布

    Returns:
        {
            'total_stocks': int,
            'total_samples': int,
            'down_count': int,
            'neutral_count': int,
            'up_count': int,
            'down_pct': float,
            'neutral_pct': float,
            'up_pct': float
        }
    """
    total_down = sum(s['down_count'] for s in stocks)
    total_neutral = sum(s['neutral_count'] for s in stocks)
    total_up = sum(s['up_count'] for s in stocks)
    total_all = total_down + total_neutral + total_up

    return {
        'total_stocks': len(stocks),
        'total_samples': total_all,
        'down_count': total_down,
        'neutral_count': total_neutral,
        'up_count': total_up,
        'down_pct': total_down / total_all if total_all > 0 else 0.0,
        'neutral_pct': total_neutral / total_all if total_all > 0 else 0.0,
        'up_pct': total_up / total_all if total_all > 0 else 0.0
    }


def calculate_deviation(
    current_dist: Tuple[float, float, float],
    target_dist: Tuple[float, float, float],
    method: str = 'l2'
) -> float:
    """
    計算當前分布與目標分布的偏差度

    Args:
        current_dist: (down%, neutral%, up%)
        target_dist: (down%, neutral%, up%)
        method: 'l2' (L2距離) 或 'kl' (KL散度)

    Returns:
        偏差度（越小越好）
    """
    c = np.array(current_dist)
    t = np.array(target_dist)

    if method == 'l2':
        # L2 距離（歐式距離）
        return float(np.sqrt(np.sum((c - t) ** 2)))
    elif method == 'kl':
        # KL 散度（對數差異）
        epsilon = 1e-10
        c = np.clip(c, epsilon, 1.0)
        t = np.clip(t, epsilon, 1.0)
        return float(np.sum(t * np.log(t / c)))
    else:
        raise ValueError(f"未知的偏差計算方法: {method}")


# ============================================================
# 智能推薦演算法（核心）
# ============================================================

def smart_minimal_recommend(
    stocks: List[Dict],
    min_samples: int = 1000000,
    target_dist: Optional[Tuple[float, float, float]] = None,
    max_days: int = 10
) -> Dict:
    """
    智能最小化推薦演算法（新版）

    目標: 在不爆量的前提下，組出達到樣本門檻且 Neff 最佳、類別平衡的訓練資料組

    演算法邏輯:
      1. 按股票分組（每檔股票包含多天數據）
      2. 逐檔累積加入股票（固定順序：按市值或代碼排序）
      3. 達標即停：total_samples >= min_samples 時停止加入下一檔
      4. 天數限制：只使用最近 max_days 天的數據
      5. 安全閾值檢查：
         - 任一類別占比不得 > 60%
         - 任一類別樣本數不得為 0
      6. 最優選擇：
         - 未指定 target_dist：選 Neff 最大且與均勻分布偏差最小的組合
         - 指定 target_dist：選 Neff 最大且與目標分布偏差最小的組合

    Args:
        stocks: 所有股票 metadata（已按日期排序）
        min_samples: 總樣本下限（必填）
        target_dist: 三分類目標分布 (down%, neutral%, up%)，None 表示使用均勻分布
        max_days: 參考的最大天數上限（預設 10 天）

    Returns:
        最佳推薦方案字典，若無法達標則返回最佳可行解（帶 warning）
    """
    # 預設使用均勻分布
    if target_dist is None:
        target_dist = (1/3, 1/3, 1/3)
        logging.info("未指定 --target-dist，使用均勻分布作為平衡目標: (0.333, 0.333, 0.333)")

    # 天數過濾：只保留最近 max_days 天的數據
    grouped_by_date = group_by_date(stocks)
    sorted_dates = sorted(grouped_by_date.keys(), reverse=True)  # 降序（最新在前）

    if len(sorted_dates) > max_days:
        selected_dates = sorted(sorted_dates[:max_days])  # 取最新 max_days 天，再升序
        logging.info(f"天數限制: 僅使用最近 {max_days} 天數據（{selected_dates[0]} - {selected_dates[-1]}）")
        filtered_stocks = [s for s in stocks if s['date'] in selected_dates]
    else:
        selected_dates = sorted_dates
        filtered_stocks = stocks
        logging.info(f"天數限制: 總共 {len(selected_dates)} 天數據，全部使用")

    # 按股票分組（每檔股票包含所有日期的數據）
    grouped_by_symbol = defaultdict(list)
    for stock in filtered_stocks:
        grouped_by_symbol[stock['symbol']].append(stock)

    # 🆕 按標籤平衡度排序（平衡度高的優先）
    # 計算每檔股票的平均平衡度分數
    symbol_balance_scores = {
        symbol: calculate_stock_group_balance(stocks)
        for symbol, stocks in grouped_by_symbol.items()
    }

    # 按平衡度降序排序（平衡度高的在前）
    sorted_symbols = sorted(
        grouped_by_symbol.keys(),
        key=lambda s: symbol_balance_scores[s],
        reverse=True
    )

    logging.info("  股票排序: 按標籤平衡度（平衡 -> 不平衡）")
    logging.info(f"  平衡度範圍: {min(symbol_balance_scores.values()):.3f} ~ {max(symbol_balance_scores.values()):.3f}")

    logging.info("\n開始智能最小化推薦：")
    logging.info(f"  目標分布: Down {target_dist[0]:.1%} | Neutral {target_dist[1]:.1%} | Up {target_dist[2]:.1%}")
    logging.info(f"  最小樣本數: {min_samples:,}")
    logging.info(f"  可用股票數: {len(sorted_symbols)} 檔")
    logging.info(f"  天數範圍: {len(selected_dates)} 天")

    # 逐檔累積
    cumulative_stocks = []
    candidates = []

    for symbol in sorted_symbols:
        # 加入當前股票的所有日期數據
        cumulative_stocks.extend(grouped_by_symbol[symbol])

        # 計算累積分布
        dist = calculate_distribution(cumulative_stocks)

        # 安全閾值檢查
        is_safe = True
        reasons = []

        # 檢查是否有類別為 0
        if dist['down_count'] == 0:
            is_safe = False
            reasons.append("Down 類別樣本數為 0")
        if dist['neutral_count'] == 0:
            is_safe = False
            reasons.append("Neutral 類別樣本數為 0")
        if dist['up_count'] == 0:
            is_safe = False
            reasons.append("Up 類別樣本數為 0")

        # 檢查是否有類別 > 80% (寬鬆閾值，適應趨勢標籤數據)
        if dist['down_pct'] > 0.80:
            is_safe = False
            reasons.append(f"Down 類別占比過高 ({dist['down_pct']:.1%} > 80%)")
        if dist['neutral_pct'] > 0.80:
            is_safe = False
            reasons.append(f"Neutral 類別占比過高 ({dist['neutral_pct']:.1%} > 80%)")
        if dist['up_pct'] > 0.80:
            is_safe = False
            reasons.append(f"Up 類別占比過高 ({dist['up_pct']:.1%} > 80%)")

        # 計算偏差和 Neff
        current_dist = (dist['down_pct'], dist['neutral_pct'], dist['up_pct'])
        deviation = calculate_deviation(current_dist, target_dist, method='l2')
        neff = calculate_neff(cumulative_stocks)

        # 記錄為候選方案（無論是否達標或安全）
        num_symbols = len(set(s['symbol'] for s in cumulative_stocks))
        dates_used = sorted(set(s['date'] for s in cumulative_stocks))

        candidate = {
            'symbols': sorted(set(s['symbol'] for s in cumulative_stocks)),
            'dates': dates_used,
            'date_range': f"{dates_used[0]}-{dates_used[-1]}",
            'num_dates': len(dates_used),
            'num_stocks': num_symbols,
            'stock_records': cumulative_stocks.copy(),
            'total_records': len(cumulative_stocks),
            'distribution': dist,
            'deviation': deviation,
            'neff': neff,
            'is_safe': is_safe,
            'is_sufficient': dist['total_samples'] >= min_samples,
            'reasons': reasons
        }

        candidates.append(candidate)

        # 達標即停（樣本數夠 + 安全）
        if dist['total_samples'] >= min_samples and is_safe:
            logging.info(f"  ✅ 達標！累積 {num_symbols} 檔股票，{dist['total_samples']:,} 樣本")
            logging.info(f"     偏差: {deviation:.4f}, Neff: {neff:,.0f}")
            break

        # 調試信息
        if dist['total_samples'] % 100000 < 50000 or not is_safe:  # 每 10 萬樣本或不安全時輸出
            status = "✅" if is_safe else "⚠️"
            logging.debug(f"  {status} {num_symbols:>3} 檔: {dist['total_samples']:>9,} 樣本, "
                         f"偏差 {deviation:.4f}, Neff {neff:>10,.0f}")
            if not is_safe:
                logging.debug(f"      原因: {', '.join(reasons)}")

    # 在所有候選中選擇最優解
    valid_candidates = [c for c in candidates if c['is_safe']]
    sufficient_candidates = [c for c in valid_candidates if c['is_sufficient']]

    if sufficient_candidates:
        # 有達標的安全方案，選 Neff 最大 + 偏差最小
        best = max(sufficient_candidates, key=lambda x: (x['neff'], -x['deviation']))
        best['status'] = 'success'
        best['message'] = f"成功達標！Neff: {best['neff']:,.0f}, 偏差: {best['deviation']:.4f}"
        logging.info(f"\n✅ {best['message']}")
    elif valid_candidates:
        # 沒有達標但有安全方案，選最接近目標的（警告）
        best = max(valid_candidates, key=lambda x: (x['neff'], -x['deviation']))
        best['status'] = 'warning'
        best['message'] = (f"未達樣本門檻 ({best['distribution']['total_samples']:,} < {min_samples:,})，"
                          f"返回最佳可行解（Neff: {best['neff']:,.0f}）")
        logging.warning(f"\n⚠️  {best['message']}")
    else:
        # 完全無安全方案，選最安全的（錯誤）
        best = min(candidates, key=lambda x: len(x['reasons']))
        best['status'] = 'error'
        best['message'] = f"無法找到符合安全閾值的方案！最接近方案問題: {', '.join(best['reasons'])}"
        logging.error(f"\n❌ {best['message']}")

    return best


def smart_recommend_datasets(
    stocks: List[Dict],
    target_dist: Tuple[float, float, float] = (0.30, 0.40, 0.30),
    min_samples: int = 100000,
    max_deviation_levels: List[float] = [0.01, 0.02, 0.03, 0.05]
) -> List[Dict]:
    """
    智能推薦數據集組合（逐日累積，多方案生成）

    演算法邏輯：
      1. 按日期分組（升序）
      2. 逐日累積加入股票
      3. 每次累積後計算標籤分布和偏差度
      4. 當達到不同偏差閾值時，記錄為候選方案
      5. 返回多個候選方案供使用者選擇

    Args:
        stocks: 所有股票 metadata（已按日期排序）
        target_dist: 目標標籤分布 (down%, neutral%, up%)
        min_samples: 最小樣本數（過濾掉樣本太少的方案）
        max_deviation_levels: 偏差閾值列表（生成多個方案）

    Returns:
        候選方案列表，按偏差度排序（最佳方案在前）
        每個方案包含：
          - dates: 日期列表
          - stocks: 股票列表
          - distribution: 標籤分布
          - deviation: 偏差度
          - description: 方案描述
    """
    grouped = group_by_date(stocks)
    sorted_dates = sorted(grouped.keys())

    logging.info(f"\n開始智能推薦：")
    logging.info(f"  目標分布: Down {target_dist[0]:.1%} | Neutral {target_dist[1]:.1%} | Up {target_dist[2]:.1%}")
    logging.info(f"  最小樣本數: {min_samples:,}")
    logging.info(f"  日期範圍: {sorted_dates[0]} - {sorted_dates[-1]} ({len(sorted_dates)} 天)")

    # 逐日累積
    cumulative_stocks = []
    candidates = []
    found_levels = set()

    for date in sorted_dates:
        # 加入當日所有股票
        cumulative_stocks.extend(grouped[date])

        # 計算累積分布
        dist = calculate_distribution(cumulative_stocks)

        # 樣本數過濾
        if dist['total_samples'] < min_samples:
            continue

        # 計算偏差度
        current_dist = (dist['down_pct'], dist['neutral_pct'], dist['up_pct'])
        deviation = calculate_deviation(current_dist, target_dist, method='l2')

        # 檢查是否符合任一偏差閾值（且尚未記錄該等級）
        for level in max_deviation_levels:
            if deviation <= level and level not in found_levels:
                found_levels.add(level)

                # 生成方案描述
                if level <= 0.01:
                    desc = "保守方案（最高精度，偏差 < 1%）"
                elif level <= 0.02:
                    desc = "平衡方案（高精度，偏差 < 2%）"
                elif level <= 0.03:
                    desc = "積極方案（中等精度，偏差 < 3%）"
                else:
                    desc = "寬鬆方案（較大樣本，偏差 < 5%）"

                # 收集日期和股票列表
                dates_used = sorted(set(s['date'] for s in cumulative_stocks))
                symbols_used = sorted(set(s['symbol'] for s in cumulative_stocks))

                candidates.append({
                    'dates': dates_used,
                    'date_range': f"{dates_used[0]}-{dates_used[-1]}",
                    'num_dates': len(dates_used),
                    'symbols': symbols_used,
                    'num_stocks': len(symbols_used),
                    'stock_records': cumulative_stocks.copy(),  # 完整記錄（含重複日期）
                    'total_records': len(cumulative_stocks),
                    'distribution': dist,
                    'deviation': deviation,
                    'level': level,
                    'description': desc
                })

                logging.info(f"  ✅ 找到候選方案（偏差 {deviation:.4f} <= {level:.2f}）: "
                           f"{len(dates_used)} 天, {len(symbols_used)} 檔, {dist['total_samples']:,} 樣本")

                break  # 一個方案只記錄最緊的閾值

    # 按偏差度排序（最佳在前）
    candidates.sort(key=lambda x: x['deviation'])

    logging.info(f"\n共生成 {len(candidates)} 個候選方案")
    return candidates


# ============================================================
# 互動式選擇界面
# ============================================================

def display_candidates(candidates: List[Dict], target_dist: Tuple[float, float, float]) -> None:
    """顯示候選方案（美化表格）"""
    print("\n" + "="*100)
    print("📋 候選數據集方案")
    print("="*100)

    print(f"\n🎯 目標分布: Down {target_dist[0]:.1%} | Neutral {target_dist[1]:.1%} | Up {target_dist[2]:.1%}")
    print(f"\n共找到 {len(candidates)} 個候選方案（按偏差度排序）：\n")

    for i, cand in enumerate(candidates, 1):
        dist = cand['distribution']
        print(f"【方案 {i}】{cand['description']}")
        print(f"  📅 日期範圍: {cand['date_range']} ({cand['num_dates']} 天)")
        print(f"  🏢 個股數量: {cand['num_stocks']} 檔")
        print(f"  📊 總樣本數: {dist['total_samples']:,}")
        print(f"  📈 標籤分布: Down {dist['down_pct']:.2%} ({dist['down_count']:,}) | "
              f"Neutral {dist['neutral_pct']:.2%} ({dist['neutral_count']:,}) | "
              f"Up {dist['up_pct']:.2%} ({dist['up_count']:,})")
        print(f"  📐 偏差度: {cand['deviation']:.4f}")
        print()


def interactive_selection(candidates: List[Dict], target_dist: Tuple[float, float, float]) -> Optional[Dict]:
    """互動式選擇候選方案"""
    if not candidates:
        logging.error("沒有候選方案可供選擇！")
        return None

    display_candidates(candidates, target_dist)

    while True:
        try:
            choice = input(f"請選擇方案 (1-{len(candidates)})，或輸入 'q' 退出: ").strip()

            if choice.lower() == 'q':
                logging.info("使用者取消選擇")
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
            else:
                print(f"❌ 無效選擇，請輸入 1-{len(candidates)}")
        except ValueError:
            print("❌ 請輸入有效數字")
        except KeyboardInterrupt:
            print("\n使用者中斷")
            return None


# ============================================================
# 選取後的詳細報告
# ============================================================

def print_selection_report(selected: Dict, target_dist: Tuple[float, float, float]) -> None:
    """列印選取方案的詳細報告"""
    dist = selected['distribution']

    print("\n" + "="*100)
    print("[SELECTED] 已選取數據集")
    print("="*100)

    # 顯示方案描述和偏差度（如果存在）
    if 'description' in selected:
        print(f"\n[Description] {selected['description']}")
    print(f"[Deviation] {selected['deviation']:.4f}")
    if 'neff' in selected:
        print(f"[Neff] {selected['neff']:,.0f}\n")
    else:
        print()

    print("[Date List]")
    dates = selected['dates']
    print(f"  Range: {dates[0]} - {dates[-1]}")
    print(f"  Days: {len(dates)}")
    print(f"  Details: {', '.join(dates)}\n")

    print("[Stock List]")
    symbols = selected['symbols']
    print(f"  Count: {len(symbols)} stocks (unique)")
    if len(symbols) <= 20:
        print(f"  Details: {', '.join(symbols)}\n")
    else:
        print(f"  First 20: {', '.join(symbols[:20])} ...")
        print("  (See full list in output JSON)\n")

    print("[File Pairs]")
    print(f"  Total: {selected['total_records']} records (date x symbol pairs)")
    print("  Note: Each pair corresponds to one NPZ file")

    # 顯示前 10 個配對範例
    stock_records = selected['stock_records']
    print("  Sample (first 10):")
    for i, record in enumerate(stock_records[:10], 1):
        print(f"    {i:>2}. {record['date']}-{record['symbol']:<6} -> {record['file_path']}")
    if len(stock_records) > 10:
        print(f"    ... ({len(stock_records) - 10} more pairs, see JSON)\n")
    else:
        print()

    print("[Total Samples]")
    print(f"  {dist['total_samples']:,} samples (sum of all pairs)\n")

    print("[Label Distribution]")
    print(f"  Down:    {dist['down_count']:>12,} ({dist['down_pct']:>6.2%})  [Target: {target_dist[0]:.2%}, Deviation: {dist['down_pct'] - target_dist[0]:+.2%}]")
    print(f"  Neutral: {dist['neutral_count']:>12,} ({dist['neutral_pct']:>6.2%})  [Target: {target_dist[1]:.2%}, Deviation: {dist['neutral_pct'] - target_dist[1]:+.2%}]")
    print(f"  Up:      {dist['up_count']:>12,} ({dist['up_pct']:>6.2%})  [Target: {target_dist[2]:.2%}, Deviation: {dist['up_pct'] - target_dist[2]:+.2%}]")

    print("\n" + "="*100 + "\n")


def save_selection_to_json(selected: Dict, output_path: str) -> None:
    """保存選取結果到 JSON"""
    # 從 stock_records 生成「日期+股票」配對列表
    stock_records = selected['stock_records']
    file_list = [
        {
            'date': record['date'],
            'symbol': record['symbol'],
            'file_path': record['file_path'],
            'n_points': record['n_points'],
            'total_labels': record['total_labels'],
            'down_count': record['down_count'],
            'neutral_count': record['neutral_count'],
            'up_count': record['up_count']
        }
        for record in stock_records
    ]

    # 按照 (日期, 股票代碼) 排序
    file_list.sort(key=lambda x: (x['date'], x['symbol']))

    output_data = {
        'description': selected.get('description', 'Minimal recommend result'),
        'date_range': selected['date_range'],
        'dates': selected['dates'],
        'num_dates': selected['num_dates'],
        'symbols': selected['symbols'],
        'num_stocks': selected['num_stocks'],
        'total_records': selected['total_records'],
        'file_list': file_list,  # 完整的「日期+股票」配對列表
        'distribution': selected['distribution'],
        'deviation': selected['deviation'],
        'level': selected.get('level', 'N/A'),
        'neff': selected.get('neff', 0),
        'status': selected.get('status', 'unknown'),
        'mode': selected.get('mode', 'N/A'),
        'max_days': selected.get('max_days', 'N/A')
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logging.info(f"[SAVED] 選取結果已保存到: {output_path}")


# ============================================================
# 傳統分析模式（保留原功能）
# ============================================================

def analyze_overall_distribution(stocks: List[Dict]) -> Dict:
    """分析整體標籤分布（舊版相容）"""
    return calculate_distribution(stocks)


def group_by_neutral_ratio(stocks: List[Dict]) -> Dict[str, List[Dict]]:
    """按持平比例分組"""
    groups = {
        'very_low': [],    # < 10%
        'low': [],         # 10-30%
        'medium': [],      # 30-50%
        'high': [],        # 50-70%
        'very_high': []    # > 70%
    }

    for stock in stocks:
        neutral_pct = stock['neutral_pct']
        if neutral_pct < 0.10:
            groups['very_low'].append(stock)
        elif neutral_pct < 0.30:
            groups['low'].append(stock)
        elif neutral_pct < 0.50:
            groups['medium'].append(stock)
        elif neutral_pct < 0.70:
            groups['high'].append(stock)
        else:
            groups['very_high'].append(stock)

    return groups


def print_summary_report(stocks: List[Dict]):
    """列印摘要報告（舊版相容）"""
    print("\n" + "="*80)
    print("標籤分布分析報告")
    print("="*80)

    # 整體分布
    overall = analyze_overall_distribution(stocks)
    print(f"\n📊 整體標籤分布 ({overall['total_stocks']} 檔股票):")
    print(f"   總樣本數: {overall['total_samples']:,}")
    print(f"   Down:    {overall['down_count']:>10,} ({overall['down_pct']:>6.2%})")
    print(f"   Neutral: {overall['neutral_count']:>10,} ({overall['neutral_pct']:>6.2%})")
    print(f"   Up:      {overall['up_count']:>10,} ({overall['up_pct']:>6.2%})")

    # 按日期分組
    grouped = group_by_date(stocks)
    print(f"\n📅 按日期分組 ({len(grouped)} 天):")
    for date, stocks_in_date in list(grouped.items())[:5]:  # 只顯示前5天
        dist = calculate_distribution(stocks_in_date)
        print(f"   {date}: {len(stocks_in_date)} 檔, {dist['total_samples']:,} 樣本, "
              f"Down {dist['down_pct']:.1%} | Neutral {dist['neutral_pct']:.1%} | Up {dist['up_pct']:.1%}")
    if len(grouped) > 5:
        print(f"   ... (還有 {len(grouped) - 5} 天)")

    # 按持平比例分組
    groups = group_by_neutral_ratio(stocks)
    print(f"\n📈 按持平比例分組:")
    for group_name, group_stocks in groups.items():
        if not group_stocks:
            continue
        group_dist = calculate_distribution(group_stocks)
        print(f"   {group_name.upper()} ({len(group_stocks)} 檔):")
        print(f"      Down: {group_dist['down_pct']:.1%} | "
              f"Neutral: {group_dist['neutral_pct']:.1%} | "
              f"Up: {group_dist['up_pct']:.1%}")

    # 極端案例
    print(f"\n⚠️  極端案例:")
    very_low_neutral = [s for s in stocks if s['neutral_pct'] < 0.10]
    very_high_neutral = [s for s in stocks if s['neutral_pct'] > 0.70]

    if very_low_neutral:
        print(f"   持平 < 10%: {len(very_low_neutral)} 檔")
        for s in very_low_neutral[:3]:
            print(f"      {s['date']}-{s['symbol']}: Down {s['down_pct']:.1%} | "
                  f"Neutral {s['neutral_pct']:.1%} | Up {s['up_pct']:.1%}")

    if very_high_neutral:
        print(f"   持平 > 70%: {len(very_high_neutral)} 檔")
        for s in very_high_neutral[:3]:
            print(f"      {s['date']}-{s['symbol']}: Down {s['down_pct']:.1%} | "
                  f"Neutral {s['neutral_pct']:.1%} | Up {s['up_pct']:.1%}")

    print("\n" + "="*80 + "\n")


# ============================================================
# 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="智能標籤分布分析與數據集選取工具 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法：
  # 基礎分析（所有數據）
  python scripts/analyze_label_distribution.py --preprocessed-dir data/preprocessed_v5 --mode analyze

  # 智能推薦（自動生成候選方案）
  python scripts/analyze_label_distribution.py --preprocessed-dir data/preprocessed_v5 --mode smart_recommend --start-date 20250901 --output selection.json

  # 互動模式（顯示候選方案並讓使用者選擇）
  python scripts/analyze_label_distribution.py --preprocessed-dir data/preprocessed_v5 --mode interactive --start-date 20250901
        """
    )

    parser.add_argument("--preprocessed-dir", required=True, help="預處理數據目錄")
    parser.add_argument("--mode", default="analyze",
                       choices=["analyze", "smart_recommend", "interactive", "minimal"],
                       help="執行模式: analyze=分析全部, smart_recommend=自動推薦, interactive=互動選擇, minimal=最小化推薦（新版）")
    parser.add_argument("--start-date", help="起始日期 (YYYYMMDD)，例如: 20250901")
    parser.add_argument("--target-dist", default=None,
                       help="目標標籤分布 (down,neutral,up)，例如: 0.30,0.40,0.30。若未指定，則使用整體分布")
    parser.add_argument("--min-samples", type=int, default=100000,
                       help="最小樣本數（過濾掉太少的方案）")
    parser.add_argument("--days", type=int, default=10,
                       help="最大天數上限（minimal 模式專用，預設 10 天）")
    parser.add_argument("--output", help="輸出 JSON 檔案路徑")

    args = parser.parse_args()

    # 載入數據
    logging.info(f"載入預處理數據...")
    stocks = load_all_stocks_metadata(args.preprocessed_dir, args.start_date)

    if not stocks:
        logging.error("找不到有效的股票數據！")
        return

    # 解析目標分布
    if args.target_dist is None:
        # 未指定 target-dist，使用整體分布
        overall = calculate_distribution(stocks)
        target_dist = (overall['down_pct'], overall['neutral_pct'], overall['up_pct'])
        logging.info("未指定 --target-dist，使用整體標籤分布:")
        logging.info(f"  Down: {target_dist[0]:.4f} ({target_dist[0]:.2%})")
        logging.info(f"  Neutral: {target_dist[1]:.4f} ({target_dist[1]:.2%})")
        logging.info(f"  Up: {target_dist[2]:.4f} ({target_dist[2]:.2%})")
    else:
        # 手動指定 target-dist
        target_dist = tuple(map(float, args.target_dist.split(',')))
        if len(target_dist) != 3 or abs(sum(target_dist) - 1.0) > 0.01:
            logging.error(f"目標分布格式錯誤: {args.target_dist}，應為三個加總為1的數字")
            return
        logging.info(f"使用指定的目標分布: Down {target_dist[0]:.2%} | Neutral {target_dist[1]:.2%} | Up {target_dist[2]:.2%}")

    # 模式 1: 基礎分析
    if args.mode == "analyze":
        print_summary_report(stocks)

    # 模式 2: 智能推薦（自動生成但不選擇）
    elif args.mode == "smart_recommend":
        candidates = smart_recommend_datasets(
            stocks=stocks,
            target_dist=target_dist,
            min_samples=args.min_samples
        )

        if not candidates:
            logging.error("無法生成任何候選方案！請降低 min_samples 或放寬偏差閾值")
            return

        # 顯示候選方案
        display_candidates(candidates, target_dist)

        # 自動選擇最佳方案（偏差最小）
        best = candidates[0]
        logging.info(f"\n自動選擇最佳方案（偏差 {best['deviation']:.4f}）")

        print_selection_report(best, target_dist)

        # 保存結果
        if args.output:
            save_selection_to_json(best, args.output)

    # 模式 3: 互動選擇
    elif args.mode == "interactive":
        candidates = smart_recommend_datasets(
            stocks=stocks,
            target_dist=target_dist,
            min_samples=args.min_samples
        )

        if not candidates:
            logging.error("無法生成任何候選方案！請降低 min_samples 或放寬偏差閾值")
            return

        # 使用者選擇
        selected = interactive_selection(candidates, target_dist)

        if selected:
            print_selection_report(selected, target_dist)

            # 保存結果
            if args.output:
                save_selection_to_json(selected, args.output)
            else:
                # 預設輸出路徑
                default_output = f"dataset_selection_{selected['date_range']}.json"
                save_selection_to_json(selected, default_output)

    # 模式 4: 最小化推薦（新版）
    elif args.mode == "minimal":
        # 解析目標分布（minimal 模式允許不指定，預設均勻分布）
        if args.target_dist is None:
            target_dist_minimal = None  # 使用均勻分布
        else:
            target_dist_minimal = tuple(map(float, args.target_dist.split(',')))
            if len(target_dist_minimal) != 3 or abs(sum(target_dist_minimal) - 1.0) > 0.01:
                logging.error(f"目標分布格式錯誤: {args.target_dist}，應為三個加總為1的數字")
                return

        # 調用最小化推薦演算法
        best = smart_minimal_recommend(
            stocks=stocks,
            min_samples=args.min_samples,
            target_dist=target_dist_minimal,
            max_days=args.days
        )

        # 顯示結果
        print("\n" + "="*100)
        print("[Minimal Mode] 最小化推薦結果")
        print("="*100)

        # 狀態顯示
        if best['status'] == 'success':
            print(f"\n[SUCCESS] {best['message']}\n")
        elif best['status'] == 'warning':
            print(f"\n[WARNING] {best['message']}\n")
        else:
            print(f"\n[ERROR] {best['message']}\n")

        # 使用現有的報告函數（與其他模式兼容）
        # 需要轉換格式以符合 print_selection_report 的預期
        final_target_dist = target_dist_minimal if target_dist_minimal else (1/3, 1/3, 1/3)

        # 只有在有有效數據時才顯示詳細報告
        if best.get('stock_records') and len(best['stock_records']) > 0:
            print_selection_report(best, final_target_dist)
        else:
            print("\n[ERROR] 無有效數據可供顯示\n")

        # 保存結果
        if args.output:
            # 添加額外信息到輸出
            best_with_meta = best.copy()
            best_with_meta['mode'] = 'minimal'
            best_with_meta['max_days'] = args.days
            save_selection_to_json(best_with_meta, args.output)
        else:
            # 預設輸出路徑
            default_output = f"dataset_selection_minimal_{best['date_range']}.json"
            best_with_meta = best.copy()
            best_with_meta['mode'] = 'minimal'
            best_with_meta['max_days'] = args.days
            save_selection_to_json(best_with_meta, default_output)

    print("[DONE] 完成\n")


if __name__ == "__main__":
    main()
