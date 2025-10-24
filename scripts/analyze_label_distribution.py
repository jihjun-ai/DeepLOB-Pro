# -*- coding: utf-8 -*-
"""
analyze_label_distribution.py - 智能標籤分布分析與數據集選取工具 v2.0
=============================================================================
【核心功能】
  1. 自動從起始日期開始，逐日遞增掃描所有預處理 NPZ 數據
  2. 基於標籤分布，智能組合出最適合學習的日期+個股組合
  3. 自動計算所需數量，確保達到目標標籤分布（可完整學習）
  4. 互動式選擇界面（顯示多個候選方案，讓使用者選擇）
  5. 選取後生成詳細報告（日期列表、個股ID、數值比例）

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

【版本】v2.0
【更新】2025-10-23
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
    print("✅ 已選取數據集")
    print("="*100)

    print(f"\n📋 方案描述: {selected['description']}")
    print(f"📐 偏差度: {selected['deviation']:.4f}\n")

    print("【日期列表】")
    dates = selected['dates']
    print(f"  範圍: {dates[0]} - {dates[-1]}")
    print(f"  天數: {len(dates)}")
    print(f"  明細: {', '.join(dates)}\n")

    print("【個股列表】")
    symbols = selected['symbols']
    print(f"  數量: {len(symbols)} 檔（不重複）")
    if len(symbols) <= 20:
        print(f"  明細: {', '.join(symbols)}\n")
    else:
        print(f"  前20檔: {', '.join(symbols[:20])} ...")
        print(f"  (完整列表請查看輸出 JSON)\n")

    print("【檔案配對列表】")
    print(f"  總記錄數: {selected['total_records']} 個（日期×股票配對）")
    print("  說明: 每個配對對應一個 NPZ 檔案")

    # 顯示前 10 個配對範例
    stock_records = selected['stock_records']
    print("  範例前10個:")
    for i, record in enumerate(stock_records[:10], 1):
        print(f"    {i:>2}. {record['date']}-{record['symbol']:<6} → {record['file_path']}")
    if len(stock_records) > 10:
        print(f"    ... (還有 {len(stock_records) - 10} 個配對，詳見 JSON)\n")
    else:
        print()

    print("【總樣本數】")
    print(f"  {dist['total_samples']:,} 個樣本（所有配對加總）\n")

    print("【標籤分布】")
    print(f"  Down:    {dist['down_count']:>12,} ({dist['down_pct']:>6.2%})  [目標: {target_dist[0]:.2%}, 偏差: {dist['down_pct'] - target_dist[0]:+.2%}]")
    print(f"  Neutral: {dist['neutral_count']:>12,} ({dist['neutral_pct']:>6.2%})  [目標: {target_dist[1]:.2%}, 偏差: {dist['neutral_pct'] - target_dist[1]:+.2%}]")
    print(f"  Up:      {dist['up_count']:>12,} ({dist['up_pct']:>6.2%})  [目標: {target_dist[2]:.2%}, 偏差: {dist['up_pct'] - target_dist[2]:+.2%}]")

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

    output_data = {
        'description': selected['description'],
        'date_range': selected['date_range'],
        'dates': selected['dates'],
        'num_dates': selected['num_dates'],
        'symbols': selected['symbols'],
        'num_stocks': selected['num_stocks'],
        'total_records': selected['total_records'],
        'file_list': file_list,  # 🆕 完整的「日期+股票」配對列表
        'distribution': selected['distribution'],
        'deviation': selected['deviation'],
        'level': selected['level']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ 選取結果已保存到: {output_path}")


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
                       choices=["analyze", "smart_recommend", "interactive"],
                       help="執行模式: analyze=分析全部, smart_recommend=自動推薦, interactive=互動選擇")
    parser.add_argument("--start-date", help="起始日期 (YYYYMMDD)，例如: 20250901")
    parser.add_argument("--target-dist", default=None,
                       help="目標標籤分布 (down,neutral,up)，例如: 0.30,0.40,0.30。若未指定，則使用整體分布")
    parser.add_argument("--min-samples", type=int, default=100000,
                       help="最小樣本數（過濾掉太少的方案）")
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

    print("✅ 完成\n")


if __name__ == "__main__":
    main()
