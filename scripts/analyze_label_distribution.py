# -*- coding: utf-8 -*-
"""
analyze_label_distribution.py - 分析並可視化標籤分布
=============================================================================
用途：幫助你理解數據的標籤分布，並生成股票選取建議

使用方式：
  # 分析所有數據
  python scripts/analyze_label_distribution.py \
      --preprocessed-dir data/preprocessed_v5 \
      --output analysis_report.json

  # 生成股票選取清單
  python scripts/analyze_label_distribution.py \
      --preprocessed-dir data/preprocessed_v5 \
      --mode recommend \
      --target-dist "0.30,0.40,0.30" \
      --output stock_selection.json

功能：
  1. 分析所有股票的標籤分布
  2. 按不同維度分組（持平比例、波動率等）
  3. 生成股票選取建議（達到目標標籤分布）
  4. 可視化標籤分布（直方圖、散點圖）
"""

import os
import json
import glob
import argparse
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_all_stocks_metadata(preprocessed_dir: str) -> List[Dict]:
    """載入所有股票的 metadata"""
    all_metadata = []

    # 找到所有 NPZ 檔案
    npz_pattern = os.path.join(preprocessed_dir, "daily", "*", "*.npz")
    npz_files = glob.glob(npz_pattern)

    logging.info(f"找到 {len(npz_files)} 個 NPZ 檔案")

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            metadata = json.loads(str(data['metadata']))

            # 只保留通過過濾且有標籤預覽的股票
            if metadata.get('pass_filter', False) and metadata.get('label_preview') is not None:
                lp = metadata['label_preview']
                all_metadata.append({
                    'symbol': metadata['symbol'],
                    'date': metadata['date'],
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
            logging.warning(f"讀取 {npz_file} 失敗: {e}")
            continue

    logging.info(f"載入 {len(all_metadata)} 檔有效股票")
    return all_metadata


def analyze_overall_distribution(stocks: List[Dict]) -> Dict:
    """分析整體標籤分布"""
    total_down = sum(s['down_count'] for s in stocks)
    total_neutral = sum(s['neutral_count'] for s in stocks)
    total_up = sum(s['up_count'] for s in stocks)
    total_all = total_down + total_neutral + total_up

    return {
        'total_stocks': len(stocks),
        'total_labels': total_all,
        'down_count': total_down,
        'neutral_count': total_neutral,
        'up_count': total_up,
        'down_pct': total_down / total_all if total_all > 0 else 0,
        'neutral_pct': total_neutral / total_all if total_all > 0 else 0,
        'up_pct': total_up / total_all if total_all > 0 else 0
    }


def group_by_neutral_ratio(stocks: List[Dict]) -> Dict[str, List[Dict]]:
    """按持平比例分組"""
    groups = {
        'very_low': [],    # < 10%
        'low': [],         # 10-20%
        'medium': [],      # 20-30%
        'high': [],        # 30-40%
        'very_high': []    # > 40%
    }

    for stock in stocks:
        neutral_pct = stock['neutral_pct']
        if neutral_pct < 0.10:
            groups['very_low'].append(stock)
        elif neutral_pct < 0.20:
            groups['low'].append(stock)
        elif neutral_pct < 0.30:
            groups['medium'].append(stock)
        elif neutral_pct < 0.40:
            groups['high'].append(stock)
        else:
            groups['very_high'].append(stock)

    return groups


def recommend_stock_selection(
    stocks: List[Dict],
    target_dist: Tuple[float, float, float] = (0.30, 0.40, 0.30)
) -> Dict:
    """
    根據目標標籤分布，推薦股票選取策略

    Args:
        stocks: 所有股票 metadata
        target_dist: 目標分布 (down%, neutral%, up%)

    Returns:
        推薦結果，包含：
        - 選取的股票清單
        - 預期達到的標籤分布
        - 選取邏輯說明
    """
    target_down, target_neutral, target_up = target_dist

    # 當前整體分布
    current = analyze_overall_distribution(stocks)
    current_dist = (current['down_pct'], current['neutral_pct'], current['up_pct'])

    logging.info(f"\n目標分布: Down {target_down:.1%} | Neutral {target_neutral:.1%} | Up {target_up:.1%}")
    logging.info(f"當前分布: Down {current_dist[0]:.1%} | Neutral {current_dist[1]:.1%} | Up {current_dist[2]:.1%}")

    # 分析缺口
    neutral_gap = target_neutral - current_dist[1]
    up_gap = target_up - current_dist[2]
    down_gap = target_down - current_dist[0]

    recommendations = []

    # 策略 1：如果持平類不足，優先選持平比例高的股票
    if neutral_gap > 0.05:
        high_neutral_stocks = [s for s in stocks if s['neutral_pct'] > 0.25]
        recommendations.append({
            'strategy': 'boost_neutral',
            'reason': f'持平類不足 ({neutral_gap:.1%})，選取持平比例 > 25% 的股票',
            'stocks': [s['symbol'] for s in high_neutral_stocks],
            'count': len(high_neutral_stocks)
        })

    # 策略 2：如果上漲類不足，選上漲比例高的股票
    if up_gap > 0.05:
        high_up_stocks = [s for s in stocks if s['up_pct'] > 0.45]
        recommendations.append({
            'strategy': 'boost_up',
            'reason': f'上漲類不足 ({up_gap:.1%})，選取上漲比例 > 45% 的股票',
            'stocks': [s['symbol'] for s in high_up_stocks],
            'count': len(high_up_stocks)
        })

    # 策略 3：如果下跌類不足，選下跌比例高的股票
    if down_gap > 0.05:
        high_down_stocks = [s for s in stocks if s['down_pct'] > 0.45]
        recommendations.append({
            'strategy': 'boost_down',
            'reason': f'下跌類不足 ({down_gap:.1%})，選取下跌比例 > 45% 的股票',
            'stocks': [s['symbol'] for s in high_down_stocks],
            'count': len(high_down_stocks)
        })

    # 策略 4：如果已經很平衡，選取中等比例的股票
    if abs(neutral_gap) < 0.05 and abs(up_gap) < 0.05 and abs(down_gap) < 0.05:
        balanced_stocks = [s for s in stocks
                          if 0.20 <= s['neutral_pct'] <= 0.35
                          and 0.25 <= s['up_pct'] <= 0.45
                          and 0.25 <= s['down_pct'] <= 0.45]
        recommendations.append({
            'strategy': 'maintain_balance',
            'reason': '當前已平衡，選取中等比例的股票維持平衡',
            'stocks': [s['symbol'] for s in balanced_stocks],
            'count': len(balanced_stocks)
        })

    return {
        'target_dist': {
            'down': target_down,
            'neutral': target_neutral,
            'up': target_up
        },
        'current_dist': {
            'down': current_dist[0],
            'neutral': current_dist[1],
            'up': current_dist[2]
        },
        'gap_analysis': {
            'down_gap': down_gap,
            'neutral_gap': neutral_gap,
            'up_gap': up_gap
        },
        'recommendations': recommendations
    }


def print_summary_report(stocks: List[Dict]):
    """列印摘要報告"""
    print("\n" + "="*80)
    print("標籤分布分析報告")
    print("="*80)

    # 整體分布
    overall = analyze_overall_distribution(stocks)
    print(f"\n📊 整體標籤分布 ({overall['total_stocks']} 檔股票):")
    print(f"   總標籤數: {overall['total_labels']:,}")
    print(f"   Down:    {overall['down_count']:>10,} ({overall['down_pct']:>6.2%})")
    print(f"   Neutral: {overall['neutral_count']:>10,} ({overall['neutral_pct']:>6.2%})")
    print(f"   Up:      {overall['up_count']:>10,} ({overall['up_pct']:>6.2%})")

    # 按持平比例分組
    groups = group_by_neutral_ratio(stocks)
    print(f"\n📈 按持平比例分組:")
    for group_name, group_stocks in groups.items():
        if not group_stocks:
            continue
        group_dist = analyze_overall_distribution(group_stocks)
        print(f"\n   {group_name.upper()} ({len(group_stocks)} 檔):")
        print(f"      Down: {group_dist['down_pct']:.1%} | "
              f"Neutral: {group_dist['neutral_pct']:.1%} | "
              f"Up: {group_dist['up_pct']:.1%}")

    # 極端案例
    print(f"\n⚠️  極端案例:")
    very_low_neutral = [s for s in stocks if s['neutral_pct'] < 0.05]
    very_high_neutral = [s for s in stocks if s['neutral_pct'] > 0.50]

    if very_low_neutral:
        print(f"   持平 < 5%: {len(very_low_neutral)} 檔")
        for s in very_low_neutral[:3]:
            print(f"      {s['symbol']}: Down {s['down_pct']:.1%} | "
                  f"Neutral {s['neutral_pct']:.1%} | Up {s['up_pct']:.1%}")

    if very_high_neutral:
        print(f"   持平 > 50%: {len(very_high_neutral)} 檔")
        for s in very_high_neutral[:3]:
            print(f"      {s['symbol']}: Down {s['down_pct']:.1%} | "
                  f"Neutral {s['neutral_pct']:.1%} | Up {s['up_pct']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="分析標籤分布")
    parser.add_argument("--preprocessed-dir", required=True, help="預處理數據目錄")
    parser.add_argument("--mode", default="analyze", choices=["analyze", "recommend"],
                       help="模式: analyze=分析, recommend=推薦選取")
    parser.add_argument("--target-dist", default="0.30,0.40,0.30",
                       help="目標分布 (down,neutral,up), 例如: 0.30,0.40,0.30")
    parser.add_argument("--output", help="輸出 JSON 檔案路徑")

    args = parser.parse_args()

    # 載入數據
    stocks = load_all_stocks_metadata(args.preprocessed_dir)

    if not stocks:
        logging.error("找不到有效的股票數據！")
        return

    # 模式 1: 分析
    if args.mode == "analyze":
        print_summary_report(stocks)

    # 模式 2: 推薦選取
    elif args.mode == "recommend":
        target_dist = tuple(map(float, args.target_dist.split(',')))
        result = recommend_stock_selection(stocks, target_dist)

        print(f"\n{'='*80}")
        print("股票選取建議")
        print(f"{'='*80}")

        print(f"\n目標分布: Down {result['target_dist']['down']:.1%} | "
              f"Neutral {result['target_dist']['neutral']:.1%} | "
              f"Up {result['target_dist']['up']:.1%}")

        print(f"\n當前分布: Down {result['current_dist']['down']:.1%} | "
              f"Neutral {result['current_dist']['neutral']:.1%} | "
              f"Up {result['current_dist']['up']:.1%}")

        print(f"\n缺口分析:")
        print(f"   Down gap:    {result['gap_analysis']['down_gap']:>+7.1%}")
        print(f"   Neutral gap: {result['gap_analysis']['neutral_gap']:>+7.1%}")
        print(f"   Up gap:      {result['gap_analysis']['up_gap']:>+7.1%}")

        print(f"\n推薦策略:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"\n   策略 {i}: {rec['strategy']}")
            print(f"      理由: {rec['reason']}")
            print(f"      股票數: {rec['count']}")
            print(f"      股票代碼: {', '.join(rec['stocks'][:10])}" +
                  (f" ... ({rec['count'] - 10} 檔更多)" if rec['count'] > 10 else ""))

        # 保存結果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"結果已保存到: {args.output}")

    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
