#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_test_trend_stable_v2.py - 快速測試 Trend Stable v2.0 修補效果

【測試目標】
驗證 v2.0 三大改進是否生效：
  1. 絕對門檻地板：避免低波動誤判
  2. 進/出一致性：減少震盪邊緣抖動
  3. 方向一致性：過濾箱型內假趨勢

【驗證指標】
  - Neutral 佔比：震盪日應 >70%
  - 切換次數：相比 v1.0 應下降 70-85%
  - 趨勢段落：方向穩定，無頻繁翻轉

【使用方式】
python scripts/quick_test_trend_stable_v2.py --input data/temp/20250901.txt
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.financial_engineering import (
    ewma_volatility_professional,
    trend_labels_stable
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_single_txt(txt_path: str) -> pd.DataFrame:
    """快速解析單一 TXT（只取一檔股票測試）"""
    data = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 32:
                continue
            symbol = parts[0]
            timestamp = int(parts[1])

            # 只取第一檔股票
            if not data:
                target_symbol = symbol
            elif symbol != target_symbol:
                break

            # 解析 LOB 數據（簡化版，只取 mid price）
            try:
                bid1_p = float(parts[2])
                ask1_p = float(parts[12])
                mid = (bid1_p + ask1_p) / 2.0

                if mid > 0:
                    data.append({
                        'timestamp': timestamp,
                        'mid': mid
                    })
            except:
                continue

    df = pd.DataFrame(data)
    logging.info(f"載入 {len(df)} 筆數據")
    return df


def count_transitions(labels: np.ndarray) -> int:
    """計算標籤切換次數"""
    return np.sum(np.diff(labels) != 0)


def analyze_label_distribution(labels: pd.Series) -> dict:
    """分析標籤分布"""
    total = len(labels)
    counts = labels.value_counts()

    return {
        'down_count': int(counts.get(-1, 0)),
        'down_pct': float(counts.get(-1, 0) / total * 100),
        'neutral_count': int(counts.get(0, 0)),
        'neutral_pct': float(counts.get(0, 0) / total * 100),
        'up_count': int(counts.get(1, 0)),
        'up_pct': float(counts.get(1, 0) / total * 100),
        'transitions': count_transitions(labels.values)
    }


def test_trend_stable_v2(df: pd.DataFrame) -> None:
    """測試 Trend Stable v2.0"""

    # 1. 計算波動率
    close = df['mid']
    vol = ewma_volatility_professional(close, halflife=60, min_periods=20)

    # 2. v1.0 參數（舊版，較鬆）
    logging.info("\n" + "="*60)
    logging.info("【v1.0】舊版參數（較鬆）")
    logging.info("="*60)

    labels_v1 = trend_labels_stable(
        close=close,
        volatility=vol,
        lookforward=120,
        vol_multiplier=2.5,
        hysteresis_ratio=0.6,
        smooth_window=15,
        min_trend_duration=30,
        abs_floor_enter=0.0,  # 舊版無地板
        abs_floor_exit=0.0,
        dir_consistency=0.0   # 舊版無方向一致性
    )

    stats_v1 = analyze_label_distribution(labels_v1)
    logging.info(f"Down:    {stats_v1['down_count']:>6} ({stats_v1['down_pct']:>5.1f}%)")
    logging.info(f"Neutral: {stats_v1['neutral_count']:>6} ({stats_v1['neutral_pct']:>5.1f}%)")
    logging.info(f"Up:      {stats_v1['up_count']:>6} ({stats_v1['up_pct']:>5.1f}%)")
    logging.info(f"切換次數: {stats_v1['transitions']}")

    # 3. v2.0 參數（新版，較嚴）
    logging.info("\n" + "="*60)
    logging.info("【v2.0】新版參數（較嚴，含地板+方向一致性）")
    logging.info("="*60)

    labels_v2 = trend_labels_stable(
        close=close,
        volatility=vol,
        lookforward=120,
        vol_multiplier=2.5,
        hysteresis_ratio=0.6,
        smooth_window=21,      # ⬆️ 從15調到21
        min_trend_duration=45,  # ⬆️ 從30調到45
        abs_floor_enter=0.0020,  # 🆕 0.20% 地板
        abs_floor_exit=0.0010,   # 🆕 0.10% 地板
        dir_consistency=0.60     # 🆕 60% 方向一致性
    )

    stats_v2 = analyze_label_distribution(labels_v2)
    logging.info(f"Down:    {stats_v2['down_count']:>6} ({stats_v2['down_pct']:>5.1f}%)")
    logging.info(f"Neutral: {stats_v2['neutral_count']:>6} ({stats_v2['neutral_pct']:>5.1f}%)")
    logging.info(f"Up:      {stats_v2['up_count']:>6} ({stats_v2['up_pct']:>5.1f}%)")
    logging.info(f"切換次數: {stats_v2['transitions']}")

    # 4. 對比效果
    logging.info("\n" + "="*60)
    logging.info("【改進效果】v2.0 vs v1.0")
    logging.info("="*60)

    neutral_increase = stats_v2['neutral_pct'] - stats_v1['neutral_pct']
    transition_reduction = (stats_v1['transitions'] - stats_v2['transitions']) / stats_v1['transitions'] * 100

    logging.info(f"Neutral 提升: {neutral_increase:+.1f}% (目標: 震盪日 >+10%)")
    logging.info(f"切換次數減少: {transition_reduction:.1f}% (目標: >70%)")

    if neutral_increase >= 10 and transition_reduction >= 70:
        logging.info("✅ 修補成功！震盪抑制效果顯著")
    elif neutral_increase >= 5 and transition_reduction >= 50:
        logging.info("⚠️ 修補部分生效，建議進一步調嚴參數")
    else:
        logging.info("❌ 修補效果不明顯，可能需要更嚴格參數")

    # 5. 視覺化對比（可選）
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # 價格
        axes[0].plot(close.values, linewidth=0.8, alpha=0.7)
        axes[0].set_ylabel('Mid Price')
        axes[0].set_title('Price vs Label Comparison')
        axes[0].grid(True, alpha=0.3)

        # v1.0 標籤
        axes[1].plot(labels_v1.values, linewidth=0.8, alpha=0.7, color='orange')
        axes[1].set_ylabel('v1.0 Label')
        axes[1].set_ylim(-1.5, 1.5)
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].text(0.02, 0.95, f'切換: {stats_v1["transitions"]}次',
                     transform=axes[1].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # v2.0 標籤
        axes[2].plot(labels_v2.values, linewidth=0.8, alpha=0.7, color='green')
        axes[2].set_ylabel('v2.0 Label')
        axes[2].set_xlabel('Time Index')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2].grid(True, alpha=0.3)
        axes[2].text(0.02, 0.95, f'切換: {stats_v2["transitions"]}次 ({transition_reduction:.0f}%↓)',
                     transform=axes[2].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()

        output_path = Path('results/trend_stable_v2_comparison.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"\n📊 視覺化結果已保存: {output_path}")

    except Exception as e:
        logging.warning(f"視覺化失敗（可忽略）: {e}")


def main():
    parser = argparse.ArgumentParser(description='快速測試 Trend Stable v2.0 修補效果')
    parser.add_argument('--input', type=str, required=True, help='TXT 檔案路徑')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"檔案不存在: {args.input}")
        sys.exit(1)

    logging.info(f"開始測試: {args.input}")

    # 載入數據
    df = parse_single_txt(args.input)

    if len(df) < 500:
        logging.error(f"數據點過少 ({len(df)})，無法有效測試")
        sys.exit(1)

    # 執行測試
    test_trend_stable_v2(df)

    logging.info("\n✅ 測試完成！")


if __name__ == '__main__':
    main()
