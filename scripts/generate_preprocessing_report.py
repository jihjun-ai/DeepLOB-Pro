# -*- coding: utf-8 -*-
"""
generate_preprocessing_report.py - 預處理整體報告生成工具
=============================================================================
【更新日期】2025-10-21
【版本說明】v1.0 - 彙總所有預處理結果

功能：
  1. 掃描 preprocessed_v5/daily/ 中的所有 summary.json
  2. 彙總統計資訊
  3. 生成 CSV 和 JSON 報告
  4. 輸出控制台摘要

使用方式：
  python scripts/generate_preprocessing_report.py \
      --preprocessed-dir ./data/preprocessed_v5

輸出：
  - data/preprocessed_v5/reports/overall_summary.json
  - data/preprocessed_v5/reports/daily_statistics.csv
  - data/preprocessed_v5/reports/filter_decisions.csv

版本：v1.0
更新：2025-10-21
"""

import os
import json
import argparse
import logging
import glob
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_all_summaries(preprocessed_dir: str) -> List[Dict[str, Any]]:
    """載入所有 summary.json"""
    daily_dir = os.path.join(preprocessed_dir, "daily")

    if not os.path.exists(daily_dir):
        logging.error(f"目錄不存在: {daily_dir}")
        return []

    summaries = []

    for summary_file in sorted(glob.glob(os.path.join(daily_dir, "*", "summary.json"))):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
                summaries.append(summary)
        except Exception as e:
            logging.warning(f"無法讀取 {summary_file}: {e}")

    logging.info(f"載入了 {len(summaries)} 個每日摘要")

    return summaries


def generate_overall_report(summaries: List[Dict], output_dir: str):
    """生成整體報告"""
    if not summaries:
        logging.warning("無可用摘要，跳過報告生成")
        return

    # 彙總統計
    total_symbols = sum(s['total_symbols'] for s in summaries)
    total_passed = sum(s['passed_filter'] for s in summaries)
    total_filtered = sum(s['filtered_out'] for s in summaries)

    # 每日統計 DataFrame
    daily_stats = []

    for s in summaries:
        daily_stats.append({
            'date': s['date'],
            'total_symbols': s['total_symbols'],
            'passed_filter': s['passed_filter'],
            'filtered_out': s['filtered_out'],
            'pass_rate': s['passed_filter'] / s['total_symbols'] * 100 if s['total_symbols'] > 0 else 0,
            'filter_threshold': s['filter_threshold'],
            'filter_method': s['filter_method'],
            'range_min': s['volatility_distribution']['min'],
            'range_max': s['volatility_distribution']['max'],
            'range_mean': s['volatility_distribution']['mean'],
            'range_P50': s['volatility_distribution']['P50'],
            'pred_down': s['predicted_label_dist']['down'],
            'pred_neutral': s['predicted_label_dist']['neutral'],
            'pred_up': s['predicted_label_dist']['up']
        })

    df_daily = pd.DataFrame(daily_stats)

    # 過濾決策 DataFrame
    filter_decisions = []

    for s in summaries:
        filter_decisions.append({
            'date': s['date'],
            'filter_method': s['filter_method'],
            'filter_threshold': s['filter_threshold'],
            'P10': s['volatility_distribution']['P10'],
            'P25': s['volatility_distribution']['P25'],
            'P50': s['volatility_distribution']['P50'],
            'P75': s['volatility_distribution']['P75'],
            'pred_neutral_pct': s['predicted_label_dist']['neutral'] * 100
        })

    df_filter = pd.DataFrame(filter_decisions)

    # 整體摘要
    overall_summary = {
        "generated_at": datetime.now().isoformat(),
        "n_days": len(summaries),
        "date_range": {
            "start": summaries[0]['date'] if summaries else None,
            "end": summaries[-1]['date'] if summaries else None
        },

        "total_statistics": {
            "total_symbols": int(total_symbols),
            "passed_filter": int(total_passed),
            "filtered_out": int(total_filtered),
            "overall_pass_rate": float(total_passed / total_symbols * 100) if total_symbols > 0 else 0
        },

        "filter_threshold_distribution": {
            "methods": df_filter['filter_method'].value_counts().to_dict(),
            "threshold_mean": float(df_filter['filter_threshold'].mean()),
            "threshold_std": float(df_filter['filter_threshold'].std()),
            "threshold_min": float(df_filter['filter_threshold'].min()),
            "threshold_max": float(df_filter['filter_threshold'].max())
        },

        "volatility_statistics": {
            "range_mean_avg": float(df_daily['range_mean'].mean()),
            "range_mean_std": float(df_daily['range_mean'].std()),
            "range_P50_avg": float(df_daily['range_P50'].mean())
        },

        "predicted_label_distribution": {
            "down_avg": float(df_daily['pred_down'].mean()),
            "neutral_avg": float(df_daily['pred_neutral'].mean()),
            "up_avg": float(df_daily['pred_up'].mean())
        }
    }

    # 保存報告
    os.makedirs(output_dir, exist_ok=True)

    # JSON 摘要
    json_path = os.path.join(output_dir, "overall_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    logging.info(f"✅ 已保存: {json_path}")

    # 每日統計 CSV
    csv_daily_path = os.path.join(output_dir, "daily_statistics.csv")
    df_daily.to_csv(csv_daily_path, index=False, encoding='utf-8-sig')
    logging.info(f"✅ 已保存: {csv_daily_path}")

    # 過濾決策 CSV
    csv_filter_path = os.path.join(output_dir, "filter_decisions.csv")
    df_filter.to_csv(csv_filter_path, index=False, encoding='utf-8-sig')
    logging.info(f"✅ 已保存: {csv_filter_path}")

    # 控制台輸出
    print_console_summary(overall_summary, df_daily, df_filter)


def print_console_summary(summary: Dict, df_daily: pd.DataFrame, df_filter: pd.DataFrame):
    """控制台輸出摘要"""
    logging.info("\n" + "="*60)
    logging.info("📊 預處理整體報告")
    logging.info("="*60)

    logging.info(f"處理期間: {summary['date_range']['start']} ~ {summary['date_range']['end']}")
    logging.info(f"總天數: {summary['n_days']} 天")

    logging.info(f"\n📈 股票統計:")
    logging.info(f"  總 symbol-day 數: {summary['total_statistics']['total_symbols']:,}")
    logging.info(f"  通過過濾: {summary['total_statistics']['passed_filter']:,} ({summary['total_statistics']['overall_pass_rate']:.1f}%)")
    logging.info(f"  被過濾: {summary['total_statistics']['filtered_out']:,}")

    logging.info(f"\n🎯 過濾閾值統計:")
    logging.info(f"  平均閾值: {summary['filter_threshold_distribution']['threshold_mean']:.4f}")
    logging.info(f"  閾值範圍: [{summary['filter_threshold_distribution']['threshold_min']:.4f}, {summary['filter_threshold_distribution']['threshold_max']:.4f}]")
    logging.info(f"  選擇方法分布:")
    for method, count in summary['filter_threshold_distribution']['methods'].items():
        logging.info(f"    {method}: {count} 天")

    logging.info(f"\n📊 波動率統計:")
    logging.info(f"  平均震盪幅度: {summary['volatility_statistics']['range_mean_avg']*100:.2f}%")
    logging.info(f"  中位數震盪 (P50): {summary['volatility_statistics']['range_P50_avg']*100:.2f}%")

    logging.info(f"\n🏷️ 預測標籤分布 (平均):")
    pred = summary['predicted_label_distribution']
    logging.info(f"  Down: {pred['down_avg']*100:.1f}%")
    logging.info(f"  Neutral: {pred['neutral_avg']*100:.1f}%")
    logging.info(f"  Up: {pred['up_avg']*100:.1f}%")

    # 檢查是否符合目標
    target_neutral = 40.0
    actual_neutral = pred['neutral_avg'] * 100

    if 30 <= actual_neutral <= 45:
        logging.info(f"  ✅ Neutral 比例符合目標 (30-45%)")
    elif actual_neutral > 50:
        logging.warning(f"  ⚠️ Neutral 比例過高 ({actual_neutral:.1f}% > 50%)，建議提高過濾閾值")
    elif actual_neutral < 20:
        logging.warning(f"  ⚠️ Neutral 比例過低 ({actual_neutral:.1f}% < 20%)，建議降低過濾閾值")

    logging.info("="*60 + "\n")


def parse_args():
    p = argparse.ArgumentParser("generate_preprocessing_report", description="預處理整體報告生成工具")
    p.add_argument("--preprocessed-dir", default="./data/preprocessed_v5", help="預處理結果目錄")
    return p.parse_args()


def main():
    args = parse_args()

    preprocessed_dir = args.preprocessed_dir

    if not os.path.exists(preprocessed_dir):
        logging.error(f"預處理目錄不存在: {preprocessed_dir}")
        return 1

    # 載入摘要
    summaries = load_all_summaries(preprocessed_dir)

    if not summaries:
        logging.error("沒有可用的摘要數據")
        return 1

    # 生成報告
    reports_dir = os.path.join(preprocessed_dir, "reports")
    generate_overall_report(summaries, reports_dir)

    logging.info("\n✅ 報告生成完成！\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
