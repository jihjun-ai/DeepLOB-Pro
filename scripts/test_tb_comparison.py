"""
Triple-Barrier 標籤方法比較工具
==================================================
比較不同參數設定對「平」標籤比例的影響

使用方式：
    python scripts/test_tb_comparison.py --config configs/config_pro_v5_ml_optimal.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

# 添加項目根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from scripts.extract_tw_stock_data_v5 import (
    parse_line, dedup_by_timestamp_keep_last, aggregate_chunks_of_10,
    ewma_vol, tb_labels, load_config
)
from scripts.tb_labels_improved import tb_labels_v2, label_horizon


def load_sample_data(data_dir: str, n_files: int = 2):
    """載入範例數據"""
    data_files = sorted(Path(data_dir).glob('*.txt'))[:n_files]

    if not data_files:
        raise FileNotFoundError(f"在 {data_dir} 找不到數據文件")

    all_mids = []

    for file_path in data_files:
        logging.info(f"載入: {file_path.name}")

        symbol_data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sym, t, rec = parse_line(line)
                if rec is None:
                    continue
                if sym not in symbol_data:
                    symbol_data[sym] = []
                symbol_data[sym].append((t, rec))

        # 取第一個股票
        if symbol_data:
            sample_symbol = list(symbol_data.keys())[0]
            rows = symbol_data[sample_symbol]
            rows_dedup = dedup_by_timestamp_keep_last(rows)
            Xd, mids = aggregate_chunks_of_10(rows_dedup)

            if len(mids) >= 100:
                all_mids.extend(mids)
                logging.info(f"  {sample_symbol}: {len(mids)} 個點")

    if not all_mids:
        raise ValueError("沒有足夠的數據")

    return pd.Series(all_mids, name='close')


def test_labeling_method(close: pd.Series, vol: pd.Series, method_name: str,
                         config: dict, **kwargs) -> dict:
    """測試單一標籤方法"""

    if method_name == 'original':
        # 原始方法
        tb_df = tb_labels(
            close=close,
            vol=vol,
            pt_mult=config['pt_multiplier'],
            sl_mult=config['sl_multiplier'],
            max_holding=config['max_holding'],
            min_return=config['min_return'],
            day_end_idx=None
        )

    elif method_name == 'improved':
        # 改進方法（PT/SL 也檢查 min_return）
        tb_df = tb_labels_v2(
            close=close,
            vol=vol,
            pt_mult=config['pt_multiplier'],
            sl_mult=config['sl_multiplier'],
            max_holding=config['max_holding'],
            min_return=config['min_return'],
            pt_sl_check_min_return=True,
            mode='path'
        )

    elif method_name == 'horizon':
        # 終點導向
        tb_df = label_horizon(
            close=close,
            horizon=config['max_holding'],
            min_return=config['min_return']
        )

    else:
        raise ValueError(f"未知方法: {method_name}")

    # 統計
    label_counts = tb_df['y'].value_counts()
    total = len(tb_df)

    result = {
        'method': method_name,
        'total': total,
        'down': label_counts.get(-1, 0),
        'neutral': label_counts.get(0, 0),
        'up': label_counts.get(1, 0),
        'down_pct': label_counts.get(-1, 0) / total * 100,
        'neutral_pct': label_counts.get(0, 0) / total * 100,
        'up_pct': label_counts.get(1, 0) / total * 100,
    }

    # 觸發原因（如果有）
    if 'why' in tb_df.columns:
        why_counts = tb_df['why'].value_counts()
        result['trigger_up'] = why_counts.get('up', 0)
        result['trigger_down'] = why_counts.get('down', 0)
        result['trigger_time'] = why_counts.get('time', 0)
        result['trigger_both'] = why_counts.get('both', 0)

    return result, tb_df


def main():
    parser = argparse.ArgumentParser(description='Triple-Barrier 標籤方法比較')
    parser.add_argument('--config', default='./configs/config_pro_v5_ml_optimal.yaml',
                       help='配置文件')
    parser.add_argument('--data-dir', default='./data/temp', help='數據目錄')
    parser.add_argument('--sample-files', type=int, default=2, help='使用幾個文件')
    args = parser.parse_args()

    # 載入配置
    config = load_config(args.config)
    tb_cfg = config['triple_barrier']

    print("\n" + "="*80)
    print("📊 Triple-Barrier 標籤方法比較測試")
    print("="*80)

    print(f"\n【當前配置】")
    print(f"  配置文件: {args.config}")
    print(f"  PT 倍數: {tb_cfg['pt_multiplier']}")
    print(f"  SL 倍數: {tb_cfg['sl_multiplier']}")
    print(f"  最大持有: {tb_cfg['max_holding']} bars")
    print(f"  最小報酬: {tb_cfg['min_return']} ({tb_cfg['min_return']*100:.2f}%)")

    # 載入數據
    print(f"\n【載入數據】")
    close = load_sample_data(args.data_dir, args.sample_files)
    print(f"  總數據點: {len(close)}")
    print(f"  價格範圍: {close.min():.2f} - {close.max():.2f}")

    # 計算波動率
    vol = ewma_vol(close, halflife=config['volatility']['halflife'])
    print(f"  波動率 (均值): {vol.mean()*100:.4f}%")

    # 測試不同方法
    print(f"\n{'='*80}")
    print("【方法比較】")
    print("="*80)

    results = []
    dataframes = {}

    methods = [
        ('original', '原始 TB（PT/SL 一律 ±1）'),
        ('improved', '改進 TB（PT/SL 也檢查 min_return）'),
        ('horizon', '終點導向（固定視窗）')
    ]

    for method_id, method_desc in methods:
        print(f"\n測試方法: {method_desc}")
        result, tb_df = test_labeling_method(close, vol, method_id, tb_cfg)
        results.append(result)
        dataframes[method_id] = tb_df

        print(f"  ✓ 完成")

    # 製作比較表
    print(f"\n{'='*80}")
    print("【標籤分布比較】")
    print("="*80)

    table_data = []
    for r in results:
        table_data.append([
            r['method'],
            f"{r['down']} ({r['down_pct']:.1f}%)",
            f"{r['neutral']} ({r['neutral_pct']:.1f}%)",
            f"{r['up']} ({r['up_pct']:.1f}%)",
            r['total']
        ])

    headers = ['方法', '下跌 (-1)', '持平 (0)', '上漲 (+1)', '總計']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # 觸發原因比較
    print(f"\n{'='*80}")
    print("【觸發原因分布】")
    print("="*80)

    trigger_data = []
    for r in results:
        if 'trigger_up' in r:
            total = r['total']
            trigger_data.append([
                r['method'],
                f"{r['trigger_up']} ({r['trigger_up']/total*100:.1f}%)",
                f"{r['trigger_down']} ({r['trigger_down']/total*100:.1f}%)",
                f"{r['trigger_time']} ({r['trigger_time']/total*100:.1f}%)",
                f"{r.get('trigger_both', 0)} ({r.get('trigger_both', 0)/total*100:.1f}%)"
            ])

    if trigger_data:
        headers = ['方法', 'PT 觸發', 'SL 觸發', '時間到期', '同時觸發']
        print(tabulate(trigger_data, headers=headers, tablefmt='grid'))

    # 關鍵發現
    print(f"\n{'='*80}")
    print("🔍 關鍵發現")
    print("="*80)

    original_neutral = results[0]['neutral_pct']
    improved_neutral = results[1]['neutral_pct']
    horizon_neutral = results[2]['neutral_pct']

    print(f"\n1. 「持平」標籤比例：")
    print(f"   原始方法:   {original_neutral:.1f}%")
    print(f"   改進方法:   {improved_neutral:.1f}% (增加 {improved_neutral-original_neutral:+.1f}%)")
    print(f"   終點導向:   {horizon_neutral:.1f}% (增加 {horizon_neutral-original_neutral:+.1f}%)")

    if improved_neutral < 25:
        print(f"\n⚠️  改進方法的「持平」仍然偏低 (<25%)")
        print(f"   建議：")
        print(f"   - 提高 min_return 到 0.003-0.005 (0.3%-0.5%)")
        print(f"   - 或放寬 PT/SL 倍數到 5.0-6.0")
    elif improved_neutral > 50:
        print(f"\n⚠️  改進方法的「持平」過高 (>50%)")
        print(f"   建議：")
        print(f"   - 降低 min_return 到 0.001-0.0015 (0.1%-0.15%)")
        print(f"   - 或縮小 PT/SL 倍數到 3.0-3.5")
    else:
        print(f"\n✅ 改進方法的「持平」比例合理 (25-50%)")

    # 參數建議
    print(f"\n{'='*80}")
    print("💡 參數調整建議")
    print("="*80)

    if improved_neutral < 30:
        new_min_return = tb_cfg['min_return'] * 1.5
        print(f"\n目標：增加「持平」到 30-40%")
        print(f"建議配置：")
        print(f"  min_return: {new_min_return:.4f} (當前 {tb_cfg['min_return']:.4f} × 1.5)")
        print(f"  pt_multiplier: {tb_cfg['pt_multiplier']*1.2:.1f} (當前 {tb_cfg['pt_multiplier']} × 1.2)")
        print(f"  sl_multiplier: {tb_cfg['sl_multiplier']*1.2:.1f} (當前 {tb_cfg['sl_multiplier']} × 1.2)")
    elif improved_neutral > 45:
        new_min_return = tb_cfg['min_return'] * 0.7
        print(f"\n目標：減少「持平」到 30-40%")
        print(f"建議配置：")
        print(f"  min_return: {new_min_return:.4f} (當前 {tb_cfg['min_return']:.4f} × 0.7)")
        print(f"  pt_multiplier: {tb_cfg['pt_multiplier']*0.85:.1f} (當前 {tb_cfg['pt_multiplier']} × 0.85)")
        print(f"  sl_multiplier: {tb_cfg['sl_multiplier']*0.85:.1f} (當前 {tb_cfg['sl_multiplier']} × 0.85)")
    else:
        print(f"\n✅ 當前參數已經不錯！")
        print(f"   可以直接使用「改進方法」生成數據")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

