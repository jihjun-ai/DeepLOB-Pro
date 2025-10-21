"""
批量分析多支股票的標籤分布
==================================================
檢查不同股票的「平」標籤比例差異

使用方式：
    python scripts/analyze_multi_stock_labels.py --config configs/config_pro_v5_ml_optimal.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(message)s')

from scripts.extract_tw_stock_data_v5 import (
    parse_line, dedup_by_timestamp_keep_last, aggregate_chunks_of_10,
    ewma_vol, tb_labels, load_config
)


def analyze_file(file_path: str, config: dict, max_stocks: int = 10):
    """分析單一文件中多支股票的標籤分布"""

    logging.info(f"\n{'='*60}")
    logging.info(f"分析文件: {os.path.basename(file_path)}")
    logging.info(f"{'='*60}")

    # 讀取數據（按股票分組）
    symbol_data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sym, t, rec = parse_line(line)
            if rec is None:
                continue
            if sym not in symbol_data:
                symbol_data[sym] = []
            symbol_data[sym].append((t, rec))

    if not symbol_data:
        logging.warning("沒有有效數據")
        return []

    logging.info(f"找到 {len(symbol_data)} 支股票")

    # 分析每支股票
    results = []
    tb_cfg = config['triple_barrier']

    for idx, (sym, rows) in enumerate(list(symbol_data.items())[:max_stocks]):
        # 去重和聚合
        rows_dedup = dedup_by_timestamp_keep_last(rows)
        Xd, mids = aggregate_chunks_of_10(rows_dedup)

        if len(mids) < 100:
            continue

        # 計算標籤
        close = pd.Series(mids, name='close')
        vol = ewma_vol(close, halflife=config['volatility']['halflife'])

        try:
            tb_df = tb_labels(
                close=close,
                vol=vol,
                pt_mult=tb_cfg['pt_multiplier'],
                sl_mult=tb_cfg['sl_multiplier'],
                max_holding=tb_cfg['max_holding'],
                min_return=tb_cfg['min_return'],
                day_end_idx=len(close) - 1
            )
        except Exception as e:
            logging.warning(f"  {sym}: 標籤計算失敗 ({e})")
            continue

        # 統計
        label_counts = tb_df['y'].value_counts()
        why_counts = tb_df['why'].value_counts()
        total = len(tb_df)

        vol_mean = vol.replace([np.inf, -np.inf], np.nan).dropna().mean()

        result = {
            'symbol': sym,
            'n_points': len(mids),
            'vol_mean': vol_mean,
            'label_-1': label_counts.get(-1, 0),
            'label_0': label_counts.get(0, 0),
            'label_+1': label_counts.get(1, 0),
            'pct_-1': label_counts.get(-1, 0) / total * 100,
            'pct_0': label_counts.get(0, 0) / total * 100,
            'pct_+1': label_counts.get(1, 0) / total * 100,
            'trigger_up': why_counts.get('up', 0),
            'trigger_down': why_counts.get('down', 0),
            'trigger_time': why_counts.get('time', 0),
            'time_pct': why_counts.get('time', 0) / total * 100
        }

        results.append(result)

        logging.info(f"  {sym}: {len(mids)} 點, vol={vol_mean*100:.3f}%, "
                    f"平={result['pct_0']:.1f}%, time={result['time_pct']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='批量分析股票標籤分布')
    parser.add_argument('--config', default='./configs/config_pro_v5_ml_optimal.yaml')
    parser.add_argument('--data-dir', default='./data/temp')
    parser.add_argument('--sample-files', type=int, default=3)
    parser.add_argument('--max-stocks', type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "="*80)
    print("📊 多股票標籤分布分析")
    print("="*80)

    # 找數據文件
    data_files = sorted(Path(args.data_dir).glob('*.txt'))[:args.sample_files]

    if not data_files:
        logging.error(f"在 {args.data_dir} 找不到數據")
        return

    logging.info(f"將分析 {len(data_files)} 個文件，每個文件最多 {args.max_stocks} 支股票")

    # 分析所有文件
    all_results = []
    for file_path in data_files:
        results = analyze_file(str(file_path), config, args.max_stocks)
        all_results.extend(results)

    if not all_results:
        logging.error("沒有有效結果")
        return

    # 彙總統計
    df = pd.DataFrame(all_results)

    print(f"\n{'='*80}")
    print("【標籤分布統計】")
    print("="*80)

    print(f"\n總共分析 {len(df)} 支股票")

    # 「平」標籤分布
    print(f"\n【「持平」(label=0) 比例分布】")
    print(f"  平均值: {df['pct_0'].mean():.1f}%")
    print(f"  中位數: {df['pct_0'].median():.1f}%")
    print(f"  最小值: {df['pct_0'].min():.1f}% ({df.loc[df['pct_0'].idxmin(), 'symbol']})")
    print(f"  最大值: {df['pct_0'].max():.1f}% ({df.loc[df['pct_0'].idxmax(), 'symbol']})")
    print(f"  標準差: {df['pct_0'].std():.1f}%")

    # 分組統計
    low_neutral = df[df['pct_0'] < 20]
    mid_neutral = df[(df['pct_0'] >= 20) & (df['pct_0'] <= 50)]
    high_neutral = df[df['pct_0'] > 50]

    print(f"\n【分組統計】")
    print(f"  低「平」(<20%):  {len(low_neutral)} 支 ({len(low_neutral)/len(df)*100:.1f}%)")
    print(f"  中「平」(20-50%): {len(mid_neutral)} 支 ({len(mid_neutral)/len(df)*100:.1f}%)")
    print(f"  高「平」(>50%):  {len(high_neutral)} 支 ({len(high_neutral)/len(df)*100:.1f}%)")

    # 波動率 vs 標籤分布
    print(f"\n【波動率 vs 「持平」比例】")
    corr = df['vol_mean'].corr(df['pct_0'])
    print(f"  相關係數: {corr:.3f}")

    if abs(corr) > 0.5:
        if corr > 0:
            print(f"  ⚠️  波動率越高，「持平」越多（異常）")
        else:
            print(f"  ✓ 波動率越高，「持平」越少（正常）")

    # Time trigger vs 標籤
    print(f"\n【Time Trigger 比例 vs 「持平」比例】")
    corr_time = df['time_pct'].corr(df['pct_0'])
    print(f"  相關係數: {corr_time:.3f}")
    print(f"  平均 time trigger: {df['time_pct'].mean():.1f}%")

    # 顯示問題股票（「平」過低）
    if len(low_neutral) > 0:
        print(f"\n{'='*80}")
        print(f"⚠️  「持平」過低的股票 (<20%)")
        print("="*80)
        for _, row in low_neutral.head(10).iterrows():
            print(f"  {row['symbol']}: 平={row['pct_0']:.1f}%, "
                  f"vol={row['vol_mean']*100:.3f}%, time={row['time_pct']:.1f}%")

    # 顯示問題股票（「平」過高）
    if len(high_neutral) > 0:
        print(f"\n{'='*80}")
        print(f"⚠️  「持平」過高的股票 (>50%)")
        print("="*80)
        for _, row in high_neutral.head(10).iterrows():
            print(f"  {row['symbol']}: 平={row['pct_0']:.1f}%, "
                  f"vol={row['vol_mean']*100:.3f}%, time={row['time_pct']:.1f}%")

    # 總體建議
    print(f"\n{'='*80}")
    print("💡 總體建議")
    print("="*80)

    avg_neutral = df['pct_0'].mean()

    if avg_neutral < 25:
        print(f"\n平均「持平」比例 {avg_neutral:.1f}% 偏低")
        print(f"建議：")
        print(f"  1. 提高 min_return 到 0.002-0.003")
        print(f"  2. 放寬 PT/SL 倍數到 4.5-5.0")
        print(f"  3. 啟用 pt_sl_check_min_return（改進版函數）")
    elif avg_neutral > 45:
        print(f"\n平均「持平」比例 {avg_neutral:.1f}% 偏高")
        print(f"建議：")
        print(f"  1. 降低 min_return 到 0.001-0.0012")
        print(f"  2. 縮小 PT/SL 倍數到 2.5-3.0")
    else:
        print(f"\n✅ 平均「持平」比例 {avg_neutral:.1f}% 合理")
        print(f"但注意：不同股票差異很大（標準差 {df['pct_0'].std():.1f}%）")
        print(f"考慮：")
        print(f"  - 使用自適應 barrier（根據個股波動率調整）")
        print(f"  - 或接受這種異質性（不同股票本來就不同）")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

