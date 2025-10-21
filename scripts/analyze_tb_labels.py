"""
Triple-Barrier 標籤診斷工具
==================================================
分析當前 tb_labels 的觸發模式和標籤分布

使用方式：
    python scripts/analyze_tb_labels.py --data-dir ./data/temp --sample-days 3
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 添加項目根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 從 extract_tw_stock_data_v5.py 導入必要函數
from scripts.extract_tw_stock_data_v5 import (
    parse_line, dedup_by_timestamp_keep_last, aggregate_chunks_of_10,
    ewma_vol, tb_labels, load_config,
    TRADING_START, TRADING_END
)


def analyze_single_day(file_path: str, config: dict) -> dict:
    """分析單日數據的 tb_labels 分布"""

    logging.info(f"\n{'='*60}")
    logging.info(f"分析文件: {os.path.basename(file_path)}")
    logging.info(f"{'='*60}")

    # 讀取並解析數據
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
        return None

    # 分析第一個股票（示範）
    sample_symbol = list(symbol_data.keys())[0]
    rows = symbol_data[sample_symbol]

    logging.info(f"示範股票: {sample_symbol}, 原始事件數: {len(rows)}")

    # 去重和聚合
    rows_dedup = dedup_by_timestamp_keep_last(rows)
    Xd, mids = aggregate_chunks_of_10(rows_dedup)

    if len(mids) < 100:
        logging.warning(f"聚合後數據點太少: {len(mids)}")
        return None

    logging.info(f"聚合後數據點: {len(mids)}")

    # 計算波動率和標籤
    close = pd.Series(mids, name='close')
    vol = ewma_vol(close, halflife=config['volatility']['halflife'])

    tb_cfg = config['triple_barrier']

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
        logging.error(f"tb_labels 失敗: {e}")
        return None

    # 統計分析
    stats = {
        'symbol': sample_symbol,
        'n_points': len(close),
        'n_labels': len(tb_df),
    }

    # 觸發原因分布
    why_counts = tb_df['why'].value_counts().to_dict()
    stats['trigger_counts'] = why_counts
    stats['trigger_pct'] = {k: v/len(tb_df)*100 for k, v in why_counts.items()}

    # 標籤分布
    label_counts = tb_df['y'].value_counts().to_dict()
    stats['label_counts'] = label_counts
    stats['label_pct'] = {k: v/len(tb_df)*100 for k, v in label_counts.items()}

    # 每種觸發方式的標籤分布
    stats['trigger_label_breakdown'] = {}
    for trigger in ['up', 'down', 'time']:
        if trigger in tb_df['why'].values:
            subset = tb_df[tb_df['why'] == trigger]
            label_dist = subset['y'].value_counts().to_dict()
            stats['trigger_label_breakdown'][trigger] = label_dist

    # time trigger 的報酬分布
    time_subset = tb_df[tb_df['why'] == 'time']
    if len(time_subset) > 0:
        ret_abs = time_subset['ret'].abs()
        stats['time_ret_stats'] = {
            'mean': float(ret_abs.mean()),
            'median': float(ret_abs.median()),
            'p25': float(ret_abs.quantile(0.25)),
            'p75': float(ret_abs.quantile(0.75)),
            'p90': float(ret_abs.quantile(0.90)),
            'below_min_return': int((ret_abs < tb_cfg['min_return']).sum()),
            'total': len(time_subset)
        }

    # 波動率統計
    vol_clean = vol.replace([np.inf, -np.inf], np.nan).dropna()
    stats['vol_stats'] = {
        'mean': float(vol_clean.mean()),
        'median': float(vol_clean.median()),
        'min': float(vol_clean.min()),
        'max': float(vol_clean.max()),
        'near_zero': int((vol_clean < 1e-6).sum()),
        'total': len(vol_clean)
    }

    # 列印詳細報告
    print_analysis_report(stats, tb_cfg)

    return stats


def print_analysis_report(stats: dict, config: dict):
    """列印分析報告"""

    print(f"\n{'='*60}")
    print(f"📊 Triple-Barrier 標籤診斷報告")
    print(f"{'='*60}")

    print(f"\n【基本資訊】")
    print(f"  股票代碼: {stats['symbol']}")
    print(f"  數據點數: {stats['n_points']}")
    print(f"  標籤數量: {stats['n_labels']}")

    print(f"\n【當前參數】")
    print(f"  PT 倍數: {config['pt_multiplier']}")
    print(f"  SL 倍數: {config['sl_multiplier']}")
    print(f"  最大持有: {config['max_holding']} bars")
    print(f"  最小報酬: {config['min_return']} ({config['min_return']*100:.3f}%)")

    print(f"\n【觸發原因分布】")
    for trigger, count in stats['trigger_counts'].items():
        pct = stats['trigger_pct'][trigger]
        print(f"  {trigger:>6}: {count:>6} ({pct:>5.1f}%)")

    print(f"\n【標籤分布（整體）】")
    for label, count in sorted(stats['label_counts'].items()):
        pct = stats['label_pct'][label]
        label_name = {-1: '下跌', 0: '持平', 1: '上漲'}[label]
        print(f"  {label_name} ({label:+2}): {count:>6} ({pct:>5.1f}%)")

    print(f"\n【各觸發方式的標籤分布】")
    for trigger, label_dist in stats['trigger_label_breakdown'].items():
        total = sum(label_dist.values())
        print(f"  {trigger} (n={total}):")
        for label in sorted(label_dist.keys()):
            count = label_dist[label]
            pct = count / total * 100
            label_name = {-1: '下跌', 0: '持平', 1: '上漲'}[label]
            print(f"    {label_name} ({label:+2}): {count:>5} ({pct:>5.1f}%)")

    if 'time_ret_stats' in stats:
        print(f"\n【time trigger 報酬統計（絕對值）】")
        trs = stats['time_ret_stats']
        print(f"  平均值: {trs['mean']*100:.4f}%")
        print(f"  中位數: {trs['median']*100:.4f}%")
        print(f"  25% 分位: {trs['p25']*100:.4f}%")
        print(f"  75% 分位: {trs['p75']*100:.4f}%")
        print(f"  90% 分位: {trs['p90']*100:.4f}%")
        print(f"  低於 min_return: {trs['below_min_return']}/{trs['total']} ({trs['below_min_return']/trs['total']*100:.1f}%)")

    print(f"\n【波動率統計】")
    vs = stats['vol_stats']
    print(f"  平均值: {vs['mean']*100:.4f}%")
    print(f"  中位數: {vs['median']*100:.4f}%")
    print(f"  最小值: {vs['min']*100:.6f}%")
    print(f"  最大值: {vs['max']*100:.4f}%")
    print(f"  接近零 (<1e-6): {vs['near_zero']}/{vs['total']}")

    print(f"\n{'='*60}")
    print(f"🔍 關鍵發現：")
    print(f"{'='*60}")

    # 自動診斷
    issues = []

    # 檢查 "平" 比例
    if 0 in stats['label_pct']:
        zero_pct = stats['label_pct'][0]
        if zero_pct < 10:
            issues.append(f"❌ '持平' 標籤比例過低 ({zero_pct:.1f}%)")
        elif zero_pct < 20:
            issues.append(f"⚠️  '持平' 標籤比例偏低 ({zero_pct:.1f}%)")
        else:
            issues.append(f"✅ '持平' 標籤比例合理 ({zero_pct:.1f}%)")
    else:
        issues.append(f"❌ 沒有 '持平' 標籤！")

    # 檢查 PT/SL 觸發率
    pt_sl_pct = stats['trigger_pct'].get('up', 0) + stats['trigger_pct'].get('down', 0)
    if pt_sl_pct > 80:
        issues.append(f"⚠️  PT/SL 觸發率過高 ({pt_sl_pct:.1f}%)，可能門檻太窄")

    # 檢查波動率
    if vs['near_zero'] > vs['total'] * 0.1:
        issues.append(f"⚠️  {vs['near_zero']/vs['total']*100:.1f}% 的波動率接近零，可能導致門檻過小")

    # 檢查 time trigger 的利用
    if 'time_ret_stats' in stats:
        trs = stats['time_ret_stats']
        if trs['below_min_return'] / trs['total'] < 0.3:
            issues.append(f"⚠️  只有 {trs['below_min_return']/trs['total']*100:.1f}% 的 time trigger 低於 min_return")
            issues.append(f"    建議：提高 min_return 到 {trs['p75']*100:.4f}% (75分位) 或 {trs['p90']*100:.4f}% (90分位)")

    for issue in issues:
        print(f"  {issue}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Triple-Barrier 標籤診斷工具')
    parser.add_argument('--data-dir', default='./data/temp', help='原始數據目錄')
    parser.add_argument('--config', default='./configs/config_pro_v5.yaml', help='配置文件')
    parser.add_argument('--sample-days', type=int, default=3, help='分析天數')
    args = parser.parse_args()

    # 載入配置
    config = load_config(args.config)

    # 查找數據文件
    data_files = sorted(Path(args.data_dir).glob('*.txt'))[:args.sample_days]

    if not data_files:
        logging.error(f"在 {args.data_dir} 找不到 .txt 文件")
        return

    logging.info(f"找到 {len(data_files)} 個數據文件，將分析前 {args.sample_days} 天")

    # 分析每一天
    all_stats = []
    for file_path in data_files:
        stats = analyze_single_day(str(file_path), config)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        logging.error("沒有成功分析的數據")
        return

    # 彙總報告
    print(f"\n{'='*60}")
    print(f"📈 多日彙總統計")
    print(f"{'='*60}")

    # 平均標籤分布
    avg_labels = {-1: 0, 0: 0, 1: 0}
    for stats in all_stats:
        for label, pct in stats['label_pct'].items():
            avg_labels[label] += pct

    n = len(all_stats)
    print(f"\n【平均標籤分布】(基於 {n} 天)")
    for label in sorted(avg_labels.keys()):
        pct = avg_labels[label] / n
        label_name = {-1: '下跌', 0: '持平', 1: '上漲'}[label]
        print(f"  {label_name} ({label:+2}): {pct:>5.1f}%")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()

