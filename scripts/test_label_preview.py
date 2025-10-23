# -*- coding: utf-8 -*-
"""
test_label_preview.py - 測試標籤預覽功能
=============================================================================
用途：驗證 preprocess_single_day.py 是否正確計算標籤分布

使用方式：
  python scripts/test_label_preview.py

測試內容：
  1. 讀取一個已存在的 NPZ 檔案
  2. 檢查 metadata 中的 label_preview 欄位
  3. 讀取對應的 summary.json
  4. 驗證標籤統計的一致性
"""

import os
import json
import glob
import numpy as np
from pathlib import Path

def test_npz_label_preview():
    """測試 NPZ 中的標籤預覽資訊"""
    print("="*70)
    print("測試 NPZ 標籤預覽功能")
    print("="*70)

    # 找到最新的預處理數據目錄
    preprocessed_dirs = glob.glob("data/preprocessed_v5*/daily/20250901/*.npz")

    if not preprocessed_dirs:
        print("❌ 找不到預處理數據！請先執行 preprocess_single_day.py")
        return

    # 取前 5 個測試
    test_files = preprocessed_dirs[:5]

    for npz_file in test_files:
        print(f"\n{'='*70}")
        print(f"檔案: {os.path.basename(npz_file)}")
        print(f"{'='*70}")

        # 讀取 NPZ
        data = np.load(npz_file, allow_pickle=True)

        # 檢查必要欄位
        print(f"\n📦 NPZ 內容:")
        print(f"   Keys: {list(data.keys())}")
        print(f"   Features shape: {data['features'].shape}")
        print(f"   Mids shape: {data['mids'].shape}")

        # 解析 metadata
        metadata = json.loads(str(data['metadata']))

        print(f"\n📋 Metadata:")
        print(f"   Symbol: {metadata['symbol']}")
        print(f"   Date: {metadata['date']}")
        print(f"   Pass filter: {metadata['pass_filter']}")
        print(f"   n_points: {metadata['n_points']}")

        # 🆕 檢查標籤預覽
        if 'label_preview' in metadata and metadata['label_preview'] is not None:
            lp = metadata['label_preview']
            print(f"\n✅ 標籤預覽:")
            print(f"   總標籤數: {lp['total_labels']:,}")
            print(f"   Down:    {lp['down_count']:>6,} ({lp['down_pct']:>6.2%})")
            print(f"   Neutral: {lp['neutral_count']:>6,} ({lp['neutral_pct']:>6.2%})")
            print(f"   Up:      {lp['up_count']:>6,} ({lp['up_pct']:>6.2%})")

            # 驗證總和
            total_check = lp['down_count'] + lp['neutral_count'] + lp['up_count']
            if total_check == lp['total_labels']:
                print(f"   ✓ 總和驗證通過")
            else:
                print(f"   ✗ 總和驗證失敗: {total_check} != {lp['total_labels']}")
        else:
            print(f"\n⚠️  無標籤預覽（可能未通過過濾或計算失敗）")


def test_summary_label_stats():
    """測試 summary.json 中的標籤統計"""
    print(f"\n\n{'='*70}")
    print("測試 summary.json 標籤統計")
    print(f"{'='*70}")

    # 找到最新的 summary.json
    summary_files = glob.glob("data/preprocessed_v5*/daily/*/summary.json")

    if not summary_files:
        print("❌ 找不到 summary.json！")
        return

    # 取最新的
    summary_file = sorted(summary_files)[-1]
    print(f"\n檔案: {summary_file}")

    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    print(f"\n📅 日期: {summary['date']}")
    print(f"   總股票數: {summary['total_symbols']}")
    print(f"   通過過濾: {summary['passed_filter']}")
    print(f"   被過濾: {summary['filtered_out']}")

    # 🆕 檢查實際標籤統計
    if 'actual_label_stats' in summary and summary['actual_label_stats'] is not None:
        ls = summary['actual_label_stats']
        print(f"\n✅ 實際標籤統計:")
        print(f"   有標籤的股票數: {ls['stocks_with_labels']}")
        print(f"   總標籤數: {ls['total_labels']:,}")
        print(f"   Down:    {ls['down_count']:>8,} ({ls['down_pct']:>6.2%})")
        print(f"   Neutral: {ls['neutral_count']:>8,} ({ls['neutral_pct']:>6.2%})")
        print(f"   Up:      {ls['up_count']:>8,} ({ls['up_pct']:>6.2%})")
    else:
        print(f"\n⚠️  無實際標籤統計（可能是舊版本數據）")

    # 對比預測標籤分布
    if 'predicted_label_dist' in summary:
        pd_dist = summary['predicted_label_dist']
        print(f"\n📊 預測標籤分布（啟發式）:")
        print(f"   Down:    {pd_dist['down']:.2%}")
        print(f"   Neutral: {pd_dist['neutral']:.2%}")
        print(f"   Up:      {pd_dist['up']:.2%}")


def test_label_filtering_logic():
    """測試基於標籤的過濾邏輯（示範）"""
    print(f"\n\n{'='*70}")
    print("示範：基於標籤分布的股票篩選")
    print(f"{'='*70}")

    # 找到所有 NPZ
    preprocessed_dirs = glob.glob("data/preprocessed_v5*/daily/20250901/*.npz")

    if not preprocessed_dirs:
        print("❌ 找不到數據！")
        return

    # 設定過濾條件
    min_neutral_pct = 0.10  # 持平類至少 10%
    max_neutral_pct = 0.50  # 持平類最多 50%
    min_total_labels = 1000  # 至少 1000 個標籤

    print(f"\n過濾條件:")
    print(f"   持平比例: {min_neutral_pct:.0%} - {max_neutral_pct:.0%}")
    print(f"   最少標籤數: {min_total_labels:,}")

    passed_stocks = []
    failed_stocks = []

    for npz_file in preprocessed_dirs:
        data = np.load(npz_file, allow_pickle=True)
        metadata = json.loads(str(data['metadata']))
        symbol = metadata['symbol']

        if 'label_preview' not in metadata or metadata['label_preview'] is None:
            failed_stocks.append((symbol, "無標籤預覽"))
            continue

        lp = metadata['label_preview']

        # 檢查條件
        if lp['total_labels'] < min_total_labels:
            failed_stocks.append((symbol, f"標籤數不足 ({lp['total_labels']})"))
            continue

        if lp['neutral_pct'] < min_neutral_pct:
            failed_stocks.append((symbol, f"持平比例太低 ({lp['neutral_pct']:.1%})"))
            continue

        if lp['neutral_pct'] > max_neutral_pct:
            failed_stocks.append((symbol, f"持平比例太高 ({lp['neutral_pct']:.1%})"))
            continue

        passed_stocks.append((symbol, lp))

    print(f"\n結果:")
    print(f"   通過: {len(passed_stocks)} 檔")
    print(f"   未通過: {len(failed_stocks)} 檔")

    if passed_stocks:
        print(f"\n✅ 通過的股票（前 10 檔）:")
        for symbol, lp in passed_stocks[:10]:
            print(f"   {symbol}: Down {lp['down_pct']:.1%} | Neutral {lp['neutral_pct']:.1%} | Up {lp['up_pct']:.1%}")

    if failed_stocks:
        print(f"\n❌ 未通過的股票（前 5 檔）:")
        for symbol, reason in failed_stocks[:5]:
            print(f"   {symbol}: {reason}")


if __name__ == "__main__":
    test_npz_label_preview()
    test_summary_label_stats()
    test_label_filtering_logic()

    print(f"\n\n{'='*70}")
    print("✅ 測試完成")
    print(f"{'='*70}")
