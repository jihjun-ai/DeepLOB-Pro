#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
find_null_prices.py - 檢查哪些日期/股票的 NPZ 缺少 last_prices 欄位

快速診斷預處理數據完整性
"""
import numpy as np
import glob
import os
import sys
from collections import defaultdict

def find_null_prices(preprocessed_dir):
    """找出缺少 last_prices 的日期和股票"""

    npz_pattern = os.path.join(preprocessed_dir, "daily", "*", "*.npz")
    npz_files = sorted(glob.glob(npz_pattern))

    if not npz_files:
        print(f"[ERROR] 未找到任何 NPZ 檔案: {npz_pattern}")
        return

    print(f"掃描目錄: {preprocessed_dir}")
    print(f"找到 {len(npz_files)} 個 NPZ 檔案\n")
    print("=" * 80)

    # 統計
    total_files = 0
    files_with_prices = 0
    files_without_prices = 0

    # 按日期分組
    dates_with_null = defaultdict(list)  # date -> [symbols]
    dates_with_prices = defaultdict(list)

    for npz_file in npz_files:
        total_files += 1

        # 提取日期和股票代碼
        parts = npz_file.replace('\\', '/').split('/')
        date = parts[-2]  # 倒數第二個是日期
        symbol = os.path.basename(npz_file).replace('.npz', '')

        try:
            data = np.load(npz_file, allow_pickle=True)

            has_last_prices = 'last_prices' in data
            has_last_volumes = 'last_volumes' in data
            has_total_volumes = 'total_volumes' in data

            if has_last_prices and has_last_volumes and has_total_volumes:
                files_with_prices += 1
                dates_with_prices[date].append(symbol)
            else:
                files_without_prices += 1
                dates_with_null[date].append(symbol)

        except Exception as e:
            print(f"[ERROR] 無法讀取 {npz_file}: {e}")
            files_without_prices += 1
            dates_with_null[date].append(symbol)

    # 報告結果
    print(f"總檔案數: {total_files}")
    print(f"  ✓ 包含 prices/volumes: {files_with_prices} ({files_with_prices/total_files*100:.1f}%)")
    print(f"  ✗ 缺少 prices/volumes: {files_without_prices} ({files_without_prices/total_files*100:.1f}%)")
    print("=" * 80)

    if dates_with_null:
        print(f"\n[WARNING] 發現 {len(dates_with_null)} 個日期的股票缺少 prices/volumes:\n")

        for date in sorted(dates_with_null.keys()):
            symbols = dates_with_null[date]
            print(f"日期: {date}")
            print(f"  缺少 prices 的股票數: {len(symbols)}")
            print(f"  股票代碼: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                print(f"  ... 還有 {len(symbols) - 10} 個")
            print()

        print("=" * 80)
        print("\n[ACTION] 需要重新預處理以下日期的數據:")
        for date in sorted(dates_with_null.keys()):
            print(f"  - {date}")

        print("\n執行命令 (Windows):")
        for date in sorted(dates_with_null.keys()):
            txt_file = f"data/raw/tw_stock/{date}.txt"
            print(f"python scripts/preprocess_single_day.py --input {txt_file} --output-dir data/preprocessed_v5 --config configs/config_pro_v5_ml_optimal.yaml")

    else:
        print("\n[SUCCESS] ✓ 所有檔案都包含完整的 prices/volumes 欄位！")

    if dates_with_prices:
        print(f"\n[INFO] 已包含 prices 的日期 ({len(dates_with_prices)} 個):")
        for date in sorted(dates_with_prices.keys())[:5]:
            print(f"  ✓ {date} ({len(dates_with_prices[date])} 檔股票)")
        if len(dates_with_prices) > 5:
            print(f"  ... 還有 {len(dates_with_prices) - 5} 個日期")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        preprocessed_dir = sys.argv[1]
    else:
        preprocessed_dir = "data/preprocessed_v5"

    find_null_prices(preprocessed_dir)
