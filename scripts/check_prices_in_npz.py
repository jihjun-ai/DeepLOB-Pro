#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
檢查預處理 NPZ 中的 last_prices 欄位
快速診斷為何 last_prices 可能為 None 或包含 0 值
"""
import numpy as np
import glob
import os
import sys

def check_npz_prices(npz_dir):
    """檢查 NPZ 目錄中的 prices 欄位"""
    npz_files = glob.glob(os.path.join(npz_dir, "daily", "*", "*.npz"))

    if not npz_files:
        print(f"❌ 未找到 NPZ 檔案: {npz_dir}")
        return

    print(f"找到 {len(npz_files)} 個 NPZ 檔案")
    print("=" * 80)

    total_files = 0
    has_last_prices = 0
    has_last_volumes = 0
    has_total_volumes = 0

    files_with_zeros = []
    files_without_prices = []

    for npz_file in npz_files[:20]:  # 只檢查前 20 個
        total_files += 1

        try:
            data = np.load(npz_file, allow_pickle=True)

            # 檢查是否有新欄位
            has_lp = 'last_prices' in data
            has_lv = 'last_volumes' in data
            has_tv = 'total_volumes' in data

            if has_lp:
                has_last_prices += 1
            if has_lv:
                has_last_volumes += 1
            if has_tv:
                has_total_volumes += 1

            symbol = os.path.basename(npz_file).replace('.npz', '')

            if has_lp:
                last_prices = data['last_prices']

                # 檢查是否有 0 值
                zero_count = np.sum(last_prices == 0.0)
                zero_pct = zero_count / len(last_prices) * 100 if len(last_prices) > 0 else 0

                if zero_count > 0:
                    files_with_zeros.append((symbol, zero_count, zero_pct, len(last_prices)))

                print(f"✅ {symbol:8s}: last_prices 存在 (長度={len(last_prices):6d}, 0值={zero_count:5d}, {zero_pct:5.1f}%)")
            else:
                files_without_prices.append(symbol)
                print(f"❌ {symbol:8s}: last_prices 不存在")

        except Exception as e:
            print(f"⚠️ 無法讀取 {npz_file}: {e}")

    print("=" * 80)
    print(f"\n統計摘要:")
    print(f"  檢查檔案數: {total_files}")
    print(f"  有 last_prices: {has_last_prices} ({has_last_prices/total_files*100:.1f}%)")
    print(f"  有 last_volumes: {has_last_volumes} ({has_last_volumes/total_files*100:.1f}%)")
    print(f"  有 total_volumes: {has_total_volumes} ({has_total_volumes/total_files*100:.1f}%)")

    if files_without_prices:
        print(f"\n❌ 缺少 last_prices 的檔案 ({len(files_without_prices)}):")
        for sym in files_without_prices[:10]:
            print(f"  - {sym}")

    if files_with_zeros:
        print(f"\n⚠️ 包含 0 值的檔案 ({len(files_with_zeros)}):")
        for sym, zero_count, zero_pct, total in sorted(files_with_zeros, key=lambda x: -x[2])[:10]:
            print(f"  - {sym}: {zero_count:5d} / {total:6d} ({zero_pct:5.1f}%)")

    # 診斷建議
    print("\n" + "=" * 80)
    print("診斷建議:")

    if has_last_prices == 0:
        print("  ❌ 所有檔案都缺少 last_prices 欄位")
        print("  → 需要重新運行 preprocess_single_day.py")
    elif has_last_prices < total_files:
        print(f"  ⚠️ 部分檔案缺少 last_prices ({total_files - has_last_prices} 個)")
        print("  → 這些檔案是舊版本預處理的，需要重新處理")
    else:
        print("  ✅ 所有檔案都包含 last_prices 欄位")

    if files_with_zeros:
        print(f"\n  ⚠️ {len(files_with_zeros)} 個檔案的 last_prices 包含 0 值")
        print("  可能原因:")
        print("    1. 開盤前沒有成交價 → last_valid_price 初始化為 0.0")
        print("    2. ffill 使用了初始值 0.0")
        print("    3. 原始數據中 LastPrice 欄位為 0")
        print("  解決方案:")
        print("    - 修改 aggregate_to_1hz() 初始化邏輯")
        print("    - 從第一個有效事件提取初始 last_valid_price")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        npz_dir = sys.argv[1]
    else:
        npz_dir = "data/preprocessed_v5"

    print(f"檢查目錄: {npz_dir}\n")
    check_npz_prices(npz_dir)
