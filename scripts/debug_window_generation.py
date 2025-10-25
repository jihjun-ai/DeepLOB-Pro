#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_window_generation.py - 調試窗口生成邏輯

使用方式：
    # 在 extract_tw_stock_data_v7.py 第 826 行後添加以下代碼：
    if sym == symbols[0]:  # 只調試第一個股票
        print(f"\n=== 調試股票 {sym} ===")
        print(f"該股票有 {len(file_data_list)} 個日期文件")
        for idx, (date, features, labels, ...) in enumerate(file_data_list):
            T = len(features)
            windows = max(0, T - 100 + 1) if T >= 100 else 0
            print(f"  文件 {idx+1}: {date}, {T} 筆數據, 生成 {windows} 個窗口")
"""

import numpy as np
import sys
from pathlib import Path

# 簡單的測試邏輯
def test_window_generation():
    """模擬窗口生成邏輯"""

    print("=" * 80)
    print("模擬窗口生成邏輯測試")
    print("=" * 80)

    # 模擬一個股票的 3 天數據
    stock_data = {
        '2330': [
            ('20250901', np.random.randn(400, 20)),  # 9/1: 400 筆
            ('20250902', np.random.randn(420, 20)),  # 9/2: 420 筆
            ('20250903', np.random.randn(390, 20)),  # 9/3: 390 筆
        ]
    }

    SEQ_LEN = 100

    # V7.0 錯誤邏輯
    print("\n【V7.0 錯誤邏輯】- 合併所有文件")
    all_features = []
    for date, features in stock_data['2330']:
        all_features.append(features)

    concat_features = np.vstack(all_features)
    T_total = len(concat_features)
    windows_v70 = T_total - SEQ_LEN

    print(f"  合併後總長度: {T_total}")
    print(f"  生成窗口數: {windows_v70}")
    print(f"  [WARNING] 問題窗口示例:")
    print(f"    窗口 [301:401]: 來自 9/1 最後 99 筆 + 9/2 第 1 筆 [X]")
    print(f"    窗口 [721:821]: 來自 9/2 最後 99 筆 + 9/3 第 1 筆 [X]")

    # V7.1 正確邏輯
    print("\n【V7.1 正確邏輯】- 逐文件處理")
    total_windows = 0
    for idx, (date, features) in enumerate(stock_data['2330'], 1):
        T = len(features)
        if T < SEQ_LEN:
            windows = 0
        else:
            windows = T - SEQ_LEN + 1

        total_windows += windows
        print(f"  文件 {idx} ({date}): {T} 筆 -> {windows} 個窗口 [OK]")

    print(f"\n  總窗口數: {total_windows}")
    print(f"  減少數量: {windows_v70 - total_windows} 個窗口（都是跨文件的錯誤窗口）")
    print(f"  減少比例: {(windows_v70 - total_windows) / windows_v70 * 100:.2f}%")

    # 驗證
    print("\n【驗證】")
    expected_reduction = (len(stock_data['2330']) - 1) * (SEQ_LEN - 1)
    print(f"  預期減少: {expected_reduction} 個窗口（{len(stock_data['2330']) - 1} 個文件邊界 x 99）")
    print(f"  實際減少: {windows_v70 - total_windows} 個窗口")
    if expected_reduction == windows_v70 - total_windows:
        print(f"  [OK] 驗證通過！")
    else:
        print(f"  [FAIL] 驗證失敗！")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_window_generation()
