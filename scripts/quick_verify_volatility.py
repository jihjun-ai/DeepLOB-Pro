#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quick_verify_volatility.py - 快速驗證震盪統計功能
=============================================================================
【用途】使用模擬數據快速驗證統計功能，無需實際數據

【使用方式】
python scripts/quick_verify_volatility.py
=============================================================================
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加專案根目錄
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_volatility_calculation():
    """測試震盪計算函數"""
    print("="*60)
    print("測試 1: 震盪計算函數")
    print("="*60)

    # 導入函數
    from scripts.extract_tw_stock_data_v5 import calculate_intraday_volatility

    # 測試案例 1: 正常震盪（2%）
    mids1 = np.array([100.0, 101.0, 102.0, 101.5, 100.5, 101.0])  # 開100, 高102, 低100, 收101
    result1 = calculate_intraday_volatility(mids1, "20241018", "TEST1")

    print(f"\n案例 1（正常震盪）:")
    print(f"  數據: 開={result1['open']:.2f}, 高={result1['high']:.2f}, 低={result1['low']:.2f}, 收={result1['close']:.2f}")
    print(f"  震盪幅度: {result1['range_pct']*100:.2f}%")
    print(f"  漲跌幅: {result1['return_pct']*100:.2f}%")
    print(f"  預期震盪: 2.00% (實際: {result1['range_pct']*100:.2f}%)")

    assert abs(result1['range_pct'] - 0.02) < 0.001, "震盪計算錯誤！"
    assert abs(result1['return_pct'] - 0.01) < 0.001, "報酬計算錯誤！"
    print("  ✅ 測試通過")

    # 測試案例 2: 高震盪（5%）
    mids2 = np.array([100.0, 105.0, 103.0, 102.0, 98.0, 101.0])  # 開100, 高105, 低98, 收101
    result2 = calculate_intraday_volatility(mids2, "20241018", "TEST2")

    print(f"\n案例 2（高震盪）:")
    print(f"  數據: 開={result2['open']:.2f}, 高={result2['high']:.2f}, 低={result2['low']:.2f}, 收={result2['close']:.2f}")
    print(f"  震盪幅度: {result2['range_pct']*100:.2f}%")
    print(f"  預期震盪: 7.00% (實際: {result2['range_pct']*100:.2f}%)")

    assert abs(result2['range_pct'] - 0.07) < 0.001, "震盪計算錯誤！"
    print("  ✅ 測試通過")

    # 測試案例 3: 低震盪（0.5%）
    mids3 = np.array([100.0, 100.5, 100.3, 100.2, 99.8, 99.9])  # 開100, 高100.5, 低99.8, 收99.9
    result3 = calculate_intraday_volatility(mids3, "20241018", "TEST3")

    print(f"\n案例 3（低震盪）:")
    print(f"  數據: 開={result3['open']:.2f}, 高={result3['high']:.2f}, 低={result3['low']:.2f}, 收={result3['close']:.2f}")
    print(f"  震盪幅度: {result3['range_pct']*100:.2f}%")
    print(f"  預期震盪: 0.70% (實際: {result3['range_pct']*100:.2f}%)")

    assert abs(result3['range_pct'] - 0.007) < 0.001, "震盪計算錯誤！"
    print("  ✅ 測試通過")

    print("\n" + "="*60)
    print("✅ 所有震盪計算測試通過！")
    print("="*60 + "\n")


def test_volatility_report():
    """測試震盪報告生成"""
    print("="*60)
    print("測試 2: 震盪報告生成")
    print("="*60)

    from scripts.extract_tw_stock_data_v5 import generate_volatility_report
    import tempfile

    # 生成模擬數據
    vol_stats = []

    # 模擬 100 個 symbol-day
    np.random.seed(42)

    for i in range(100):
        symbol = f"TEST{i % 10:04d}"  # 10 檔股票
        date = f"2024101{i % 10}"

        # 模擬震盪分布（平均 2%，標準差 1%）
        range_pct = max(0.002, np.random.normal(0.02, 0.01))
        return_pct = np.random.normal(0.0, 0.015)

        vol_stats.append({
            'date': date,
            'symbol': symbol,
            'range_pct': range_pct,
            'return_pct': return_pct,
            'high': 100 * (1 + range_pct / 2),
            'low': 100 * (1 - range_pct / 2),
            'open': 100.0,
            'close': 100 * (1 + return_pct),
            'n_points': 100
        })

    # 生成報告
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n生成報告到臨時目錄: {tmpdir}\n")

        generate_volatility_report(vol_stats, tmpdir)

        # 檢查文件是否生成
        csv_path = os.path.join(tmpdir, "volatility_stats.csv")
        json_path = os.path.join(tmpdir, "volatility_summary.json")

        assert os.path.exists(csv_path), "CSV 文件未生成！"
        assert os.path.exists(json_path), "JSON 文件未生成！"

        print(f"\n✅ 報告生成成功:")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")

        # 驗證 CSV 內容
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 100, "CSV 記錄數量錯誤！"
        print(f"\n✅ CSV 驗證通過（{len(df)} 筆記錄）")

        # 驗證 JSON 內容
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        assert summary['total_samples'] == 100, "JSON 樣本數錯誤！"
        assert 'range_pct' in summary, "JSON 缺少震盪統計！"
        assert 'threshold_stats' in summary, "JSON 缺少閾值統計！"
        print(f"✅ JSON 驗證通過")

    print("\n" + "="*60)
    print("✅ 震盪報告生成測試通過！")
    print("="*60 + "\n")


def main():
    """主函數"""
    print("\n" + "="*60)
    print("🧪 震盪統計功能快速驗證")
    print("="*60 + "\n")

    try:
        # 測試 1: 震盪計算
        test_volatility_calculation()

        # 測試 2: 報告生成
        test_volatility_report()

        print("="*60)
        print("🎉 所有測試通過！")
        print("="*60)
        print("\n下一步:")
        print("  1. 執行完整統計分析:")
        print("     python scripts/extract_tw_stock_data_v5.py \\")
        print("         --input-dir ./data/temp \\")
        print("         --output-dir ./data/processed_v5_stats")
        print("\n  2. 查看震盪分布報告:")
        print("     - volatility_stats.csv")
        print("     - volatility_summary.json")
        print("\n  3. 參考指南決定閾值:")
        print("     - VOLATILITY_ANALYSIS_GUIDE.md")
        print("="*60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
