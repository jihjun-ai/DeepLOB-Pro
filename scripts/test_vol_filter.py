"""
快速測試波動率過濾功能
==================================================
驗證 volatility_filter 是否正確運作

使用方式：
    python scripts/test_vol_filter.py --config configs/config_pro_v5_vol_filtered.yaml
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.extract_tw_stock_data_v5 import load_config, ewma_vol

def test_vol_filter_logic():
    """測試波動率過濾邏輯"""

    print("="*60)
    print("📊 波動率過濾功能測試")
    print("="*60)

    # 測試案例 1：低波動率（應被過濾）
    print("\n【測試案例 1】低波動率股票日")
    # 使用更小的價格變動來產生真正的低波動率
    close_low = pd.Series([100.0] * 10 + [100.01] * 10 + [100.005] * 10 + [100.008] * 20 + [100.01] * 50)
    vol_low = ewma_vol(close_low, halflife=60)
    vol_clean = vol_low.replace([np.inf, -np.inf], np.nan).dropna()
    vol_mean_low = vol_clean.mean()

    print(f"  價格範圍: {close_low.min():.3f} - {close_low.max():.3f}")
    print(f"  平均波動率: {vol_mean_low*100:.4f}%")
    print(f"  判定: ", end="")

    threshold = 0.0005
    if vol_mean_low < threshold:
        print(f"✅ 會被過濾 (< {threshold*100:.2f}%)")
    else:
        print(f"❌ 不會被過濾 (>= {threshold*100:.2f}%)")

    # 測試案例 2：中等波動率（應保留）
    print("\n【測試案例 2】中等波動率股票日")
    close_med = pd.Series([100.0, 100.5, 99.8, 100.3, 99.9, 100.4] * 20)
    vol_med = ewma_vol(close_med, halflife=60)
    vol_mean_med = vol_med.mean()

    print(f"  價格範圍: {close_med.min():.2f} - {close_med.max():.2f}")
    print(f"  平均波動率: {vol_mean_med*100:.4f}%")
    print(f"  判定: ", end="")

    if vol_mean_med < 0.0005:
        print(f"❌ 會被過濾 (< 0.05%)")
    else:
        print(f"✅ 不會被過濾 (>= 0.05%)")

    # 測試案例 3：高波動率（應保留）
    print("\n【測試案例 3】高波動率股票日")
    close_high = pd.Series([100.0, 102.0, 98.5, 101.5, 97.8, 103.2] * 20)
    vol_high = ewma_vol(close_high, halflife=60)
    vol_mean_high = vol_high.mean()

    print(f"  價格範圍: {close_high.min():.2f} - {close_high.max():.2f}")
    print(f"  平均波動率: {vol_mean_high*100:.4f}%")
    print(f"  判定: ", end="")

    if vol_mean_high < 0.0005:
        print(f"❌ 會被過濾 (< 0.05%)")
    else:
        print(f"✅ 不會被過濾 (>= 0.05%)")

    # 總結
    print("\n" + "="*60)
    print("📋 測試總結")
    print("="*60)

    test_results = [
        ("低波動率", vol_mean_low, vol_mean_low < threshold, True),
        ("中等波動率", vol_mean_med, vol_mean_med < threshold, False),
        ("高波動率", vol_mean_high, vol_mean_high < threshold, False),
    ]

    all_passed = True
    for name, vol_val, filtered, should_filter in test_results:
        status = "✅ PASS" if (filtered == should_filter) else "❌ FAIL"
        print(f"{status} - {name}: {vol_val*100:.4f}% (過濾={filtered})")
        if filtered != should_filter:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有測試通過！波動率過濾邏輯正確運作")
    else:
        print("⚠️  有測試失敗，請檢查實作")
    print("="*60 + "\n")

    return all_passed


def test_config_loading():
    """測試配置文件載入"""

    print("="*60)
    print("📄 配置文件測試")
    print("="*60)

    config_path = "./configs/config_pro_v5_vol_filtered.yaml"

    try:
        config = load_config(config_path)
        print(f"\n✅ 配置文件載入成功: {config_path}")

        # 檢查關鍵參數
        vol_filter = config.get('volatility_filter', {})

        print(f"\n【波動率過濾設定】")
        print(f"  enabled: {vol_filter.get('enabled', False)}")
        print(f"  min_daily_vol: {vol_filter.get('min_daily_vol', 0.0)}")

        if vol_filter.get('enabled'):
            threshold = vol_filter.get('min_daily_vol', 0.0)
            print(f"\n✅ 波動率過濾已啟用")
            print(f"   閾值: {threshold*100:.3f}%")

            # 給出建議
            if threshold < 0.0003:
                print(f"   建議：閾值偏低，過濾效果可能不明顯")
            elif threshold > 0.001:
                print(f"   建議：閾值偏高，可能過濾過多數據")
            else:
                print(f"   建議：閾值設定合理")
        else:
            print(f"\n⚠️  波動率過濾未啟用")

        # 檢查其他相關設定
        tb_cfg = config.get('triple_barrier', {})
        print(f"\n【Triple-Barrier 設定】")
        print(f"  pt_multiplier: {tb_cfg.get('pt_multiplier', 'N/A')}")
        print(f"  sl_multiplier: {tb_cfg.get('sl_multiplier', 'N/A')}")
        print(f"  max_holding: {tb_cfg.get('max_holding', 'N/A')}")
        print(f"  min_return: {tb_cfg.get('min_return', 'N/A')}")

        print("\n" + "="*60 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ 配置文件載入失敗: {e}")
        print("="*60 + "\n")
        return False


def main():
    """主測試流程"""

    print("\n" + "🚀 開始波動率過濾功能測試\n")

    # 測試 1：邏輯測試
    logic_ok = test_vol_filter_logic()

    # 測試 2：配置測試
    config_ok = test_config_loading()

    # 總結
    print("="*60)
    print("🏁 測試完成")
    print("="*60)

    if logic_ok and config_ok:
        print("\n✅ 所有測試通過！")
        print("\n下一步：")
        print("  1. 運行數據分析:")
        print("     python scripts/extract_tw_stock_data_v5.py \\")
        print("         --input-dir ./data/temp \\")
        print("         --output-dir ./data/vol_analysis \\")
        print("         --config configs/config_pro_v5_ml_optimal.yaml \\")
        print("         --stats-only")
        print("\n  2. 根據分析結果調整 min_daily_vol")
        print("\n  3. 生成過濾後的數據:")
        print("     python scripts/extract_tw_stock_data_v5.py \\")
        print("         --input-dir ./data/temp \\")
        print("         --output-dir ./data/processed_v5_vol_filtered \\")
        print("         --config configs/config_pro_v5_vol_filtered.yaml")
        return 0
    else:
        print("\n❌ 有測試失敗，請檢查")
        return 1


if __name__ == '__main__':
    sys.exit(main())
