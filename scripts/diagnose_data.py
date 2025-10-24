#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_data.py - 快速診斷預處理數據狀態

用途：
  1. 檢查 NPZ 檔案數量
  2. 查看標籤分布
  3. 計算總樣本數
  4. 找出為何無法生成候選方案
"""

import os
import json
import glob
import numpy as np
from pathlib import Path

def main():
    preprocessed_dir = "data/preprocessed_v5"
    start_date = "20250901"

    print("="*80)
    print("預處理數據診斷")
    print("="*80)

    # 1. 檢查 NPZ 檔案
    npz_pattern = os.path.join(preprocessed_dir, "daily", "*", "*.npz")
    npz_files = glob.glob(npz_pattern)

    print(f"\n1. NPZ 檔案掃描:")
    print(f"   路徑: {npz_pattern}")
    print(f"   找到: {len(npz_files)} 個檔案")

    if len(npz_files) == 0:
        print("\n   ❌ 錯誤: 找不到任何 NPZ 檔案！")
        print("   可能原因:")
        print("     1. 預處理目錄路徑不正確")
        print("     2. 尚未執行預處理 (batch_preprocess.bat)")
        print("     3. 檔案在不同目錄")

        # 檢查目錄是否存在
        if not Path(preprocessed_dir).exists():
            print(f"\n   ⚠️  目錄不存在: {Path(preprocessed_dir).absolute()}")
        else:
            print(f"\n   ✅ 目錄存在: {Path(preprocessed_dir).absolute()}")

            # 列出子目錄
            daily_dir = Path(preprocessed_dir) / "daily"
            if daily_dir.exists():
                subdirs = list(daily_dir.glob("*"))
                print(f"   找到 {len(subdirs)} 個日期子目錄:")
                for subdir in sorted(subdirs)[:5]:
                    files = list(subdir.glob("*.npz"))
                    print(f"     {subdir.name}: {len(files)} 個 NPZ")
        return

    # 2. 載入有效股票（通過過濾且有標籤預覽）
    print(f"\n2. 載入股票 metadata (日期 >= {start_date}):")

    valid_stocks = []
    all_stocks = []

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            metadata = json.loads(str(data['metadata']))

            all_stocks.append(metadata)

            # 檢查是否通過過濾
            if not metadata.get('pass_filter', False):
                continue

            # 檢查是否有標籤預覽
            if metadata.get('label_preview') is None:
                continue

            # 日期過濾
            if metadata['date'] < start_date:
                continue

            valid_stocks.append(metadata)

        except Exception as e:
            continue

    print(f"   總檔案數: {len(npz_files)}")
    print(f"   可載入: {len(all_stocks)}")
    print(f"   通過 pass_filter: {sum(1 for s in all_stocks if s.get('pass_filter', False))}")
    print(f"   有 label_preview: {sum(1 for s in all_stocks if s.get('label_preview') is not None)}")
    print(f"   日期 >= {start_date}: {len(valid_stocks)}")

    if len(valid_stocks) == 0:
        print("\n   ❌ 錯誤: 沒有有效的股票數據！")

        # 診斷原因
        if len(all_stocks) == 0:
            print("   原因: 無法載入任何 metadata")
        else:
            no_filter = sum(1 for s in all_stocks if not s.get('pass_filter', False))
            no_preview = sum(1 for s in all_stocks if s.get('label_preview') is None)
            before_date = sum(1 for s in all_stocks if s.get('pass_filter', False)
                            and s.get('label_preview') is not None
                            and s['date'] < start_date)

            print(f"   原因分析:")
            print(f"     未通過 pass_filter: {no_filter} 檔")
            print(f"     沒有 label_preview: {no_preview} 檔")
            print(f"     日期早於 {start_date}: {before_date} 檔")

            # 顯示日期範圍
            dates = sorted(set(s['date'] for s in all_stocks))
            if dates:
                print(f"\n   可用日期範圍: {dates[0]} - {dates[-1]}")
        return

    # 3. 計算標籤分布
    print(f"\n3. 標籤分布分析 ({len(valid_stocks)} 檔):")

    total_down = 0
    total_neutral = 0
    total_up = 0

    for stock in valid_stocks:
        lp = stock['label_preview']
        total_down += lp['down_count']
        total_neutral += lp['neutral_count']
        total_up += lp['up_count']

    total_all = total_down + total_neutral + total_up

    print(f"   總樣本數: {total_all:,}")
    print(f"   Down:     {total_down:>10,} ({total_down/total_all*100:>5.1f}%)")
    print(f"   Neutral:  {total_neutral:>10,} ({total_neutral/total_all*100:>5.1f}%)")
    print(f"   Up:       {total_up:>10,} ({total_up/total_all*100:>5.1f}%)")

    # 4. 計算偏差度
    target_dist = (0.30, 0.40, 0.30)
    current_dist = (total_down/total_all, total_neutral/total_all, total_up/total_all)

    deviation = np.sqrt(sum((c - t)**2 for c, t in zip(current_dist, target_dist)))

    print(f"\n4. 與目標分布的偏差:")
    print(f"   目標: Down {target_dist[0]:.1%} | Neutral {target_dist[1]:.1%} | Up {target_dist[2]:.1%}")
    print(f"   當前: Down {current_dist[0]:.1%} | Neutral {current_dist[1]:.1%} | Up {current_dist[2]:.1%}")
    print(f"   偏差度 (L2): {deviation:.4f}")

    # 判斷偏差等級
    if deviation <= 0.01:
        level = "保守方案 (< 1%)"
    elif deviation <= 0.02:
        level = "平衡方案 (< 2%)"
    elif deviation <= 0.03:
        level = "積極方案 (< 3%)"
    elif deviation <= 0.05:
        level = "寬鬆方案 (< 5%)"
    else:
        level = "❌ 超出所有閾值 (> 5%)"

    print(f"   評級: {level}")

    # 5. 按日期分組統計
    from collections import defaultdict
    grouped = defaultdict(list)
    for stock in valid_stocks:
        grouped[stock['date']].append(stock)

    print(f"\n5. 按日期分組統計:")
    print(f"   日期範圍: {min(grouped.keys())} - {max(grouped.keys())}")
    print(f"   天數: {len(grouped)}")
    print(f"\n   前5天明細:")

    for date in sorted(grouped.keys())[:5]:
        stocks_on_date = grouped[date]
        samples = sum(s['label_preview']['total_labels'] for s in stocks_on_date)
        d = sum(s['label_preview']['down_count'] for s in stocks_on_date)
        n = sum(s['label_preview']['neutral_count'] for s in stocks_on_date)
        u = sum(s['label_preview']['up_count'] for s in stocks_on_date)
        print(f"     {date}: {len(stocks_on_date):>3} 檔, {samples:>8,} 樣本, "
              f"D {d/samples*100:>4.1f}% | N {n/samples*100:>4.1f}% | U {u/samples*100:>4.1f}%")

    # 6. 診斷建議
    print(f"\n6. 診斷結果與建議:")
    print("="*80)

    if deviation > 0.05:
        print("❌ 問題: 標籤分布偏差太大 (> 5%)")
        print("\n   原因: 當前標籤分布與目標 30/40/30 差異過大")
        print("\n   建議:")
        print("   1. 調整目標分布更接近實際:")
        actual_target = f"{current_dist[0]:.2f},{current_dist[1]:.2f},{current_dist[2]:.2f}"
        print(f"      --target-dist \"{actual_target}\"")
        print("\n   2. 或者調整預處理參數 (如果需要更多 Neutral):")
        print("      config_pro_v5_ml_optimal.yaml:")
        print("        vol_multiplier: 2.5 → 3.0 (提高趨勢門檻)")
        print("        min_trend_duration: 45 → 60 (延長持續性)")
    elif total_all < 100000:
        print("⚠️  問題: 總樣本數不足 (< 100,000)")
        print(f"\n   當前樣本數: {total_all:,}")
        print("\n   建議:")
        print("   1. 降低 --min-samples:")
        print(f"      --min-samples {total_all // 2}")
        print("\n   2. 或增加日期範圍:")
        print("      --start-date 更早的日期")
    else:
        print("✅ 數據狀態良好")
        print(f"\n   總樣本數: {total_all:,}")
        print(f"   偏差度: {deviation:.4f} ({level})")
        print("\n   建議使用:")
        print(f"   --min-samples {total_all // 2}")
        actual_target = f"{current_dist[0]:.2f},{current_dist[1]:.2f},{current_dist[2]:.2f}"
        print(f"   --target-dist \"{actual_target}\"")

    print("="*80)


if __name__ == "__main__":
    main()
