#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_npz_lengths.py - 檢查 NPZ 檔案中各個欄位的長度是否一致
"""
import numpy as np
import glob
import os
from collections import defaultdict

def check_npz_lengths(npz_dir):
    """檢查所有 NPZ 檔案的欄位長度"""

    npz_pattern = os.path.join(npz_dir, "daily", "*", "*.npz")
    npz_files = sorted(glob.glob(npz_pattern))

    if not npz_files:
        print(f"[ERROR] 未找到 NPZ 檔案: {npz_pattern}")
        return

    print(f"檢查目錄: {npz_dir}")
    print(f"找到 {len(npz_files)} 個 NPZ 檔案")
    print("=" * 80)

    # 統計長度分布
    length_stats = defaultdict(lambda: defaultdict(set))  # field -> length -> set(files)

    # 記錄異常檔案
    inconsistent_files = []

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)

            # 提取檔案資訊
            parts = npz_file.replace('\\', '/').split('/')
            date = parts[-2]
            symbol = os.path.basename(npz_file).replace('.npz', '')

            # 獲取所有欄位的長度
            field_lengths = {}
            for key in data.files:
                if key == 'metadata':
                    continue
                arr = data[key]
                length = len(arr) if hasattr(arr, '__len__') else 0
                field_lengths[key] = length
                length_stats[key][length].add((date, symbol))

            # 檢查該檔案內部各欄位長度是否一致
            lengths = set(field_lengths.values())
            if len(lengths) > 1:
                inconsistent_files.append((date, symbol, field_lengths))

        except Exception as e:
            print(f"[ERROR] 讀取失敗: {npz_file}")
            print(f"        錯誤: {e}")

    # 報告結果
    print(f"\n{'='*80}")
    print("欄位長度分布統計:")
    print(f"{'='*80}\n")

    for field in sorted(length_stats.keys()):
        length_dist = length_stats[field]
        print(f"欄位: {field}")

        if len(length_dist) == 1:
            # 所有檔案長度一致
            length = list(length_dist.keys())[0]
            count = len(list(length_dist.values())[0])
            print(f"  ✅ 所有檔案長度一致: {length} (共 {count} 個檔案)")
        else:
            # 長度不一致
            print(f"  ⚠️ 長度不一致！發現 {len(length_dist)} 種不同長度:")
            for length in sorted(length_dist.keys()):
                files = length_dist[length]
                print(f"     長度 {length}: {len(files)} 個檔案")

                # 顯示前 5 個檔案
                for i, (date, symbol) in enumerate(sorted(files)[:5]):
                    print(f"       - {date}/{symbol}.npz")
                if len(files) > 5:
                    print(f"       ... 還有 {len(files) - 5} 個")
        print()

    # 報告檔案內部不一致的情況
    if inconsistent_files:
        print(f"{'='*80}")
        print(f"⚠️ 發現 {len(inconsistent_files)} 個檔案內部欄位長度不一致:")
        print(f"{'='*80}\n")

        for date, symbol, field_lengths in inconsistent_files[:10]:
            print(f"檔案: {date}/{symbol}.npz")
            for field, length in sorted(field_lengths.items()):
                print(f"  {field:20s}: {length:8d}")
            print()

        if len(inconsistent_files) > 10:
            print(f"... 還有 {len(inconsistent_files) - 10} 個檔案有問題")
    else:
        print(f"{'='*80}")
        print("✅ 所有檔案內部欄位長度都一致")
        print(f"{'='*80}")

    # 檢查是否有 last_prices, last_volumes, total_volumes
    print(f"\n{'='*80}")
    print("價格/成交量欄位檢查:")
    print(f"{'='*80}\n")

    fields_to_check = ['last_prices', 'last_volumes', 'total_volumes', 'volume_deltas']
    for field in fields_to_check:
        if field in length_stats:
            count = sum(len(files) for files in length_stats[field].values())
            print(f"  ✅ {field:20s}: 存在於 {count} 個檔案")
        else:
            print(f"  ❌ {field:20s}: 不存在")

    # 總結
    print(f"\n{'='*80}")
    print("總結:")
    print(f"{'='*80}")
    print(f"  檢查檔案數: {len(npz_files)}")
    print(f"  內部不一致: {len(inconsistent_files)}")

    # 檢查是否所有必要欄位都存在
    required_fields = ['features', 'labels', 'mids']
    missing_required = [f for f in required_fields if f not in length_stats]
    if missing_required:
        print(f"  ⚠️ 缺少必要欄位: {', '.join(missing_required)}")
    else:
        print(f"  ✅ 所有必要欄位都存在")

    # 檢查新增欄位
    new_fields = ['last_prices', 'last_volumes', 'total_volumes']
    has_new_fields = all(f in length_stats for f in new_fields)
    if has_new_fields:
        print(f"  ✅ 所有新增欄位都存在 (last_prices, last_volumes, total_volumes)")
    else:
        missing_new = [f for f in new_fields if f not in length_stats]
        print(f"  ⚠️ 缺少新增欄位: {', '.join(missing_new)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        npz_dir = sys.argv[1]
    else:
        npz_dir = "data/preprocessed_swing"

    check_npz_lengths(npz_dir)
