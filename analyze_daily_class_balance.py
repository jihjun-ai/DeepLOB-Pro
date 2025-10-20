#!/usr/bin/env python3
"""
分析：按日計算類別權重是否會導致極端值
"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

print("=" * 70)
print("模擬：單日樣本數不同時的類別權重")
print("=" * 70)

# 模擬不同情況
scenarios = [
    ("正常日", [50, 40, 60]),  # 三類都有足夠樣本
    ("Class 1 稀少", [50, 5, 60]),  # Class 1 很少
    ("Class 1 極少", [50, 1, 60]),  # Class 1 只有 1 個
    ("Class 1 缺席", [50, 0, 60]),  # Class 1 完全沒有！
]

for name, counts in scenarios:
    print(f"\n{name}:")
    print(f"  Class 0: {counts[0]} 樣本")
    print(f"  Class 1: {counts[1]} 樣本")
    print(f"  Class 2: {counts[2]} 樣本")

    # 生成標籤
    y = np.concatenate([
        np.full(counts[0], 0),
        np.full(counts[1], 1),
        np.full(counts[2], 2),
    ])

    if len(y) == 0:
        print("  ❌ 該日無樣本，跳過")
        continue

    # 檢查是否有類別缺席
    unique_classes = np.unique(y)
    all_classes = np.array([0, 1, 2])

    try:
        # 嘗試計算權重（sklearn 的方式）
        cls_w = compute_class_weight('balanced', classes=all_classes, y=y)

        print(f"  類別權重:")
        for i, w in enumerate(cls_w):
            print(f"    Class {i}: {w:.6f}")

        # 檢查極端值
        if cls_w.max() > 10:
            print(f"  ⚠️  最大權重 {cls_w.max():.2f} 超過 10")
        if cls_w.min() < 0.1:
            print(f"  ⚠️  最小權重 {cls_w.min():.2f} 小於 0.1")

    except Exception as e:
        print(f"  ❌ 計算失敗: {e}")

print("\n" + "=" * 70)
print("結論分析")
print("=" * 70)
print("1. 如果某類別缺席，sklearn 會怎麼處理？")
print("2. 類別權重是否會產生極端值？")
print("3. 是否需要 Laplace 平滑？")
