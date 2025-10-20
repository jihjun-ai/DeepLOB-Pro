#!/usr/bin/env python3
"""
緊急權重診斷腳本
檢查 npz 檔案中的 weights 是否導致 Class 1 被過度壓制
"""
import numpy as np
import sys

def diagnose_weights(npz_path):
    """診斷權重分布"""
    print("=" * 70)
    print(f"診斷檔案: {npz_path}")
    print("=" * 70)

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"❌ 無法載入檔案: {e}")
        return False

    # 1. 檢查基本資訊
    print(f"\n1️⃣  檔案內容:")
    print(f"   Keys: {list(data.keys())}")

    if 'y' not in data:
        print("❌ 缺少標籤 'y'")
        return False

    if 'weights' not in data:
        print("❌ 缺少權重 'weights'（但 config 啟用了 use_sample_weights）")
        return False

    # 2. 標籤分布
    labels = data['y']
    weights = data['weights']

    print(f"\n2️⃣  標籤分布:")
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for u, c in zip(unique, counts):
        pct = 100 * c / total
        print(f"   Class {u}: {c:,} ({pct:.2f}%)")

    # 3. 權重基本統計
    print(f"\n3️⃣  權重統計:")
    print(f"   Shape: {weights.shape}")
    print(f"   Mean: {weights.mean():.4f} (應≈1.0)")
    print(f"   Std: {weights.std():.4f}")
    print(f"   Min: {weights.min():.4f}")
    print(f"   Max: {weights.max():.4f}")
    print(f"   Median: {np.median(weights):.4f}")

    # 檢查異常值
    nan_count = np.isnan(weights).sum()
    inf_count = np.isinf(weights).sum()
    neg_count = (weights < 0).sum()
    zero_count = (weights == 0).sum()

    if nan_count > 0:
        print(f"   ❌ NaN: {nan_count}")
    if inf_count > 0:
        print(f"   ❌ Inf: {inf_count}")
    if neg_count > 0:
        print(f"   ❌ 負值: {neg_count}")
    if zero_count > 0:
        print(f"   ⚠️  零值: {zero_count}")

    # 4. 各類別的權重統計（關鍵！）
    print(f"\n4️⃣  各類別權重統計:")
    for u in unique:
        class_mask = (labels == u)
        class_weights = weights[class_mask]
        class_count = class_mask.sum()

        # 計算加權後的「有效樣本數」
        effective_samples = class_weights.sum()
        avg_weight = class_weights.mean()

        print(f"   Class {u}:")
        print(f"      實際樣本數: {class_count:,}")
        print(f"      平均權重: {avg_weight:.4f}")
        print(f"      加權後有效樣本數: {effective_samples:,.1f}")
        print(f"      權重範圍: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
        print(f"      權重中位數: {np.median(class_weights):.4f}")

    # 5. 關鍵診斷：類別間的權重比
    print(f"\n5️⃣  類別間權重比（關鍵診斷）:")
    class_effective = {}
    for u in unique:
        class_mask = (labels == u)
        class_effective[u] = weights[class_mask].sum()

    # 計算比例
    total_effective = sum(class_effective.values())
    for u in unique:
        pct = 100 * class_effective[u] / total_effective
        print(f"   Class {u}: {pct:.2f}% (加權後)")

    # 檢查是否極度不平衡
    min_pct = min(class_effective.values()) / total_effective * 100
    max_pct = max(class_effective.values()) / total_effective * 100
    imbalance_ratio = max_pct / min_pct

    print(f"\n6️⃣  不平衡度:")
    print(f"   最小類別佔比: {min_pct:.2f}%")
    print(f"   最大類別佔比: {max_pct:.2f}%")
    print(f"   不平衡比: {imbalance_ratio:.2f}x")

    if imbalance_ratio > 10:
        print(f"   ❌ 嚴重不平衡！某類別被過度壓制")
        return False
    elif imbalance_ratio > 5:
        print(f"   ⚠️  中度不平衡")
        return True
    else:
        print(f"   ✅ 平衡度可接受")
        return True

    return True

if __name__ == "__main__":
    # 診斷訓練集
    train_path = "data/processed_v5_fixed/npz/stock_embedding_train.npz"

    print("\n" + "=" * 70)
    print("開始診斷訓練集權重")
    print("=" * 70)

    success = diagnose_weights(train_path)

    if not success:
        print("\n" + "=" * 70)
        print("❌ 診斷失敗：權重設定有嚴重問題")
        print("=" * 70)
        print("\n建議：")
        print("1. 重新生成數據（關閉 sample_weights）")
        print("2. 或者在配置中設定 use_sample_weights: false")
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("✅ 診斷完成")
        print("=" * 70)
        sys.exit(0)
