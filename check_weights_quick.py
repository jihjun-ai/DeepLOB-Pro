#!/usr/bin/env python3
"""快速檢查權重設定"""
import numpy as np

# 載入訓練數據
data = np.load('data/processed_v5_fixed/npz/stock_embedding_train.npz')

print("=" * 70)
print("資料層檢查 - NPZ 檔案內容")
print("=" * 70)
print(f"Keys: {list(data.keys())}")
print(f"Labels shape: {data['y'].shape}")
print(f"Weights: {'✅ EXISTS' if 'weights' in data else '❌ NOT FOUND'}")

# 標籤分布
labels = data['y']
unique, counts = np.unique(labels, return_counts=True)
total = len(labels)
print(f"\nLabel distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c:,} ({100*c/total:.2f}%)")

# 權重統計
if 'weights' in data:
    weights = data['weights']
    print(f"\n權重統計:")
    print(f"  Shape: {weights.shape}")
    print(f"  Mean: {weights.mean():.4f} (應≈1.0)")
    print(f"  Min: {weights.min():.4f}")
    print(f"  Max: {weights.max():.4f}")
    print(f"  NaN count: {np.isnan(weights).sum()}")
    print(f"  Inf count: {np.isinf(weights).sum()}")
    print(f"  Negative count: {(weights < 0).sum()}")

    # 每個類別的平均權重
    print(f"\n各類別平均權重:")
    for u in unique:
        class_weights = weights[labels == u]
        print(f"  Class {u}: mean={class_weights.mean():.4f}, "
              f"min={class_weights.min():.4f}, max={class_weights.max():.4f}")
else:
    print("\n❌ 警告: weights 欄位不存在！")
    print("   → use_sample_weights=true 但數據沒有 weights")
