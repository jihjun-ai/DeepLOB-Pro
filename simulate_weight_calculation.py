#!/usr/bin/env python3
"""
模擬權重計算過程，找出為什麼 Class 1 權重極低
"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ===== 模擬參數 =====
tau = 80.0
scale = 1.0
use_log_scale = True

# ===== 模擬 Class 1 的典型樣本 =====
# 從診斷結果反推：權重中位數 0.0019

# 假設 1: Class 1 是「時間到」觸發（vertical barrier）
# 這些樣本的收益極小（< min_return = 0.0015）

# 模擬 1000 個 Class 1 樣本
np.random.seed(42)
n_samples = 1000

# Class 1 的收益分布（絕對值）
# 大部分極小（< 0.0015），少數中等（0.0015~0.005）
ret_class1 = np.concatenate([
    np.random.uniform(0.00001, 0.0005, 700),  # 70% 極小收益
    np.random.uniform(0.0005, 0.0015, 200),   # 20% 小收益
    np.random.uniform(0.0015, 0.005, 100),    # 10% 中等收益
])

# 觸發時間（大部分是 max_holding = 40）
tt_class1 = np.concatenate([
    np.full(800, 40),  # 80% 時間到
    np.random.randint(20, 40, 150),  # 15% 接近時間到
    np.random.randint(1, 20, 50),    # 5% 快速觸發
])

print("=" * 70)
print("Class 1 樣本分布（假設）")
print("=" * 70)
print(f"收益統計:")
print(f"  Min: {ret_class1.min():.6f}")
print(f"  Median: {np.median(ret_class1):.6f}")
print(f"  Mean: {ret_class1.mean():.6f}")
print(f"  Max: {ret_class1.max():.6f}")
print(f"\n觸發時間統計:")
print(f"  Min: {tt_class1.min()}")
print(f"  Median: {np.median(tt_class1):.0f}")
print(f"  Mean: {tt_class1.mean():.1f}")
print(f"  Max: {tt_class1.max()}")

# ===== 步驟 1: 收益權重 =====
print("\n" + "=" * 70)
print("步驟 1: 計算收益權重")
print("=" * 70)

if use_log_scale:
    ret_weight = np.log1p(np.abs(ret_class1) * 1000) * scale
    print(f"公式: log(1 + |ret| * 1000) * {scale}")
else:
    ret_weight = np.abs(ret_class1) * scale
    print(f"公式: |ret| * {scale}")

print(f"\n收益權重統計:")
print(f"  Min: {ret_weight.min():.6f}")
print(f"  Median: {np.median(ret_weight):.6f}")
print(f"  Mean: {ret_weight.mean():.6f}")
print(f"  Max: {ret_weight.max():.6f}")

# 顯示幾個典型例子
print(f"\n典型例子:")
for i in [0, 500, 700, 900, 999]:
    ret = ret_class1[i]
    rw = ret_weight[i]
    print(f"  ret={ret:.6f} → ret_weight={rw:.6f}")

# ===== 步驟 2: 時間衰減 =====
print("\n" + "=" * 70)
print("步驟 2: 應用時間衰減")
print("=" * 70)
print(f"公式: ret_weight × exp(-tt / {tau})")

time_decay = np.exp(-tt_class1 / float(tau))
base = ret_weight * time_decay

print(f"\n時間衰減統計:")
print(f"  Min: {time_decay.min():.6f}")
print(f"  Median: {np.median(time_decay):.6f}")
print(f"  Mean: {time_decay.mean():.6f}")
print(f"  Max: {time_decay.max():.6f}")

print(f"\n基礎權重統計 (ret_weight × time_decay):")
print(f"  Min: {base.min():.6f}")
print(f"  Median: {np.median(base):.6f}")
print(f"  Mean: {base.mean():.6f}")
print(f"  Max: {base.max():.6f}")

# ===== 步驟 3: 類別平衡 =====
print("\n" + "=" * 70)
print("步驟 3: 類別平衡權重")
print("=" * 70)

# 模擬三個類別的分布（從診斷數據）
y_all = np.concatenate([
    np.full(444314, 0),  # Class 0: 39.30%
    np.full(316564, 1),  # Class 1: 28.00%
    np.full(369699, 2),  # Class 2: 32.70%
])

classes = np.array([0, 1, 2])
cls_w = compute_class_weight('balanced', classes=classes, y=y_all)

print(f"類別分布:")
print(f"  Class 0: {444314:,} (39.30%)")
print(f"  Class 1: {316564:,} (28.00%)")
print(f"  Class 2: {369699:,} (32.70%)")

print(f"\n類別平衡權重:")
for i, w in enumerate(cls_w):
    print(f"  Class {i}: {w:.6f}")

# 對 Class 1 應用類別權重
class1_weight_multiplier = cls_w[1]
final_weight = base * class1_weight_multiplier

print(f"\n最終權重統計 (base × class_weight[1]):")
print(f"  Class 1 權重倍數: {class1_weight_multiplier:.6f}")
print(f"  Min: {final_weight.min():.6f}")
print(f"  Median: {np.median(final_weight):.6f}")
print(f"  Mean: {final_weight.mean():.6f}")
print(f"  Max: {final_weight.max():.6f}")

# ===== 步驟 4: 歸一化 =====
print("\n" + "=" * 70)
print("步驟 4: 歸一化（這裡只針對 Class 1 樣本）")
print("=" * 70)
print("注意：實際上會與所有類別一起歸一化")

normalized = final_weight / final_weight.mean()
print(f"\nClass 1 內部歸一化後:")
print(f"  Min: {normalized.min():.6f}")
print(f"  Median: {np.median(normalized):.6f}")
print(f"  Mean: {normalized.mean():.6f}")
print(f"  Max: {normalized.max():.6f}")

# ===== 對比診斷結果 =====
print("\n" + "=" * 70)
print("與診斷結果對比")
print("=" * 70)
print(f"診斷報告顯示:")
print(f"  Class 1 平均權重: 0.2158")
print(f"  Class 1 權重中位數: 0.0019  ← 極低！")
print(f"  Class 1 權重範圍: [0.0004, 16.5129]")

print(f"\n模擬結果（Class 1 內部歸一化）:")
print(f"  平均權重: {normalized.mean():.6f}")
print(f"  權重中位數: {np.median(normalized):.6f}")
print(f"  權重範圍: [{normalized.min():.6f}, {normalized.max():.6f}]")

print("\n" + "=" * 70)
print("結論分析")
print("=" * 70)
print("如果模擬中位數遠大於 0.0019，說明：")
print("1. Class 1 的實際收益分布比假設更極端（更多 < 0.0001 的樣本）")
print("2. 或者有其他因素影響權重計算")
print("\n如果模擬中位數接近 0.0019，說明：")
print("1. Class 1 確實大部分是極小收益樣本")
print("2. 對數縮放仍無法解決「收益太小」的本質問題")
