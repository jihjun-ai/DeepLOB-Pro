#!/usr/bin/env python3
"""模擬權重計算流程,驗證裁剪邏輯"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# 模擬一天的標籤分布（Class 1 樣本很少的極端情況）
y = np.array([0]*100 + [1]*1 + [2]*120)  # Class 1 只有 1 個樣本
print(f"標籤分布: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}, Class 2={np.sum(y==2)}")

# 模擬收益（Class 1 收益極小）
ret = np.concatenate([
    np.random.uniform(0.002, 0.005, 100),  # Class 0
    np.array([0.0001]),                     # Class 1 (極小)
    np.random.uniform(0.002, 0.005, 120)   # Class 2
])

# 模擬時間衰減
tt = np.random.uniform(0, 40, len(y))

# ===== 按照修改後的邏輯計算權重 =====

# 1. 基礎權重（對數縮放 + 最小值保護）
use_log_scale = True
scale = 1.0
tau = 80.0

if use_log_scale:
    ret_weight = np.log1p(np.abs(ret) * 1000) * scale
    ret_weight = np.maximum(ret_weight, 0.1)  # 🔧 最小權重 0.1
else:
    ret_weight = np.abs(ret) * scale

# 2. 時間衰減
time_decay = np.exp(-tt / tau)
base = ret_weight * time_decay
base = np.clip(base, 0.05, None)

print(f"\nBase 權重統計:")
print(f"  範圍: [{base.min():.4f}, {base.max():.4f}]")
print(f"  均值: {base.mean():.4f}")
print(f"  Class 1 base 權重: {base[y==1]}")

# 3. 類別平衡權重（修改後：裁剪 + 歸一化）
classes = np.array([0, 1, 2])
cls_w = compute_class_weight('balanced', classes=classes, y=y)

print(f"\n類別權重（原始 sklearn 計算）:")
print(f"  Class 0: {cls_w[0]:.4f}")
print(f"  Class 1: {cls_w[1]:.4f}")  # ← 這裡會非常大！
print(f"  Class 2: {cls_w[2]:.4f}")

# 🔧 關鍵修正：裁剪類別權重
cls_w_clipped = np.clip(cls_w, 0.5, 3.0)
cls_w_clipped = cls_w_clipped / cls_w_clipped.mean()

print(f"\n類別權重（裁剪 + 歸一化後）:")
print(f"  Class 0: {cls_w_clipped[0]:.4f}")
print(f"  Class 1: {cls_w_clipped[1]:.4f}")  # ← 應該被限制在 3.0 內
print(f"  Class 2: {cls_w_clipped[2]:.4f}")

# 4. 合併權重
w_map_old = dict(zip(classes, cls_w))
cw_old = np.array([w_map_old[yi] for yi in y])
w_old = base * cw_old

w_map_new = dict(zip(classes, cls_w_clipped))
cw_new = np.array([w_map_new[yi] for yi in y])
w_new = base * cw_new

# 5. 最終裁剪
w_old = np.clip(w_old, 0.1, 5.0)
w_new = np.clip(w_new, 0.1, 5.0)

# 6. 歸一化
w_old = w_old / w_old.mean()
w_new = w_new / w_new.mean()

print(f"\n最終權重對比:")
print(f"\n【未裁剪類別權重】:")
print(f"  Class 1 權重範圍: [{w_old[y==1].min():.4f}, {w_old[y==1].max():.4f}]")
print(f"  Class 1 平均權重: {w_old[y==1].mean():.4f}")

print(f"\n【已裁剪類別權重】:")
print(f"  Class 1 權重範圍: [{w_new[y==1].min():.4f}, {w_new[y==1].max():.4f}]")
print(f"  Class 1 平均權重: {w_new[y==1].mean():.4f}")

# 計算加權後的類別分布
def weighted_dist(y, w):
    total = 0
    for c in [0, 1, 2]:
        total += w[y==c].sum()

    for c in [0, 1, 2]:
        pct = w[y==c].sum() / total * 100
        print(f"  Class {c}: {pct:.2f}%")

print(f"\n【未裁剪】加權後類別分布:")
weighted_dist(y, w_old)

print(f"\n【已裁剪】加權後類別分布:")
weighted_dist(y, w_new)

print(f"\n結論:")
print(f"  如果診斷結果顯示 Class 1 平均權重 < 0.5,")
print(f"  說明數據生成時使用的是【未裁剪】版本的腳本！")
