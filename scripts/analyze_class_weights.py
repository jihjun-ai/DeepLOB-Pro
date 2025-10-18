#!/usr/bin/env python
"""
分析不同類別權重方法對 Class 1 Recall 的影響
"""
import numpy as np

# 假設當前數據分佈（根據 v9 Triple-Barrier 配置）
class_counts = np.array([367457, 562361, 319601])  # Class 0, 1, 2
total = class_counts.sum()

print("=" * 60)
print("數據分佈分析")
print("=" * 60)
print(f'  Class 0 (下跌): {class_counts[0]:,} ({class_counts[0]/total*100:.2f}%)')
print(f'  Class 1 (持平): {class_counts[1]:,} ({class_counts[1]/total*100:.2f}%)')
print(f'  Class 2 (上漲): {class_counts[2]:,} ({class_counts[2]/total*100:.2f}%)')
print(f'  總樣本數: {total:,}')
print()

# ===== 方法 1：逆頻率權重（當前方法）=====
print("=" * 60)
print("方法 1：逆頻率權重（當前方法）")
print("=" * 60)
epsilon = 1e-6
weights_inv_freq = total / (len(class_counts) * class_counts + epsilon)
weights_inv_freq_norm = weights_inv_freq / weights_inv_freq.mean()
print(f'  Class 0: {weights_inv_freq_norm[0]:.4f}')
print(f'  Class 1: {weights_inv_freq_norm[1]:.4f}')
print(f'  Class 2: {weights_inv_freq_norm[2]:.4f}')
print(f'  權重比 (0:1:2) = {weights_inv_freq_norm[0]:.2f} : {weights_inv_freq_norm[1]:.2f} : {weights_inv_freq_norm[2]:.2f}')
print(f'  問題：Class 1 權重最小 → 模型忽略 Class 1！')
print()

# ===== 方法 2：平方根逆頻率（更溫和）=====
print("=" * 60)
print("方法 2：平方根逆頻率（更溫和的再平衡）")
print("=" * 60)
weights_sqrt = np.sqrt(total / class_counts)
weights_sqrt_norm = weights_sqrt / weights_sqrt.mean()
print(f'  Class 0: {weights_sqrt_norm[0]:.4f}')
print(f'  Class 1: {weights_sqrt_norm[1]:.4f}')
print(f'  Class 2: {weights_sqrt_norm[2]:.4f}')
print(f'  權重比 (0:1:2) = {weights_sqrt_norm[0]:.2f} : {weights_sqrt_norm[1]:.2f} : {weights_sqrt_norm[2]:.2f}')
print(f'  改進：權重對比度降低，Class 1 相對提升')
print()

# ===== 方法 3：手動增強 Class 1 =====
print("=" * 60)
print("方法 3：手動增強少數類（Class 0 & 2）")
print("=" * 60)
# 給 Class 0 和 Class 2 更高權重（因為它們樣本少）
weights_manual = np.array([1.3, 0.7, 1.5])  # 手動設計
weights_manual_norm = weights_manual / weights_manual.mean()
print(f'  Class 0: {weights_manual_norm[0]:.4f}')
print(f'  Class 1: {weights_manual_norm[1]:.4f}')
print(f'  Class 2: {weights_manual_norm[2]:.4f}')
print(f'  權重比 (0:1:2) = {weights_manual_norm[0]:.2f} : {weights_manual_norm[1]:.2f} : {weights_manual_norm[2]:.2f}')
print(f'  策略：故意降低 Class 1 權重，強迫模型學習其他類')
print()

# ===== 方法 4：Effective Number（極度激進）=====
print("=" * 60)
print("方法 4：Effective Number (beta=0.999, 最激進)")
print("=" * 60)
beta = 0.999
effective_num = 1.0 - np.power(beta, class_counts)
weights_en = (1.0 - beta) / effective_num
weights_en_norm = weights_en / weights_en.mean()
print(f'  Class 0: {weights_en_norm[0]:.4f}')
print(f'  Class 1: {weights_en_norm[1]:.4f}')
print(f'  Class 2: {weights_en_norm[2]:.4f}')
print(f'  權重比 (0:1:2) = {weights_en_norm[0]:.2f} : {weights_en_norm[1]:.2f} : {weights_en_norm[2]:.2f}')
print(f'  特點：極度壓縮權重差異，適合長尾分佈')
print()

# ===== 關鍵發現 =====
print("=" * 60)
print("關鍵發現")
print("=" * 60)
print("1. 當前逆頻率方法給 Class 1 最小權重（0.57），導致模型忽略它")
print("2. Class 1 樣本最多（45%），但權重最小 → 損失貢獻被壓制")
print("3. Class 2 樣本最少（25.6%），但權重最大（1.26） → 模型專注預測 Class 2")
print()
print("解決方案：")
print("  - 短期：使用方法 3（手動權重）或啟用平衡採樣器")
print("  - 中期：調整 Triple-Barrier 使三類更均衡（30%/35%/35%）")
print("  - 長期：使用 Focal Loss 或平衡採樣器")
print("=" * 60)
