# -*- coding: utf-8 -*-
"""
analyze_current_labels.py - 分析當前標籤是否符合日內波段交易需求
=============================================================================
檢查點：
  1. 每天交易次數（目標：1-2 次/股/天）
  2. 平均持倉時間（目標：1-3 小時）
  3. 平均利潤（目標：≥1%）
  4. 標籤是否能識別趨勢

使用方式：
  python scripts/analyze_current_labels.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("分析當前標籤是否符合日內波段交易需求")
print("=" * 80)
print()

# 載入預處理數據（查看一天的標籤）
data_dir = Path("data/processed_v6/npz")

if not data_dir.exists():
    print(f"❌ 數據目錄不存在: {data_dir}")
    print("請先運行 extract_tw_stock_data_v6.py")
    sys.exit(1)

# 找到第一個 train 文件
train_files = sorted(data_dir.glob("train_*.npz"))
if not train_files:
    print("❌ 找不到訓練數據文件")
    sys.exit(1)

print(f"✓ 載入數據: {train_files[0].name}")
data = np.load(train_files[0])

# 檢查可用的鍵
print(f"✓ 可用數據鍵: {list(data.keys())}")
print()

# 提取標籤和元數據
if 'y_train' in data:
    labels = data['y_train']
elif 'labels' in data:
    labels = data['labels']
else:
    print("❌ 找不到標籤數據")
    sys.exit(1)

print(f"✓ 標籤總數: {len(labels):,}")
print()

# 分析標籤分布
unique, counts = np.unique(labels, return_counts=True)
print("【當前標籤分布】")
print("-" * 80)
label_names = {0: "下跌 (Class 0)", 1: "持平 (Class 1)", 2: "上漲 (Class 2)"}
for label, count in zip(unique, counts):
    pct = count / len(labels) * 100
    name = label_names.get(label, f"未知 ({label})")
    print(f"  {name}: {count:7,} ({pct:5.1f}%)")
print()

# 估算每天交易次數
# 假設：連續相同標籤 = 持倉中，標籤改變 = 新交易
def count_label_changes(labels):
    """計算標籤變化次數（交易次數的近似）"""
    changes = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            changes += 1
    return changes

# 計算交易信號（非持平標籤）
trade_signals = labels[labels != 1]  # 排除 Class 1（持平）
n_trades = count_label_changes(labels)

print("【交易頻率估算】")
print("-" * 80)
print(f"總樣本數: {len(labels):,}")
print(f"標籤變化次數: {n_trades:,}")
print(f"平均每次持倉樣本數: {len(labels) / (n_trades + 1):.1f}")
print()

# 假設數據覆蓋範圍
# 從文檔得知：aggregation_factor=10, seq_len=100
# 每個樣本 = 100 timesteps × 10 原始事件 = 1000 原始事件
samples_per_day_per_stock = 4.5 * 3600 / 10  # 約 1620 個聚合後的 tick

print("【日內波段交易需求檢查】")
print("-" * 80)

# 假設標籤變化 = 交易信號
trades_per_1000_samples = (n_trades / len(labels)) * 1000

print(f"✓ 每 1000 樣本的交易次數: {trades_per_1000_samples:.1f}")
print()

if trades_per_1000_samples > 50:
    print("⚠️ 警告：交易過於頻繁！")
    print("   - 當前配置可能產生過多交易信號")
    print("   - 不符合「每天 1-2 次交易」的需求")
    print("   - 建議：增加 lookforward 或 max_holding")
elif trades_per_1000_samples < 5:
    print("⚠️ 警告：交易過於稀少！")
    print("   - 當前配置可能錯過交易機會")
    print("   - 建議：降低 threshold 或 min_return")
else:
    print("✓ 交易頻率適中")

print()

# 持平比例檢查
neutral_ratio = counts[unique == 1][0] / len(labels) if 1 in unique else 0

print("【持平標籤比例】")
print("-" * 80)
print(f"持平比例: {neutral_ratio*100:.1f}%")
print()

if neutral_ratio < 0.3:
    print("⚠️ 警告：持平標籤過少！")
    print("   - 模型可能學不會「不交易」")
    print("   - 會產生過多錯誤交易")
elif neutral_ratio > 0.6:
    print("⚠️ 警告：持平標籤過多！")
    print("   - 可能錯過真正的趨勢")
    print("   - 建議：降低 threshold")
else:
    print("✓ 持平比例適中（30-60%）")

print()

# 建議
print("=" * 80)
print("建議")
print("=" * 80)
print("""
基於您的需求（日內波段，每天 1-2 次交易，≥1% 利潤）：

【當前配置問題】
  - max_holding: 40 bars → 太短（只有 4-8 分鐘）
  - min_return: 0.25%   → 太小（會被手續費吃掉）
  - 標籤方法: Triple-Barrier → 適合高頻，不適合波段

【建議改用趨勢標籤】⭐⭐⭐⭐⭐

在 config_pro_v5_ml_optimal.yaml 中修改：

方案 A: 純趨勢標籤（最適合）
  labeling_method: "trend_adaptive"
  trend_labeling:
    lookforward: 150        # 150 bars ≈ 1.5-3 小時
    threshold: 0.01         # 1% 閾值（符合利潤目標）
    volatility_adjust: true # 波動率自適應

方案 B: Triple-Barrier 重新校準（折衷）
  triple_barrier:
    max_holding: 250        # 250 bars ≈ 2.5-5 小時
    min_return: 0.01        # 1% 最小利潤
    pt_multiplier: 5.0      # 更寬止盈（讓利潤跑）
    sl_multiplier: 3.0      # 相對緊止損（控制風險）

【下一步】
  1. 修改配置文件
  2. 重新運行 preprocess_single_day.py（或 extract_tw_stock_data_v6.py）
  3. 檢查新標籤分布
  4. 重新訓練 DeepLOB
""")
print("=" * 80)
