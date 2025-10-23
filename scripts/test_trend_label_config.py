# -*- coding: utf-8 -*-
"""
test_trend_label_config.py - 測試趨勢標籤配置
=============================================================================
快速測試趨勢標籤是否正常工作

使用方式：
  python scripts/test_trend_label_config.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.utils.yaml_manager import YAMLManager
from src.utils.financial_engineering import (
    ewma_volatility_professional,
    trend_labels_adaptive
)

print("=" * 80)
print("測試趨勢標籤配置")
print("=" * 80)
print()

# 載入配置
print("✓ 載入配置: configs/config_intraday_swing.yaml")
config_path = project_root / "configs" / "config_intraday_swing.yaml"
yaml_manager = YAMLManager(config_path)
config = yaml_manager.load_config()

print(f"✓ 標籤方法: {config.get('labeling_method', 'triple_barrier')}")
print()

# 生成測試數據
print("✓ 生成測試數據（模擬日內波段）")
np.random.seed(42)
n_points = 500

# 模擬一個有趨勢的價格序列
# 前 200: 上漲趨勢 (+2%)
# 200-350: 橫盤 (±0.5%)
# 350-500: 下跌趨勢 (-1.5%)

prices = np.zeros(n_points) + 100.0
prices[0:200] = 100.0 + np.linspace(0, 2.0, 200) + np.random.randn(200) * 0.1
prices[200:350] = prices[199] + np.random.randn(150) * 0.2
prices[350:500] = prices[349] + np.linspace(0, -1.5, 150) + np.random.randn(150) * 0.1

close = pd.Series(prices, name='close')

print(f"  價格範圍: {close.min():.2f} ~ {close.max():.2f}")
print(f"  整體變化: {close.iloc[0]:.2f} → {close.iloc[-1]:.2f} ({(close.iloc[-1]/close.iloc[0]-1)*100:.2f}%)")
print()

# 計算波動率
print("✓ 計算波動率")
vol = ewma_volatility_professional(close, halflife=60)
print(f"  平均波動率: {vol.mean():.4f}")
print()

# 測試趨勢標籤
print("✓ 測試趨勢標籤函數")
trend_cfg = config.get('trend_labeling', {})
lookforward = trend_cfg.get('lookforward', 150)
vol_multiplier = trend_cfg.get('vol_multiplier', 2.0)

print(f"  lookforward: {lookforward}")
print(f"  vol_multiplier: {vol_multiplier}")
print()

try:
    labels = trend_labels_adaptive(
        close=close,
        volatility=vol,
        lookforward=lookforward,
        vol_multiplier=vol_multiplier
    )

    print("✓ 趨勢標籤生成成功！")
    print()

    # 統計標籤分布
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    label_names = {-1: "下跌", 0: "持平", 1: "上漲"}

    print("【標籤分布】")
    print("-" * 80)
    for label, count in zip(unique, counts):
        pct = count / total * 100
        name = label_names.get(label, f"未知 ({label})")
        print(f"  {name}: {count:4d} ({pct:5.1f}%)")
    print()

    # 分析各階段
    print("【各階段標籤】")
    print("-" * 80)

    # 上漲階段 (0-200)
    up_labels = labels.iloc[0:200]
    up_dist = up_labels.value_counts()
    print(f"上漲階段 (0-200):")
    print(f"  上漲: {up_dist.get(1, 0)} ({up_dist.get(1, 0)/len(up_labels)*100:.1f}%)")
    print(f"  持平: {up_dist.get(0, 0)} ({up_dist.get(0, 0)/len(up_labels)*100:.1f}%)")
    print(f"  下跌: {up_dist.get(-1, 0)} ({up_dist.get(-1, 0)/len(up_labels)*100:.1f}%)")
    print()

    # 橫盤階段 (200-350)
    neutral_labels = labels.iloc[200:350]
    neutral_dist = neutral_labels.value_counts()
    print(f"橫盤階段 (200-350):")
    print(f"  上漲: {neutral_dist.get(1, 0)} ({neutral_dist.get(1, 0)/len(neutral_labels)*100:.1f}%)")
    print(f"  持平: {neutral_dist.get(0, 0)} ({neutral_dist.get(0, 0)/len(neutral_labels)*100:.1f}%)")
    print(f"  下跌: {neutral_dist.get(-1, 0)} ({neutral_dist.get(-1, 0)/len(neutral_labels)*100:.1f}%)")
    print()

    # 下跌階段 (350-500)
    down_labels = labels.iloc[350:500]
    down_dist = down_labels.value_counts()
    print(f"下跌階段 (350-500):")
    print(f"  上漲: {down_dist.get(1, 0)} ({down_dist.get(1, 0)/len(down_labels)*100:.1f}%)")
    print(f"  持平: {down_dist.get(0, 0)} ({down_dist.get(0, 0)/len(down_labels)*100:.1f}%)")
    print(f"  下跌: {down_dist.get(-1, 0)} ({down_dist.get(-1, 0)/len(down_labels)*100:.1f}%)")
    print()

    # 評估
    print("=" * 80)
    print("評估")
    print("=" * 80)

    up_ratio = up_dist.get(1, 0) / len(up_labels)
    down_ratio = down_dist.get(-1, 0) / len(down_labels)
    neutral_ratio = neutral_dist.get(0, 0) / len(neutral_labels)

    print(f"✓ 上漲階段識別率: {up_ratio*100:.1f}% (目標 > 50%)")
    print(f"✓ 下跌階段識別率: {down_ratio*100:.1f}% (目標 > 50%)")
    print(f"✓ 橫盤階段持平率: {neutral_ratio*100:.1f}% (目標 > 50%)")
    print()

    if up_ratio > 0.5 and down_ratio > 0.5 and neutral_ratio > 0.5:
        print("✅ 趨勢標籤工作正常！")
        print("   可以用於實際數據生成")
    else:
        print("⚠️  趨勢標籤可能需要調整參數")
        if up_ratio <= 0.5:
            print("   - 上漲階段識別不足，考慮降低 vol_multiplier")
        if down_ratio <= 0.5:
            print("   - 下跌階段識別不足，考慮降低 vol_multiplier")
        if neutral_ratio <= 0.5:
            print("   - 橫盤識別不足，考慮提高 vol_multiplier")

    print()
    print("=" * 80)

except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
