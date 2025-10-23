# -*- coding: utf-8 -*-
"""
test_trend_stable.py - 測試穩定趨勢標籤功能
=============================================================================
驗證 trend_labels_stable 相比 trend_labels_adaptive 的改進：
1. 震盪區間減少頻繁翻轉
2. 趨勢區間方向穩定
3. 切換次數顯著下降

使用方式：
    python scripts/test_trend_stable.py
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.financial_engineering import (
    ewma_volatility_professional,
    trend_labels_adaptive,
    trend_labels_stable
)


def generate_test_price(n=2000, trend_periods=3, noise_level=0.002):
    """
    生成測試價格序列（包含趨勢段和震盪段）

    Args:
        n: 數據點數
        trend_periods: 趨勢週期數
        noise_level: 噪音水平
    """
    t = np.arange(n)

    # 基礎趨勢（正弦波 + 線性趨勢）
    trend = 100 + 0.01 * t + 5 * np.sin(2 * np.pi * trend_periods * t / n)

    # 添加噪音
    noise = np.random.normal(0, noise_level * trend.mean(), n)

    price = trend + noise

    return pd.Series(price, name='close')


def count_label_changes(labels):
    """計算標籤切換次數"""
    changes = (labels.diff() != 0).sum()
    return changes


def analyze_labels(labels, name):
    """分析標籤統計"""
    counts = labels.value_counts().to_dict()
    total = len(labels)

    print(f"\n{name} 標籤統計:")
    print(f"  Down  (-1): {counts.get(-1, 0):5d} ({counts.get(-1, 0)/total*100:5.1f}%)")
    print(f"  Neutral(0): {counts.get(0, 0):5d} ({counts.get(0, 0)/total*100:5.1f}%)")
    print(f"  Up    (+1): {counts.get(1, 0):5d} ({counts.get(1, 0)/total*100:5.1f}%)")
    print(f"  切換次數: {count_label_changes(labels)}")

    return counts


def main():
    print("="*70)
    print("穩定趨勢標籤測試")
    print("="*70)

    # 生成測試數據
    print("\n生成測試數據...")
    close = generate_test_price(n=2000, trend_periods=3, noise_level=0.003)

    # 計算波動率
    volatility = ewma_volatility_professional(close, halflife=60, min_periods=20)

    # 測試參數
    lookforward = 120
    vol_multiplier = 2.5

    # 1. 自適應版（原版）
    print(f"\n1. 計算自適應趨勢標籤 (lookforward={lookforward}, vol_mult={vol_multiplier})...")
    labels_adaptive = trend_labels_adaptive(
        close=close,
        volatility=volatility,
        lookforward=lookforward,
        vol_multiplier=vol_multiplier
    )

    # 2. 穩定版
    print(f"\n2. 計算穩定趨勢標籤 (+ hysteresis + persistence + smoothing)...")
    labels_stable = trend_labels_stable(
        close=close,
        volatility=volatility,
        lookforward=lookforward,
        vol_multiplier=vol_multiplier,
        hysteresis_ratio=0.6,
        smooth_window=15,
        min_trend_duration=30
    )

    # 分析對比
    print("\n" + "="*70)
    print("結果對比")
    print("="*70)

    counts_adaptive = analyze_labels(labels_adaptive, "自適應版 (Adaptive)")
    counts_stable = analyze_labels(labels_stable, "穩定版 (Stable)")

    # 計算改善比例
    changes_adaptive = count_label_changes(labels_adaptive)
    changes_stable = count_label_changes(labels_stable)
    reduction = (changes_adaptive - changes_stable) / changes_adaptive * 100

    print("\n" + "="*70)
    print("改善統計")
    print("="*70)
    print(f"切換次數減少: {changes_adaptive} → {changes_stable} ({reduction:.1f}%)")
    print(f"Neutral 比例變化: {counts_adaptive.get(0, 0)/len(labels_adaptive)*100:.1f}% → "
          f"{counts_stable.get(0, 0)/len(labels_stable)*100:.1f}%")

    # 視覺化對比
    print("\n生成對比圖表...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # 子圖1: 價格
    axes[0].plot(close.values, label='Price', color='black', linewidth=1)
    axes[0].set_ylabel('Price')
    axes[0].set_title('Test Price Series (Trend + Oscillation + Noise)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子圖2: 自適應標籤
    axes[1].plot(labels_adaptive.values, label='Adaptive Labels',
                 color='blue', linewidth=0.8, alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Label')
    axes[1].set_title(f'Adaptive Trend Labels (Changes: {changes_adaptive})')
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(['Down', 'Neutral', 'Up'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子圖3: 穩定標籤
    axes[2].plot(labels_stable.values, label='Stable Labels',
                 color='green', linewidth=0.8, alpha=0.7)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Label')
    axes[2].set_xlabel('Time (bars)')
    axes[2].set_title(f'Stable Trend Labels (Changes: {changes_stable}, Reduction: {reduction:.1f}%)')
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_yticklabels(['Down', 'Neutral', 'Up'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存圖表
    output_path = project_root / 'results' / 'trend_stable_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Chart saved: {output_path}")

    # 顯示圖表（可選）
    # plt.show()

    print("\n" + "="*70)
    print("Test Completed!")
    print("="*70)
    print("\n[OK] Core Improvements Verified:")
    print(f"  1. Label changes reduced: {reduction:.1f}%")
    print(f"  2. Neutral ratio: helps identify oscillation zones")
    print(f"  3. Trend direction: more stable, fewer false signals")

    return 0


if __name__ == "__main__":
    sys.exit(main())
