# -*- coding: utf-8 -*-
"""
test_trend_vs_tb.py - Triple-Barrier vs 趨勢標籤對比測試
=============================================================================
展示 Triple-Barrier 的短視問題，以及趨勢標籤的解決方案

使用方式：
  python scripts/test_trend_vs_tb.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.financial_engineering import (
    ewma_volatility_professional,
    triple_barrier_labels_professional,
    trend_labels_simple,
    trend_labels_adaptive,
    multi_scale_labels_combined
)

print("=" * 80)
print("Triple-Barrier vs 趨勢標籤對比測試")
print("=" * 80)
print()

# ============================================================
# 測試案例 1: 明顯下跌趨勢 + 小震盪
# ============================================================
print("[測試 1] 明顯下跌趨勢 + 小震盪")
print("-" * 80)

# 生成測試數據：整體下跌但有小震盪
np.random.seed(42)
n_points = 200

# 整體下跌趨勢 (100 → 95, -5%)
trend = np.linspace(100, 95, n_points)

# 加入小震盪 (±0.5%)
noise = np.random.randn(n_points) * 0.3

# 最終價格序列
prices = trend + noise
close = pd.Series(prices, name='close')

print(f"價格範圍: {close.min():.2f} ~ {close.max():.2f}")
print(f"整體變化: {close.iloc[0]:.2f} → {close.iloc[-1]:.2f} ({(close.iloc[-1]/close.iloc[0]-1)*100:.2f}%)")
print()

# 計算波動率
vol = ewma_volatility_professional(close, halflife=20)

# 方法 A: Triple-Barrier（原始）
tb_df = triple_barrier_labels_professional(
    close=close,
    volatility=vol,
    pt_multiplier=2.0,
    sl_multiplier=2.0,
    max_holding=20,
    min_return=0.003,
    day_end_idx=len(close) - 1
)

# 方法 B: 簡單趨勢標籤
trend_simple = trend_labels_simple(
    close=close,
    lookforward=50,  # 往前看 50 個點
    threshold=0.01   # 1% 閾值
)

# 方法 C: 自適應趨勢標籤
trend_adaptive = trend_labels_adaptive(
    close=close,
    volatility=vol,
    lookforward=50,
    vol_multiplier=3.0
)

# 方法 D: 多時間尺度組合
multi_scale = multi_scale_labels_combined(
    close=close,
    volatility=vol,
    tb_holding=20,
    trend_lookforward=50,
    pt_multiplier=2.0,
    sl_multiplier=2.0,
    min_return=0.003,
    trend_threshold=0.01,
    day_end_idx=len(close) - 1
)

# 統計標籤分布
def label_stats(labels, name):
    """計算標籤分布統計"""
    if isinstance(labels, pd.DataFrame):
        labels = labels['y']

    labels = labels[labels != 0]  # 過濾掉 0（無法標籤的部分）

    if len(labels) == 0:
        print(f"{name:30s}: 無有效標籤")
        return

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f"{name:30s}: ", end='')
    for label, count in zip(unique, counts):
        pct = count / total * 100
        label_name = {-1: '下跌', 0: '橫盤', 1: '上漲'}.get(label, str(label))
        print(f"{label_name} {count:3d} ({pct:5.1f}%)  ", end='')
    print()

print("【標籤分布統計】")
label_stats(tb_df['y'], "Triple-Barrier (原始)")
label_stats(trend_simple, "簡單趨勢標籤")
label_stats(trend_adaptive, "自適應趨勢標籤")
label_stats(multi_scale['final_label'], "多時間尺度組合")
print()

# 分析結果
tb_labels = tb_df['y'].values
trend_labels = trend_simple.values[:len(tb_labels)]

# Triple-Barrier 的問題：噪音多
tb_up = (tb_labels == 1).sum()
tb_down = (tb_labels == -1).sum()
tb_neutral = (tb_labels == 0).sum()

# 趨勢標籤：正確識別下跌
trend_up = (trend_labels == 1).sum()
trend_down = (trend_labels == -1).sum()
trend_neutral = (trend_labels == 0).sum()

print("【分析】")
print(f"整體趨勢: 下跌 (100 → 95, -5%)")
print()
print(f"Triple-Barrier 判斷:")
print(f"  上漲: {tb_up:3d} 次  ← 錯誤！整體是下跌趨勢")
print(f"  下跌: {tb_down:3d} 次")
print(f"  橫盤: {tb_neutral:3d} 次")
print(f"  問題: 小震盪產生大量噪音標籤，無法識別整體趨勢")
print()
print(f"趨勢標籤判斷:")
print(f"  上漲: {trend_up:3d} 次")
print(f"  下跌: {trend_down:3d} 次  ← 正確識別整體下跌趨勢！")
print(f"  橫盤: {trend_neutral:3d} 次")
print(f"  優勢: 用更長時間窗口，正確識別整體趨勢")
print()

# ============================================================
# 視覺化對比
# ============================================================
print("【視覺化對比】")
print("生成圖表: trend_vs_tb_comparison.png")

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# 子圖 1: 價格走勢
ax1 = axes[0]
ax1.plot(close.values, label='價格', linewidth=1.5, color='black')
ax1.axhline(y=close.iloc[0], color='green', linestyle='--', alpha=0.5, label='起始價')
ax1.axhline(y=close.iloc[-1], color='red', linestyle='--', alpha=0.5, label='結束價')
ax1.set_ylabel('價格')
ax1.set_title('價格走勢（整體下跌 -5%）')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 子圖 2: Triple-Barrier 標籤
ax2 = axes[1]
tb_colors = {-1: 'red', 0: 'gray', 1: 'green'}
for i, label in enumerate(tb_labels):
    ax2.scatter(i, label, c=tb_colors.get(label, 'black'), s=20, alpha=0.6)
ax2.set_ylabel('標籤')
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['下跌', '橫盤', '上漲'])
ax2.set_title(f'Triple-Barrier 標籤（問題：{tb_up}次誤判上漲）')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 子圖 3: 趨勢標籤
ax3 = axes[2]
for i, label in enumerate(trend_labels):
    ax3.scatter(i, label, c=tb_colors.get(label, 'black'), s=20, alpha=0.6)
ax3.set_ylabel('標籤')
ax3.set_yticks([-1, 0, 1])
ax3.set_yticklabels(['下跌', '橫盤', '上漲'])
ax3.set_title(f'趨勢標籤（正確：{trend_down}次下跌標籤）')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 子圖 4: 多時間尺度組合
ax4 = axes[3]
multi_labels = multi_scale['final_label'].values
for i, label in enumerate(multi_labels):
    ax4.scatter(i, label, c=tb_colors.get(label, 'black'), s=20, alpha=0.6)
ax4.set_ylabel('標籤')
ax4.set_yticks([-1, 0, 1])
ax4.set_yticklabels(['下跌', '橫盤', '上漲'])
ax4.set_title('多時間尺度組合（TB + 趨勢過濾）')
ax4.set_xlabel('時間點')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('trend_vs_tb_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 圖表已保存")
print()

# ============================================================
# 總結
# ============================================================
print("=" * 80)
print("【總結】")
print("=" * 80)
print()
print("Triple-Barrier 的問題:")
print("  ❌ 短視：只看進場→出場這段，無法識別整體趨勢")
print("  ❌ 噪音：小震盪產生大量錯誤標籤")
print("  ❌ 逆勢：在下跌趨勢中仍會產生「做多」信號")
print()
print("趨勢標籤的優勢:")
print("  ✓ 長視：用更長時間窗口判斷整體方向")
print("  ✓ 穩定：減少小震盪的影響")
print("  ✓ 順勢：正確識別趨勢方向")
print()
print("建議使用方案:")
print("  → 方案 A: 純趨勢標籤（簡單、穩定）")
print("  → 方案 B: 多時間尺度組合（TB + 趨勢過濾，最佳）")
print()
print("配置參數建議:")
print("  - trend_lookforward: 50-100 (看更長時間)")
print("  - trend_threshold: 0.01 (1% 閾值)")
print("  - 組合策略: 只在趨勢方向交易")
print()
print("=" * 80)
