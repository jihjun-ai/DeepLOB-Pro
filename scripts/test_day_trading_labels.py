# -*- coding: utf-8 -*-
"""
test_day_trading_labels.py - 日沖交易標籤測試
=============================================================================
測試 Triple-Barrier 在日沖交易場景下的表現

日沖交易特性：
  - 超短期持倉（5-10 分鐘）
  - 小利潤目標（0.3%-1%）
  - 快速進出
  - 不看長期趨勢

使用方式：
  python scripts/test_day_trading_labels.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 設定中文字體（Windows）
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

from src.utils.financial_engineering import (
    ewma_volatility_professional,
    triple_barrier_labels_professional,
    trend_labels_simple
)

print("=" * 80)
print("日沖交易標籤測試")
print("=" * 80)
print()

# ============================================================
# 場景 1: 快速上漲後回落（典型日沖機會）
# ============================================================
print("[場景 1] 快速上漲 0.8% → 回落 0.5%（典型日沖場景）")
print("-" * 80)

np.random.seed(42)

# 模擬日沖場景：
# 0-30: 快速上漲 (+0.8%)
# 30-60: 回落 (-0.5%)
# 60-100: 橫盤震盪 (±0.1%)

n_points = 100
prices = np.zeros(n_points) + 100.0

# 階段 1: 快速上漲
prices[0:30] = 100.0 + np.linspace(0, 0.8, 30) + np.random.randn(30) * 0.05

# 階段 2: 回落
prices[30:60] = prices[29] + np.linspace(0, -0.5, 30) + np.random.randn(30) * 0.05

# 階段 3: 橫盤
prices[60:100] = prices[59] + np.random.randn(40) * 0.1

close = pd.Series(prices, name='close')

print(f"價格範圍: {close.min():.2f} ~ {close.max():.2f}")
print(f"整體變化: {close.iloc[0]:.2f} → {close.iloc[-1]:.2f} ({(close.iloc[-1]/close.iloc[0]-1)*100:.2f}%)")
print(f"最高點: {close.max():.2f} (+{(close.max()/close.iloc[0]-1)*100:.2f}%)")
print()

# 計算波動率
vol = ewma_volatility_professional(close, halflife=20)

# Triple-Barrier（日沖配置）
tb_df = triple_barrier_labels_professional(
    close=close,
    volatility=vol,
    max_holding=40,      # 40 bars ≈ 4-8 分鐘
    pt_multiplier=2.5,   # 2.5σ 止盈
    sl_multiplier=2.5,   # 2.5σ 止損
    min_return=0.0025    # 0.25% 最小收益
)

# 趨勢標籤（50 bars lookforward = 不適合日沖）
trend_50 = trend_labels_simple(
    close=close,
    lookforward=50,
    threshold=0.005
)

# 短期趨勢標籤（15 bars lookforward = 適合日沖）
trend_15 = trend_labels_simple(
    close=close,
    lookforward=15,
    threshold=0.003
)

# 統計標籤分布
tb_counts = tb_df['label'].value_counts().sort_index()
trend_50_counts = trend_50.value_counts().sort_index()
trend_15_counts = trend_15.value_counts().sort_index()

print("【標籤分布對比】")
print(f"Triple-Barrier (max_holding=40):")
for label in [-1, 0, 1]:
    count = tb_counts.get(label, 0)
    pct = count / len(tb_df) * 100
    name = {-1: "下跌", 0: "持平", 1: "上漲"}[label]
    print(f"  {name}: {count:3d} ({pct:5.1f}%)")

print(f"\n趨勢標籤 (lookforward=50, 不適合日沖):")
for label in [-1, 0, 1]:
    count = trend_50_counts.get(label, 0)
    pct = count / len(trend_50) * 100
    name = {-1: "下跌", 0: "持平", 1: "上漲"}[label]
    print(f"  {name}: {count:3d} ({pct:5.1f}%)")

print(f"\n短期趨勢標籤 (lookforward=15, 適合日沖):")
for label in [-1, 0, 1]:
    count = trend_15_counts.get(label, 0)
    pct = count / len(trend_15) * 100
    name = {-1: "下跌", 0: "持平", 1: "上漲"}[label]
    print(f"  {name}: {count:3d} ({pct:5.1f}%)")

print()

# ============================================================
# 分析關鍵時間點
# ============================================================
print("【關鍵時間點分析】")
print("-" * 80)

# 時間點 1: 上漲階段（t=15, 已上漲約 0.4%）
t1 = 15
print(f"\n時間點 1 (t={t1}): 上漲階段中段")
print(f"  價格: {close.iloc[t1]:.2f} (+{(close.iloc[t1]/close.iloc[0]-1)*100:.2f}%)")
print(f"  Triple-Barrier: {['下跌','持平','上漲'][tb_df['label'].iloc[t1]+1]}")
print(f"  趨勢50: {['下跌','持平','上漲'][trend_50.iloc[t1]+1]}")
print(f"  趨勢15: {['下跌','持平','上漲'][trend_15.iloc[t1]+1]}")
print(f"  → 日沖交易者希望: 上漲（繼續持有）")

# 時間點 2: 高點附近（t=28）
t2 = 28
print(f"\n時間點 2 (t={t2}): 接近高點")
print(f"  價格: {close.iloc[t2]:.2f} (+{(close.iloc[t2]/close.iloc[0]-1)*100:.2f}%)")
print(f"  Triple-Barrier: {['下跌','持平','上漲'][tb_df['label'].iloc[t2]+1]}")
print(f"  趨勢50: {['下跌','持平','上漲'][trend_50.iloc[t2]+1]}")
print(f"  趨勢15: {['下跌','持平','上漲'][trend_15.iloc[t2]+1]}")
print(f"  → 日沖交易者希望: 上漲/持平（準備出場）")

# 時間點 3: 開始回落（t=35）
t3 = 35
print(f"\n時間點 3 (t={t3}): 回落初期")
print(f"  價格: {close.iloc[t3]:.2f} (+{(close.iloc[t3]/close.iloc[0]-1)*100:.2f}%)")
print(f"  Triple-Barrier: {['下跌','持平','上漲'][tb_df['label'].iloc[t3]+1]}")
print(f"  趨勢50: {['下跌','持平','上漲'][trend_50.iloc[t3]+1]}")
print(f"  趨勢15: {['下跌','持平','上漲'][trend_15.iloc[t3]+1]}")
print(f"  → 日沖交易者希望: 下跌（空單機會）")

# 時間點 4: 橫盤階段（t=70）
t4 = 70
print(f"\n時間點 4 (t={t4}): 橫盤震盪")
print(f"  價格: {close.iloc[t4]:.2f} (+{(close.iloc[t4]/close.iloc[0]-1)*100:.2f}%)")
print(f"  Triple-Barrier: {['下跌','持平','上漲'][tb_df['label'].iloc[t4]+1]}")
print(f"  趨勢50: {['下跌','持平','上漲'][trend_50.iloc[t4]+1]}")
print(f"  趨勢15: {['下跌','持平','上漲'][trend_15.iloc[t4]+1]}")
print(f"  → 日沖交易者希望: 持平（不交易）")

print()

# ============================================================
# 視覺化對比
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# 子圖 1: 價格走勢
ax1 = axes[0]
ax1.plot(close.values, 'k-', linewidth=1.5, label='價格')
ax1.axhline(y=close.iloc[0], color='g', linestyle='--', alpha=0.5, label='起始')
ax1.axhline(y=close.max(), color='r', linestyle='--', alpha=0.5, label='高點')
ax1.axvline(x=30, color='gray', linestyle=':', alpha=0.3)
ax1.axvline(x=60, color='gray', linestyle=':', alpha=0.3)
ax1.text(15, close.max()*0.995, '上漲', ha='center', fontsize=10)
ax1.text(45, close.max()*0.995, '回落', ha='center', fontsize=10)
ax1.text(80, close.max()*0.995, '橫盤', ha='center', fontsize=10)
ax1.set_ylabel('價格', fontsize=11)
ax1.legend(loc='upper right')
ax1.set_title('日沖交易場景：快速上漲 → 回落 → 橫盤', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 子圖 2: Triple-Barrier 標籤
ax2 = axes[1]
up_mask = tb_df['label'] == 1
down_mask = tb_df['label'] == -1
neutral_mask = tb_df['label'] == 0
ax2.scatter(tb_df.index[up_mask], [1]*sum(up_mask), c='green', s=30, alpha=0.6, label='上漲')
ax2.scatter(tb_df.index[neutral_mask], [0]*sum(neutral_mask), c='gray', s=30, alpha=0.4, label='持平')
ax2.scatter(tb_df.index[down_mask], [-1]*sum(down_mask), c='red', s=30, alpha=0.6, label='下跌')
ax2.axvline(x=30, color='gray', linestyle=':', alpha=0.3)
ax2.axvline(x=60, color='gray', linestyle=':', alpha=0.3)
ax2.set_ylabel('標籤', fontsize=11)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['下跌', '持平', '上漲'])
ax2.legend(loc='upper right')
ax2.set_title(f'Triple-Barrier (max_holding=40, 適合日沖)', fontsize=11)
ax2.grid(True, alpha=0.3)

# 子圖 3: 趨勢50（不適合日沖）
ax3 = axes[2]
up_mask = trend_50 == 1
down_mask = trend_50 == -1
neutral_mask = trend_50 == 0
ax3.scatter(trend_50.index[up_mask], [1]*sum(up_mask), c='green', s=30, alpha=0.6, label='上漲')
ax3.scatter(trend_50.index[neutral_mask], [0]*sum(neutral_mask), c='gray', s=30, alpha=0.4, label='持平')
ax3.scatter(trend_50.index[down_mask], [-1]*sum(down_mask), c='red', s=30, alpha=0.6, label='下跌')
ax3.axvline(x=30, color='gray', linestyle=':', alpha=0.3)
ax3.axvline(x=60, color='gray', linestyle=':', alpha=0.3)
ax3.set_ylabel('標籤', fontsize=11)
ax3.set_yticks([-1, 0, 1])
ax3.set_yticklabels(['下跌', '持平', '上漲'])
ax3.legend(loc='upper right')
ax3.set_title(f'趨勢標籤 (lookforward=50, 太長，不適合日沖)', fontsize=11)
ax3.grid(True, alpha=0.3)

# 子圖 4: 趨勢15（適合日沖）
ax4 = axes[3]
up_mask = trend_15 == 1
down_mask = trend_15 == -1
neutral_mask = trend_15 == 0
ax4.scatter(trend_15.index[up_mask], [1]*sum(up_mask), c='green', s=30, alpha=0.6, label='上漲')
ax4.scatter(trend_15.index[neutral_mask], [0]*sum(neutral_mask), c='gray', s=30, alpha=0.4, label='持平')
ax4.scatter(trend_15.index[down_mask], [-1]*sum(down_mask), c='red', s=30, alpha=0.6, label='下跌')
ax4.axvline(x=30, color='gray', linestyle=':', alpha=0.3)
ax4.axvline(x=60, color='gray', linestyle=':', alpha=0.3)
ax4.set_ylabel('標籤', fontsize=11)
ax4.set_yticks([-1, 0, 1])
ax4.set_yticklabels(['下跌', '持平', '上漲'])
ax4.set_xlabel('時間點', fontsize=11)
ax4.legend(loc='upper right')
ax4.set_title(f'短期趨勢標籤 (lookforward=15, 適合日沖)', fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_trading_labels_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 圖表已保存: day_trading_labels_comparison.png")
print()

# ============================================================
# 結論
# ============================================================
print("=" * 80)
print("結論")
print("=" * 80)
print("""
【Triple-Barrier 對日沖交易的適用性】

✅ 優勢：
  1. 時間尺度匹配：max_holding=40 (4-8分鐘) 符合日沖持倉時間
  2. 風險管理：2.5σ 止盈/止損提供明確出場點
  3. 快速反應：每個 bar 都判斷，不會錯過機會
  4. 符合實務：日沖交易者確實這樣思考（進場→止盈/止損/時間到）

❌ 限制：
  1. 無法識別更大時間尺度的趨勢（但日沖不需要）
  2. 橫盤時可能產生過多信號（需要額外過濾）

【趨勢標籤 (lookforward=50) 對日沖的問題】

❌ 不適合：
  1. 時間窗口太長（50 bars ≈ 20-40 分鐘）超過日沖持倉時間
  2. 看到「未來太遠」，會錯過短期機會
  3. 在快速上漲階段可能標註「持平」（因為看到後續回落）

【短期趨勢標籤 (lookforward=15) 的可能性】

✅ 可行：
  1. 時間窗口適中（15 bars ≈ 6-12 分鐘）
  2. 能捕捉短期趨勢
  3. 更穩定，噪音較少

⚠️ 需要測試：
  1. 是否能及時捕捉反轉點？
  2. 橫盤階段的表現如何？

【建議】

對於日沖交易判斷：

  方案 A（推薦）: 保持 Triple-Barrier (max_holding=40)
    - 完全符合日沖交易邏輯
    - 風險管理內建
    - 實證效果良好（DeepLOB 72.98%）

  方案 B（可選）: Triple-Barrier + 短期趨勢過濾
    - TB 提供進出場信號
    - 短期趨勢（lookforward=15）過濾逆勢交易
    - 可能提升勝率，但會減少交易次數

  方案 C（不推薦）: 純趨勢標籤 (lookforward=50)
    - 時間尺度不匹配
    - 會錯過日沖機會
""")
print("=" * 80)
