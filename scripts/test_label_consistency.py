# -*- coding: utf-8 -*-
"""
test_label_consistency.py - 驗證兩階段標籤生成一致性
============================================================
測試 preprocess_single_day.py 和 extract_tw_stock_data_v6.py
使用相同數據時，生成的標籤是否完全一致。

使用方式：
  python scripts/test_label_consistency.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加項目根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 導入配置
from src.utils.yaml_manager import YAMLManager

print("=" * 70)
print("標籤一致性測試")
print("=" * 70)

# 載入配置
config_path = project_root / "configs" / "config_pro_v5_ml_optimal.yaml"
yaml_manager = YAMLManager(str(config_path))
config = yaml_manager.as_dict()

print(f"\n配置檔案: {config_path}")
print(f"版本: {config.get('version')}")

# 讀取 Triple-Barrier 配置
tb_config = config['triple_barrier']
print(f"\nTriple-Barrier 參數:")
print(f"  pt_multiplier: {tb_config.get('pt_multiplier')}")
print(f"  sl_multiplier: {tb_config.get('sl_multiplier')}")
print(f"  max_holding: {tb_config.get('max_holding')}")
print(f"  min_return: {tb_config.get('min_return')}")

# 生成測試數據（模擬真實價格序列）
np.random.seed(42)
n_points = 500
base_price = 100.0
returns = np.random.randn(n_points) * 0.002  # 0.2% 標準差
prices = base_price * np.exp(np.cumsum(returns))
test_mids = prices

print(f"\n測試數據:")
print(f"  點數: {len(test_mids)}")
print(f"  價格範圍: {test_mids.min():.2f} ~ {test_mids.max():.2f}")
print(f"  平均收益率: {np.mean(returns):.6f}")
print(f"  收益率標準差: {np.std(returns):.6f}")

# ============================================================
# 方法 1: 使用 preprocess_single_day.py 的函數
# ============================================================
print("\n" + "=" * 70)
print("方法 1: preprocess_single_day.py (compute_label_preview)")
print("=" * 70)

from scripts.preprocess_single_day import ewma_vol, tb_labels

close = pd.Series(test_mids, name='close')

# 讀取配置參數
halflife = tb_config.get('ewma_halflife', 60)
pt_mult = tb_config.get('pt_multiplier', tb_config.get('pt_mult', 2.0))
sl_mult = tb_config.get('sl_multiplier', tb_config.get('sl_mult', 2.0))
max_holding = tb_config.get('max_holding', 200)
min_return = tb_config.get('min_return', 0.0001)

# 計算波動率
vol_1 = ewma_vol(close, halflife=halflife)

# 計算標籤（啟用日界保護）
tb_df_1 = tb_labels(
    close=close,
    vol=vol_1,
    pt_mult=pt_mult,
    sl_mult=sl_mult,
    max_holding=max_holding,
    min_return=min_return,
    day_end_idx=len(close) - 1  # 啟用日界保護
)

labels_1 = tb_df_1['y'].values

print(f"標籤數量: {len(labels_1)}")
print(f"標籤分布:")
for label, count in sorted(zip(*np.unique(labels_1, return_counts=True))):
    pct = count / len(labels_1) * 100
    print(f"  {label:2d}: {count:5d} ({pct:5.1f}%)")

# ============================================================
# 方法 2: 使用 extract_tw_stock_data_v6.py 的函數
# ============================================================
print("\n" + "=" * 70)
print("方法 2: extract_tw_stock_data_v6.py (ewma_vol + tb_labels)")
print("=" * 70)

from scripts.extract_tw_stock_data_v6 import ewma_vol as ewma_vol_v6
from scripts.extract_tw_stock_data_v6 import tb_labels as tb_labels_v6

close = pd.Series(test_mids, name='close')

# 計算波動率
vol_2 = ewma_vol_v6(close, halflife=halflife)

# 計算標籤（啟用日界保護）
tb_df_2 = tb_labels_v6(
    close=close,
    vol=vol_2,
    pt_mult=pt_mult,
    sl_mult=sl_mult,
    max_holding=max_holding,
    min_return=min_return,
    day_end_idx=len(close) - 1  # 啟用日界保護
)

labels_2 = tb_df_2['y'].values

print(f"標籤數量: {len(labels_2)}")
print(f"標籤分布:")
for label, count in sorted(zip(*np.unique(labels_2, return_counts=True))):
    pct = count / len(labels_2) * 100
    print(f"  {label:2d}: {count:5d} ({pct:5.1f}%)")

# ============================================================
# 一致性驗證
# ============================================================
print("\n" + "=" * 70)
print("一致性驗證")
print("=" * 70)

# 檢查波動率
vol_diff = np.abs(vol_1.values - vol_2.values).max()
print(f"\n1. 波動率差異:")
print(f"   最大絕對差異: {vol_diff:.2e}")
if vol_diff < 1e-10:
    print(f"   [OK] 波動率計算完全一致")
else:
    print(f"   [WARNING] 波動率有微小差異（可能是數值誤差）")

# 檢查標籤數量
print(f"\n2. 標籤數量:")
print(f"   方法 1: {len(labels_1)}")
print(f"   方法 2: {len(labels_2)}")
if len(labels_1) == len(labels_2):
    print(f"   [OK] 標籤數量一致")
else:
    print(f"   [ERROR] 標籤數量不一致！")

# 檢查標籤值
if len(labels_1) == len(labels_2):
    label_match = np.all(labels_1 == labels_2)
    print(f"\n3. 標籤值:")
    print(f"   完全匹配: {label_match}")

    if label_match:
        print(f"   [OK] 標籤值完全一致")
    else:
        diff_count = np.sum(labels_1 != labels_2)
        diff_pct = diff_count / len(labels_1) * 100
        print(f"   [ERROR] 有 {diff_count} 個標籤不同 ({diff_pct:.2f}%)")

        # 顯示前 10 個不同的位置
        diff_indices = np.where(labels_1 != labels_2)[0][:10]
        if len(diff_indices) > 0:
            print(f"\n   前 10 個差異位置:")
            for idx in diff_indices:
                print(f"     [{idx}] 方法1={labels_1[idx]}, 方法2={labels_2[idx]}")

# 檢查其他欄位
print(f"\n4. 其他欄位:")
ret_diff = np.abs(tb_df_1['ret'].values - tb_df_2['ret'].values).max()
tt_match = np.all(tb_df_1['tt'].values == tb_df_2['tt'].values)
why_match = np.all(tb_df_1['why'].values == tb_df_2['why'].values)

print(f"   ret (收益率) 最大差異: {ret_diff:.2e}")
print(f"   tt (持有時間) 匹配: {tt_match}")
print(f"   why (觸發原因) 匹配: {why_match}")

if ret_diff < 1e-10 and tt_match and why_match:
    print(f"   [OK] 所有欄位完全一致")

# 總結
print("\n" + "=" * 70)
print("總結")
print("=" * 70)

if vol_diff < 1e-10 and len(labels_1) == len(labels_2) and np.all(labels_1 == labels_2):
    print("\n[SUCCESS] 兩個方法生成的標籤完全一致！")
    print("\n[OK] preprocess_single_day.py 和 extract_tw_stock_data_v6.py")
    print("   現在使用相同的專業函數，標籤生成邏輯已統一。")
    exit_code = 0
else:
    print("\n[WARNING] 兩個方法生成的標籤有差異！")
    print("\n請檢查:")
    print("  1. 配置參數是否完全相同")
    print("  2. 日界保護 (day_end_idx) 是否都啟用")
    print("  3. 函數實現是否完全一致")
    exit_code = 1

print("=" * 70)
sys.exit(exit_code)
