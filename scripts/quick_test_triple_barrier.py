#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速測試 Triple-Barrier 參數配置
不實際生成數據，僅基於當前數據估算新參數的效果
"""
import sys
import io
from pathlib import Path
import numpy as np
import json

# 設置 stdout 為 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加項目根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ruamel.yaml import YAML

def estimate_distribution(current_dist, param_changes):
    """
    基於參數變化估算新的分佈

    簡化模型：
    - min_return 降低 X% → Class 1 減少約 0.5*X%, Class 0/2 各增加 0.25*X%
    - pt/sl_multiplier 降低 Y% → Class 1 減少約 0.3*Y%, Class 0/2 各增加 0.15*Y%
    - max_holding 降低 Z% → Class 1 減少約 0.2*Z%, Class 0/2 各增加 0.1*Z%
    """
    c0, c1, c2 = current_dist

    # 計算參數變化百分比
    min_return_change = param_changes.get('min_return', 0)  # 負值表示降低
    multiplier_change = param_changes.get('multiplier', 0)
    holding_change = param_changes.get('max_holding', 0)

    # 估算影響（負值表示 Class 1 減少）
    c1_delta = (
        0.5 * min_return_change +
        0.3 * multiplier_change +
        0.2 * holding_change
    )

    # 調整分佈（Class 1 的變化平均分配給 Class 0 和 Class 2）
    new_c1 = c1 + c1_delta
    delta_for_others = -c1_delta / 2
    new_c0 = c0 + delta_for_others
    new_c2 = c2 + delta_for_others

    # 歸一化到 100%
    total = new_c0 + new_c1 + new_c2
    new_c0 = (new_c0 / total) * 100
    new_c1 = (new_c1 / total) * 100
    new_c2 = (new_c2 / total) * 100

    return new_c0, new_c1, new_c2

def analyze_config(config_path, current_dist):
    """分析配置並估算效果"""
    yaml = YAML()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)

    tb_params = config['triple_barrier']

    # 當前參數（作為基準）
    baseline = {
        'pt_multiplier': 5.9,
        'sl_multiplier': 5.9,
        'max_holding': 50,
        'min_return': 0.00215
    }

    # 計算變化百分比
    min_return_pct_change = ((tb_params['min_return'] - baseline['min_return']) / baseline['min_return']) * 100
    multiplier_pct_change = ((tb_params['pt_multiplier'] - baseline['pt_multiplier']) / baseline['pt_multiplier']) * 100
    holding_pct_change = ((tb_params['max_holding'] - baseline['max_holding']) / baseline['max_holding']) * 100

    param_changes = {
        'min_return': min_return_pct_change,
        'multiplier': multiplier_pct_change,
        'max_holding': holding_pct_change
    }

    # 估算新分佈
    new_dist = estimate_distribution(current_dist, param_changes)

    return tb_params, param_changes, new_dist

def main():
    print("=" * 80)
    print("Triple-Barrier 參數快速測試工具")
    print("=" * 80)
    print()

    # 當前數據分佈（訓練集）
    print("📊 當前數據分佈（基於 config_pro_v5_ml_optimal.yaml）：")
    print("-" * 80)
    current_samples = [367457, 562361, 319601]
    total = sum(current_samples)
    current_dist = [(s/total)*100 for s in current_samples]

    print(f"  Class 0 (下跌): {current_samples[0]:>7,} ({current_dist[0]:5.2f}%)")
    print(f"  Class 1 (持平): {current_samples[1]:>7,} ({current_dist[1]:5.2f}%)")
    print(f"  Class 2 (上漲): {current_samples[2]:>7,} ({current_dist[2]:5.2f}%)")
    print(f"  總計:          {total:>7,}")
    print()

    # 測試五個新配置
    configs = [
        ('實證版 ⭐⭐⭐⭐⭐', 'configs/config_pro_v5_balanced_empirical.yaml'),
        ('最優版 ⭐⭐⭐', 'configs/config_pro_v5_balanced_optimal.yaml'),
        ('保守版', 'configs/config_pro_v5_balanced_conservative.yaml'),
        ('中等版', 'configs/config_pro_v5_balanced_moderate.yaml'),
        ('激進版', 'configs/config_pro_v5_balanced_aggressive.yaml'),
    ]

    results = []

    for name, config_path in configs:
        print("=" * 80)
        print(f"配置：{name}")
        print("=" * 80)

        try:
            tb_params, param_changes, new_dist = analyze_config(config_path, current_dist)

            print("\n📝 Triple-Barrier 參數：")
            print(f"  pt_multiplier: {tb_params['pt_multiplier']}")
            print(f"  sl_multiplier: {tb_params['sl_multiplier']}")
            print(f"  max_holding:   {tb_params['max_holding']} bars")
            print(f"  min_return:    {tb_params['min_return']:.5f} ({tb_params['min_return']*100:.3f}%)")

            print("\n📊 參數變化：")
            print(f"  min_return:    {param_changes['min_return']:+6.1f}%")
            print(f"  multiplier:    {param_changes['multiplier']:+6.1f}%")
            print(f"  max_holding:   {param_changes['max_holding']:+6.1f}%")

            print("\n🎯 預估新分佈：")
            print(f"  Class 0 (下跌): {new_dist[0]:5.2f}% (變化: {new_dist[0]-current_dist[0]:+5.2f}%)")
            print(f"  Class 1 (持平): {new_dist[1]:5.2f}% (變化: {new_dist[1]-current_dist[1]:+5.2f}%)")
            print(f"  Class 2 (上漲): {new_dist[2]:5.2f}% (變化: {new_dist[2]-current_dist[2]:+5.2f}%)")

            # 檢查是否接近目標 (30%/35%/35%)
            target = [31.5, 35.0, 33.5]
            deviation = sum(abs(new_dist[i] - target[i]) for i in range(3))

            print(f"\n📈 目標達成度 (目標: 30-33% / 33-37% / 30-33%):")
            if deviation < 5.0:
                print(f"  ✅ 優秀！偏差 = {deviation:.2f}% (接近理想分佈)")
            elif deviation < 8.0:
                print(f"  ✅ 良好！偏差 = {deviation:.2f}% (可接受範圍)")
            elif deviation < 12.0:
                print(f"  ⚠️  尚可，偏差 = {deviation:.2f}% (需微調)")
            else:
                print(f"  ❌ 偏差過大 = {deviation:.2f}% (建議調整)")

            results.append({
                'name': name,
                'new_dist': new_dist,
                'deviation': deviation,
                'params': tb_params
            })

        except Exception as e:
            print(f"❌ 載入配置失敗: {e}")

        print()

    # 總結推薦
    print("=" * 80)
    print("📋 總結與推薦")
    print("=" * 80)
    print()

    if results:
        # 找出偏差最小的配置
        best = min(results, key=lambda x: x['deviation'])

        print(f"✅ 推薦配置: {best['name']}")
        print(f"   預估分佈: Class 0={best['new_dist'][0]:.2f}%, "
              f"Class 1={best['new_dist'][1]:.2f}%, "
              f"Class 2={best['new_dist'][2]:.2f}%")
        print(f"   偏差度: {best['deviation']:.2f}%")
        print()

        print("📌 下一步操作：")
        print()
        print("1. 使用推薦配置重新生成數據：")
        print(f"   python scripts/extract_tw_stock_data_v5.py \\")
        print(f"       --input-dir ./data/temp \\")
        print(f"       --output-dir ./data/processed_v5_balanced \\")
        print(f"       --config configs/config_pro_v5_balanced_moderate.yaml")
        print()
        print("2. 驗證新數據分佈：")
        print("   檢查 ./data/processed_v5_balanced/npz/normalization_meta.json")
        print("   確認 label_dist 接近目標")
        print()
        print("3. 使用新數據訓練模型：")
        print("   python scripts/train_deeplob_v5.py \\")
        print("       --config configs/train_v5_fix_moderate.yaml \\")
        print("       --data-dir ./data/processed_v5_balanced/npz")
        print()
        print("4. 比較結果：")
        print("   期望 Class 1 Recall 提升至 30-50%（vs 當前 11.35%）")

    print()
    print("=" * 80)
    print("⚠️  注意：此工具僅提供估算，實際分佈需重新生成數據後確認")
    print("=" * 80)

if __name__ == '__main__':
    main()
