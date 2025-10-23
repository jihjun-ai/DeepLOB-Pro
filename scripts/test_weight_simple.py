"""
簡單的權重策略測試（不依賴其他模組）

作者：DeepLOB-Pro Team
日期：2025-10-23
"""

import numpy as np
import json
from pathlib import Path


def compute_all_weight_strategies(labels):
    """計算所有權重策略（簡化版）"""
    strategies = {}

    # 過濾 NaN
    valid_labels = labels[~np.isnan(labels)]

    if len(valid_labels) == 0:
        return {}

    # 計算類別分布
    unique, counts = np.unique(valid_labels, return_counts=True)
    total = len(valid_labels)
    n_classes = len(unique)

    # 確保有所有 3 個類別
    label_counts = {-1: 0, 0: 0, 1: 0}
    for label, count in zip(unique, counts):
        label_counts[int(label)] = int(count)

    # 1. Balanced
    strategies['balanced'] = {
        'class_weights': {
            label: total / (n_classes * count) if count > 0 else 1.0
            for label, count in label_counts.items()
        },
        'description': 'Standard balanced weights'
    }

    # 2. Square Root Balanced
    balanced = strategies['balanced']['class_weights']
    strategies['sqrt_balanced'] = {
        'class_weights': {
            label: np.sqrt(weight)
            for label, weight in balanced.items()
        },
        'description': 'Square root balanced (gentler)'
    }

    # 3. Effective Number (beta=0.999)
    beta = 0.999
    effective_weights = {}
    for label, count in label_counts.items():
        if count > 0:
            effective_num = (1 - beta) / (1 - beta**count)
            effective_weights[label] = 1.0 / effective_num
        else:
            effective_weights[label] = 1.0

    total_weight = sum(effective_weights.values())
    strategies['effective_num_0999'] = {
        'class_weights': {
            k: v / total_weight * len(effective_weights)
            for k, v in effective_weights.items()
        },
        'description': 'Effective Number (beta=0.999)'
    }

    # 4. Uniform
    strategies['uniform'] = {
        'class_weights': {-1: 1.0, 0: 1.0, 1: 1.0},
        'description': 'No weighting'
    }

    return strategies


def test_weight_calculation():
    """測試權重計算"""
    print("="*70)
    print("權重策略計算測試")
    print("="*70)

    # 模擬標籤分布（類似 Experiment 5）
    labels = np.array(
        [-1] * 3000 +  # Down
        [0] * 1400 +   # Neutral (少數)
        [1] * 5600 +   # Up
        [np.nan] * 100  # 邊界
    )

    print("\n標籤分布:")
    valid_labels = labels[~np.isnan(labels)]
    unique, counts = np.unique(valid_labels, return_counts=True)
    total = len(valid_labels)
    for label, count in zip(unique, counts):
        label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[int(label)]
        print(f"  {label_name:8s}: {count:5d} ({count/total:5.1%})")

    # 計算權重
    strategies = compute_all_weight_strategies(labels)

    print(f"\n計算了 {len(strategies)} 種權重策略:")
    print("-"*70)

    for name, strategy in strategies.items():
        print(f"\n策略: {name}")
        print(f"描述: {strategy['description']}")

        weights = strategy['class_weights']
        print("權重:")
        for label in [-1, 0, 1]:
            label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[label]
            weight = weights.get(label, 1.0)
            print(f"  {label_name:8s}: {weight:.4f}")

    print("\n" + "="*70)


def test_npz_storage():
    """測試 NPZ 文件中的權重策略"""
    print("\n測試 NPZ 文件權重策略")
    print("="*70)

    # 查找預處理文件
    preprocessed_dir = Path('data/preprocessed_v5_1hz/daily')

    if not preprocessed_dir.exists():
        print(f"\n找不到目錄: {preprocessed_dir}")
        print("請先運行預處理腳本")
        return

    # 找第一個NPZ
    date_dirs = [d for d in preprocessed_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        print("找不到日期目錄")
        return

    npz_files = list(date_dirs[0].glob("*.npz"))
    if not npz_files:
        print("找不到 NPZ 文件")
        return

    npz_path = npz_files[0]
    print(f"\n載入: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    if 'metadata' in data:
        metadata = json.loads(str(data['metadata']))

        if 'weight_strategies' in metadata and metadata['weight_strategies']:
            strategies = metadata['weight_strategies']
            print(f"\n找到 {len(strategies)} 種權重策略")

            # 顯示前 3 種
            for i, (name, strategy) in enumerate(list(strategies.items())[:3]):
                print(f"\n策略 {i+1}: {name}")
                weights = strategy.get('class_weights', {})
                for label_str in ['-1', '0', '1']:
                    label = int(label_str)
                    label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[label]
                    weight = weights.get(label_str, 1.0)
                    print(f"  {label_name:8s}: {weight:.4f}")
        else:
            print("\nNPZ 沒有權重策略（舊版本）")
            print("請重新運行預處理腳本")
    else:
        print("NPZ 沒有 metadata")

    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        test_weight_calculation()
        test_npz_storage()
        print("\n測試完成")
    except Exception as e:
        print(f"\n測試失敗: {e}")
        import traceback
        traceback.print_exc()
