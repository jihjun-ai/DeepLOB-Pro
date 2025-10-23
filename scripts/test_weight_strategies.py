"""
測試權重策略計算功能

用途：
1. 驗證 compute_all_weight_strategies() 函數
2. 測試 NPZ 是否正確保存權重策略
3. 展示權重策略的實際值

作者：DeepLOB-Pro Team
日期：2025-10-23
"""

import numpy as np
import json
from pathlib import Path
import sys

# 添加 scripts 目錄到路徑
sys.path.append(str(Path(__file__).parent))


def test_weight_calculation():
    """測試權重計算功能"""
    print("="*70)
    print("測試 1: 權重計算功能")
    print("="*70)

    # 模擬標籤分布（類似 Experiment 5）
    # Down: 30%, Neutral: 14%, Up: 56%（極端不平衡）
    labels = np.array(
        [-1] * 3000 +  # Down
        [0] * 1400 +   # Neutral (少數類別)
        [1] * 5600 +   # Up
        [np.nan] * 100  # 邊界點
    )

    print(f"\n標籤分布:")
    valid_labels = labels[~np.isnan(labels)]
    unique, counts = np.unique(valid_labels, return_counts=True)
    total = len(valid_labels)
    for label, count in zip(unique, counts):
        label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[int(label)]
        print(f"  {label_name:8s} ({int(label):2d}): {count:6d} ({count/total:6.1%})")

    # 導入權重計算函數
    from preprocess_single_day import compute_all_weight_strategies

    # 計算權重策略
    strategies = compute_all_weight_strategies(labels)

    print(f"\n計算了 {len(strategies)} 種權重策略:")
    print("-"*70)

    for name, strategy in strategies.items():
        print(f"\n策略: {name}")
        print(f"描述: {strategy['description']}")

        if 'class_weights' in strategy:
            weights = strategy['class_weights']
            print(f"權重:")
            for label in [-1, 0, 1]:
                label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[label]
                weight = weights.get(label, 1.0)
                print(f"  {label_name:8s} ({label:2d}): {weight:.4f}")

    print("\n" + "="*70)


def test_npz_weight_storage():
    """測試 NPZ 文件中的權重策略"""
    print("\n" + "="*70)
    print("測試 2: NPZ 文件權重策略存儲")
    print("="*70)

    # 查找一個預處理好的 NPZ 文件
    preprocessed_dir = Path('data/preprocessed_v5_1hz/daily')

    if not preprocessed_dir.exists():
        print("\n⚠️ 找不到預處理數據目錄")
        print(f"   路徑: {preprocessed_dir}")
        print("   請先運行預處理腳本")
        return

    # 找第一個日期目錄
    date_dirs = [d for d in preprocessed_dir.iterdir() if d.is_dir()]

    if not date_dirs:
        print("\n⚠️ 找不到日期目錄")
        return

    first_date = date_dirs[0]
    npz_files = list(first_date.glob("*.npz"))

    if not npz_files:
        print(f"\n⚠️ {first_date} 中找不到 NPZ 文件")
        return

    # 載入第一個 NPZ
    npz_path = npz_files[0]
    print(f"\n載入文件: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # 檢查是否有權重策略
    if 'metadata' in data:
        metadata_str = str(data['metadata'])
        metadata = json.loads(metadata_str)

        if 'weight_strategies' in metadata and metadata['weight_strategies']:
            strategies = metadata['weight_strategies']
            print(f"\n✅ 找到 {len(strategies)} 種權重策略")

            # 顯示前 3 種
            for i, (name, strategy) in enumerate(list(strategies.items())[:3]):
                print(f"\n策略 {i+1}: {name}")
                print(f"  描述: {strategy.get('description', 'N/A')}")

                if 'class_weights' in strategy:
                    weights = strategy['class_weights']
                    print(f"  權重:")
                    for label_str in ['-1', '0', '1']:
                        label = int(label_str)
                        label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[label]
                        weight = weights.get(label_str, 1.0)
                        print(f"    {label_name:8s}: {weight:.4f}")

            print(f"\n其他策略: {', '.join(list(strategies.keys())[3:])}")

        else:
            print("\n❌ NPZ 文件中沒有權重策略")
            print("   這可能是舊版本的 NPZ 文件")
            print("   請重新運行預處理腳本以生成權重策略")

    else:
        print("\n❌ NPZ 文件中沒有 metadata")

    print("\n" + "="*70)


def test_weight_comparison():
    """比較不同權重策略的效果"""
    print("\n" + "="*70)
    print("測試 3: 權重策略對比")
    print("="*70)

    # 模擬不同程度的不平衡
    test_cases = [
        {
            'name': '輕微不平衡 (30/40/30)',
            'labels': np.array([-1]*3000 + [0]*4000 + [1]*3000)
        },
        {
            'name': '中度不平衡 (30/20/50)',
            'labels': np.array([-1]*3000 + [0]*2000 + [1]*5000)
        },
        {
            'name': '嚴重不平衡 (10/10/80)',
            'labels': np.array([-1]*1000 + [0]*1000 + [1]*8000)
        },
        {
            'name': '極端不平衡 (5/5/90)',
            'labels': np.array([-1]*500 + [0]*500 + [1]*9000)
        }
    ]

    from preprocess_single_day import compute_all_weight_strategies

    for case in test_cases:
        print(f"\n{case['name']}:")
        print("-"*70)

        labels = case['labels']

        # 計算分布
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        print("分布:")
        for label, count in zip(unique, counts):
            label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[int(label)]
            print(f"  {label_name}: {count/total:.1%}")

        # 計算權重
        strategies = compute_all_weight_strategies(labels)

        # 只顯示關鍵策略
        key_strategies = ['balanced', 'sqrt_balanced', 'effective_num_0999']

        print("\n關鍵權重策略:")
        for strategy_name in key_strategies:
            if strategy_name in strategies:
                weights = strategies[strategy_name]['class_weights']
                print(f"\n  {strategy_name}:")
                for label in [-1, 0, 1]:
                    label_name = {-1: 'Down', 0: 'Neutral', 1: 'Up'}[label]
                    weight = weights.get(label, 1.0)
                    print(f"    {label_name}: {weight:.3f}")

    print("\n" + "="*70)


def main():
    """主函數"""
    print("\n" + "="*70)
    print("權重策略計算功能測試")
    print("="*70)

    try:
        # 測試 1: 基本權重計算
        test_weight_calculation()

        # 測試 2: NPZ 文件權重存儲
        test_npz_weight_storage()

        # 測試 3: 不同場景的權重對比
        test_weight_comparison()

        print("\n" + "="*70)
        print("✅ 所有測試完成")
        print("="*70)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
