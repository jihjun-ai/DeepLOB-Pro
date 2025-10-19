#!/usr/bin/env python
"""
比較三個修復配置的訓練結果
提取 Class 1 Recall 和其他關鍵指標
"""
import os
import sys
import torch
from pathlib import Path
import json

# 添加 src 到路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def load_checkpoint_info(checkpoint_path):
    """從檢查點載入訓練信息"""
    try:
        # PyTorch 2.6 安全載入
        from ruamel.yaml.scalarfloat import ScalarFloat
        from ruamel.yaml.scalarint import ScalarInt
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString

        with torch.serialization.safe_globals([ScalarFloat, ScalarInt, DoubleQuotedScalarString]):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        return checkpoint
    except Exception as e:
        print(f"❌ 無法載入檢查點 {checkpoint_path}: {e}")
        return None

def extract_metrics(checkpoint):
    """提取關鍵指標"""
    if checkpoint is None:
        return None

    metrics = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_weighted_f1': checkpoint.get('val_weighted_f1', 'N/A'),
        'val_unweighted_f1': checkpoint.get('val_unweighted_f1', 'N/A'),
        'val_unweighted_acc': checkpoint.get('val_unweighted_acc', 'N/A'),
    }

    # 提取 per-class 指標（如果有）
    if 'val_per_class_metrics' in checkpoint:
        per_class = checkpoint['val_per_class_metrics']
        for i in range(3):
            metrics[f'class_{i}_precision'] = per_class.get(f'class_{i}_precision', 'N/A')
            metrics[f'class_{i}_recall'] = per_class.get(f'class_{i}_recall', 'N/A')
            metrics[f'class_{i}_f1'] = per_class.get(f'class_{i}_f1', 'N/A')

    return metrics

def format_metric(value):
    """格式化指標值"""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)

def main():
    print("=" * 80)
    print("三個修復配置的訓練結果比較")
    print("=" * 80)
    print()

    configs = {
        '保守版': 'checkpoints/v5_fix_conservative/deeplob_v5_best.pth',
        '中等版': 'checkpoints/v5_fix_moderate/deeplob_v5_best.pth',
        '激進版': 'checkpoints/v5_fix_aggressive/deeplob_v5_best.pth',
    }

    results = {}

    for name, path in configs.items():
        print(f"\n{'='*80}")
        print(f"配置: {name}")
        print(f"{'='*80}")

        if not os.path.exists(path):
            print(f"⚠️  檢查點不存在: {path}")
            continue

        checkpoint = load_checkpoint_info(path)
        metrics = extract_metrics(checkpoint)

        if metrics:
            results[name] = metrics

            print(f"\n訓練 Epoch: {metrics['epoch']}")
            print(f"\n整體指標:")
            print(f"  Weighted F1 (驗證集):   {format_metric(metrics['val_weighted_f1'])}")
            print(f"  Unweighted F1 (驗證集): {format_metric(metrics['val_unweighted_f1'])}")
            print(f"  Unweighted Acc (驗證集): {format_metric(metrics['val_unweighted_acc'])}")

            print(f"\nPer-Class 指標:")
            for i in range(3):
                class_names = ['下跌 (Class 0)', '持平 (Class 1)', '上漲 (Class 2)']
                if f'class_{i}_recall' in metrics:
                    print(f"  {class_names[i]}:")
                    print(f"    Precision: {format_metric(metrics[f'class_{i}_precision'])}")
                    print(f"    Recall:    {format_metric(metrics[f'class_{i}_recall'])}")
                    print(f"    F1:        {format_metric(metrics[f'class_{i}_f1'])}")
        else:
            print(f"❌ 無法提取指標")

    # 總結比較
    print("\n" + "=" * 80)
    print("Class 1 (持平) Recall 比較總結")
    print("=" * 80)
    print()
    print(f"{'配置':<15} {'Epoch':<10} {'Class 1 Recall':<20} {'Unweighted F1':<20}")
    print("-" * 80)

    for name, metrics in results.items():
        if metrics and 'class_1_recall' in metrics:
            epoch = metrics['epoch']
            class_1_recall = format_metric(metrics['class_1_recall'])
            unweighted_f1 = format_metric(metrics['val_unweighted_f1'])
            print(f"{name:<15} {epoch:<10} {class_1_recall:<20} {unweighted_f1:<20}")
        else:
            print(f"{name:<15} {'N/A':<10} {'N/A':<20} {'N/A':<20}")

    print("\n" + "=" * 80)
    print("分析建議")
    print("=" * 80)

    if results:
        # 找出 Class 1 Recall 最高的配置
        best_config = None
        best_recall = -1

        for name, metrics in results.items():
            if metrics and 'class_1_recall' in metrics:
                recall = metrics['class_1_recall']
                if isinstance(recall, (int, float)) and recall > best_recall:
                    best_recall = recall
                    best_config = name

        if best_config:
            print(f"\n✅ 推薦使用: {best_config}")
            print(f"   Class 1 Recall = {best_recall:.4f}")
            print(f"   原因: Class 1 召回率最高，模型最均衡")

        # 檢查是否有配置 Class 1 Recall 仍然過低
        for name, metrics in results.items():
            if metrics and 'class_1_recall' in metrics:
                recall = metrics['class_1_recall']
                if isinstance(recall, (int, float)) and recall < 0.10:
                    print(f"\n⚠️  警告: {name} 的 Class 1 Recall 仍然過低 ({recall:.4f})")
                    print(f"   建議: 嘗試更激進的配置或調整 Triple-Barrier 參數")

if __name__ == '__main__':
    main()
