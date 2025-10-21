#!/usr/bin/env python3
"""
就地修復權重 - 將各類別平均權重拉齊
基於 ChatGPT 建議，但加入更安全的處理
"""
import numpy as np
import os
import shutil
from pathlib import Path

def fix_weights(npz_path, backup=True, target_strategy="equal_mean", clip_max=10.0):
    """
    修復 NPZ 文件中的權重

    Args:
        npz_path: NPZ 文件路徑
        backup: 是否備份原始文件
        target_strategy: 目標策略
            - "equal_mean": 各類別平均權重拉齊到全局均值
            - "equal_weighted": 各類別加權後佔比相同（1/3）
            - "preserve_ratio": 保持原始樣本比例
        clip_max: 最大權重上限
    """
    print(f"\n處理文件: {npz_path}")

    # 備份
    if backup:
        backup_path = str(npz_path).replace(".npz", "_backup.npz")
        if not os.path.exists(backup_path):
            shutil.copy2(npz_path, backup_path)
            print(f"✅ 已備份到: {backup_path}")

    # 載入
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    w = data["weights"].astype(np.float64)
    stock_ids = data["stock_ids"]

    print(f"\n原始權重統計:")
    print(f"  Shape: {w.shape}")
    print(f"  Mean: {w.mean():.4f}")
    print(f"  Min: {w.min():.4f}, Max: {w.max():.4f}")

    # 計算各類別原始統計
    n_classes = int(y.max()) + 1
    for c in range(n_classes):
        mask = (y == c)
        print(f"  Class {c}: 樣本數={mask.sum():,}, 平均權重={w[mask].mean():.4f}, "
              f"加權佔比={w[mask].sum()/w.sum()*100:.2f}%")

    # 重標定策略
    if target_strategy == "equal_mean":
        # 策略 1: 各類別平均權重拉齊到全局均值
        mu_target = w.mean()
        mu_class = np.array([w[y==c].mean() for c in range(n_classes)])
        scale = np.ones_like(mu_class)
        for c in range(n_classes):
            if mu_class[c] > 0:
                scale[c] = mu_target / mu_class[c]
        w_new = w * scale[y]
        print(f"\n策略: 各類別平均權重拉齊到 {mu_target:.4f}")

    elif target_strategy == "equal_weighted":
        # 策略 2: 各類別加權後佔比相同（1/3）
        target_ratio = 1.0 / n_classes
        current_sum = np.array([w[y==c].sum() for c in range(n_classes)])
        total_sum = w.sum()
        target_sum = total_sum * target_ratio
        scale = np.ones(n_classes)
        for c in range(n_classes):
            if current_sum[c] > 0:
                scale[c] = target_sum / current_sum[c]
        w_new = w * scale[y]
        print(f"\n策略: 各類別加權佔比均為 {target_ratio*100:.2f}%")

    elif target_strategy == "preserve_ratio":
        # 策略 3: 保持原始樣本比例
        n_samples = np.array([np.sum(y==c) for c in range(n_classes)])
        sample_ratio = n_samples / len(y)
        current_sum = np.array([w[y==c].sum() for c in range(n_classes)])
        total_sum = w.sum()
        target_sum = total_sum * sample_ratio
        scale = np.ones(n_classes)
        for c in range(n_classes):
            if current_sum[c] > 0:
                scale[c] = target_sum[c] / current_sum[c]
        w_new = w * scale[y]
        print(f"\n策略: 保持原始樣本比例 {sample_ratio*100}")

    # 安全裁剪
    w_new = np.clip(w_new, 0.05, clip_max)

    # 歸一化（均值=1）
    w_new = w_new / (w_new.mean() + 1e-12)

    # 轉換為 float32
    w_new = w_new.astype(np.float32)

    print(f"\n修復後權重統計:")
    print(f"  Mean: {w_new.mean():.4f}")
    print(f"  Min: {w_new.min():.4f}, Max: {w_new.max():.4f}")

    for c in range(n_classes):
        mask = (y == c)
        print(f"  Class {c}: 平均權重={w_new[mask].mean():.4f}, "
              f"加權佔比={w_new[mask].sum()/w_new.sum()*100:.2f}%")

    # 保存
    np.savez_compressed(npz_path, X=X, y=y, weights=w_new, stock_ids=stock_ids)
    print(f"\n✅ 已保存到: {npz_path}")


if __name__ == "__main__":
    # 處理 train/val/test 三個集合
    base_dir = Path("data/processed_v5_fixed/npz")

    for split in ["train", "val", "test"]:
        npz_path = base_dir / f"stock_embedding_{split}.npz"
        if npz_path.exists():
            fix_weights(
                npz_path,
                backup=True,
                target_strategy="equal_mean",  # 可改為 "equal_weighted" 或 "preserve_ratio"
                clip_max=10.0
            )
        else:
            print(f"⚠️  文件不存在: {npz_path}")

    print("\n" + "="*70)
    print("✅ 所有文件處理完成！")
    print("="*70)
    print("\n下一步:")
    print("1. 驗證權重: python diagnose_weights_emergency.py")
    print("2. 開始訓練: python scripts/train_deeplob_v5.py --config configs/train_v5_recovery.yaml --data-dir ./data/processed_v5_fixed/npz --epochs 50")
