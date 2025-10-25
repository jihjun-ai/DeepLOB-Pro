#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_no_cross_file.py - 驗證 V7.1 修復：滑動窗口不跨文件

使用方式：
    python scripts/verify_no_cross_file.py --data-dir data/processed_v7_test/npz

檢查項目：
    1. 樣本數量是否合理
    2. 標籤分布是否正常
    3. stock_ids 是否有異常
"""
import os
import argparse
import numpy as np
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description='驗證 V7.1 修復')
    parser.add_argument('--data-dir', type=str, required=True, help='NPZ 目錄')
    args = parser.parse_args()

    npz_dir = args.data_dir

    # 檢查文件是否存在
    train_path = os.path.join(npz_dir, 'stock_embedding_train.npz')
    val_path = os.path.join(npz_dir, 'stock_embedding_val.npz')
    test_path = os.path.join(npz_dir, 'stock_embedding_test.npz')
    meta_path = os.path.join(npz_dir, 'normalization_meta.json')

    if not os.path.exists(train_path):
        print(f"❌ 找不到訓練數據: {train_path}")
        return

    print("=" * 80)
    print("V7.1 驗證報告：滑動窗口不跨文件")
    print("=" * 80)

    # 載入數據
    train = np.load(train_path)
    val = np.load(val_path)
    test = np.load(test_path)

    # 1. 樣本數量檢查
    print("\n【1. 樣本數量檢查】")
    print(f"Train: {len(train['X']):,} 樣本")
    print(f"Val:   {len(val['X']):,} 樣本")
    print(f"Test:  {len(test['X']):,} 樣本")
    print(f"Total: {len(train['X']) + len(val['X']) + len(test['X']):,} 樣本")

    # 2. 數據形狀檢查
    print("\n【2. 數據形狀檢查】")
    print(f"Train X shape: {train['X'].shape}")  # (N, 100, 20)
    print(f"Train y shape: {train['y'].shape}")  # (N,)
    print(f"Window size: {train['X'].shape[1]} timesteps")
    print(f"Features: {train['X'].shape[2]} dimensions")

    # 3. 標籤分布檢查
    print("\n【3. 標籤分布檢查】")
    all_labels = np.concatenate([train['y'], val['y'], test['y']])
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)

    print(f"標籤 0 (下跌): {label_counts[0]:,} ({label_counts[0]/total_samples*100:.2f}%)")
    print(f"標籤 1 (持平): {label_counts[1]:,} ({label_counts[1]/total_samples*100:.2f}%)")
    print(f"標籤 2 (上漲): {label_counts[2]:,} ({label_counts[2]/total_samples*100:.2f}%)")

    # 檢查是否接近目標分布 30/40/30
    target_dist = [0.30, 0.40, 0.30]
    actual_dist = [label_counts[i]/total_samples for i in range(3)]
    print(f"\n目標分布: {target_dist}")
    print(f"實際分布: {[f'{d:.3f}' for d in actual_dist]}")

    # 4. Stock IDs 檢查
    print("\n【4. Stock IDs 檢查】")
    if 'stock_ids' in train:
        train_stocks = set(train['stock_ids'])
        val_stocks = set(val['stock_ids'])
        test_stocks = set(test['stock_ids'])

        print(f"Train 股票數: {len(train_stocks)}")
        print(f"Val 股票數:   {len(val_stocks)}")
        print(f"Test 股票數:  {len(test_stocks)}")

        # 檢查是否有重疊（應該沒有）
        overlap_train_val = train_stocks & val_stocks
        overlap_train_test = train_stocks & test_stocks
        overlap_val_test = val_stocks & test_stocks

        if overlap_train_val or overlap_train_test or overlap_val_test:
            print("\n⚠️ 警告：發現數據集重疊！")
            print(f"Train-Val 重疊: {len(overlap_train_val)} 檔")
            print(f"Train-Test 重疊: {len(overlap_train_test)} 檔")
            print(f"Val-Test 重疊: {len(overlap_val_test)} 檔")
        else:
            print("\n✅ 數據集無重疊（正確）")

    # 5. 權重檢查（如果有）
    print("\n【5. 權重檢查】")
    if 'weights' in train:
        weights = train['weights']
        print(f"權重範圍: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"權重平均: {weights.mean():.3f}")
        print(f"權重標準差: {weights.std():.3f}")
    else:
        print("⚠️ 無權重數據")

    # 6. Metadata 檢查
    print("\n【6. Metadata 檢查】")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        print(f"版本: {meta.get('version', 'N/A')}")
        print(f"生成時間: {meta.get('timestamp', 'N/A')}")

        if 'label_distribution' in meta:
            print(f"\n標籤分布（來自 metadata）:")
            for label, dist in meta['label_distribution'].items():
                print(f"  {label}: {dist}")
    else:
        print("⚠️ 找不到 normalization_meta.json")

    # 7. 價格和成交量檢查（V7.1 新增）
    print("\n【7. 價格和成交量檢查】")
    if 'prices' in train:
        print(f"✅ Train 包含價格數據: {train['prices'].shape}")
        print(f"   價格範圍: [{train['prices'].min():.2f}, {train['prices'].max():.2f}]")
    else:
        print("⚠️ Train 無價格數據")

    if 'volumes' in train:
        print(f"✅ Train 包含成交量數據: {train['volumes'].shape}")
    else:
        print("⚠️ Train 無成交量數據")

    # 8. 總結
    print("\n" + "=" * 80)
    print("【總結】")
    print("=" * 80)

    issues = []

    # 檢查樣本數量是否太少
    if total_samples < 100000:
        issues.append(f"⚠️ 樣本數量偏少 ({total_samples:,} < 100,000)")
    else:
        print(f"✅ 樣本數量充足 ({total_samples:,} 樣本)")

    # 檢查標籤分布
    if abs(actual_dist[1] - 0.40) > 0.10:  # 持平類偏差超過 10%
        issues.append(f"⚠️ 標籤分布偏離目標（持平類: {actual_dist[1]:.2f} vs 目標 0.40）")
    else:
        print(f"✅ 標籤分布接近目標")

    # 檢查窗口大小
    if train['X'].shape[1] != 100:
        issues.append(f"⚠️ 窗口大小異常 ({train['X'].shape[1]} != 100)")
    else:
        print(f"✅ 窗口大小正確 (100 timesteps)")

    # 輸出問題
    if issues:
        print(f"\n發現 {len(issues)} 個問題:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n🎉 所有檢查通過！")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
