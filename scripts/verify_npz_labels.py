"""驗證預處理 NPZ 文件是否包含 labels 欄位"""
import numpy as np
from pathlib import Path

# 掃描所有日期目錄
npz_base = Path('data/preprocessed_swing/daily')
files = []
for date_dir in sorted(npz_base.glob('*')):
    if date_dir.is_dir():
        files.extend(list(date_dir.glob('*.npz')))

print(f"找到 {len(files)} 個 NPZ 文件")
print("\n檢查前 5 個文件:")

for i, f in enumerate(files[:5]):
    with np.load(f, allow_pickle=True) as data:
        fields = list(data.keys())
        has_labels = 'labels' in fields
        metadata = data['metadata'].item() if 'metadata' in data else {}

        # Handle both dict and string metadata
        if isinstance(metadata, dict):
            npz_version = metadata.get('npz_version', 'v1.0')
        else:
            npz_version = 'unknown'

        print(f"\n{i+1}. {f.name}")
        print(f"   NPZ 版本: {npz_version}")
        print(f"   欄位: {fields}")
        print(f"   有 labels: {has_labels}")

        if has_labels:
            labels = data['labels']
            print(f"   Labels shape: {labels.shape}")
            print(f"   Labels unique: {np.unique(labels)}")

# 統計整體
with_labels = 0
without_labels = 0

for f in files:
    with np.load(f, allow_pickle=True) as data:
        if 'labels' in data:
            with_labels += 1
        else:
            without_labels += 1

print(f"\n\n統計:")
print(f"有 labels: {with_labels} ({with_labels/len(files)*100:.1f}%)")
print(f"無 labels: {without_labels} ({without_labels/len(files)*100:.1f}%)")
