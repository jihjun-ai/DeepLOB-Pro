"""檢查數據格式"""
import numpy as np
from pathlib import Path

# 檢查 processed_v5
print("=" * 60)
print("檢查 processed_v5")
print("=" * 60)
p1 = Path("../data/processed_v5/npz")
files1 = list(p1.glob("*.npz"))[:3]
print(f"檔案數量: {len(list(p1.glob('*.npz')))}")
print(f"前 3 個檔案: {[f.name for f in files1]}")
if files1:
    data1 = np.load(files1[0], allow_pickle=True)
    print(f"第一個檔案的鍵: {list(data1.keys())}")

# 檢查 processed_v5_filter2.0
print("\n" + "=" * 60)
print("檢查 processed_v5_filter2.0")
print("=" * 60)
p2 = Path("../data/processed_v5_filter2.0/npz")
files2 = list(p2.glob("*.npz"))[:3]
print(f"檔案數量: {len(list(p2.glob('*.npz')))}")
print(f"前 3 個檔案: {[f.name for f in files2]}")
if files2:
    data2 = np.load(files2[0], allow_pickle=True)
    print(f"第一個檔案的鍵: {list(data2.keys())}")
    print(f"shapes: X={data2['X'].shape if 'X' in data2 else 'N/A'}, y={data2['y'].shape if 'y' in data2 else 'N/A'}")
