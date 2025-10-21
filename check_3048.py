import numpy as np
import glob
import json

# 找到股票 3048 的數據
files = glob.glob('data/preprocessed_v5_1hz/daily/*/3048.npz')
print(f'找到 {len(files)} 個 3048 的數據文件')

if files:
    # 檢查前幾個文件
    for i, f in enumerate(files[:5]):
        print(f'\n=== 檔案 {i+1}: {f} ===')
        data = np.load(f, allow_pickle=True)
        mids = data['mids']
        mask = data['bucket_mask']

        print(f'mids shape: {mids.shape}')
        print(f'mids stats:')
        print(f'  min: {mids.min():.4f}')
        print(f'  max: {mids.max():.4f}')
        print(f'  mean: {mids.mean():.4f}')
        print(f'  nan count: {np.isnan(mids).sum()}')
        print(f'  zero count: {(mids == 0).sum()}')
        print(f'  inf count: {np.isinf(mids).sum()}')

        # 檢查前幾個值
        print(f'  前 10 個值: {mids[:10]}')

        # 檢查是否有 0 值
        if (mids == 0).sum() > 0:
            zero_indices = np.where(mids == 0)[0]
            print(f'  ⚠️ 發現 {len(zero_indices)} 個零值，索引: {zero_indices[:10]}')

        # 檢查 metadata
        meta = json.loads(str(data['metadata']))
        print(f'  date: {meta["date"]}')
        print(f'  pass_filter: {meta["pass_filter"]}')
