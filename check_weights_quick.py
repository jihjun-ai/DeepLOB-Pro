#!/usr/bin/env python3
"""Quick check of NPZ weights"""
import numpy as np
import sys

try:
    data = np.load('data/processed_v5_fixed/npz/stock_embedding_train.npz')
    print('Keys:', list(data.keys()))
    print('Labels shape:', data['y'].shape)

    if 'weights' in data:
        print('✅ Weights EXISTS')
        print('Label distribution:', np.bincount(data['y']))
        print('Label unique:', np.unique(data['y']))
        print(f'Weight stats: mean={data["weights"].mean():.4f}, min={data["weights"].min():.4f}, max={data["weights"].max():.4f}, nan={np.isnan(data["weights"]).sum()}')
        sys.exit(0)
    else:
        print('❌ Weights NOT FOUND')
        sys.exit(1)
except FileNotFoundError:
    print('❌ File not found: data/processed_v5_fixed/npz/stock_embedding_train.npz')
    print('📝 Directory is empty - data needs to be generated!')
    sys.exit(2)
