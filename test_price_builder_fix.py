# -*- coding: utf-8 -*-
"""測試修復後的 price_builder 是否能處理 V6 數據"""

import sys
sys.path.insert(0, 'label_viewer')

from utils.price_builder import reconstruct_close_price
import numpy as np
import json

# 載入 V6 metadata
with open('data/processed_v6/npz/normalization_meta.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 載入 V6 測試數據
data = np.load('data/processed_v6/npz/stock_embedding_test.npz', allow_pickle=True)
X_sample = data['X'][:10]

print(f'載入樣本形狀: {X_sample.shape}')
print(f'Normalization 方法: {metadata["normalization"]["method"]}')
print()

# 測試 reconstruct_close_price
close_prices = reconstruct_close_price(X_sample, metadata)
print(f'✅ 成功重建收盤價！')
print(f'   形狀: {close_prices.shape}')
print(f'   範圍: [{close_prices.min():.2f}, {close_prices.max():.2f}]')
print(f'   平均: {close_prices.mean():.2f}')
print()
print('修復成功！label_viewer 現在應該能正常工作了。')
