"""檢查資料維度"""
import numpy as np

data = np.load('data/processed_v5/npz/stock_embedding_train.npz', allow_pickle=True)
print(f"X shape: {data['X'].shape}")
print(f"Available keys: {list(data.keys())}")
