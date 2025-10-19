import numpy as np
# for split in ['train', 'val', 'test']:
#     d = np.load(f'data/processed_v5/npz/stock_embedding_{split}.npz')
#     vals, counts = np.unique(d['y'], return_counts=True)
#     total = len(d['y'])
#     print(f'\n{split.upper()}:')
#     for v, c in zip(vals, counts):
#         print(f'  Class {v}: {c:,} ({c/total*100:.1f}%)')
        
        

# data = np.load('data/processed_v5/npz/stock_embedding_train.npz')
# y = data['y']
# unique, counts = np.unique(y, return_counts=True)
# for label, count in zip(unique, counts):
#     print(f"Class {label}: {count:,} ({count/len(y)*100:.2f}%)")        
    
    
    
# 檢查生成的數據分佈
import json
with open('data/processed_v5_balanced/npz/normalization_meta.json', 'r') as f:
    meta = json.load(f)
    dist = meta['data_split']['results']['train']['label_dist']
    total = sum(dist)
    print(f'Class 0: {dist[0]/total*100:.2f}%')
    print(f'Class 1: {dist[1]/total*100:.2f}%')
    print(f'Class 2: {dist[2]/total*100:.2f}%')
