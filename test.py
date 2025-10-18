import numpy as np
for split in ['train', 'val', 'test']:
    d = np.load(f'data/processed_v5/npz/stock_embedding_{split}.npz')
    vals, counts = np.unique(d['y'], return_counts=True)
    total = len(d['y'])
    print(f'\n{split.upper()}:')
    for v, c in zip(vals, counts):
        print(f'  Class {v}: {c:,} ({c/total*100:.1f}%)')