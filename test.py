import numpy as np 
data = np.load('data/processed_v6_balanced/npz/stock_embedding_train.npz')
y = data['y'] 
dist = np.bincount(y) 
pct = dist / dist.sum() * 100
print(f'Down={pct[0]:.1f}%, Neutral={pct[1]:.1f}%, Up={pct[2]:.1f}%')
print(f'Imbalance: {max(dist)/min(dist):.2f}x')
