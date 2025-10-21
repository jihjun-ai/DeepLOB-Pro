import numpy as np
f = np.load(r"D:\Case-New\python\DeepLOB-Pro\data\preprocessed_v5_1hz\daily\20250901\2330.npz", allow_pickle=True)
print("Keys:", list(f.keys()))
print("features:", f['features'].shape)
print("mids:", f['mids'].shape)
print("bucket_event_count:", f['bucket_event_count'].shape)
print("bucket_mask:", f['bucket_mask'].shape)
m = f['bucket_mask']
print("\nMask 0:", (m==0).sum())
print("Mask 1:", (m==1).sum())
print("Mask 2:", (m==2).sum())
print("Mask 3:", (m==3).sum())
