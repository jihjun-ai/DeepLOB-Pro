import numpy as np
import json

# Load 2330
data = np.load(r"D:\Case-New\python\DeepLOB-Pro\data\preprocessed_v5_1hz\daily\20250901\2330.npz", allow_pickle=True)

print("Arrays in NPZ:")
print(list(data.keys()))
print()

features = data['features']
mids = data['mids']
bucket_event_count = data['bucket_event_count']
bucket_mask = data['bucket_mask']

print(f"features shape: {features.shape}")
print(f"mids shape: {mids.shape}")
print(f"bucket_event_count shape: {bucket_event_count.shape}")
print(f"bucket_mask shape: {bucket_mask.shape}")
print()

total = len(bucket_mask)
print(f"bucket_mask distribution:")
print(f"  mask=0 (single): {(bucket_mask==0).sum()} ({(bucket_mask==0).sum()/total*100:.1f}%)")
print(f"  mask=1 (ffill): {(bucket_mask==1).sum()} ({(bucket_mask==1).sum()/total*100:.1f}%)")
print(f"  mask=2 (missing): {(bucket_mask==2).sum()} ({(bucket_mask==2).sum()/total*100:.1f}%)")
print(f"  mask=3 (multi): {(bucket_mask==3).sum()} ({(bucket_mask==3).sum()/total*100:.1f}%)")
print()

print(f"bucket_event_count stats:")
print(f"  min: {bucket_event_count.min()}")
print(f"  max: {bucket_event_count.max()}")
print(f"  mean: {bucket_event_count.mean():.2f}")
print()

meta = json.loads(str(data['metadata']))
print("Metadata:")
for k, v in meta.items():
    print(f"  {k}: {v}")
