import numpy as np
import os

print("="*60)
print("Verifying V7 normalized data")
print("="*60)

npz_dir = 'data/processed_v7/npz'

# Check files
files = os.listdir(npz_dir)
print(f"\nFiles in {npz_dir}:")
for f in files:
    path = os.path.join(npz_dir, f)
    size = os.path.getsize(path)
    print(f"  {f}: {size:,} bytes")

# Load and check normalization
for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()} Data:")
    try:
        data = np.load(f'{npz_dir}/stock_embedding_{split}.npz')
        X = data['X']
        print(f"  Shape: {X.shape}")
        print(f"  Min: {X.min():.2f}")
        print(f"  Max: {X.max():.2f}")
        print(f"  Mean: {X.mean():.2f}")
        print(f"  Std: {X.std():.2f}")

        # Verify
        if abs(X.mean()) < 1 and 0.5 < X.std() < 2:
            print(f"  Status: OK (normalized)")
        else:
            print(f"  Status: FAILED (not normalized)")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*60)
