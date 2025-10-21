import numpy as np
import json
from pathlib import Path

output_dir = Path(r"D:\Case-New\python\DeepLOB-Pro\data\preprocessed_v5_1hz\daily\20250901")

# Analyze 2330
npz_file = output_dir / "2330.npz"
data = np.load(npz_file, allow_pickle=True)

features = data['features']
mids = data['mids']
bucket_event_count = data['bucket_event_count']
bucket_mask = data['bucket_mask']
meta = json.loads(str(data['metadata']))

print("=== 2330 Taiwan Semiconductor ===")
print(f"\nShapes:")
print(f"  features: {features.shape}")
print(f"  mids: {mids.shape}")
print(f"  bucket_event_count: {bucket_event_count.shape}")
print(f"  bucket_mask: {bucket_mask.shape}")

total = len(bucket_mask)
mask_0 = (bucket_mask == 0).sum()
mask_1 = (bucket_mask == 1).sum()
mask_2 = (bucket_mask == 2).sum()
mask_3 = (bucket_mask == 3).sum()

print(f"\nbucket_mask Distribution:")
print(f"  Total seconds: {total}")
print(f"  Single event (mask=0): {mask_0} ({mask_0/total*100:.1f}%)")
print(f"  ffill (mask=1): {mask_1} ({mask_1/total*100:.1f}%)")
print(f"  missing (mask=2): {mask_2} ({mask_2/total*100:.1f}%)")
print(f"  multi-event (mask=3): {mask_3} ({mask_3/total*100:.1f}%)")

print(f"\nEvent Count Stats:")
print(f"  Min: {bucket_event_count.min()}")
print(f"  Max: {bucket_event_count.max()}")
print(f"  Mean: {bucket_event_count.mean():.2f}")
print(f"  Median: {np.median(bucket_event_count):.0f}")

zero_events = (bucket_event_count == 0).sum()
single_events = (bucket_event_count == 1).sum()
multi_events = (bucket_event_count >= 2).sum()

print(f"\n  Zero-event seconds: {zero_events} ({zero_events/total*100:.1f}%)")
print(f"  Single-event seconds: {single_events} ({single_events/total*100:.1f}%)")
print(f"  Multi-event seconds: {multi_events} ({multi_events/total*100:.1f}%)")

print(f"\nValidation Tests:")
shape_ok = (features.shape[0] == mids.shape[0] == bucket_event_count.shape[0] == bucket_mask.shape[0])
print(f"  Shape alignment: {'PASS' if shape_ok else 'FAIL'}")

zero_event_mask = bucket_mask[bucket_event_count == 0]
zero_ok = len(zero_event_mask) == 0 or all((zero_event_mask == 1) | (zero_event_mask == 2))
print(f"  Zero-event marking: {'PASS' if zero_ok else 'FAIL'}")

multi_event_mask = bucket_mask[bucket_event_count >= 2]
multi_ok = len(multi_event_mask) == 0 or all(multi_event_mask == 3)
print(f"  Multi-event marking: {'PASS' if multi_ok else 'FAIL'}")

print(f"\nKey Metadata:")
print(f"  symbol: {meta.get('symbol', 'N/A')}")
print(f"  date: {meta.get('date', 'N/A')}")
print(f"  sampling_mode: {meta.get('sampling_mode', 'N/A')}")
print(f"  bucket_seconds: {meta.get('bucket_seconds', 'N/A')}")
print(f"  ffill_limit: {meta.get('ffill_limit', 'N/A')}")
print(f"  agg_reducer: {meta.get('agg_reducer', 'N/A')}")
print(f"  n_seconds: {meta.get('n_seconds', 'N/A')}")
print(f"  ffill_ratio: {meta.get('ffill_ratio', 0.0):.2%}")
print(f"  missing_ratio: {meta.get('missing_ratio', 0.0):.2%}")
print(f"  multi_event_ratio: {meta.get('multi_event_ratio', 0.0):.2%}")
print(f"  max_gap_sec: {meta.get('max_gap_sec', 0)}")

valid_mids = mids[bucket_mask != 2]
if len(valid_mids) > 0:
    print(f"\nMid Price Stats (excluding missing):")
    print(f"  Min: {valid_mids.min():.2f}")
    print(f"  Max: {valid_mids.max():.2f}")
    print(f"  Mean: {valid_mids.mean():.2f}")
    range_pct = (valid_mids.max() - valid_mids.min()) / valid_mids.mean() * 100
    print(f"  Range: {range_pct:.2f}%")

# Sample a few other stocks
print(f"\n\n{'='*60}")
print("Sample Analysis (Multiple Stocks)")
print(f"{'='*60}")

samples = ['0050', '1101', '2454', '2317']
print(f"\n{'Symbol':<8} {'Seconds':<8} {'Single%':<10} {'ffill%':<10} {'miss%':<10} {'multi%':<10}")
print("-" * 70)

for symbol in samples:
    npz_path = output_dir / f"{symbol}.npz"
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        mask = d['bucket_mask']
        total = len(mask)
        m0 = (mask == 0).sum()
        m1 = (mask == 1).sum()
        m2 = (mask == 2).sum()
        m3 = (mask == 3).sum()
        print(f"{symbol:<8} {total:<8} {m0/total*100:<10.1f} {m1/total*100:<10.1f} {m2/total*100:<10.1f} {m3/total*100:<10.1f}")
