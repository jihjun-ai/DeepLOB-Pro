# -*- coding: utf-8 -*-
"""
analyze_1hz_output.py - 分析 1Hz 聚合輸出
"""

import numpy as np
import json
from pathlib import Path

def analyze_npz(npz_path):
    """分析單個 NPZ 文件"""
    data = np.load(npz_path, allow_pickle=True)

    features = data['features']
    mids = data['mids']
    bucket_event_count = data['bucket_event_count']
    bucket_mask = data['bucket_mask']
    meta = json.loads(str(data['metadata']))

    print(f"\n{'='*60}")
    print(f"檔案: {npz_path.name}")
    print(f"{'='*60}")

    # 1. 形狀資訊
    print(f"\n[形狀資訊]")
    print(f"  features: {features.shape}")
    print(f"  mids: {mids.shape}")
    print(f"  bucket_event_count: {bucket_event_count.shape}")
    print(f"  bucket_mask: {bucket_mask.shape}")

    # 2. bucket_mask 分布
    total = len(bucket_mask)
    mask_0 = (bucket_mask == 0).sum()
    mask_1 = (bucket_mask == 1).sum()
    mask_2 = (bucket_mask == 2).sum()
    mask_3 = (bucket_mask == 3).sum()

    print(f"\n[bucket_mask 分布]")
    print(f"  總秒數: {total}")
    print(f"  單事件 (mask=0): {mask_0} ({mask_0/total*100:.1f}%)")
    print(f"  ffill (mask=1): {mask_1} ({mask_1/total*100:.1f}%)")
    print(f"  缺失 (mask=2): {mask_2} ({mask_2/total*100:.1f}%)")
    print(f"  多事件 (mask=3): {mask_3} ({mask_3/total*100:.1f}%)")

    # 3. bucket_event_count 統計
    print(f"\n[事件計數統計]")
    print(f"  最小: {bucket_event_count.min()}")
    print(f"  最大: {bucket_event_count.max()}")
    print(f"  平均: {bucket_event_count.mean():.2f}")
    print(f"  中位數: {np.median(bucket_event_count):.0f}")

    zero_events = (bucket_event_count == 0).sum()
    single_events = (bucket_event_count == 1).sum()
    multi_events = (bucket_event_count >= 2).sum()

    print(f"\n  無事件秒 (count=0): {zero_events} ({zero_events/total*100:.1f}%)")
    print(f"  單事件秒 (count=1): {single_events} ({single_events/total*100:.1f}%)")
    print(f"  多事件秒 (count≥2): {multi_events} ({multi_events/total*100:.1f}%)")

    # 4. 驗收測試
    print(f"\n[驗收測試]")

    # 測試1: 形狀對齊
    shape_ok = (features.shape[0] == mids.shape[0] ==
                bucket_event_count.shape[0] == bucket_mask.shape[0])
    print(f"  ✅ 形狀對齊: {shape_ok}")

    # 測試2: 無事件標記
    zero_event_mask = bucket_mask[bucket_event_count == 0]
    zero_ok = all((zero_event_mask == 1) | (zero_event_mask == 2))
    print(f"  ✅ 無事件標記正確: {zero_ok}")

    # 測試3: 多事件標記
    multi_event_mask = bucket_mask[bucket_event_count >= 2]
    multi_ok = len(multi_event_mask) == 0 or all(multi_event_mask == 3)
    print(f"  ✅ 多事件標記正確: {multi_ok}")

    # 測試4: 統計一致性
    ffill_ratio = mask_1 / total
    missing_ratio = mask_2 / total
    stat_ok = ffill_ratio + missing_ratio <= 1.0
    print(f"  ✅ 統計一致性: {stat_ok}")

    # 5. Metadata
    print(f"\n[Metadata 關鍵欄位]")
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

    # 6. 中間價統計
    valid_mids = mids[bucket_mask != 2]
    if len(valid_mids) > 0:
        print(f"\n[中間價統計 (排除缺失)]")
        print(f"  最小: {valid_mids.min():.2f}")
        print(f"  最大: {valid_mids.max():.2f}")
        print(f"  平均: {valid_mids.mean():.2f}")
        print(f"  震盪幅度: {(valid_mids.max() - valid_mids.min()) / valid_mids.mean() * 100:.2f}%")

    return {
        'symbol': meta.get('symbol', 'N/A'),
        'total_seconds': total,
        'mask_0': mask_0,
        'mask_1': mask_1,
        'mask_2': mask_2,
        'mask_3': mask_3,
        'ffill_ratio': ffill_ratio,
        'missing_ratio': missing_ratio,
        'multi_event_ratio': mask_3 / total,
        'all_tests_pass': shape_ok and zero_ok and multi_ok and stat_ok
    }


if __name__ == "__main__":
    # 分析目錄
    output_dir = Path(r"D:\Case-New\python\DeepLOB-Pro\data\preprocessed_v5_1hz\daily\20250901")

    # 1. 分析幾個代表性股票
    sample_symbols = ['2330', '2454', '0050', '1101']

    results = []
    for symbol in sample_symbols:
        npz_file = output_dir / f"{symbol}.npz"
        if npz_file.exists():
            result = analyze_npz(npz_file)
            results.append(result)

    # 2. 整體統計
    print(f"\n\n{'='*60}")
    print("整體統計摘要")
    print(f"{'='*60}")

    all_npz = list(output_dir.glob("*.npz"))
    print(f"\n總檔案數: {len(all_npz)}")
    print(f"\n樣本分析:")
    print(f"  {'股票':<10} {'總秒':<8} {'單事件%':<10} {'ffill%':<10} {'缺失%':<10} {'多事件%':<10} {'測試'}")
    print(f"  {'-'*80}")

    for r in results:
        status = "✅" if r['all_tests_pass'] else "❌"
        print(f"  {r['symbol']:<10} {r['total_seconds']:<8} "
              f"{r['mask_0']/r['total_seconds']*100:<10.1f} "
              f"{r['ffill_ratio']*100:<10.1f} "
              f"{r['missing_ratio']*100:<10.1f} "
              f"{r['multi_event_ratio']*100:<10.1f} "
              f"{status}")

    print(f"\n✅ 1Hz 聚合輸出分析完成！")
