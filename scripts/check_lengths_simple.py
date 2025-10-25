import numpy as np
import glob
import os

npz_files = sorted(glob.glob('data/preprocessed_swing/daily/*/*.npz'))
print(f'Found {len(npz_files)} NPZ files\n')

# Check first 20 files in detail
print('Checking first 20 files:')
print('='*80)

for npz_file in npz_files[:20]:
    data = np.load(npz_file, allow_pickle=True)
    filename = os.path.basename(npz_file)

    print(f'\n{filename}:')
    for key in sorted(data.files):
        if key == 'metadata':
            continue
        arr = data[key]
        print(f'  {key:20s}: {len(arr):8d}')

# Check all files for consistency
print('\n' + '='*80)
print('Checking all files for length consistency...')
print('='*80 + '\n')

from collections import defaultdict
length_stats = defaultdict(lambda: defaultdict(list))

for npz_file in npz_files:
    data = np.load(npz_file, allow_pickle=True)
    filename = os.path.basename(npz_file)

    for key in data.files:
        if key == 'metadata':
            continue
        arr = data[key]
        length = len(arr)
        length_stats[key][length].append(filename)

# Report
for field in sorted(length_stats.keys()):
    lengths = length_stats[field]
    print(f'\nField: {field}')
    if len(lengths) == 1:
        length = list(lengths.keys())[0]
        count = len(list(lengths.values())[0])
        print(f'  OK: All files have length {length} ({count} files)')
    else:
        print(f'  WARNING: Found {len(lengths)} different lengths!')
        for length in sorted(lengths.keys()):
            files = lengths[length]
            print(f'    Length {length}: {len(files)} files')
            for f in files[:3]:
                print(f'      - {f}')
