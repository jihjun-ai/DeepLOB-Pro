import torch
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

configs = {
    '保守版': 'checkpoints/v5_fix_conservative/deeplob_v5_best.pth',
    '中等版': 'checkpoints/v5_fix_moderate/deeplob_v5_best.pth',
    '激進版': 'checkpoints/v5_fix_aggressive/deeplob_v5_best.pth',
}

print('=' * 80)
print('三個修復配置的訓練結果比較')
print('=' * 80)

for name, path in configs.items():
    print(f'\n{name}:')
    try:
        with torch.serialization.safe_globals([ScalarFloat, ScalarInt, DoubleQuotedScalarString]):
            ckpt = torch.load(path, map_location='cpu')

        epoch = ckpt.get("epoch", "N/A")
        val_weighted_f1 = ckpt.get("val_weighted_f1", None)
        val_unweighted_acc = ckpt.get("val_unweighted_acc", None)

        print(f'  Epoch: {epoch}')
        print(f'  Val Weighted F1: {f"{val_weighted_f1:.4f}" if val_weighted_f1 is not None else "N/A"}')
        print(f'  Val Unweighted Acc: {f"{val_unweighted_acc:.4f}" if val_unweighted_acc is not None else "N/A"}')
        print(f'  Available keys: {list(ckpt.keys())[:10]}...')
    except Exception as e:
        print(f'  Error: {e}')
