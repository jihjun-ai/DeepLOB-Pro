import torch

ckpt = torch.load('checkpoints/v5/deeplob_v5_best.pth', weights_only=False)
print('檢查點包含的鍵:', list(ckpt.keys()))

if 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt

print('\n前 20 個權重鍵:')
for i, k in enumerate(list(state_dict.keys())[:20]):
    print(f'  {i+1}. {k}')

print(f'\n總共 {len(state_dict)} 個權重')

# 檢查是否有 config
if 'config' in ckpt:
    print('\n配置信息:', ckpt['config'])
