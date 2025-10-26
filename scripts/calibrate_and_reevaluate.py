"""
校準 DeepLOB 並重新評估
======================

目的: 修復模型輸出機率過於保守的問題
方法: Temperature Scaling + 分桶重新評估

使用:
    python scripts/calibrate_and_reevaluate.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.deeplob import DeepLOB

# ============================================================================
# Temperature Scaling
# ============================================================================
class TemperatureScaler(nn.Module):
    """Temperature Scaling for calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def calibrate_model(model, val_loader, device):
    """校準模型"""
    print("\n" + "="*80)
    print("Temperature Scaling 校準")
    print("="*80)

    # 收集驗證集的 logits
    model.eval()
    all_logits = []
    all_labels = []

    print("收集驗證集預測...")
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_labels.append(y)

            if (batch_idx + 1) % 100 == 0:
                print(f"  {batch_idx + 1}/{len(val_loader)}")

    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    print(f"\n收集完成: {len(all_logits):,} 樣本")

    # Temperature Scaling
    scaler = TemperatureScaler().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def eval_fn():
        optimizer.zero_grad()
        scaled_logits = scaler(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    print("\n優化 Temperature...")
    optimizer.step(eval_fn)

    optimal_temp = scaler.temperature.item()
    print(f"\n最優 Temperature: {optimal_temp:.4f}")

    # 評估校準前後
    with torch.no_grad():
        # 校準前
        probs_before = torch.softmax(all_logits, dim=1)
        conf_before = torch.max(probs_before, dim=1)[0]

        # 校準後
        scaled_logits = all_logits / optimal_temp
        probs_after = torch.softmax(scaled_logits, dim=1)
        conf_after = torch.max(probs_after, dim=1)[0]

    print(f"\n校準前信心統計:")
    print(f"  平均: {conf_before.mean():.4f}")
    print(f"  中位數: {conf_before.median():.4f}")
    print(f"  Max: {conf_before.max():.4f}")
    print(f"  ≥0.8: {(conf_before >= 0.8).sum():,} 樣本 ({(conf_before >= 0.8).float().mean()*100:.2f}%)")

    print(f"\n校準後信心統計:")
    print(f"  平均: {conf_after.mean():.4f}")
    print(f"  中位數: {conf_after.median():.4f}")
    print(f"  Max: {conf_after.max():.4f}")
    print(f"  ≥0.8: {(conf_after >= 0.8).sum():,} 樣本 ({(conf_after >= 0.8).float().mean()*100:.2f}%)")

    return optimal_temp, probs_after.cpu().numpy(), all_labels.cpu().numpy()

# ============================================================================
# 分桶評估
# ============================================================================
def evaluate_bins(probs, labels):
    """分桶評估"""
    from sklearn.metrics import accuracy_score

    confidence = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("\n" + "="*80)
    print("校準後分桶評估")
    print("="*80)
    print(f"{'信心區間':<15} {'樣本數':>10} {'佔比':>8} {'準確率':>10}")
    print("-"*80)

    results = []
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (confidence >= low) & (confidence < high)
        n_samples = mask.sum()

        if n_samples > 0:
            acc = accuracy_score(labels[mask], preds[mask])
            pct = n_samples / len(confidence) * 100
            print(f"{low:.1f}-{high:.1f}         {n_samples:>10,}   {pct:>6.1f}%   {acc*100:>8.2f}%")

            results.append({
                'bin': f'{low:.1f}-{high:.1f}',
                'n_samples': int(n_samples),
                'percentage': float(pct),
                'accuracy': float(acc)
            })

    print("-"*80)
    overall_acc = accuracy_score(labels, preds)
    print(f"{'Overall':<15} {len(labels):>10,}  {100:>6.1f}%   {overall_acc*100:>8.2f}%")
    print("="*80)

    # 高信心區域
    high_conf_mask = confidence >= 0.8
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(labels[high_conf_mask], preds[high_conf_mask])
        high_conf_pct = high_conf_mask.sum() / len(confidence) * 100

        print(f"\n⭐ 高信心區域 (≥0.8):")
        print(f"   樣本數: {high_conf_mask.sum():,} ({high_conf_pct:.1f}%)")
        print(f"   準確率: {high_conf_acc*100:.2f}%")

        if high_conf_acc >= 0.70:
            print(f"   ✅ 結論: 高信心區域準確率 ≥ 70%, 可用於 RL!")
            return True
        elif high_conf_acc >= 0.65:
            print(f"   ⚠️ 結論: 高信心區域準確率 65-70%, 勉強可用")
            return True
        else:
            print(f"   ❌ 結論: 高信心區域準確率 < 65%")
            return False
    else:
        print(f"\n⚠️ 高信心區域仍然沒有樣本")
        return False

    return results

# ============================================================================
# 主程序
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 載入數據
    print("\n載入數據...")
    val_data = np.load('data/processed_v7/npz/stock_embedding_val.npz')
    X_val = torch.FloatTensor(val_data['X'])
    y_val = torch.LongTensor(val_data['y'])

    from torch.utils.data import TensorDataset, DataLoader
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 載入模型
    print("\n載入模型...")
    checkpoint = torch.load('checkpoints/v5/deeplob_v5_best.pth',
                           map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = DeepLOB(
        num_classes=3,
        input_shape=(100, 20),
        conv1_filters=model_config.get('conv1_filters', 32),
        conv2_filters=model_config.get('conv2_filters', 32),
        conv3_filters=model_config.get('conv3_filters', 32),
        lstm_hidden_size=model_config.get('lstm_hidden_size', 48),
        fc_hidden_size=model_config.get('fc_hidden_size', 48),
        dropout=model_config.get('dropout', 0.78)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 校準
    optimal_temp, probs_after, labels = calibrate_model(model, val_loader, device)

    # 重新評估
    can_use_for_rl = evaluate_bins(probs_after, labels)

    # 保存校準參數
    output = {
        'optimal_temperature': float(optimal_temp),
        'can_use_for_rl': can_use_for_rl
    }

    output_file = Path('results/deeplob_confidence_eval/calibration_result.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n校準參數已保存: {output_file}")

    # 下一步建議
    print("\n" + "="*80)
    print("下一步建議")
    print("="*80)

    if can_use_for_rl:
        print("\n✅ 校準成功! 可以開始 RL 訓練")
        print("\n執行命令:")
        print(f"    # 使用校準後的 temperature={optimal_temp:.4f}")
        print("    # 快速測試")
        print("    python scripts/train_sb3_deeplob.py --timesteps 10000 --test")
    else:
        print("\n⚠️ 校準後仍不理想，建議:")
        print("    1. 降低 Label Smoothing (0.028 → 0.01)")
        print("    2. 降低 Dropout (0.78 → 0.65)")
        print("    3. 重新訓練模型")

if __name__ == "__main__":
    main()
