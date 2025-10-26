"""
DeepLOB 分桶評估腳本
====================

目的: 檢查 DeepLOB 在不同信心區間的準確率
關鍵問題: 高信心區域 (>0.8) 準確率是否 > 70%?

使用方法:
    conda activate deeplob-pro
    python scripts/evaluate_deeplob_by_confidence.py \
        --checkpoint checkpoints/v5/deeplob_v5_best.pth \
        --data-dir data/processed_v7/npz
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 從實際訓練腳本導入模型
from src.models.deeplob import DeepLOB

# ============================================================================
# 如果導入失敗，使用備用模型定義
# ============================================================================
class DeepLOB_Backup(nn.Module):
    def __init__(self, num_classes=3, conv_filters=32, lstm_hidden=48, dropout=0.78):
        super().__init__()

        # Conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_filters, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
        )

        # Conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
        )

        # Conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 5)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        # FC
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len=100, features=20)
        x = x.unsqueeze(1)  # (batch, 1, 100, 20)

        # Conv layers
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        # Reshape for LSTM: (batch, seq, features)
        h = h.squeeze(-1).permute(0, 2, 1)  # (batch, seq, conv_filters)

        # LSTM
        h, _ = self.lstm(h)
        h = h[:, -1, :]  # Take last timestep

        # FC
        out = self.fc(h)
        return out

# ============================================================================
# 分桶評估函數
# ============================================================================
def evaluate_by_confidence(model, data_loader, device, bins):
    """按信心區間評估模型"""
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_confidence = []

    print("\n推理中...")
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_confidence.append(confidence.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"  處理批次 {batch_idx + 1}/{len(data_loader)}")

    # 合併結果
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_confidence = np.concatenate(all_confidence, axis=0)

    # 分桶統計
    print("\n" + "="*80)
    print("信心區間評估")
    print("="*80)
    print(f"{'信心區間':<15} {'樣本數':>10} {'佔比':>8} {'準確率':>10}")
    print("-"*80)

    results = []
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (all_confidence >= low) & (all_confidence < high)
        n_samples = mask.sum()

        if n_samples > 0:
            acc = accuracy_score(all_labels[mask], all_preds[mask])
            pct = n_samples / len(all_confidence) * 100

            print(f"{low:.1f}-{high:.1f}         {n_samples:>10,}   {pct:>6.1f}%   {acc*100:>8.2f}%")

            results.append({
                'bin': f'{low:.1f}-{high:.1f}',
                'n_samples': int(n_samples),
                'percentage': float(pct),
                'accuracy': float(acc)
            })
        else:
            print(f"{low:.1f}-{high:.1f}         {n_samples:>10,}   {0:>6.1f}%   {'N/A':>8}")

    print("-"*80)
    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"{'Overall':<15} {len(all_labels):>10,}  {100:>6.1f}%   {overall_acc*100:>8.2f}%")
    print("="*80)

    # 高信心區域評估
    high_conf_mask = all_confidence >= 0.8
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(all_labels[high_conf_mask], all_preds[high_conf_mask])
        high_conf_pct = high_conf_mask.sum() / len(all_confidence) * 100

        print(f"\n⭐ 高信心區域 (≥0.8):")
        print(f"   樣本數: {high_conf_mask.sum():,} ({high_conf_pct:.1f}%)")
        print(f"   準確率: {high_conf_acc*100:.2f}%")

        if high_conf_acc >= 0.70:
            print(f"   ✅ 結論: 高信心區域準確率 ≥ 70%, 可用於 RL!")
        elif high_conf_acc >= 0.65:
            print(f"   ⚠️ 結論: 高信心區域準確率 65-70%, 勉強可用")
        else:
            print(f"   ❌ 結論: 高信心區域準確率 < 65%, 建議先改進 DeepLOB")

    # 各類別在高信心區的表現
    if high_conf_mask.sum() > 0:
        print(f"\n高信心區域各類別表現:")
        for cls in range(3):
            cls_name = ['下跌', '持平', '上漲'][cls]
            cls_mask = (all_labels[high_conf_mask] == cls)
            if cls_mask.sum() > 0:
                cls_acc = accuracy_score(
                    all_labels[high_conf_mask][cls_mask],
                    all_preds[high_conf_mask][cls_mask]
                )
                print(f"   {cls_name}: {cls_acc*100:.2f}% ({cls_mask.sum():,} 樣本)")

    return {
        'bins': results,
        'overall_accuracy': float(overall_acc),
        'high_confidence': {
            'threshold': 0.8,
            'n_samples': int(high_conf_mask.sum()),
            'percentage': float(high_conf_pct) if high_conf_mask.sum() > 0 else 0,
            'accuracy': float(high_conf_acc) if high_conf_mask.sum() > 0 else 0
        }
    }

# ============================================================================
# 主程序
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='DeepLOB 分桶評估')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/v5/deeplob_v5_best.pth',
                       help='模型檢查點路徑')
    parser.add_argument('--data-dir', type=str,
                       default='data/processed_v7/npz',
                       help='數據目錄')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='設備 (cuda/cpu)')
    args = parser.parse_args()

    # 設備
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")

    # 載入數據
    print(f"\n載入數據: {args.data_dir}")
    val_data = np.load(Path(args.data_dir) / 'stock_embedding_val.npz')
    X_val = torch.FloatTensor(val_data['X'])
    y_val = torch.LongTensor(val_data['y'])

    print(f"驗證集大小: {len(X_val):,} 樣本")
    print(f"輸入形狀: {X_val.shape}")

    # 創建 DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 載入模型
    print(f"\n載入模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 從檢查點獲取配置（如果有）
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # 創建模型（使用配置中的參數）
    model = DeepLOB(
        num_classes=model_config.get('num_classes', 3),
        input_shape=tuple(model_config.get('input', {}).get('shape', [100, 20])),
        conv1_filters=model_config.get('conv1_filters', 32),
        conv2_filters=model_config.get('conv2_filters', 32),
        conv3_filters=model_config.get('conv3_filters', 32),
        lstm_hidden_size=model_config.get('lstm_hidden_size', 48),
        fc_hidden_size=model_config.get('fc_hidden_size', 48),
        dropout=model_config.get('dropout', 0.78)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型參數: {sum(p.numel() for p in model.parameters()):,}")

    # 定義分桶
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 評估
    results = evaluate_by_confidence(model, val_loader, device, bins)

    # 保存結果
    output_dir = Path("results/deeplob_confidence_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "confidence_evaluation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已保存: {output_file}")

    # 建議下一步
    print("\n" + "="*80)
    print("下一步建議")
    print("="*80)

    high_conf_acc = results['high_confidence']['accuracy']

    if high_conf_acc >= 0.70:
        print("\n✅ 建議: 直接開始 RL 訓練")
        print("\n執行命令:")
        print("    # 快速測試 (10K steps, 10 分鐘)")
        print("    python scripts/train_sb3_deeplob.py --timesteps 10000 --test")
        print("\n    # 完整訓練 (1M steps, 4-8 小時)")
        print("    python scripts/train_sb3_deeplob.py --timesteps 1000000")
    elif high_conf_acc >= 0.65:
        print("\n⚠️ 建議: 可以嘗試 RL, 但可能需要改進")
        print("\n執行命令:")
        print("    # 先快速測試")
        print("    python scripts/train_sb3_deeplob.py --timesteps 10000 --test")
        print("\n    # 若效果不佳, 再改進 DeepLOB")
    else:
        print("\n❌ 建議: 先改進 DeepLOB")
        print("\n改進方向:")
        print("    1. 收緊持平定義 (持平類 43% → 30%)")
        print("    2. 增加模型容量 (conv 48, lstm 64)")
        print("    3. 資料降噪")

if __name__ == "__main__":
    main()
