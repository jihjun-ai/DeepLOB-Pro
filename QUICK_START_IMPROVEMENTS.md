# DeepLOB 改進方案 - 快速上手指南

**日期**: 2025-10-19
**目標**: 突破 45-46% 訓練平台，達到 65% 準確率

---

## 📋 總覽

基於 ChatGPT 的 7 個診斷問題，我已經為你創建了完整的改進方案：

### ✅ 已完成的實作

1. **診斷文檔** - [TRAINING_DIAGNOSIS_AND_IMPROVEMENTS.md](TRAINING_DIAGNOSIS_AND_IMPROVEMENTS.md)
2. **驗證腳本** - [scripts/verify_class_weights.py](scripts/verify_class_weights.py)
3. **改進版模型** - [src/models/deeplob_improved.py](src/models/deeplob_improved.py)
4. **Focal Loss** - [src/models/focal_loss.py](src/models/focal_loss.py)
5. **ECE 評估** - [src/evaluation/calibration.py](src/evaluation/calibration.py)

---

## 🚀 立即開始（3 步驟）

### 階段 1: 驗證當前配置（5 分鐘）

```bash
# 啟動環境
conda activate deeplob-pro

# 驗證 class weights 計算
python scripts/verify_class_weights.py
```

**預期輸出**:
```
✅ 類別權重計算正確
✅ 損失函數正確應用權重
✅ 樣本權重已禁用（避免問題）
```

---

### 階段 2: 測試改進版模型（10 分鐘）

#### 方案 A: 增加模型容量（最簡單，最可能有效）

```bash
# 1. 複製配置
cp configs/train_v5_recovery.yaml configs/train_v5_capacity_test.yaml

# 2. 手動編輯 configs/train_v5_capacity_test.yaml
#    修改以下兩行:
#      lstm_hidden_size: 64  # 從 48 增加到 64
#      fc_hidden_size: 64    # 從 48 增加到 64

# 3. 快速訓練測試（10 epochs）
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_capacity_test.yaml \
    --epochs 10
```

**判斷標準**:
- ✅ 如果 10 epochs 後驗證準確率 > 48%，繼續完整訓練
- ❌ 如果仍卡在 45-46%，進入方案 B

---

#### 方案 B: 使用改進版 DeepLOB（LayerNorm + Attention）

**測試模型**（獨立運行）:

```bash
python -c "
from src.models.deeplob_improved import DeepLOBImproved
import torch

model = DeepLOBImproved(
    num_classes=3,
    lstm_hidden=64,
    dropout=0.5,
    use_layer_norm=True,
    use_attention=True
)

x = torch.randn(16, 100, 20)
logits = model(x)
print(f'✅ 模型測試成功: {logits.shape}')
"
```

**整合到訓練**（需要修改訓練腳本）:

1. 打開 `scripts/train_deeplob_v5.py`
2. 在 imports 部分加入:
   ```python
   from src.models.deeplob_improved import DeepLOBImproved
   ```
3. 在配置中加入模型類型選項:
   ```yaml
   model:
     type: "deeplob_improved"  # 或 "deeplob" (原版)
     use_layer_norm: true
     use_attention: true
   ```
4. 修改模型初始化邏輯（見下方完整修改）

---

### 階段 3: 試用 Focal Loss（20 分鐘）

**測試 Focal Loss**:

```bash
python -c "
from src.models.focal_loss import FocalLoss
import torch

criterion = FocalLoss(gamma=2.0, label_smoothing=0.15)

logits = torch.randn(32, 3)
labels = torch.randint(0, 3, (32,))

loss = criterion(logits, labels)
print(f'✅ Focal Loss 測試成功: {loss.item():.4f}')
"
```

**整合到訓練**（見下方完整修改）

---

## 📝 完整訓練腳本修改指南

### 修改 `scripts/train_deeplob_v5.py`

#### 1️⃣ 加入新的 imports

在文件開頭的 imports 區域加入:

```python
# 新增：改進版模組
from src.models.deeplob_improved import DeepLOBImproved
from src.models.focal_loss import FocalLoss
from src.evaluation.calibration import evaluate_calibration
```

#### 2️⃣ 修改模型初始化（約在第 700-800 行）

找到原本的模型創建部分:

```python
# 原代碼（大約在 main() 函數中）
model = DeepLOB(
    num_classes=config['model']['num_classes'],
    ...
)
```

**替換為**:

```python
# 新代碼：支持多種模型
model_type = config['model'].get('type', 'deeplob')  # 默認原版

if model_type == 'deeplob_improved':
    logger.info("使用改進版 DeepLOB（LayerNorm + Attention）")
    model = DeepLOBImproved(
        num_classes=config['model']['num_classes'],
        conv_filters=config['model'].get('conv1_filters', 32),
        lstm_hidden=config['model']['lstm_hidden_size'],
        fc_hidden=config['model']['fc_hidden_size'],
        dropout=config['model']['dropout'],
        use_layer_norm=config['model'].get('use_layer_norm', True),
        use_attention=config['model'].get('use_attention', True),
        input_shape=tuple(config['model']['input']['shape'])
    )
elif model_type == 'deeplob':
    logger.info("使用原版 DeepLOB")
    model = DeepLOB(
        num_classes=config['model']['num_classes'],
        # ... 原有參數
    )
else:
    raise ValueError(f"Unknown model type: {model_type}")
```

#### 3️⃣ 修改損失函數（約在第 600-650 行）

找到 `train_epoch` 函數中的損失計算部分:

```python
# 原代碼
loss_per_sample = nn.functional.cross_entropy(
    logits, labels,
    reduction='none',
    label_smoothing=label_smoothing,
    weight=class_weights
)
```

**替換為**:

```python
# 新代碼：支持 Focal Loss
loss_type = config['loss'].get('type', 'ce')

if loss_type == 'focal':
    # 使用 Focal Loss
    if not hasattr(train_epoch, 'focal_criterion'):
        # 只初始化一次
        gamma = config['loss'].get('gamma', 2.0)
        train_epoch.focal_criterion = FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
            reduction='none'
        ).to(device)

    loss_per_sample = train_epoch.focal_criterion(logits, labels)

elif loss_type == 'ce':
    # 原有 CE Loss
    loss_per_sample = nn.functional.cross_entropy(
        logits, labels,
        reduction='none',
        label_smoothing=label_smoothing,
        weight=class_weights
    )
else:
    raise ValueError(f"Unknown loss type: {loss_type}")
```

#### 4️⃣ 加入 ECE 評估（在評估函數中）

在 `evaluate()` 函數返回結果前加入:

```python
# 在 evaluate() 函數末尾，返回 results 之前加入:

if config.get('calibration', {}).get('enabled', False):
    # 評估校準品質
    all_logits_tensor = torch.FloatTensor(all_logits)
    all_labels_tensor = torch.LongTensor(all_labels)

    from src.evaluation.calibration import compute_ece

    ece = compute_ece(all_logits_tensor, all_labels_tensor, n_bins=15)
    results['calibration'] = {'ece': ece}

    logger.info(f"  ECE: {ece:.4f}")
```

---

## 🔧 配置文件範例

### 方案 1: 增加容量 + Focal Loss

**創建**: `configs/train_v5_improved_v1.yaml`

```yaml
# 基於 train_v5_recovery.yaml，修改以下部分:

model:
  arch: "DeepLOB"
  type: "deeplob"  # 保持原版架構
  lstm_hidden_size: 64  # ✨ 48 → 64
  fc_hidden_size: 64    # ✨ 48 → 64
  dropout: 0.6

loss:
  type: "focal"         # ✨ ce → focal
  gamma: 2.0            # ✨ 新增
  class_weights: "auto"
  label_smoothing:
    global: 0.15

calibration:
  enabled: true         # ✨ 啟用 ECE 計算

train:
  early_stop:
    patience: 20        # ✨ 15 → 20
```

---

### 方案 2: 改進版 DeepLOB + Focal Loss

**創建**: `configs/train_v5_improved_v2.yaml`

```yaml
model:
  type: "deeplob_improved"  # ✨ 使用改進版
  use_layer_norm: true      # ✨ 新增
  use_attention: true       # ✨ 新增
  lstm_hidden_size: 64
  fc_hidden_size: 64
  dropout: 0.6

loss:
  type: "focal"
  gamma: 2.0
  class_weights: "auto"
  label_smoothing:
    global: 0.15

calibration:
  enabled: true

train:
  epochs: 50
  early_stop:
    patience: 20
```

---

## 🎯 執行計劃

### Day 1（今天）

#### ✅ 任務 1: 驗證基礎配置（已完成）
```bash
conda activate deeplob-pro
python scripts/verify_class_weights.py
```

#### ⏳ 任務 2: 快速容量測試（10 分鐘）
```bash
# 測試增加容量的效果
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_capacity_test.yaml \
    --epochs 10
```

**決策點**:
- 如果驗證損失改善 → 繼續完整訓練
- 如果沒改善 → 嘗試改進版模型

---

### Day 2（明天）

#### 任務 3: 完整訓練（最佳配置）

基於 Day 1 結果選擇:

**選項 A（如果容量測試有效）**:
```bash
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_improved_v1.yaml \
    --epochs 50
```

**選項 B（如果需要架構改進）**:
```bash
# 先完成訓練腳本修改（見上方），然後運行:
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_improved_v2.yaml \
    --epochs 50
```

---

### Day 3（後天）

#### 任務 4: A/B 測試與超參數優化

```bash
# 測試不同 gamma 值
for gamma in 1.5 2.0 2.5; do
    python scripts/train_deeplob_v5.py \
        --config configs/train_v5_improved_v1.yaml \
        --override loss.gamma=$gamma \
        --epochs 50
done

# 測試不同 dropout
for dropout in 0.5 0.6 0.7; do
    python scripts/train_deeplob_v5.py \
        --config configs/train_v5_improved_v1.yaml \
        --override model.dropout=$dropout \
        --epochs 50
done
```

---

## 📊 預期改進效果

| 改進項目 | 當前值 | 建議值 | 預期提升 |
|---------|--------|--------|---------|
| LSTM Hidden | 48 | 64 | +2-3% |
| FC Hidden | 48 | 64 | +1-2% |
| 損失函數 | CE | Focal (γ=2.0) | +1-2% |
| 架構 | 原版 | LayerNorm + Attention | +3-5% |
| Early Stop Patience | 15 | 20 | 避免過早停止 |

**累計預期**: 45-46% → **52-58%**（保守估計）

---

## 🐛 故障排除

### 問題 1: 改進版模型找不到

**錯誤**:
```
ImportError: cannot import name 'DeepLOBImproved'
```

**解決**:
```bash
# 確認文件存在
ls src/models/deeplob_improved.py

# 測試導入
python -c "from src.models.deeplob_improved import DeepLOBImproved; print('OK')"
```

---

### 問題 2: Focal Loss 損失值異常

**現象**: Loss 變成 NaN 或 Inf

**解決**:
1. 降低 gamma（從 2.0 → 1.5）
2. 降低學習率（從 8e-5 → 5e-5）
3. 檢查 logits 裁剪是否生效

---

### 問題 3: ECE 計算失敗

**錯誤**:
```
RuntimeError: Expected all tensors to be on the same device
```

**解決**:
```python
# 在 calibration.py 中確保所有 tensor 在 CPU
logits = logits.cpu()
labels = labels.cpu()
```

---

## 📚 相關資源

### 必讀文檔
1. [TRAINING_DIAGNOSIS_AND_IMPROVEMENTS.md](TRAINING_DIAGNOSIS_AND_IMPROVEMENTS.md) - 完整診斷報告
2. [CLAUDE.md](CLAUDE.md) - 專案總覽

### 參考論文
1. **Focal Loss**: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
2. **Temperature Scaling**: [Guo et al., 2017](https://arxiv.org/abs/1706.04599)
3. **DeepLOB**: [Zhang et al., 2019](https://arxiv.org/abs/1808.03668)

---

## ✅ 檢查清單

### 在開始訓練前，確認:

- [ ] ✅ 已運行 `verify_class_weights.py`
- [ ] ✅ 已測試改進版模型（如使用）
- [ ] ✅ 已測試 Focal Loss（如使用）
- [ ] ✅ 配置文件中 `use_sample_weights=false`
- [ ] ✅ 配置文件中 `balance_sampler.enabled=false`
- [ ] ✅ GPU 可用（`torch.cuda.is_available()`）
- [ ] ✅ 數據文件存在且完整

---

## 🎓 下一步

完成 Day 1-3 的訓練後:

1. **分析結果**: 查看 TensorBoard 日誌
2. **比較配置**: 哪個配置效果最好？
3. **錯誤分析**: 查看混淆矩陣，哪個類別仍然困難？
4. **進一步優化**:
   - 資料增強
   - 集成學習
   - 調整 triple-barrier 參數

---

**最後更新**: 2025-10-19
**版本**: v1.0
**狀態**: 可立即執行 ✅
