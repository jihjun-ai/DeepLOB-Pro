# DeepLOB 訓練診斷與改進方案
**日期**: 2025-10-19
**當前狀態**: 訓練卡在 45-46% 平台
**目標**: 突破 65% 準確率

---

## 📊 問題診斷總結

### 當前配置狀態
- ✅ **已正確關閉** `use_sample_weights=false`
- ✅ **使用** `class_weights='auto'`（PyTorch 逆頻率計算）
- ✅ **Label smoothing** = 0.15（合理）
- ⚠️ **模型容量偏小**: `lstm_hidden=48`, `fc_hidden=48`
- ⚠️ **缺少** LayerNorm 和 Attention Pooling
- ⚠️ **缺少** ECE 計算和校準驗證

### 7 個關鍵問題解答

#### 1️⃣ 權重合成是否合理？
**✅ 當前實現正確**
- 你的 `cross_entropy(..., weight=class_weights)` 是 PyTorch 標準用法
- 與 label smoothing 完全兼容

**未來如果重啟 sample_weights，建議**:
```python
# 對數壓縮 + clip + normalize
w = np.log1p(w)
w = np.clip(w, 0.1, 5.0)
w = w / (w.mean() + 1e-8)
```

**更推薦: Focal Loss（自動關注難樣本）**
- 無需手動 sample weights
- 自動平衡簡單/困難樣本
- 見下方實作

---

#### 2️⃣ 損失計算的位置與次序
**✅ 你的實現完全正確**

PyTorch 內部處理流程:
1. 計算 label smoothing 分佈
2. 計算 log softmax
3. 應用 class weights（對每類加權）
4. 返回加權損失

你的代碼已經正確使用 PyTorch 內建功能，無需修改。

---

#### 3️⃣ 早停與指標一致性
**✅ 使用 `f1_macro_unweighted` 是正確的**

**經驗法則**:

| 場景 | 早停指標 | Patience |
|------|---------|----------|
| 數據不平衡（你的情況） | `f1_macro` | 15-20 |
| 數據平衡 | `f1_weighted` 或 `acc` | 10-15 |
| 極度不平衡（100:1） | `balanced_accuracy` | 20-25 |
| 金融應用（重視少數類） | `f1_macro` 或 `min_class_recall` | 15-20 |

**建議調整**:
```yaml
early_stop:
  metric: "val.f1_macro_unweighted"
  patience: 20  # 增加耐心（從 15 → 20）
  min_delta: 0.001  # 新增：只有改善 >0.1% 才算進步
```

---

#### 4️⃣ Label Smoothing 取值
**當前值 0.15 合理**

**建議: 階段性調整**
```python
def get_adaptive_smoothing(epoch, total_epochs):
    """前期 0.15，後期漸進到 0.25"""
    if epoch < total_epochs * 0.3:
        return 0.15
    else:
        progress = (epoch - total_epochs * 0.3) / (total_epochs * 0.7)
        return 0.15 + 0.10 * progress  # 0.15 → 0.25
```

**必須新增: ECE 計算**（見下方實作）
- 評估校準品質
- 與溫度縮放結合
- 監控 overconfidence

---

#### 5️⃣ Sample Weights 安全守則
**✅ 你已正確禁用平衡採樣器**
```yaml
balance_sampler:
  enabled: false  # ✅ 正確
```

**關鍵原則: 避免雙重加權**
```python
# ❌ 錯誤: 雙重放大
train_loader = DataLoader(dataset, sampler=WeightedRandomSampler(...))
loss = (loss * sample_weights).mean()  # 結果: 權重 × 權重

# ✅ 正確: 二選一
# 方案 A: 只用 loss weighting（推薦）
train_loader = DataLoader(dataset, shuffle=True)
loss = (loss * sample_weights).mean()

# 方案 B: 只用採樣加權
train_loader = DataLoader(dataset, sampler=WeightedRandomSampler(...))
loss = loss.mean()
```

---

#### 6️⃣ 模型端的 Shape/正則
**⚠️ 當前容量可能不足**
```yaml
lstm_hidden_size: 48  # 建議 → 64
fc_hidden_size: 48    # 建議 → 64
dropout: 0.6          # ✅ 已足夠
```

**a) 建議立即嘗試: 增加容量**
```yaml
model:
  lstm_hidden_size: 64  # 48 → 64
  fc_hidden_size: 64    # 48 → 64
  dropout: 0.6
```

**b) 架構改進: LayerNorm + Attention Pooling**
- 改善梯度流
- 更好的長程依賴建模
- 見下方完整實作

**c) 備選方案: 輕量 Transformer**
- 單層 Transformer Encoder
- 替代 LSTM
- A/B 測試用

---

#### 7️⃣ Metrics 穩定性
**✅ 你使用 sklearn（已處理邊界情況）**

你的實現:
```python
precision, recall, _, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, zero_division=0  # ✅ 安全
)
```

sklearn 已自動處理:
- 除零保護（`zero_division=0`）
- 缺失類別
- 空樣本

**無需修改**，但如果要手寫版本，見下方實作。

---

## 🎯 立即行動計劃（優先順序）

### 階段 1: 快速驗證（今天完成）

#### 1.1 增加模型容量（最可能有效）
```bash
# 複製配置文件
cp configs/train_v5_recovery.yaml configs/train_v5_recovery_v2.yaml

# 修改以下參數:
# lstm_hidden_size: 48 → 64
# fc_hidden_size: 48 → 64
# patience: 15 → 20

# 訓練 10 epochs 快速測試
conda activate deeplob-pro
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_recovery_v2.yaml \
    --epochs 10
```

**預期結果**:
- 如果 10 epochs 後驗證準確率 > 48%，則值得繼續
- 觀察是否仍卡在平台

---

#### 1.2 驗證 Class Weights 計算
```bash
conda activate deeplob-pro
python scripts/verify_class_weights.py
```

見下方腳本實作。

---

### 階段 2: 架構改進（明天）

#### 2.1 實作改進版 DeepLOB
- 新增 LayerNorm
- 新增 Attention Pooling
- 保持相同訓練流程

```bash
# 使用改進版模型
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_improved_model.yaml \
    --epochs 50
```

---

#### 2.2 實作 Focal Loss
```bash
# A/B 測試: Focal Loss vs CE Loss
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_focal_loss.yaml \
    --epochs 50
```

---

### 階段 3: 精細調優（後天）

#### 3.1 加入 ECE 監控
- 評估校準品質
- 優化溫度縮放

#### 3.2 超參數搜索
- Learning rate: [5e-5, 8e-5, 1e-4]
- Dropout: [0.5, 0.6, 0.7]
- Label smoothing: [0.15, 0.20, 0.25]

---

## 📦 完整實作代碼

### 1️⃣ 驗證腳本: `scripts/verify_class_weights.py`
見下方文件

### 2️⃣ 改進版 DeepLOB: `src/models/deeplob_improved.py`
見下方文件

### 3️⃣ Focal Loss: `src/models/focal_loss.py`
見下方文件

### 4️⃣ ECE 計算: `src/evaluation/calibration.py`
見下方文件

### 5️⃣ 穩健 Metrics: `src/evaluation/robust_metrics.py`
見下方文件

---

## 📈 預期改進效果

| 改進項目 | 預期提升 | 實施難度 | 優先級 |
|---------|---------|---------|--------|
| 增加模型容量（48→64） | +2-3% | ⭐ 簡單 | 🔥 最高 |
| LayerNorm + Attention | +3-5% | ⭐⭐ 中等 | 🔥 高 |
| Focal Loss | +1-2% | ⭐⭐ 中等 | 🔥 中 |
| 調整 Label Smoothing | +1-2% | ⭐ 簡單 | 🔥 中 |
| 溫度縮放 + ECE | +0-1% | ⭐ 簡單 | 🔥 低 |

**累計預期**: 從 45-46% → **52-58%**（保守估計）

---

## 🚨 常見陷阱

### ❌ 不要做
1. ⛔ **不要同時啟用** `WeightedRandomSampler` + `sample_weights`
2. ⛔ **不要過早增加** label smoothing（先確認模型收斂）
3. ⛔ **不要同時改多個參數**（無法歸因改進來源）
4. ⛔ **不要在數據不平衡時用** `accuracy` 早停

### ✅ 應該做
1. ✅ **每次只改一個變量**（對照實驗）
2. ✅ **保留所有訓練日誌**（便於回溯）
3. ✅ **監控梯度範數**（檢測數值穩定性）
4. ✅ **定期檢查混淆矩陣**（了解模型行為）

---

## 📚 相關文檔

- [QUICK_START_FIX.md](QUICK_START_FIX.md) - 當前問題修復
- [VOLATILITY_ANALYSIS_GUIDE.md](VOLATILITY_ANALYSIS_GUIDE.md) - 波動率分析
- [TRAINING_PLAN_20251019.md](TRAINING_PLAN_20251019.md) - 訓練計劃

---

**下一步**: 執行階段 1.1（增加模型容量）
