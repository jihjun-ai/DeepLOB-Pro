# DeepLOB 訓練快速開始

**版本**: v1.0 | **最後更新**: 2025-10-23

---

## 🚀 一鍵啟動（推薦新手）

```bash
# 完整流程（自動化）
cd D:\Case-New\python\DeepLOB-Pro
conda activate deeplob-pro

# 步驟 1: 預處理（首次，30 分鐘）
scripts\batch_preprocess.bat

# 步驟 2: 生成訓練數據（5-10 分鐘）
python scripts\extract_tw_stock_data_v6.py --preprocessed-dir data\preprocessed_v5_1hz --output-dir data\processed_v6 --config configs\config_pro_v5_ml_optimal.yaml

# 步驟 3: 訓練模型（2-4 小時）
python scripts\train_deeplob_v5.py --config configs\train_v5.yaml
```

---

## 📋 三階段處理流程

```
原始 TXT → [階段1] → 預處理 NPZ → [階段2] → 訓練 NPZ → [階段3] → 訓練模型
  (30分)     清洗聚合      (5-10分)    時間窗口       (2-4小時)   DeepLOB
```

---

## 🎯 關鍵決策點

### 1. 選擇標籤方法

| 方法 | 適合交易類型 | 特點 |
|-----|-----------|------|
| **Triple-Barrier** | 高頻交易 | 10-20 次/天，0.3-0.5% 利潤 |
| **趨勢標籤** | 波段交易 | 1-2 次/天，≥1% 利潤 |

**配置位置**: `configs/config_pro_v5_ml_optimal.yaml`
```yaml
labeling:
  method: "triple_barrier"  # 或 "trend"
```

---

### 2. 選擇權重策略

**推薦**: `effective_num_0999` ⭐⭐⭐⭐⭐

**配置位置**: `configs/train_v5.yaml`
```yaml
training:
  weight_strategy: "effective_num_0999"
```

**查看權重**: 使用 Label Viewer
```bash
cd label_viewer
python app_preprocessed.py
# 勾選「權重策略對比」
```

---

## 📊 驗收標準

| 指標 | 最低要求 | 良好 | 優秀 |
|-----|---------|------|------|
| 測試準確率 | > 65% | > 70% ✅ | > 75% ⭐ |
| Macro F1 | > 65% | > 70% ✅ | > 75% ⭐ |

---

## 🔍 檢查數據質量

```bash
# 使用 Label Viewer
cd label_viewer
python app_preprocessed.py

# 檢查項目:
✅ 標籤分布（30/40/30 最佳）
✅ 事件數量（平均 > 5）
✅ 有效比例（> 95%）
```

---

## 🛠️ 常見問題

### 記憶體不足
```yaml
# configs/train_v5.yaml
training:
  batch_size: 64  # 降低（原本 128）
```

### 準確率停滯
```yaml
# 調整權重策略
training:
  weight_strategy: "effective_num_0999"  # 從 uniform 改為此
```

### 找不到數據
```bash
# 檢查路徑
dir data\preprocessed_v5_1hz\daily\
dir data\processed_v6\npz\
```

---

## 📚 完整文檔

**詳細指南**: [PREPROCESSED_TO_TRAINING_GUIDE.md](docs/PREPROCESSED_TO_TRAINING_GUIDE.md) ⭐⭐⭐⭐⭐

**其他文檔**:
- [PREPROCESSED_DATA_SPECIFICATION.md](docs/PREPROCESSED_DATA_SPECIFICATION.md) - NPZ 格式
- [WEIGHT_STRATEGIES_GUIDE.md](label_viewer/WEIGHT_STRATEGIES_GUIDE.md) - 權重策略
- [TREND_LABELING_IMPLEMENTATION.md](docs/TREND_LABELING_IMPLEMENTATION.md) - 標籤方法

---

## 💡 提示

**提示 1**: 先用 1-2 天數據測試流程（30 分鐘）
**提示 2**: 使用 Label Viewer 檢查標籤分布再訓練
**提示 3**: 記錄每次實驗的參數和結果
**提示 4**: TensorBoard 監控訓練過程
**提示 5**: 權重策略選擇 `effective_num_0999` 最穩

---

**快速支援**: 查看 [PREPROCESSED_TO_TRAINING_GUIDE.md](docs/PREPROCESSED_TO_TRAINING_GUIDE.md)
