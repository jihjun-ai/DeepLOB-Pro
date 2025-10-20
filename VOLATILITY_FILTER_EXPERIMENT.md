# 震盪篩選實驗執行指南

## 📋 實驗概述

根據實際數據統計結果（2,145 個 symbol-day，平均震盪 3.25%）設計的 4 組對照實驗。

**實驗目的**：驗證震盪篩選對 DeepLOB 模型準確率的影響

---

## 🎯 實驗組設計

| 組別 | 配置文件 | 震盪範圍 | 預期保留 | 說明 |
|------|---------|---------|---------|------|
| **基準組** | `config_pro_v5_baseline.yaml` | 無限制 | 100% (2,145) | 對照基準 |
| **組1** | `config_pro_v5_filter1.5.yaml` | 1.5% ~ 10% | ~65% (1,397) | 溫和篩選 |
| **組2** | `config_pro_v5_filter2.0.yaml` | **2.0% ~ 10%** | **~57% (1,219)** | **推薦** ⭐ |
| **組3** | `config_pro_v5_filter2.5.yaml` | 2.5% ~ 10% | ~49% (1,060) | 嚴格篩選 |

---

## 🚀 快速執行指令

### 一鍵執行腳本

```bash
# 激活環境
conda activate deeplob-pro

# 基準組（無篩選）
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_baseline \
    --config configs/config_pro_v5_baseline.yaml \
    --make-npz

# 組1 - 溫和篩選（1.5%）
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_filter1.5 \
    --config configs/config_pro_v5_filter1.5.yaml \
    --make-npz

# 組2 - 推薦篩選（2.0%）⭐ 重點
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_filter2.0 \
    --config configs/config_pro_v5_filter2.0.yaml \
    --make-npz

# 組3 - 嚴格篩選（2.5%）
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_filter2.5 \
    --config configs/config_pro_v5_filter2.5.yaml \
    --make-npz
```

---

## 📊 執行監控與驗證

### 執行時會顯示

```
============================================================
📊 震盪幅度統計報告（Intraday Range Analysis）
============================================================
總樣本數: 2,145 個 symbol-day 組合

閾值篩選統計（震盪 ≥ X% 的樣本數）:
  2.0% |  1,219 |  56.8%  ← 實際保留
============================================================

⚡ 震盪篩選啟用：
  - 下限: 2.0%
  - 上限: 10.0%
  - 過濾樣本: 926 個 (43.2%)
============================================================
```

### 驗證輸出

每組執行完成後檢查：

```bash
# 檢查輸出檔案
ls -lh data/processed_v5_filter2.0/npz/

# 應該看到：
# stock_embedding_train.npz  (數 GB)
# stock_embedding_val.npz
# stock_embedding_test.npz
# normalization_meta.json
```

### 查看樣本數

```python
import numpy as np

# 載入訓練數據
data = np.load('data/processed_v5_filter2.0/npz/stock_embedding_train.npz')
print(f"訓練樣本數: {data['X'].shape[0]:,}")
# 預期：數百萬樣本（取決於具體數據量）
```

---

## ⏱️ 時間估算

| 組別 | 預估時間 | 備註 |
|------|---------|------|
| 基準組 | 45-90 分鐘 | 最多樣本 |
| 組1 (1.5%) | 35-70 分鐘 | ~65% 樣本 |
| 組2 (2.0%) | 30-60 分鐘 | ~57% 樣本 |
| 組3 (2.5%) | 25-50 分鐘 | ~49% 樣本 |
| **總計** | **3-6 小時** | 循序執行 |

**加速建議**：使用 4 個終端並行執行，總時間約 1-2 小時

---

## 📁 輸出結構

```
data/
├── processed_v5_baseline/
│   ├── volatility_stats.csv
│   ├── volatility_summary.json
│   └── npz/
│       ├── stock_embedding_train.npz
│       ├── stock_embedding_val.npz
│       ├── stock_embedding_test.npz
│       └── normalization_meta.json
├── processed_v5_filter1.5/
├── processed_v5_filter2.0/  ⭐ 推薦組
└── processed_v5_filter2.5/
```

---

## 🎯 下一步：訓練模型

```bash
# 訓練基準組
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v5_baseline/npz \
    --checkpoint-dir ./checkpoints/deeplob_baseline

# 訓練組2（推薦）
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v5_filter2.0/npz \
    --checkpoint-dir ./checkpoints/deeplob_filter2.0
```

---

## 📝 檢查清單

### 執行前

- [ ] `conda activate deeplob-pro`
- [ ] 確認數據目錄：`ls ./data/temp`
- [ ] 磁碟空間充足（>20GB）

### 執行後

- [ ] 4 組數據都已生成
- [ ] 檢查樣本數是否符合預期
- [ ] 查看震盪統計報告

---

**創建日期**：2025-10-19
**狀態**：準備執行
**預估完成時間**：3-6 小時
