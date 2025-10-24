# 從預處理數據到 DeepLOB 訓練完整指南

**版本**: v1.0
**最後更新**: 2025-10-23
**適用範圍**: preprocess_single_day.py → extract_tw_stock_data_v6.py → train_deeplob_v5.py

---

## 📋 目錄

1. [流程概述](#流程概述)
2. [階段 1: 預處理（已完成）](#階段-1-預處理已完成)
3. [階段 2: 生成訓練數據](#階段-2-生成訓練數據)
4. [階段 3: 訓練 DeepLOB 模型](#階段-3-訓練-deeplob-模型)
5. [完整範例](#完整範例)
6. [常見問題](#常見問題)

---

## 流程概述

### 三階段處理流程

```
┌─────────────────────────────────────────────────────────────────┐
│  階段 1: 預處理（preprocess_single_day.py）                      │
├─────────────────────────────────────────────────────────────────┤
│  輸入: 原始 TXT 文件                                             │
│  處理: 清洗、聚合、動態過濾、標籤預覽                             │
│  輸出: NPZ 文件（每檔股票一個）                                   │
│       └─ data/preprocessed_v5_1hz/daily/20250901/2330.npz      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  階段 2: 生成訓練數據（extract_tw_stock_data_v6.py）             │
├─────────────────────────────────────────────────────────────────┤
│  輸入: 預處理 NPZ 文件（多檔股票）                                │
│  處理: Z-Score 標準化、時間序列窗口、標籤生成                     │
│  輸出: 訓練/驗證/測試 NPZ                                        │
│       ├─ stock_embedding_train.npz                              │
│       ├─ stock_embedding_val.npz                                │
│       ├─ stock_embedding_test.npz                               │
│       └─ normalization_meta.json                                │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  階段 3: 訓練模型（train_deeplob_v5.py）                         │
├─────────────────────────────────────────────────────────────────┤
│  輸入: 訓練數據 NPZ                                              │
│  處理: DeepLOB 訓練（CNN-LSTM）                                  │
│  輸出: 訓練好的模型                                              │
│       ├─ checkpoints/best_model.pth                             │
│       └─ logs/training_metrics.json                             │
└─────────────────────────────────────────────────────────────────┘
```

### 為什麼需要三個階段？

| 階段 | 為什麼需要？ | 優點 |
|-----|-----------|------|
| 階段 1 | 原始數據質量差，需要清洗和過濾 | ✅ 一次處理，多次使用<br>✅ 快速檢查標籤分布 |
| 階段 2 | DeepLOB 需要特定格式（時間序列窗口） | ✅ 快速調整參數<br>✅ 支援多種標籤方法 |
| 階段 3 | 模型訓練需要優化和調參 | ✅ 專注模型架構<br>✅ 獨立調參 |

---

## 階段 1: 預處理（已完成）

### 如果您已經運行過 `preprocess_single_day.py`

✅ **恭喜！您已經完成第一階段。**

您應該有以下輸出：

```
data/preprocessed_v5_1hz/
└── daily/
    ├── 20250901/
    │   ├── 0050.npz
    │   ├── 2330.npz
    │   ├── 2317.npz
    │   ├── ... (更多股票)
    │   └── summary.json
    ├── 20250902/
    │   └── ...
    └── ...
```

### 檢查預處理結果

```bash
# 使用 Label Viewer 檢查
cd label_viewer
python app_preprocessed.py

# 在瀏覽器中打開 http://localhost:8051
# 載入目錄: data/preprocessed_v5_1hz/daily/20250901
# 查看標籤分布、數據質量
```

### 如果尚未預處理

```bash
# 批次預處理所有歷史數據
scripts\batch_preprocess.bat

# 或單獨處理某一天
python scripts/preprocess_single_day.py \
    --input data/raw/Ticker_20250901.txt \
    --output data/preprocessed_v5_1hz \
    --config configs/config_pro_v5_ml_optimal.yaml
```

---

## 階段 2: 生成訓練數據

### 目標

將預處理的 NPZ 文件轉換為 DeepLOB 訓練格式：
- 時間序列窗口（100 timesteps）
- Z-Score 標準化
- 按股票劃分 train/val/test (70/15/15)

### 使用腳本

**腳本**: `scripts/extract_tw_stock_data_v6.py`

**基本用法**:

```bash
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5_1hz \
    --output-dir data/processed_v6 \
    --config configs/config_pro_v5_ml_optimal.yaml
```

### 參數說明

| 參數 | 說明 | 範例 |
|-----|------|------|
| `--preprocessed-dir` | 預處理數據根目錄 | `data/preprocessed_v5_1hz` |
| `--output-dir` | 輸出目錄 | `data/processed_v6` |
| `--config` | 配置文件 | `configs/config_pro_v5_ml_optimal.yaml` |
| `--date-pattern` | 日期過濾（可選） | `2025090*`（9月所有日期）|

### 配置文件要點

**文件**: `configs/config_pro_v5_ml_optimal.yaml`

**關鍵設置**:

```yaml
# 標籤方法選擇（二選一）
labeling:
  method: "triple_barrier"  # 或 "trend"

  # Triple-Barrier 配置（高頻交易）
  triple_barrier:
    pt_mult: 2.0
    sl_mult: 2.0
    max_holding: 200
    min_return: 0.0001
    ewma_halflife: 60

  # 趨勢標籤配置（波段交易）
  trend:
    lookback_seconds: 600
    looforward_seconds: 900
    profit_threshold: 0.01
    volume_percentile: 50

# 時間序列窗口
preprocessing:
  seq_len: 100  # DeepLOB 固定使用 100

# 數據劃分
split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
```

### 標籤方法選擇 ⭐

根據 [TREND_LABELING_IMPLEMENTATION.md](TREND_LABELING_IMPLEMENTATION.md)：

#### Triple-Barrier（推薦：高頻交易）

**特點**:
- 捕捉短期價格變化（10-20 秒）
- 適合高頻交易（每天 10-20 次交易）
- 目標利潤：0.3-0.5%

**配置**:
```yaml
labeling:
  method: "triple_barrier"
```

**適用場景**:
- 日內高頻交易
- 需要快速進出
- 追求小幅但穩定的利潤

---

#### 趨勢標籤（推薦：波段交易）

**特點**:
- 捕捉中期趨勢（10-15 分鐘）
- 適合日內波段交易（每天 1-2 次交易）
- 目標利潤：≥ 1%

**配置**:
```yaml
labeling:
  method: "trend"
```

**適用場景**:
- 日內波段交易
- 追求較大利潤
- 交易次數少但勝率高

---

### 執行階段 2

```bash
# 1. 確認預處理數據存在
dir data\preprocessed_v5_1hz\daily\

# 2. 執行數據生成（約 5-10 分鐘）
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5_1hz \
    --output-dir data/processed_v6 \
    --config configs/config_pro_v5_ml_optimal.yaml

# 3. 檢查輸出
dir data\processed_v6\npz\
```

### 預期輸出

```
data/processed_v6/
└── npz/
    ├── stock_embedding_train.npz     # 訓練集
    ├── stock_embedding_val.npz       # 驗證集
    ├── stock_embedding_test.npz      # 測試集
    └── normalization_meta.json       # 標準化參數
```

### 輸出 NPZ 格式

每個 NPZ 文件包含：

```python
import numpy as np

data = np.load('data/processed_v6/npz/stock_embedding_train.npz')

# 必備欄位
X = data['X']              # shape: (N, 100, 20) - 特徵序列
y = data['y']              # shape: (N,) - 標籤（0=Down, 1=Neutral, 2=Up）

# 可選欄位
sample_weights = data.get('sample_weights')  # shape: (N,) - 樣本權重
stock_ids = data.get('stock_ids')           # shape: (N,) - 股票ID
```

### 檢查數據質量

```bash
# 使用 Python 快速檢查
python -c "
import numpy as np
data = np.load('data/processed_v6/npz/stock_embedding_train.npz')
print(f'樣本數: {len(data[\"X\"])}')
print(f'特徵形狀: {data[\"X\"].shape}')
print(f'標籤分布: {np.bincount(data[\"y\"])}')
"
```

**預期輸出**:
```
樣本數: 5584553
特徵形狀: (5584553, 100, 20)
標籤分布: [1675366 2233821 1675366]  # Down/Neutral/Up
```

---

## 階段 3: 訓練 DeepLOB 模型

### 目標

使用階段 2 生成的訓練數據訓練 DeepLOB 模型。

### 使用腳本

**腳本**: `scripts/train_deeplob_v5.py`

**基本用法**:

```bash
python scripts/train_deeplob_v5.py \
    --config configs/train_v5.yaml
```

### 訓練配置

**文件**: `configs/train_v5.yaml`

**關鍵設置**:

```yaml
# 數據路徑
data:
  train_npz: "data/processed_v6/npz/stock_embedding_train.npz"
  val_npz: "data/processed_v6/npz/stock_embedding_val.npz"
  test_npz: "data/processed_v6/npz/stock_embedding_test.npz"
  norm_meta: "data/processed_v6/npz/normalization_meta.json"

# 模型架構
model:
  name: "DeepLOB"          # 或 "DeepLOBImproved"
  num_classes: 3
  dropout_rate: 0.7
  use_stock_embedding: false  # 是否使用股票嵌入

# 訓練參數
training:
  batch_size: 128
  epochs: 50
  learning_rate: 0.001
  optimizer: "Adam"

  # 權重策略（從預處理數據中選擇）
  weight_strategy: "effective_num_0999"  # 推薦

  # Early Stopping
  early_stopping:
    patience: 10
    min_delta: 0.001

# 輸出路徑
output:
  checkpoint_dir: "checkpoints/deeplob_v5"
  log_dir: "logs/deeplob_v5"
  tensorboard_dir: "runs/deeplob_v5"
```

### 權重策略選擇

根據 [WEIGHT_STRATEGIES_GUIDE.md](../label_viewer/WEIGHT_STRATEGIES_GUIDE.md)：

**推薦策略**:

| 策略 | 適用場景 | 推薦度 |
|-----|---------|--------|
| `effective_num_0999` | 標準場景（最常用）| ⭐⭐⭐⭐⭐ |
| `sqrt_balanced` | 訓練穩定優先 | ⭐⭐⭐⭐ |
| `balanced` | 不平衡嚴重 | ⭐⭐⭐⭐ |
| `cb_focal_0999` | 難分樣本多 | ⭐⭐⭐⭐⭐ |
| `uniform` | 類別平衡（基準）| ⭐⭐ |

**如何選擇**:
1. 使用 Label Viewer 查看標籤分布
2. 如果 Neutral 比例 > 50% → 使用 `effective_num_0999`
3. 如果訓練不穩定 → 使用 `sqrt_balanced`
4. 如果準確率低 → 使用 `cb_focal_0999`

### 執行訓練

```bash
# 1. 確認訓練數據存在
dir data\processed_v6\npz\

# 2. 啟動訓練
python scripts/train_deeplob_v5.py --config configs/train_v5.yaml

# 3. 監控訓練（另開終端）
tensorboard --logdir runs/deeplob_v5
# 瀏覽器打開 http://localhost:6006
```

### 訓練輸出

```
checkpoints/deeplob_v5/
├── best_model.pth              # 最佳模型（驗證準確率最高）
├── last_model.pth              # 最後一個 epoch
└── checkpoint_epoch_XX.pth     # 定期保存

logs/deeplob_v5/
├── training.log                # 訓練日誌
├── metrics.json                # 指標記錄
└── confusion_matrix_test.png   # 測試集混淆矩陣

runs/deeplob_v5/
└── events.out.tfevents.*       # TensorBoard 日誌
```

### 訓練指標

**訓練過程中會顯示**:

```
Epoch 10/50:
  Train Loss: 0.6542, Train Acc: 71.23%
  Val Loss: 0.6891, Val Acc: 70.15%
  [Down] Precision: 68.5%, Recall: 65.2%, F1: 66.8%
  [Neutral] Precision: 72.1%, Recall: 78.9%, F1: 75.3%
  [Up] Precision: 69.8%, Recall: 66.7%, F1: 68.2%
```

**測試結果**:

```
Test Results:
  Overall Accuracy: 72.98%
  Macro F1 Score: 73.24%

Per-Class Metrics:
  [Down]    Precision: 70.2%, Recall: 68.5%, F1: 69.3%
  [Neutral] Precision: 73.8%, Recall: 80.2%, F1: 76.9%
  [Up]      Precision: 75.6%, Recall: 70.1%, F1: 72.7%
```

---

## 完整範例

### 範例 1: 從零開始（完整流程）

```bash
# ===============================================
# 階段 1: 預處理原始數據
# ===============================================

# 批次預處理所有歷史數據（首次，約 30 分鐘）
cd D:\Case-New\python\DeepLOB-Pro
conda activate deeplob-pro
scripts\batch_preprocess.bat

# 檢查輸出
dir data\preprocessed_v5_1hz\daily\

# 使用 Label Viewer 檢查標籤分布
cd label_viewer
python app_preprocessed.py
# 瀏覽器: http://localhost:8051
# 載入: data/preprocessed_v5_1hz/daily/20250901

# ===============================================
# 階段 2: 生成訓練數據
# ===============================================

cd D:\Case-New\python\DeepLOB-Pro

# 生成訓練數據（約 5-10 分鐘）
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5_1hz \
    --output-dir data/processed_v6 \
    --config configs/config_pro_v5_ml_optimal.yaml

# 檢查輸出
dir data\processed_v6\npz\

# 快速檢查數據質量
python -c "import numpy as np; data = np.load('data/processed_v6/npz/stock_embedding_train.npz'); print(f'樣本數: {len(data[\"X\"])}, 標籤分布: {np.bincount(data[\"y\"])}')"

# ===============================================
# 階段 3: 訓練 DeepLOB 模型
# ===============================================

# 訓練模型（約 2-4 小時，取決於 GPU）
python scripts/train_deeplob_v5.py --config configs/train_v5.yaml

# 另開終端監控訓練
tensorboard --logdir runs/deeplob_v5
# 瀏覽器: http://localhost:6006

# 訓練完成後查看結果
type logs\deeplob_v5\metrics.json
```

---

### 範例 2: 測試不同標籤方法

```bash
# ===============================================
# 測試 Triple-Barrier（高頻交易）
# ===============================================

# 修改配置: configs/config_pro_v5_ml_optimal.yaml
# labeling.method: "triple_barrier"

python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5_1hz \
    --output-dir data/processed_v6_tb \
    --config configs/config_pro_v5_ml_optimal.yaml

python scripts/train_deeplob_v5.py --config configs/train_v5.yaml

# ===============================================
# 測試趨勢標籤（波段交易）
# ===============================================

# 修改配置: configs/config_pro_v5_ml_optimal.yaml
# labeling.method: "trend"

python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5_1hz \
    --output-dir data/processed_v6_trend \
    --config configs/config_pro_v5_ml_optimal.yaml

python scripts/train_deeplob_v5.py --config configs/train_v5_trend.yaml

# 對比結果
echo "Triple-Barrier 結果:"
type logs\deeplob_v5_tb\metrics.json

echo "趨勢標籤結果:"
type logs\deeplob_v5_trend\metrics.json
```

---

### 範例 3: 調整權重策略

```bash
# ===============================================
# 使用 Label Viewer 選擇權重策略
# ===============================================

cd label_viewer
python app_preprocessed.py
# 勾選「權重策略對比」
# 查看 effective_num_0999 的權重值

# ===============================================
# 修改訓練配置
# ===============================================

# 編輯 configs/train_v5.yaml
training:
  weight_strategy: "effective_num_0999"  # 或其他策略

# 重新訓練
cd D:\Case-New\python\DeepLOB-Pro
python scripts/train_deeplob_v5.py --config configs/train_v5.yaml
```

---

## 常見問題

### 問題 1: 找不到預處理數據

**錯誤訊息**:
```
FileNotFoundError: data/preprocessed_v5_1hz not found
```

**解決方案**:
```bash
# 檢查目錄是否存在
dir data\preprocessed_v5_1hz\daily\

# 如果不存在，執行階段 1
scripts\batch_preprocess.bat
```

---

### 問題 2: 標籤分布不平衡

**症狀**: Neutral 比例過高（> 60%）

**解決方案**:

**方法 1: 調整過濾閾值**
```yaml
# configs/config_pro_v5_ml_optimal.yaml
filter:
  target_neutral_pct: 0.35  # 降低 Neutral 目標比例
```

**方法 2: 使用權重策略**
```yaml
# configs/train_v5.yaml
training:
  weight_strategy: "effective_num_0999"  # 使用權重平衡
```

---

### 問題 3: 訓練數據太少

**症狀**: 樣本數 < 100 萬

**原因**: 過濾太嚴格，很多股票被排除

**解決方案**:

```yaml
# configs/config_pro_v5_ml_optimal.yaml
filter:
  # 放寬過濾條件
  min_range_pct: 0.003  # 降低最小波動率
```

---

### 問題 4: 訓練過程中記憶體不足

**錯誤訊息**:
```
RuntimeError: CUDA out of memory
```

**解決方案**:

```yaml
# configs/train_v5.yaml
training:
  batch_size: 64  # 減小 batch size（原本 128）
```

或使用梯度累積：

```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 2  # 等效於 batch_size=128
```

---

### 問題 5: 模型準確率停滯不前

**症狀**: 準確率卡在 60-65%

**可能原因**:
1. 標籤質量差
2. 權重策略不當
3. 學習率過大

**解決方案**:

**檢查標籤質量**:
```bash
# 使用 Label Viewer 檢查
cd label_viewer
python app_preprocessed.py
# 查看標籤預覽、事件數量、時間桶遮罩
```

**調整權重策略**:
```yaml
# 從 uniform 改為 effective_num_0999
training:
  weight_strategy: "effective_num_0999"
```

**降低學習率**:
```yaml
training:
  learning_rate: 0.0005  # 從 0.001 降低
```

---

### 問題 6: 如何知道訓練是否成功？

**驗收標準**:

| 指標 | 目標值 | 說明 |
|-----|--------|------|
| 測試準確率 | > 65% | 最低要求 |
| 測試準確率 | > 70% | 良好 ✅ |
| 測試準確率 | > 75% | 優秀 ⭐ |
| Macro F1 | > 70% | 確保類別平衡 |
| Neutral 召回率 | < 95% | 避免過度預測 Neutral |
| Down/Up 精確率 | > 65% | 確保信號可靠 |

**查看結果**:
```bash
type logs\deeplob_v5\metrics.json
```

---

## 📚 相關文檔

### 階段 1 文檔
- [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md) - NPZ 格式規格
- [LABEL_PREVIEW_GUIDE.md](LABEL_PREVIEW_GUIDE.md) - 標籤預覽指南
- [APP_PREPROCESSED_GUIDE.md](../label_viewer/APP_PREPROCESSED_GUIDE.md) - Label Viewer 指南

### 階段 2 文檔
- [TREND_LABELING_IMPLEMENTATION.md](TREND_LABELING_IMPLEMENTATION.md) - 趨勢標籤說明
- scripts/extract_tw_stock_data_v6.py - 腳本源碼

### 階段 3 文檔
- [WEIGHT_STRATEGIES_GUIDE.md](../label_viewer/WEIGHT_STRATEGIES_GUIDE.md) - 權重策略指南
- scripts/train_deeplob_v5.py - 訓練腳本源碼
- [1.DeepLOB 台股模型訓練最終報告.md](1.DeepLOB 台股模型訓練最終報告.md) - 訓練報告

---

## 🎯 最佳實踐

### 1. 數據質量優先

```
先檢查 → 再訓練
使用 Label Viewer 檢查：
  - 標籤分布（是否平衡）
  - 事件數量（是否充足）
  - 時間桶遮罩（數據完整性）
  - 權重策略（選擇合適策略）
```

### 2. 小規模測試

```
測試流程：
  1. 選擇 1-2 天數據
  2. 快速生成訓練數據（5 分鐘）
  3. 訓練 10 個 epoch（30 分鐘）
  4. 檢查指標
  5. 如果 OK，擴展到全部數據
```

### 3. 記錄實驗

```
每次實驗記錄：
  - 標籤方法（Triple-Barrier / Trend）
  - 權重策略（effective_num_0999 / ...）
  - 訓練參數（lr, batch_size, epochs）
  - 最終指標（準確率, F1, 各類別指標）
```

### 4. 版本管理

```
data/
├── preprocessed_v5_1hz/          # 階段 1 輸出（固定）
├── processed_v6_tb/              # 階段 2: Triple-Barrier
├── processed_v6_trend/           # 階段 2: 趨勢標籤
└── processed_v6_tb_aggressive/   # 階段 2: 激進參數
```

---

## 🎓 總結

### 三階段流程總覽

| 階段 | 輸入 | 輸出 | 時間 | 頻率 |
|-----|------|------|------|------|
| 1. 預處理 | 原始 TXT | NPZ（每檔股票）| 30 分 | 一次 |
| 2. 生成訓練數據 | 預處理 NPZ | 訓練 NPZ | 5-10 分 | 多次（調參）|
| 3. 訓練模型 | 訓練 NPZ | 模型檢查點 | 2-4 小時 | 多次（調參）|

### 快速開始命令

```bash
# 完整流程（一次性執行）
scripts\batch_preprocess.bat && \
python scripts/extract_tw_stock_data_v6.py --preprocessed-dir data/preprocessed_v5_1hz --output-dir data/processed_v6 --config configs/config_pro_v5_ml_optimal.yaml && \
python scripts/train_deeplob_v5.py --config configs/train_v5.yaml
```

### 下一步

完成訓練後，可以：
1. 評估模型性能（`scripts/evaluate_model.py`）
2. 回測交易策略（使用 Stable-Baselines3）
3. 部署到生產環境

---

**最後更新**: 2025-10-23
**文檔版本**: v1.0
**狀態**: ✅ 完整
