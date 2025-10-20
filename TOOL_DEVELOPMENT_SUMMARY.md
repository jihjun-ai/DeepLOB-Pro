# 訓練數據標籤視覺化工具 - 開發完成報告

## 📋 開發概要

**開發日期**：2025-10-20
**開發目的**：檢查 `extract_tw_stock_data_v5.py` 產生的訓練數據標籤正確性
**工具版本**：v1.0
**狀態**：✅ 開發完成並測試通過

---

## 🎯 工具功能

### 核心功能
1. **收盤價與標籤趨勢對照圖**
   - 繪製收盤價曲線
   - 用顏色標註標籤（紅=下跌, 灰=持平, 綠=上漲）
   - 檢查標籤與價格趨勢是否一致

2. **標籤分布統計**
   - 整體標籤分布（檢查類別平衡）
   - 個股標籤分布（檢查個別股票異常）
   - 前 20 檔股票的堆疊分布圖

3. **樣本權重分析**
   - 權重分布直方圖
   - 權重統計摘要（均值、標準差、極值）
   - 檢查 Triple-Barrier 樣本權重分配

4. **股票樣本數統計**
   - 每檔股票的樣本數分布
   - 檢查數據覆蓋是否均勻

---

## 📁 新增文件列表

### 1. 核心腳本
- **`scripts/visualize_training_labels.py`** (主要工具)
  - 載入 NPZ 數據與 metadata
  - 重建收盤價（反向 Z-Score）
  - 生成視覺化圖表
  - 輸出統計報告

### 2. 說明文檔
- **`docs/visualize_training_labels_usage.md`** (詳細使用說明)
  - 完整參數說明
  - 使用範例
  - 判斷標準
  - 常見問題排查
  - 進階使用技巧

- **`LABEL_VISUALIZATION_README.md`** (快速上手指南)
  - 快速開始（3 步驟）
  - 視覺化範例解讀
  - 最佳實踐建議
  - 快速指令參考

### 3. 批次腳本
- **`check_labels.bat`** (Windows 快速啟動腳本)
  - 一鍵執行視覺化檢查
  - 支援自定義參數

---

## 🔧 技術實現細節

### 數據載入與處理
```python
# 載入 NPZ 數據
data = np.load(npz_path, allow_pickle=True)
X = data['X']        # (N, 100, 20) 時間序列特徵
y = data['y']        # (N,) 標籤 {0, 1, 2}
weights = data['weights']  # (N,) 樣本權重
stock_ids = data['stock_ids']  # (N,) 股票代碼

# 載入 metadata（Z-Score 參數）
metadata = json.load(meta_path)
mu = metadata['normalization']['feature_means']
sd = metadata['normalization']['feature_stds']
```

### 收盤價重建
```python
# 反向 Z-Score（取最後一個時間點）
X_last = X[:, -1, :]  # (N, 20)
X_denorm = X_last * sd + mu

# 計算中間價（近似收盤價）
bid1 = X_denorm[:, 0]   # 第 1 檔買價
ask1 = X_denorm[:, 5]   # 第 1 檔賣價
close = (bid1 + ask1) / 2.0
```

### 視覺化設計
```python
# 標籤顏色映射
LABEL_COLORS = {
    0: '#e74c3c',  # 紅色（下跌）
    1: '#95a5a6',  # 灰色（持平）
    2: '#2ecc71'   # 綠色（上漲）
}

# 主圖：收盤價曲線 + 標籤背景
ax.plot(close, color='#3498db', label='收盤價')
for i in range(len(close)):
    label = int(labels[i])
    ax.axvspan(i-0.5, i+0.5, alpha=0.2, color=LABEL_COLORS[label])
```

---

## 📊 輸出圖表說明

### 整體統計圖 (`overall_statistics.png`)

**布局**：2 行 3 列
```
+-------------------+-------------------+-------------------+
|  標籤分布（整體） |  股票樣本數分布   |  樣本權重分布     |
+-------------------+-------------------+-------------------+
|  前 20 檔股票的標籤分布（堆疊圖）                       |
+--------------------------------------------------------+
```

**檢查重點**：
- 標籤分布是否平衡（理想：下跌 20-35%, 持平 30-50%, 上漲 20-35%）
- 股票樣本數是否有極端異常（某股票樣本數過少）
- 權重均值是否接近 1.0

### 個股詳細圖 (`stock_*.png`)

**布局**：3 行 2 列
```
+--------------------------------------------------------+
|  收盤價與標籤趨勢（主圖，最重要）                       |
+--------------------------------------------------------+
|  標籤序列（時間軸）                                    |
+--------------------------------------------------------+
|  標籤分布統計     |  樣本權重分布                        |
+-------------------+-----------------------------------+
```

**檢查重點**：
- 主圖：標籤顏色與價格趨勢是否一致
- 時間軸：標籤是否過於跳動或過於連續
- 分布統計：個股標籤分布是否合理

---

## ✅ 測試結果

### 測試環境
- 數據集：`data/processed_v5/npz` (訓練集)
- 樣本數：1,249,419 個
- 股票數：256 檔
- 測試股票：前 3 檔（2449, 2498, 2634）

### 測試輸出
```
2025-10-20 07:19:55 - INFO - 載入 train 數據集...
2025-10-20 07:19:59 - INFO - ✅ 已載入 train 數據: 1,249,419 個樣本
2025-10-20 07:19:59 - INFO -    形狀: X=(1249419, 100, 20), y=(1249419,), weights=(1249419,)
2025-10-20 07:19:59 - INFO - 重建收盤價序列...
2025-10-20 07:19:59 - INFO - 生成整體統計圖...
2025-10-20 07:20:00 - INFO - ✅ 已保存整體統計圖: results/label_visualization/train/overall_statistics.png
2025-10-20 07:20:00 - INFO - 生成前 3 檔股票的詳細圖表...
2025-10-20 07:20:00 - INFO -   [1/3] 處理 2449（5,286 個樣本）...
2025-10-20 07:20:00 - INFO -   ✅ 已保存: results/label_visualization/train/stock_2449.png
2025-10-20 07:20:00 - INFO -   [2/3] 處理 2498（5,090 個樣本）...
2025-10-20 07:20:00 - INFO -   ✅ 已保存: results/label_visualization/train/stock_2498.png
2025-10-20 07:20:00 - INFO -   [3/3] 處理 2634（5,046 個樣本）...
2025-10-20 07:20:00 - INFO -   ✅ 已保存: results/label_visualization/train/stock_2634.png
```

### 生成文件
- ✅ `overall_statistics.png` (124 KB)
- ✅ `stock_2449.png` (133 KB)
- ✅ `stock_2498.png` (142 KB)
- ✅ `stock_2634.png` (135 KB)

### 執行時間
- 總時間：約 5 秒
- 載入數據：約 4 秒
- 生成圖表：約 1 秒

---

## 🎯 使用場景

### 場景 1: 每次生成訓練數據後立即檢查
```bash
# 1. 生成訓練數據
python scripts/extract_tw_stock_data_v5.py \
    --config configs/config_pro_v5.yaml

# 2. 立即視覺化檢查（必須！）
python scripts/visualize_training_labels.py --n-stocks 10

# 3. 確認標籤正確後再訓練
python scripts/train_deeplob_v5.py
```

### 場景 2: 調試 Triple-Barrier 參數
```bash
# 嘗試不同參數配置
for config in config_v5_*.yaml; do
    # 生成數據
    python scripts/extract_tw_stock_data_v5.py --config configs/$config

    # 視覺化檢查
    python scripts/visualize_training_labels.py

    # 人工檢查圖表，選擇最佳配置
done
```

### 場景 3: 對比不同過濾策略
```bash
# 對比 3 種震盪過濾配置
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_filter1.5/npz \
    --output-dir ./results/viz_filter1.5

python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_filter2.0/npz \
    --output-dir ./results/viz_filter2.0

python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_filter2.5/npz \
    --output-dir ./results/viz_filter2.5

# 人工對比三者的標籤質量
```

---

## 💡 標籤正確性判斷標準

### ✅ 良好標籤的特徵
1. **趨勢一致性**
   - 價格上升段：主要是綠色（上漲標籤）
   - 價格下降段：主要是紅色（下跌標籤）
   - 價格橫盤段：主要是灰色（持平標籤）

2. **分布平衡**
   - 下跌：20-35%
   - 持平：30-50%
   - 上漲：20-35%

3. **連續性適中**
   - 標籤有適度連續性（同一趨勢持續數個時間點）
   - 偶爾有切換（趨勢反轉）

4. **權重合理**
   - 均值接近 1.0（±0.2）
   - 大部分權重在 0.1 ~ 5.0 之間

### ❌ 需要調整的症狀
1. **標籤與趨勢相反**
   - 原因：Triple-Barrier 參數過大
   - 解決：降低 `pt_multiplier` 和 `sl_multiplier`

2. **標籤過於集中在持平**
   - 原因：`min_return` 閾值過高
   - 解決：降低 `min_return`（從 0.002 → 0.0005）

3. **標籤過於跳動**
   - 原因：`max_holding` 過短或波動率過大
   - 解決：增加 `max_holding`（從 50 → 200）

4. **權重極端不均**
   - 原因：`return_scaling` 或 `tau` 不當
   - 解決：調整 `sample_weights` 參數

---

## 🔄 後續改進方向

### 優先級 1（必要）
- ✅ 已完成：基本視覺化功能
- ✅ 已完成：標籤分布統計
- ✅ 已完成：樣本權重分析

### 優先級 2（有用）
- ⏳ 待實作：標籤轉換點分析（標籤切換時的價格變化）
- ⏳ 待實作：觸發原因統計（up/down/time 的分布）
- ⏳ 待實作：持有期分布直方圖（`tt` 觸發時間統計）

### 優先級 3（進階）
- ⏳ 待實作：交互式圖表（使用 Plotly）
- ⏳ 待實作：自動異常檢測（標籤與趨勢不一致的自動標記）
- ⏳ 待實作：HTML 報告生成（包含所有圖表的完整報告）

---

## 📚 相關文檔

### 使用說明
- [LABEL_VISUALIZATION_README.md](LABEL_VISUALIZATION_README.md) - 快速上手（推薦首先閱讀）
- [docs/visualize_training_labels_usage.md](docs/visualize_training_labels_usage.md) - 詳細使用說明

### 技術文檔
- [scripts/extract_tw_stock_data_v5.py](scripts/extract_tw_stock_data_v5.py) - 數據生成腳本
- [configs/config_pro_v5.yaml](configs/config_pro_v5.yaml) - 配置文件範例

### 參考論文
- Triple-Barrier Method: "Advances in Financial Machine Learning" (M. López de Prado, 2018)
- Sample Weighting: "Meta-Labeling: How to Build Robust Investment Strategies" (López de Prado, 2019)

---

## 🎉 總結

### 開發成果
- ✅ 工具開發完成並測試通過
- ✅ 生成高質量視覺化圖表
- ✅ 提供完整使用說明文檔
- ✅ 支援批次腳本快速啟動

### 使用建議
1. **每次生成訓練數據後必須檢查**（不可跳過）
2. **重點檢查前 10 檔股票的主圖**（收盤價與標籤趨勢）
3. **對比不同配置版本**（找出最佳 Triple-Barrier 參數）
4. **保存檢查報告**（用於後續訓練效果分析）

### 預期效果
- 避免使用錯誤標籤訓練（可節省數小時訓練時間）
- 快速發現 Triple-Barrier 參數問題
- 提高訓練數據質量（進而提高模型準確率）

---

**開發者**：DeepLOB-Pro Team
**工具版本**：v1.0
**開發日期**：2025-10-20
**狀態**：✅ 生產就緒（Production Ready）
