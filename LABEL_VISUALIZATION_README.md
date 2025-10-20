# 訓練數據標籤視覺化工具 - 快速上手

## 🎯 工具目的

檢查 `extract_tw_stock_data_v5.py` 產生的訓練數據標籤（Triple-Barrier 標籤）是否正確，確保：
- 標籤與價格趨勢一致（上漲=綠色，下跌=紅色，持平=灰色）
- 標籤分布平衡（避免過度集中在某一類別）
- 樣本權重合理（均值接近 1.0）

---

## 🚀 快速開始（3 步驟）

### 1. 使用批次腳本（最簡單）

```batch
# Windows
check_labels.bat

# 或指定參數
check_labels.bat data/processed_v5/npz train 5
```

### 2. 直接使用 Python（推薦）

```bash
conda activate deeplob-pro

# 檢查訓練集（預設）
python scripts/visualize_training_labels.py

# 檢查驗證集
python scripts/visualize_training_labels.py --split val --n-stocks 5

# 檢查不同配置版本
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5_balanced/npz
```

### 3. 查看結果

圖表保存在：`results/label_visualization/[split]/`

- `overall_statistics.png` - 整體統計圖（必看）
- `stock_*.png` - 個股詳細圖（重點檢查）

---

## 📊 如何判斷標籤正確性

### ✅ 正確標籤的特徵

打開 `stock_*.png` 檢查**主圖（收盤價與標籤趨勢）**：

1. **價格上升階段** → 背景主要是綠色（上漲標籤）
2. **價格下降階段** → 背景主要是紅色（下跌標籤）
3. **價格橫盤階段** → 背景主要是灰色（持平標籤）

### ❌ 錯誤標籤的症狀

1. **標籤與趨勢相反**
   - 價格明顯上升，但標籤為紅色
   - 價格明顯下降，但標籤為綠色
   - **原因**：Triple-Barrier 參數設置錯誤

2. **標籤過於集中在「持平」**
   - 持平類別 > 70%
   - **原因**：`min_return` 閾值設置過高

3. **標籤過於跳動**
   - 每個時間點標籤都不同（像彩虹）
   - **原因**：波動率估計過大或 `max_holding` 過短

4. **權重分布極端**
   - 權重均值遠離 1.0
   - 大量極端權重（> 100）
   - **原因**：`return_scaling` 或 `tau` 參數不當

---

## 🔧 常見問題排查

### Q1: 標籤與趨勢完全相反？

**解決方案**：調整 `configs/config_pro_v5.yaml`

```yaml
triple_barrier:
  pt_multiplier: 2.0   # ← 降低此值（從 5.9 → 2.0）
  sl_multiplier: 2.0   # ← 降低此值
  max_holding: 200     # ← 增加持有期（從 50 → 200）
  min_return: 0.0001   # ← 降低閾值（從 0.00215 → 0.0001）
```

然後重新生成訓練數據：

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_fixed \
    --config configs/config_pro_v5.yaml
```

### Q2: 標籤過於集中在「持平」（> 50%）？

**解決方案**：降低 `min_return` 閾值

```yaml
triple_barrier:
  min_return: 0.0005  # ← 從 0.00215 降至 0.0005
```

### Q3: 如何檢查多個配置版本？

使用循環腳本：

```bash
# 檢查所有版本
for DIR in data/processed_v5*/npz; do
    echo "Checking $DIR..."
    python scripts/visualize_training_labels.py --data-dir $DIR --n-stocks 5
done
```

---

## 📁 輸出文件說明

### 1. overall_statistics.png（整體統計圖）

包含 4 個子圖：

- **左上**: 標籤分布（整體）
  - 理想分布：下跌 20-35%, 持平 30-50%, 上漲 20-35%

- **中上**: 股票樣本數分布
  - 檢查是否有股票樣本數過少（< 100）

- **右上**: 樣本權重分布
  - 理想權重均值：0.8 ~ 1.2

- **底部**: 前 20 檔股票的標籤分布（堆疊圖）
  - 檢查不同股票的標籤分布是否一致

### 2. stock_*.png（個股詳細圖）

包含 4 個子圖：

- **頂部大圖**: 收盤價與標籤趨勢（**最重要！**）
  - 藍色線：收盤價曲線
  - 背景顏色：標籤（紅=下跌, 灰=持平, 綠=上漲）
  - **檢查重點**：顏色與趨勢是否一致

- **中間**: 標籤序列（時間軸）
  - 檢查標籤連續性（不應過於跳動或過於連續）

- **左下**: 標籤分布統計
  - 檢查該股票的類別平衡

- **右下**: 樣本權重分布
  - 檢查權重分配是否合理

---

## 🎨 視覺化範例解讀

### 正確範例（理想狀態）

```
收盤價曲線：  ／￣＼＿／￣
標籤顏色：   綠灰綠紅綠灰綠
            ↑   ↑ ↑ ↑   ↑
            上漲時是綠色，下跌時是紅色，橫盤時是灰色
```

### 錯誤範例（需要修正）

```
收盤價曲線：  ／￣＼＿／￣
標籤顏色：   紅灰紅綠紅灰紅
            ↑      ↑
            上漲卻是紅色（錯！）
            下跌卻是綠色（錯！）
```

---

## 💡 最佳實踐建議

### 1. 每次生成訓練數據後立即檢查

```bash
# 1. 生成數據
python scripts/extract_tw_stock_data_v5.py --config configs/config_pro_v5.yaml

# 2. 立即視覺化檢查（不要跳過！）
python scripts/visualize_training_labels.py --n-stocks 10

# 3. 確認無誤後再開始訓練
python scripts/train_deeplob_v5.py
```

### 2. 重點檢查前 10 檔股票

```bash
python scripts/visualize_training_labels.py --n-stocks 10
```

樣本數多的股票通常更有代表性，優先檢查這些股票。

### 3. 對比不同配置版本

生成多個配置版本的視覺化報告，找出最佳參數：

```bash
# 配置 1: 激進（小閾值）
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_aggressive/npz \
    --output-dir ./results/label_viz_aggressive

# 配置 2: 保守（大閾值）
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_conservative/npz \
    --output-dir ./results/label_viz_conservative

# 配置 3: 平衡（中等閾值）
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_balanced/npz \
    --output-dir ./results/label_viz_balanced
```

對比三者，選擇標籤最合理的配置。

---

## 📚 相關文檔

- 詳細使用說明：[docs/visualize_training_labels_usage.md](docs/visualize_training_labels_usage.md)
- Triple-Barrier 原理：參見 `extract_tw_stock_data_v5.py` 註解
- 配置文件說明：[configs/config_pro_v5.yaml](configs/config_pro_v5.yaml)

---

## 🐛 疑難排解

### 問題 1: 中文顯示亂碼

**症狀**：圖表標題和標籤顯示方框

**解決**：
- Windows：圖表雖有警告但仍可正常生成，可忽略
- Linux/Mac：安裝中文字型或使用英文標籤

### 問題 2: 找不到 NPZ 文件

**錯誤訊息**：`FileNotFoundError: 找不到數據文件`

**解決**：
1. 確認數據目錄路徑正確
2. 確認已運行過 `extract_tw_stock_data_v5.py`
3. 檢查 `--data-dir` 參數是否正確

### 問題 3: 記憶體不足

**症狀**：載入大數據集時程式崩潰

**解決**：
- 使用 `--max-points 300` 減少每檔股票的顯示點數
- 使用 `--n-stocks 3` 減少檢查的股票數量

---

## ⚡ 快速指令參考

```bash
# 基本使用
python scripts/visualize_training_labels.py

# 檢查驗證集
python scripts/visualize_training_labels.py --split val

# 檢查測試集
python scripts/visualize_training_labels.py --split test

# 檢查更多股票
python scripts/visualize_training_labels.py --n-stocks 10

# 檢查不同配置
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5_balanced/npz

# 自定義輸出目錄
python scripts/visualize_training_labels.py --output-dir ./my_results

# 減少顯示點數（提高速度）
python scripts/visualize_training_labels.py --max-points 200
```

---

**工具版本**：v1.0
**更新日期**：2025-10-20
**建議使用頻率**：每次生成訓練數據後必須檢查
**檢查時間**：約 1-3 分鐘（取決於 `--n-stocks` 參數）
