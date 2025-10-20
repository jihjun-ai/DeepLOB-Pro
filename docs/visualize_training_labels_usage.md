# 訓練數據標籤視覺化工具使用說明

## 工具概述

`visualize_training_labels.py` 是一個專門用於檢查 `extract_tw_stock_data_v5.py` 產生的訓練數據標籤正確性的視覺化工具。

### 主要功能

1. **收盤價與標籤趨勢對照**：繪製收盤價曲線，並用顏色標註標籤（紅=下跌, 灰=持平, 綠=上漲）
2. **標籤分布統計**：檢查標籤類別平衡（0:下跌, 1:持平, 2:上漲）
3. **樣本權重分析**：檢查 Triple-Barrier 樣本權重分配
4. **整體統計報告**：跨股票的標籤分布與樣本數統計

---

## 使用方式

### 基本使用（檢查訓練集）

```bash
conda activate deeplob-pro
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5/npz \
    --split train \
    --n-stocks 5
```

### 檢查驗證集

```bash
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5/npz \
    --split val \
    --n-stocks 3
```

### 檢查測試集

```bash
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5/npz \
    --split test \
    --n-stocks 3
```

### 檢查其他配置版本

```bash
# 檢查 balanced 版本
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_balanced/npz \
    --split train \
    --n-stocks 5

# 檢查特定過濾配置
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5-44.61/npz \
    --split train \
    --n-stocks 5
```

---

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--data-dir` | NPZ 數據目錄路徑 | `./data/processed_v5/npz` |
| `--split` | 數據集劃分 (`train`, `val`, `test`) | `train` |
| `--n-stocks` | 顯示前 N 檔股票的詳細圖表 | `3` |
| `--output-dir` | 圖表輸出目錄 | `./results/label_visualization` |
| `--max-points` | 單檔股票最多顯示的時間點數 | `500` |

---

## 輸出結果

### 1. 整體統計圖 (`overall_statistics.png`)

包含以下子圖：

- **標籤分布（整體）**：顯示整個數據集的標籤類別分布
- **股票樣本數分布**：每檔股票的樣本數分布直方圖
- **樣本權重分布**：Triple-Barrier 樣本權重分配
- **前 20 檔股票的標籤分布（堆疊圖）**：檢查不同股票的標籤分布是否一致

### 2. 個股詳細圖 (`stock_<股票代碼>.png`)

每檔股票包含以下子圖：

- **收盤價與標籤趨勢（主圖）**：
  - 藍色線：收盤價曲線
  - 背景顏色：標籤類別（紅=下跌, 灰=持平, 綠=上漲）
  - **檢查重點**：標籤顏色與價格趨勢是否一致

- **標籤序列（時間軸）**：
  - 顏色條帶：每個時間點的標籤
  - **檢查重點**：標籤是否過於連續或跳動

- **標籤分布統計**：
  - 柱狀圖：該股票的標籤類別計數與百分比
  - **檢查重點**：是否有極端不平衡

- **樣本權重分布**：
  - 直方圖：該股票的樣本權重分配
  - **檢查重點**：權重是否合理（均值應接近 1.0）

---

## 如何判斷標籤正確性

### 1. 標籤與趨勢一致性（主圖）

**✅ 正確範例**：
- 價格上升階段：背景主要為綠色（上漲標籤）
- 價格下降階段：背景主要為紅色（下跌標籤）
- 價格橫盤階段：背景主要為灰色（持平標籤）

**❌ 錯誤範例**：
- 價格明顯上升，但標籤為紅色（下跌）
- 價格明顯下降，但標籤為綠色（上漲）
- 標籤顏色與價格趨勢完全相反

### 2. 標籤連續性（時間軸圖）

**✅ 正常模式**：
- 標籤有適度連續性（同一趨勢持續數個時間點）
- 偶爾有標籤切換（趨勢反轉）

**❌ 異常模式**：
- 標籤過於跳動（每個時間點都不同）
- 標籤過於連續（整段時間只有一種標籤）

### 3. 標籤分布平衡

**✅ 良好平衡**（參考 V5 設計目標）：
- 下跌：20-35%
- 持平：30-50%
- 上漲：20-35%

**⚠️ 需要調整**：
- 某類別 < 10% 或 > 60%
- 持平類別過少（< 20%）或過多（> 70%）

### 4. 樣本權重合理性

**✅ 正常權重**：
- 均值接近 1.0（±0.2）
- 大部分權重在 0.1 ~ 5.0 之間
- 少數高權重樣本（用於強調重要趨勢）

**❌ 異常權重**：
- 均值遠離 1.0（> 2.0 或 < 0.5）
- 大量極端權重（> 100）

---

## 常見問題與排查

### Q1: 標籤與趨勢完全相反？

**可能原因**：
- Triple-Barrier 參數設置錯誤（止盈/止損倍數過大或過小）
- `min_return` 閾值設置不當

**解決方案**：
檢查 `config_pro_v5.yaml` 中的 `triple_barrier` 參數：

```yaml
triple_barrier:
  pt_multiplier: 2.0  # 止盈倍數（建議 1.5-3.0）
  sl_multiplier: 2.0  # 止損倍數（建議 1.5-3.0）
  max_holding: 200    # 最大持有期（建議 50-200）
  min_return: 0.0001  # 最小報酬閾值（建議 0.0001-0.002）
```

### Q2: 標籤過於集中在「持平」？

**可能原因**：
- `min_return` 閾值設置過高，導致大部分樣本被歸類為持平
- 波動率估計過小

**解決方案**：
1. 降低 `min_return` 閾值（例如從 0.002 降至 0.0005）
2. 調整波動率估計方法（嘗試 `garch` 代替 `ewma`）

### Q3: 某些股票樣本數過少？

**可能原因**：
- 該股票交易不活躍（成交量小）
- 數據品質問題（過多異常值被過濾）

**解決方案**：
檢查 `volatility_stats.csv` 確認該股票的震盪幅度統計

### Q4: 權重分布極端不均？

**可能原因**：
- `return_scaling` 或 `tau` 參數設置不當
- 某些股票有極端收益事件

**解決方案**：
調整 `config_pro_v5.yaml` 中的 `sample_weights` 參數：

```yaml
sample_weights:
  enabled: true
  tau: 100.0           # 時間衰減參數（建議 50-200）
  return_scaling: 10.0 # 收益縮放係數（建議 5-20）
  balance_classes: true
```

---

## 輸出範例

### 控制台輸出

```
2025-10-20 14:30:15 - INFO - ✅ 已載入 train 數據: 1,249,419 個樣本
2025-10-20 14:30:15 - INFO -    形狀: X=(1249419, 100, 20), y=(1249419,), weights=(1249419,)
2025-10-20 14:30:18 - INFO - 重建收盤價序列...
2025-10-20 14:30:20 - INFO -
生成整體統計圖...
2025-10-20 14:30:25 - INFO - ✅ 已保存整體統計圖: results/label_visualization/train/overall_statistics.png
2025-10-20 14:30:25 - INFO -
生成前 5 檔股票的詳細圖表...
2025-10-20 14:30:25 - INFO -   [1/5] 處理 2330（8,543 個樣本）...
2025-10-20 14:30:27 - INFO -   ✅ 已保存: results/label_visualization/train/stock_2330.png
...
2025-10-20 14:30:45 - INFO -
============================================================
2025-10-20 14:30:45 - INFO - 📊 標籤檢查摘要報告
2025-10-20 14:30:45 - INFO - ============================================================
2025-10-20 14:30:45 - INFO - 數據集: train
2025-10-20 14:30:45 - INFO - 總樣本數: 1,249,419
2025-10-20 14:30:45 - INFO - 股票數量: 256
2025-10-20 14:30:45 - INFO -
標籤分布:
2025-10-20 14:30:45 - INFO -   下跌: 367,457 (29.41%)
2025-10-20 14:30:45 - INFO -   持平: 562,361 (45.01%)
2025-10-20 14:30:45 - INFO -   上漲: 319,601 (25.58%)
2025-10-20 14:30:45 - INFO -
樣本權重統計:
2025-10-20 14:30:45 - INFO -   均值: 0.993
2025-10-20 14:30:45 - INFO -   標準差: 2.644
2025-10-20 14:30:45 - INFO -   最大值: 238.655
2025-10-20 14:30:45 - INFO -   最小值: 0.001
2025-10-20 14:30:45 - INFO -
圖表已保存至: results/label_visualization/train
2025-10-20 14:30:45 - INFO - ============================================================
```

---

## 批次檢查多個配置

建議使用批次腳本一次檢查所有配置版本：

### Windows 批次腳本 (`check_all_labels.bat`)

```batch
@echo off
echo 檢查所有訓練數據標籤...

echo.
echo [1/3] 檢查 processed_v5...
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5/npz --split train --n-stocks 5

echo.
echo [2/3] 檢查 processed_v5_balanced...
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5_balanced/npz --split train --n-stocks 5

echo.
echo [3/3] 檢查 processed_v5-44.61...
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5-44.61/npz --split train --n-stocks 5

echo.
echo 完成！圖表已保存至 results/label_visualization/
pause
```

### Linux/Mac Shell 腳本 (`check_all_labels.sh`)

```bash
#!/bin/bash
echo "檢查所有訓練數據標籤..."

echo ""
echo "[1/3] 檢查 processed_v5..."
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5/npz --split train --n-stocks 5

echo ""
echo "[2/3] 檢查 processed_v5_balanced..."
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5_balanced/npz --split train --n-stocks 5

echo ""
echo "[3/3] 檢查 processed_v5-44.61..."
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5-44.61/npz --split train --n-stocks 5

echo ""
echo "完成！圖表已保存至 results/label_visualization/"
```

---

## 整合到訓練流程

建議在每次重新生成訓練數據後，立即執行視覺化檢查：

```bash
# 1. 生成訓練數據
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_new \
    --config configs/config_pro_v5_new.yaml

# 2. 視覺化檢查（立即檢查標籤正確性）
python scripts/visualize_training_labels.py \
    --data-dir ./data/processed_v5_new/npz \
    --split train \
    --n-stocks 10

# 3. 檢查圖表後，確認無誤再開始訓練
python scripts/train_deeplob_v5.py \
    --config configs/train_v5_new.yaml
```

---

## 進階使用：自定義圖表

如需自定義圖表樣式或新增統計項目，可修改 `visualize_training_labels.py` 中的以下函數：

- `plot_stock_labels()`: 個股詳細圖表
- `plot_overall_statistics()`: 整體統計圖表
- `reconstruct_close_price()`: 價格重建邏輯

---

## 依賴套件

工具已使用標準數據科學套件，應該已包含在 `deeplob-pro` 環境中：

- `numpy`: 數據處理
- `pandas`: 統計分析
- `matplotlib`: 圖表繪製
- `seaborn`: 進階視覺化（可選）

如需安裝缺少的套件：

```bash
conda activate deeplob-pro
pip install matplotlib seaborn
```

---

**版本**：v1.0
**更新日期**：2025-10-20
**作者**：DeepLOB-Pro Team
