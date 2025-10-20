# V5 資料健康檢查工具使用說明

## 概述

`check_data_health_v5.py` 是一個全面的資料品質檢查工具，用於評估 `extract_tw_stock_data_v5.py` 產生的訓練資料是否健康並適合用於模型訓練。

## 功能特色

### 1. 基礎檢查 (Basic Validation)
- ✅ 檔案存在性檢查（train/val/test.npz + metadata.json）
- ✅ 資料維度正確性（期望: N × 100 × 20）
- ✅ 樣本數量一致性（X, y, weights 長度相同）

### 2. 資料品質檢查 (Data Quality)
- ✅ NaN/Inf 檢測
- ✅ 數值範圍檢查（Z-score 應在合理範圍）
- ✅ 標準化狀態驗證（訓練集 mean≈0, std≈1）
- ✅ 樣本權重合理性（正值、無極端值）

### 3. 標籤分布檢查 (Label Distribution)
- ✅ 標籤有效性（只包含 0, 1, 2）
- ✅ 類別平衡檢查（任何類別 < 20% 發出警告）
- ✅ 各 split 的標籤分布統計

### 4. 訓練適用性檢查 (Training Suitability)
- ✅ 訓練集大小（建議 > 100K 樣本）
- ✅ 驗證集大小（建議 > 10K 樣本）
- ✅ 股票覆蓋度（有效股票數量）
- ✅ 資料切分比例（訓練集應 > 60%）
- ✅ V5 配置摘要（波動率方法、Triple-Barrier 參數）

### 5. 統計分析 (Statistical Analysis)
- ✅ 20 個 LOB 特徵的統計量（均值、標準差、分位數）
- ✅ 特徵分布檢查

### 6. 視覺化（可選）
- 📊 標籤分布圖（train/val/test）
- 📊 權重分布圖（train/val/test）

---

## 使用方式

### 方法一：批次檔（最簡單）

```bash
# Windows 用戶
check_health.bat
```

這會自動檢查 `./data/processed_v5/npz` 並生成報告。

### 方法二：Python 命令（推薦）

```bash
# 基本使用（檢查 processed_v5）
conda activate deeplob-pro
python scripts/check_data_health_v5.py --data-dir ./data/processed_v5/npz

# 檢查其他資料夾（如 processed_v5_balanced）
python scripts/check_data_health_v5.py --data-dir ./data/processed_v5_balanced/npz

# 檢查並保存 JSON 報告
python scripts/check_data_health_v5.py \
    --data-dir ./data/processed_v5/npz \
    --save-report

# 顯示詳細統計資訊
python scripts/check_data_health_v5.py \
    --data-dir ./data/processed_v5/npz \
    --save-report \
    --verbose

# 生成視覺化圖表（需要 matplotlib）
python scripts/check_data_health_v5.py \
    --data-dir ./data/processed_v5/npz \
    --save-report \
    --plot
```

### 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--data-dir` | NPZ 資料目錄 | `./data/processed_v5/npz` |
| `--save-report` | 保存 JSON 報告 | `True` |
| `--plot` | 生成視覺化圖表 | `False` |
| `--verbose` | 顯示詳細統計資訊 | `False` |

---

## 輸出內容

### 1. 控制台輸出（彩色）

```
======================================================================
V5 資料健康檢查工具
======================================================================

資料目錄: ./data/processed_v5/npz

======================================================================
1. 基礎檢查 (Basic Validation)
======================================================================

✅ train 集檔案 - 存在
✅ val 集檔案 - 存在
✅ test 集檔案 - 存在
✅ Metadata 檔案 - 版本: 5.0.0
✅ train 集形狀 - (1,249,419, 100, 20)
✅ train 集樣本數一致 - 1,249,419 樣本
...

======================================================================
總結報告 (Summary)
======================================================================

整體健康狀態: ✅ 健康

建議:
  資料品質良好，適合用於訓練模型。

檢查項目統計:
  總計: 45 項
  ✅ 通過: 42
  ⚠️ 警告: 3
  ❌ 失敗: 0

✅ 報告已保存: ./data/processed_v5/npz/health_report.json
```

### 2. JSON 報告（machine-readable）

保存於 `{data_dir}/health_report.json`，包含：

```json
{
  "timestamp": "2025-10-20T15:30:00",
  "overall_status": "pass",
  "results": {
    "basic": {
      "status": "pass",
      "checks": [...]
    },
    "quality": {
      "status": "pass",
      "checks": [...]
    },
    "labels": {
      "status": "pass",
      "distributions": {
        "train": {
          "class_0": 397396,
          "class_1": 506030,
          "class_2": 345993,
          "pct_0": 31.81,
          "pct_1": 40.50,
          "pct_2": 27.69
        },
        ...
      }
    },
    "suitability": {
      "status": "pass",
      "statistics": {
        "train_size": 1249419,
        "val_size": 184950,
        "test_size": 419324,
        "train_stocks": 136,
        "val_stocks": 29,
        "test_stocks": 29,
        "split_ratios": {
          "train": 0.674,
          "val": 0.100,
          "test": 0.226
        }
      }
    },
    ...
  }
}
```

### 3. 視覺化圖表（可選）

保存於 `{data_dir}/health_visualizations.png`，包含：
- 左上/中/右：Train/Val/Test 標籤分布柱狀圖
- 左下/中/右：Train/Val/Test 權重分布直方圖

---

## 判斷標準

### ✅ 健康 (Pass)
- 所有檢查項目通過
- 資料品質良好，適合直接訓練

**建議行動**：
- 直接開始訓練模型
- 使用現有配置即可

### ⚠️ 部分健康 (Warning)
- 存在一些警告項目，但不影響訓練
- 常見警告原因：
  - 類別不平衡（某類別 < 20%）
  - 訓練集偏小（< 100K 樣本）
  - 驗證集偏小（< 10K 樣本）
  - 數值範圍異常（但在可接受範圍內）

**建議行動**：
- 檢查警告項目
- 考慮以下調整：
  - **類別不平衡**：啟用 `sample_weights.balance_classes: true`
  - **資料量小**：調整 `intraday_volatility_filter` 以包含更多資料
  - **切分比例**：調整 `split.train_ratio/val_ratio/test_ratio`
- 可嘗試訓練，但需密切監控

### ❌ 不健康 (Fail)
- 存在嚴重問題，不建議直接訓練
- 常見失敗原因：
  - NaN/Inf 存在
  - 標籤無效（非 0/1/2）
  - 權重為負值
  - 維度不符

**建議行動**：
- **立即停止**訓練計劃
- 檢查 `extract_tw_stock_data_v5.py` 配置
- 重新生成資料
- 再次執行健康檢查

---

## 實際案例

### 案例一：健康資料

```bash
$ python scripts/check_data_health_v5.py --data-dir ./data/processed_v5/npz

# 輸出摘要
整體健康狀態: ✅ 健康
檢查項目統計: 42/42 通過

# 建議
資料品質良好，適合用於訓練模型。

# 行動
直接開始訓練：
conda activate deeplob-pro
python scripts/train_deeplob_v5.py --config configs/config_pro_v5.yaml
```

### 案例二：類別不平衡警告

```bash
$ python scripts/check_data_health_v5.py --data-dir ./data/processed_v5_filter2.0/npz

# 輸出摘要
整體健康狀態: ⚠️ 部分健康
⚠️ train 集類別平衡 - 最小類別佔比 15.2%（建議 > 20%）

# 標籤分布
Class 0 (下跌): 189,234 (15.2%)  ⚠️ 偏低
Class 1 (持平): 512,849 (41.0%)
Class 2 (上漲): 547,336 (43.8%)

# 建議
啟用樣本權重平衡：
configs/config_pro_v5.yaml:
  sample_weights:
    enabled: true
    balance_classes: true  # 啟用類別平衡
```

### 案例三：資料量不足警告

```bash
$ python scripts/check_data_health_v5.py --data-dir ./data/processed_v5_filter2.5/npz

# 輸出摘要
整體健康狀態: ⚠️ 部分健康
⚠️ 訓練集大小 - 87,234 樣本（建議 > 100K）

# 建議
1. 放寬震盪篩選條件：
   configs/config_pro_v5.yaml:
     intraday_volatility_filter:
       enabled: true
       min_range_pct: 0.015  # 從 0.025 降低到 0.015

2. 或增加輸入資料：
   - 增加交易日數
   - 增加股票數量
```

### 案例四：嚴重錯誤（NaN 存在）

```bash
$ python scripts/check_data_health_v5.py --data-dir ./data/processed_v5_broken/npz

# 輸出摘要
整體健康狀態: ❌ 不健康
❌ train 集 NaN - 12,458 個 (0.8%)

# 建議
資料存在嚴重問題，不建議直接用於訓練。

# 行動
1. 檢查輸入資料品質
2. 重新執行資料處理：
   python scripts/extract_tw_stock_data_v5.py \
       --input-dir ./data/temp \
       --output-dir ./data/processed_v5_fixed \
       --config configs/config_pro_v5.yaml
```

---

## 常見問題

### Q1: 如何判斷資料是否可用於訓練？

**A:** 遵循以下原則：
- ✅ **健康**：直接開始訓練
- ⚠️ **警告**：根據警告類型決定
  - 類別不平衡：啟用 `balance_classes`
  - 資料量小：可訓練但需監控過擬合
  - 其他警告：視情況調整
- ❌ **失敗**：**必須**重新生成資料

### Q2: 類別不平衡怎麼辦？

**A:** 兩種解決方案：
1. **啟用樣本權重平衡**（推薦）：
   ```yaml
   # configs/config_pro_v5.yaml
   sample_weights:
     enabled: true
     balance_classes: true  # ✅ 啟用
   ```

2. **調整 Triple-Barrier 參數**：
   ```yaml
   triple_barrier:
     min_return: 0.0001  # 降低閾值（增加上漲/下跌標籤）
   ```

### Q3: 訓練集太小（< 100K）怎麼辦？

**A:** 三種方法：
1. **放寬震盪篩選**（推薦）：
   ```yaml
   intraday_volatility_filter:
     enabled: true
     min_range_pct: 0.010  # 降低最小震盪要求
     max_range_pct: 1.000  # 提高最大震盪限制
   ```

2. **增加輸入資料**：
   - 下載更多交易日數據
   - 增加股票覆蓋範圍

3. **調整切分比例**：
   ```yaml
   split:
     train_ratio: 0.80  # 從 0.70 提高到 0.80
     val_ratio: 0.10
     test_ratio: 0.10
   ```

### Q4: 如何比較多個資料集的品質？

**A:** 批量檢查：
```bash
# 檢查所有 processed_v5* 資料夾
for dir in data/processed_v5*/npz; do
    echo "檢查 $dir..."
    python scripts/check_data_health_v5.py --data-dir $dir --save-report
done

# 比較報告
cat data/processed_v5/npz/health_report.json | jq '.results.suitability.statistics.train_size'
cat data/processed_v5_balanced/npz/health_report.json | jq '.results.suitability.statistics.train_size'
```

### Q5: 權重極端值（max > 100）是否正常？

**A:** 視情況而定：
- **正常情況**：某些高報酬樣本權重自然較高
- **異常情況**：配置錯誤（如 `return_scaling` 過大）

**檢查步驟**：
1. 查看 `sample_weights.return_scaling`（建議 10-100）
2. 查看 `sample_weights.tau`（建議 50-200）
3. 檢查 metadata 中的 `weight_stats.max`

**調整範例**：
```yaml
sample_weights:
  return_scaling: 10.0  # 降低（從 50.0）
  tau: 100.0           # 增加（從 50.0）
```

---

## 進階使用

### 1. 批量檢查多個資料集

```bash
#!/bin/bash
# check_all_datasets.sh

DATASETS=(
    "processed_v5"
    "processed_v5_balanced"
    "processed_v5_filter1.5"
    "processed_v5_filter2.0"
)

for dataset in "${DATASETS[@]}"; do
    echo "========================================="
    echo "檢查 $dataset"
    echo "========================================="
    python scripts/check_data_health_v5.py \
        --data-dir ./data/$dataset/npz \
        --save-report \
        --verbose
    echo ""
done

echo "所有檢查完成！"
```

### 2. 自動化決策腳本

```python
# auto_decide_training.py
import json
import sys

def should_train(report_path):
    """根據健康報告自動決定是否可訓練"""
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    status = report['overall_status']

    if status == 'fail':
        print("❌ 資料不健康，禁止訓練")
        return False
    elif status == 'warning':
        # 檢查具體警告
        train_size = report['results']['suitability']['statistics'].get('train_size', 0)
        if train_size < 50000:
            print("⚠️ 訓練集過小，不建議訓練")
            return False
        else:
            print("⚠️ 存在警告，但可嘗試訓練")
            return True
    else:
        print("✅ 資料健康，可開始訓練")
        return True

if __name__ == "__main__":
    report_path = sys.argv[1]
    can_train = should_train(report_path)
    sys.exit(0 if can_train else 1)
```

使用：
```bash
python auto_decide_training.py ./data/processed_v5/npz/health_report.json
if [ $? -eq 0 ]; then
    echo "開始訓練..."
    python scripts/train_deeplob_v5.py --config configs/config_pro_v5.yaml
else
    echo "資料不符訓練條件，終止"
fi
```

### 3. 整合到 CI/CD

```yaml
# .github/workflows/data_quality.yml
name: Data Quality Check

on:
  push:
    paths:
      - 'data/processed_v5/**'

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          pip install numpy pandas

      - name: Run health check
        run: |
          python scripts/check_data_health_v5.py \
              --data-dir ./data/processed_v5/npz \
              --save-report

      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: health-report
          path: ./data/processed_v5/npz/health_report.json

      - name: Fail if unhealthy
        run: |
          STATUS=$(cat ./data/processed_v5/npz/health_report.json | jq -r '.overall_status')
          if [ "$STATUS" = "fail" ]; then
            echo "❌ 資料健康檢查失敗"
            exit 1
          fi
```

---

## 參考指標

### 健康資料的典型特徵

| 指標 | 良好範圍 | 警告範圍 | 失敗條件 |
|------|----------|----------|----------|
| 訓練集樣本數 | > 100K | 50K - 100K | < 50K |
| 驗證集樣本數 | > 10K | 5K - 10K | < 5K |
| 類別最小佔比 | > 20% | 15% - 20% | < 15% |
| 標準化均值 | [-0.05, 0.05] | [-0.1, 0.1] | 其他 |
| 標準化標準差 | [0.9, 1.1] | [0.8, 1.2] | 其他 |
| 權重最大值 | < 50 | 50 - 100 | > 100 |
| NaN/Inf 數量 | 0 | 0 | > 0 |

### V5 最佳配置參考

```yaml
# configs/config_pro_v5.yaml（推薦配置）

volatility:
  method: ewma
  halflife: 60

triple_barrier:
  pt_multiplier: 2.0
  sl_multiplier: 2.0
  max_holding: 200
  min_return: 0.0001  # 0.01%

sample_weights:
  enabled: true
  tau: 100.0
  return_scaling: 10.0
  balance_classes: true  # ✅ 推薦啟用

intraday_volatility_filter:
  enabled: true
  min_range_pct: 0.015  # 1.5%
  max_range_pct: 1.000  # 100%

split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

---

## 總結

`check_data_health_v5.py` 提供全面的資料品質檢查，幫助您：

1. ✅ **快速判斷**資料是否適合訓練
2. ⚠️ **發現潛在問題**並提供解決建議
3. 📊 **量化資料品質**（JSON 報告 + 視覺化）
4. 🔧 **指導配置調整**（針對性建議）

**使用流程**：
1. 生成資料 → 2. 健康檢查 → 3. 根據結果調整 → 4. 開始訓練

**記住**：
- ✅ 健康 = 直接訓練
- ⚠️ 警告 = 檢查並調整
- ❌ 失敗 = 重新生成

祝訓練順利！ 🚀
