# 智能標籤分布分析與數據集選取工具 v2.0 使用指南

## 📋 目錄
- [功能概述](#功能概述)
- [核心改進](#核心改進)
- [使用方式](#使用方式)
- [演算法說明](#演算法說明)
- [輸出格式](#輸出格式)
- [實際範例](#實際範例)
- [常見問題](#常見問題)

---

## 功能概述

`analyze_label_distribution.py v2.0` 是一個智能化的數據集選取工具，幫助你從大量預處理數據中，**自動組合出最適合 DeepLOB 模型學習的日期+個股組合**。

### 五大核心功能

1. **自動從起始日期開始掃描**
   - 逐日遞增掃描所有預處理 NPZ 數據
   - 支援日期過濾（只載入指定日期之後的數據）

2. **智能組合最佳數據集**
   - 基於標籤分布（Down/Neutral/Up），自動計算最優組合
   - 確保達到目標標籤分布（預設 30%/40%/30%）

3. **自動計算所需數量**
   - 逐日累積，當達到目標分布時停止
   - 確保樣本數足夠（可設定最小樣本數）

4. **互動式選擇界面**
   - 顯示 3-5 個候選方案（保守/平衡/積極）
   - 讓使用者根據實際需求選擇

5. **詳細選取報告**
   - 日期列表（範圍、天數、明細）
   - 個股列表（數量、代碼）
   - 標籤分布（數量、比例、與目標偏差）

---

## 核心改進

### 相比 v1.0 的提升

| 功能 | v1.0 | v2.0 |
|------|------|------|
| **自動日期匹配** | ❌ 手動篩選 | ✅ 從起始日期逐日累積 |
| **智能組合演算法** | ❌ 無 | ✅ L2距離/KL散度最小化 |
| **多方案生成** | ❌ 單一推薦 | ✅ 保守/平衡/積極多方案 |
| **互動選擇** | ❌ 無 | ✅ 使用者可選擇偏好方案 |
| **偏差度量化** | ❌ 僅顯示百分比 | ✅ 精確計算偏差度（0.0001-0.05） |
| **輸出格式** | ⚠️ 僅文字 | ✅ JSON + 詳細報告 |

---

## 使用方式

### 三種執行模式

#### 模式 1: 基礎分析 (`analyze`)
**用途**: 快速查看所有數據的標籤分布概況

```bash
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze
```

**輸出範例**:
```
================================================================================
標籤分布分析報告
================================================================================

📊 整體標籤分布 (1,234 檔股票):
   總樣本數: 5,678,901
   Down:      1,703,670 ( 30.0%)
   Neutral:   2,271,560 ( 40.0%)
   Up:        1,703,671 ( 30.0%)

📅 按日期分組 (15 天):
   20250901: 82 檔, 378,234 樣本, Down 28.5% | Neutral 42.3% | Up 29.2%
   20250902: 85 檔, 392,145 樣本, Down 31.2% | Neutral 38.9% | Up 29.9%
   ...

📈 按持平比例分組:
   LOW (123 檔):
      Down: 35.2% | Neutral: 24.5% | Up: 40.3%
   MEDIUM (567 檔):
      Down: 30.1% | Neutral: 39.8% | Up: 30.1%
   ...
```

---

#### 模式 2: 智能推薦 (`smart_recommend`)
**用途**: 自動生成候選方案，並選擇最佳（偏差最小）

```bash
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 100000 \
    --output dataset_selection.json
```

**參數說明**:
- `--start-date`: 起始日期（YYYYMMDD），從此日期開始逐日累積
- `--target-dist`: 目標標籤分布（down,neutral,up），加總須為 1.0
- `--min-samples`: 最小樣本數，過濾掉樣本太少的方案
- `--output`: 輸出 JSON 檔案路徑

**輸出範例**:
```
開始智能推薦：
  目標分布: Down 30.0% | Neutral 40.0% | Up 30.0%
  最小樣本數: 100,000
  日期範圍: 20250901 - 20250915 (15 天)

  ✅ 找到候選方案（偏差 0.0089 <= 0.01）: 5 天, 145 檔, 234,567 樣本
  ✅ 找到候選方案（偏差 0.0156 <= 0.02）: 8 天, 198 檔, 456,789 樣本
  ✅ 找到候選方案（偏差 0.0278 <= 0.03）: 12 天, 267 檔, 789,012 樣本

共生成 3 個候選方案

====================================================================================================
📋 候選數據集方案
====================================================================================================

🎯 目標分布: Down 30.0% | Neutral 40.0% | Up 30.0%

共找到 3 個候選方案（按偏差度排序）：

【方案 1】保守方案（最高精度，偏差 < 1%）
  📅 日期範圍: 20250901-20250905 (5 天)
  🏢 個股數量: 145 檔
  📊 總樣本數: 234,567
  📈 標籤分布: Down 30.45% (71,453) | Neutral 39.23% (92,012) | Up 30.32% (71,102)
  📐 偏差度: 0.0089

【方案 2】平衡方案（高精度，偏差 < 2%）
  📅 日期範圍: 20250901-20250908 (8 天)
  🏢 個股數量: 198 檔
  📊 總樣本數: 456,789
  📈 標籤分布: Down 29.87% (136,463) | Neutral 41.23% (188,303) | Up 28.90% (132,023)
  📐 偏差度: 0.0156

【方案 3】積極方案（中等精度，偏差 < 3%）
  📅 日期範圍: 20250901-20250912 (12 天)
  🏢 個股數量: 267 檔
  📊 總樣本數: 789,012
  📈 標籤分布: Down 28.56% (225,330) | Neutral 42.67% (336,715) | Up 28.77% (226,967)
  📐 偏差度: 0.0278

自動選擇最佳方案（偏差 0.0089）

====================================================================================================
✅ 已選取數據集
====================================================================================================

📋 方案描述: 保守方案（最高精度，偏差 < 1%）
📐 偏差度: 0.0089

【日期列表】
  範圍: 20250901 - 20250905
  天數: 5
  明細: 20250901, 20250902, 20250903, 20250904, 20250905

【個股列表】
  數量: 145 檔
  前20檔: 1101, 1102, 1216, 1301, 1303, 1326, 2002, 2301, 2303, 2308, 2317, 2330, 2357, 2382, 2454, 2603, 2609, 2801, 2880, 2881 ...
  (完整列表請查看輸出 JSON)

【總樣本數】
  234,567 個樣本

【標籤分布】
  Down:           71,453 (30.45%)  [目標: 30.00%, 偏差: +0.45%]
  Neutral:        92,012 (39.23%)  [目標: 40.00%, 偏差: -0.77%]
  Up:             71,102 (30.32%)  [目標: 30.00%, 偏差: +0.32%]

====================================================================================================

✅ 選取結果已保存到: dataset_selection.json
```

---

#### 模式 3: 互動模式 (`interactive`)
**用途**: 顯示候選方案，讓使用者手動選擇偏好方案

```bash
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode interactive \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 100000
```

**操作流程**:
1. 工具顯示 3-5 個候選方案
2. 使用者輸入方案編號 (1-5) 或 'q' 退出
3. 顯示詳細報告並保存結果

**互動範例**:
```
請選擇方案 (1-3)，或輸入 'q' 退出: 2

====================================================================================================
✅ 已選取數據集
====================================================================================================

📋 方案描述: 平衡方案（高精度，偏差 < 2%）
📐 偏差度: 0.0156

【日期列表】
  範圍: 20250901 - 20250908
  天數: 8
  明細: 20250901, 20250902, 20250903, 20250904, 20250905, 20250906, 20250907, 20250908

【個股列表】
  數量: 198 檔
  前20檔: 1101, 1102, 1216, 1301, 1303, 1326, 2002, 2301, 2303, 2308, 2317, 2330, 2357, 2382, 2412, 2454, 2603, 2609, 2615, 2801 ...

【總樣本數】
  456,789 個樣本

【標籤分布】
  Down:          136,463 (29.87%)  [目標: 30.00%, 偏差: -0.13%]
  Neutral:       188,303 (41.23%)  [目標: 40.00%, 偏差: +1.23%]
  Up:            132,023 (28.90%)  [目標: 30.00%, 偏差: -1.10%]

====================================================================================================

✅ 選取結果已保存到: dataset_selection_20250901-20250908.json
```

---

## 演算法說明

### 智能推薦演算法（逐日累積 + 偏差最小化）

#### 步驟 1: 日期排序
```python
# 按日期升序排列（從起始日期開始）
sorted_dates = ['20250901', '20250902', '20250903', ...]
```

#### 步驟 2: 逐日累積
```python
cumulative_stocks = []
for date in sorted_dates:
    # 加入當日所有股票
    cumulative_stocks.extend(stocks_on_date[date])

    # 計算累積標籤分布
    dist = calculate_distribution(cumulative_stocks)
    # dist = {'down_pct': 0.305, 'neutral_pct': 0.392, 'up_pct': 0.303}
```

#### 步驟 3: 偏差評估
使用 **L2 距離（歐式距離）** 計算與目標分布的偏差：

```python
def calculate_deviation(current, target):
    # current = (0.305, 0.392, 0.303)
    # target  = (0.30,  0.40,  0.30)

    # L2 距離 = sqrt((0.305-0.30)^2 + (0.392-0.40)^2 + (0.303-0.30)^2)
    #         = sqrt(0.000025 + 0.000064 + 0.000009)
    #         = sqrt(0.000098)
    #         = 0.0099 ✅ 非常接近目標！
    return sqrt(sum((c - t)^2 for c, t in zip(current, target)))
```

**偏差度解讀**:
- `< 0.01` (1%): 非常精確，保守方案
- `< 0.02` (2%): 高精度，平衡方案
- `< 0.03` (3%): 中等精度，積極方案
- `< 0.05` (5%): 較大樣本，寬鬆方案

#### 步驟 4: 多方案生成
當累積分布的偏差度達到不同閾值時，記錄為候選方案：

```python
max_deviation_levels = [0.01, 0.02, 0.03, 0.05]

for date in sorted_dates:
    cumulative_stocks.extend(stocks[date])
    dist = calculate_distribution(cumulative_stocks)
    deviation = calculate_deviation(dist, target)

    # 檢查是否符合任一閾值
    if deviation <= 0.01:
        candidates.append({'desc': '保守方案', 'deviation': 0.0089, ...})
    elif deviation <= 0.02:
        candidates.append({'desc': '平衡方案', 'deviation': 0.0156, ...})
    ...
```

#### 步驟 5: 使用者選擇
- **智能推薦模式**: 自動選擇偏差最小的方案
- **互動模式**: 顯示所有候選方案，讓使用者選擇

---

## 輸出格式

### JSON 輸出結構

選取結果會保存為 JSON 格式，包含以下欄位：

```json
{
  "description": "平衡方案（高精度，偏差 < 2%）",
  "date_range": "20250901-20250905",
  "dates": ["20250901", "20250902", "20250903", "20250904", "20250905"],
  "num_dates": 5,
  "symbols": ["1101", "1102", "1216", "1301", "2002", "2301", "2330", "2454"],
  "num_stocks": 8,
  "total_records": 40,
  "file_list": [
    {
      "date": "20250901",
      "symbol": "1101",
      "file_path": "data/preprocessed_v5/daily/20250901/1101.npz",
      "n_points": 4567,
      "total_labels": 4467,
      "down_count": 1340,
      "neutral_count": 1787,
      "up_count": 1340
    },
    {
      "date": "20250901",
      "symbol": "1102",
      "file_path": "data/preprocessed_v5/daily/20250901/1102.npz",
      "n_points": 3891,
      "total_labels": 3791,
      "down_count": 1137,
      "neutral_count": 1517,
      "up_count": 1137
    }
  ],
  "distribution": {
    "total_stocks": 8,
    "total_samples": 234567,
    "down_count": 71453,
    "neutral_count": 92012,
    "up_count": 71102,
    "down_pct": 0.3045,
    "neutral_pct": 0.3923,
    "up_pct": 0.3032
  },
  "deviation": 0.0089,
  "level": 0.01
}
```

**核心改進** ⭐: 新增 `file_list` 欄位，包含完整的「日期+股票」配對列表，每個配對對應一個 NPZ 檔案，可直接用於訓練數據載入。

### 欄位說明

完整的欄位說明請參閱：[DATASET_SELECTION_JSON_FORMAT.md](DATASET_SELECTION_JSON_FORMAT.md)

**核心欄位摘要**:

| 欄位 | 說明 | 範例 |
|------|------|------|
| `description` | 方案描述 | "平衡方案（高精度，偏差 < 2%）" |
| `date_range` | 日期範圍 | "20250901-20250905" |
| `dates` | 日期列表（不重複） | ["20250901", "20250902", ...] |
| `num_dates` | 天數 | 5 |
| `symbols` | 個股代碼列表（不重複） | ["1101", "1102", ...] |
| `num_stocks` | 個股數量 | 8 |
| `total_records` | 總配對數（日期×股票） | 40 |
| `file_list` | **「日期+股票」配對列表** ⭐ | 見 JSON 範例 |
| `distribution.total_samples` | 總樣本數 | 234567 |
| `distribution.down_pct` | Down 標籤比例 | 0.3045 (30.45%) |
| `deviation` | 偏差度（L2距離） | 0.0089 |
| `level` | 偏差閾值 | 0.01 |

---

## 實際範例

### 範例 1: 快速查看數據概況

```bash
# 分析所有數據
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze
```

**用途**: 了解整體數據量、標籤分布、極端案例

---

### 範例 2: 自動選取最佳數據集（推薦）

```bash
# 智能推薦（目標 30/40/30，最少 10 萬樣本）
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 100000 \
    --output results/dataset_selection_optimal.json
```

**用途**: 自動化選取，適合批次處理或 CI/CD 流程

---

### 範例 3: 手動選擇偏好方案

```bash
# 互動模式（顯示候選方案讓我選擇）
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode interactive \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 50000
```

**用途**: 彈性選擇，可根據樣本數、天數、偏差度自由決定

---

### 範例 4: 不同目標分布

```bash
# 目標分布：25/50/25（更多 Neutral）
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --start-date 20250901 \
    --target-dist "0.25,0.50,0.25" \
    --min-samples 100000 \
    --output results/dataset_selection_neutral_heavy.json
```

**用途**: 調整目標分布以符合特定交易策略（如減少誤交易）

---

### 範例 5: 調整最小樣本數

```bash
# 最少 50 萬樣本（適合大規模訓練）
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 500000 \
    --output results/dataset_selection_large.json
```

**用途**: 確保充足樣本數，提升模型泛化能力

---

## 常見問題

### Q1: 無法生成任何候選方案？

**錯誤訊息**:
```
無法生成任何候選方案！請降低 min_samples 或放寬偏差閾值
```

**原因**:
- 數據量不足（預處理數據太少）
- `min_samples` 設定過高
- 起始日期太晚（沒有足夠天數）

**解決方案**:
1. 降低 `--min-samples`（如從 100000 降到 50000）
2. 提前 `--start-date`（如從 20250910 改為 20250901）
3. 檢查預處理數據是否完整（運行 `--mode analyze` 查看）

---

### Q2: 偏差度太大，無法達到目標分布？

**現象**: 所有候選方案的偏差度都 > 0.05

**原因**:
- 目標分布與實際數據分布差異過大
- 數據本身標籤不平衡

**解決方案**:
1. 調整 `--target-dist` 更接近實際分布
2. 先運行 `--mode analyze` 查看整體分布
3. 考慮調整預處理參數（如 `vol_multiplier`、`min_trend_duration`）

---

### Q3: 如何使用選取結果進行訓練？

**步驟**:
1. 運行智能推薦或互動模式，生成 JSON 檔案
2. 編寫訓練腳本，載入 JSON 中的 `file_list`（日期+股票配對）
3. 直接從 `file_path` 載入對應的 NPZ 檔案

**範例代碼**:
```python
import json
import numpy as np
from pathlib import Path

# 載入選取結果
with open('dataset_selection.json', 'r', encoding='utf-8') as f:
    selection = json.load(f)

# 使用 file_list（已包含完整的日期+股票配對）
file_list = selection['file_list']

data_list = []
for item in file_list:
    file_path = item['file_path']

    if Path(file_path).exists():
        data = np.load(file_path, allow_pickle=True)
        data_list.append(data)
        print(f"✅ {item['date']}-{item['symbol']}: {item['total_labels']} 樣本")
    else:
        print(f"⚠️  檔案不存在: {file_path}")

print(f"\n✅ 成功載入 {len(data_list)}/{len(file_list)} 個檔案")
```

**詳細範例**: 請參閱 [DATASET_SELECTION_JSON_FORMAT.md](DATASET_SELECTION_JSON_FORMAT.md) 中的完整使用範例（包含按日期/個股分組載入、並行載入等）

---

### Q4: 偏差度的意義是什麼？

**偏差度** = 當前分布與目標分布的 L2 距離（歐式距離）

**數學定義**:
```
deviation = sqrt((down_curr - down_target)^2
                + (neutral_curr - neutral_target)^2
                + (up_curr - up_target)^2)
```

**實際意義**:
- `< 0.01`: 每個類別平均偏差 < 0.6%（非常精確）
- `< 0.02`: 每個類別平均偏差 < 1.2%（高精度）
- `< 0.03`: 每個類別平均偏差 < 1.8%（中等精度）
- `< 0.05`: 每個類別平均偏差 < 3.0%（可接受）

---

### Q5: 互動模式中如何選擇最佳方案？

**考量因素**:

1. **偏差度**（最重要）
   - 越小越好（< 0.02 為佳）

2. **樣本數**
   - 至少 10 萬樣本（DeepLOB 訓練建議）
   - 50 萬+ 樣本更佳（充足數據）

3. **天數**
   - 太少（< 3 天）: 可能過擬合特定日期
   - 太多（> 20 天）: 可能包含不同市場狀態

4. **個股數量**
   - 至少 100 檔（多樣性）
   - 300+ 檔更佳（泛化能力）

**推薦策略**:
- **保守**: 偏差 < 0.01，樣本數夠用即可
- **平衡**: 偏差 < 0.02，樣本數適中（最推薦）
- **積極**: 偏差 < 0.03，樣本數較多

---

## 總結

### v2.0 的核心價值

✅ **自動化**: 從起始日期逐日累積，無需手動篩選
✅ **智能化**: L2 距離最小化，精確匹配目標分布
✅ **多方案**: 保守/平衡/積極三種選擇
✅ **互動式**: 使用者可根據實際需求自由選擇
✅ **可追溯**: JSON 輸出，完整記錄選取邏輯

### 建議工作流程

```
1. 預處理數據 (batch_preprocess.bat)
   ↓
2. 基礎分析 (--mode analyze)
   → 了解整體數據分布
   ↓
3. 智能推薦 (--mode smart_recommend)
   → 自動生成最佳方案
   ↓
4. 互動選擇 (--mode interactive)
   → 根據需求手動調整
   ↓
5. 訓練模型 (使用選取的 dates + symbols)
   → 確保標籤分布平衡
```

---

**版本**: v2.0
**更新**: 2025-10-23
**作者**: DeepLOB-Pro Team
**相關文檔**:
- [V7 Quick Start](V7_QUICK_START.md)
- [Trend Labeling Implementation](TREND_LABELING_IMPLEMENTATION.md)
- [Preprocessed Data Specification](PREPROCESSED_DATA_SPECIFICATION.md)
