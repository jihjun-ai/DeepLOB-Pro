# 標籤預覽功能使用指南

## 功能概述

在 `preprocess_single_day.py` 預處理階段，系統會自動計算每個個股的 **Triple-Barrier 標籤分布**（預覽模式），並保存到：

1. **NPZ metadata**：每個股票的 `label_preview` 欄位
2. **summary.json**：當天所有股票的 `actual_label_stats` 統計

### 核心價值：知情選擇，而非硬性過濾 ⭐

標籤預覽的目的**不是硬性過濾**股票，而是提供**全局視野**，讓你可以：

- 📊 **看到全貌**：了解每個股票的標籤分布特性
- 🎯 **靈活組合**：根據訓練目標主動挑選不同特性的股票
- 🚫 **避免盲目**：不會因為少數極端股票（如持平只有 3%）破壞整體訓練集平衡
- 🔧 **針對性補償**：缺什麼類別就選什麼特性的股票

**關鍵理念**：
> 每個個股已經有標籤預覽，在選用資料時可以**依需求選取**，而不會讓資料過於集中或過於平均。

---

## 完整工作流程

### 步驟 1：預處理（生成標籤預覽）

```bash
# 批次預處理所有歷史數據
conda activate deeplob-pro
scripts\batch_preprocess.bat
```

預處理階段會自動計算每個股票的標籤分布，無需額外配置。

**輸出**：
- `data/preprocessed_v5/daily/{date}/{symbol}.npz` - 包含 `label_preview`
- `data/preprocessed_v5/daily/{date}/summary.json` - 包含 `actual_label_stats`

---

### 步驟 2：分析標籤分布（分析模式）

使用 `analyze_label_distribution.py` 工具查看整體標籤分布：

```bash
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze
```

#### 輸出範例

```
================================================================================
標籤分布分析報告
================================================================================

📊 整體標籤分布 (211 檔股票):
   總標籤數: 3,012,456
   Down:      942,334 ( 31.28%)
   Neutral:   451,233 ( 14.98%)  ← 持平類嚴重不足！
   Up:      1,618,889 ( 53.74%)  ← 上漲類過多！

📈 按持平比例分組:

   VERY_LOW (32 檔):  ← 持平 < 10%（強趨勢股）
      Down: 40.2% | Neutral: 5.8% | Up: 54.0%

   LOW (68 檔):  ← 持平 10-20%
      Down: 35.1% | Neutral: 15.2% | Up: 49.7%

   MEDIUM (85 檔):  ← 持平 20-30%（相對平衡）
      Down: 28.5% | Neutral: 25.0% | Up: 46.5%

   HIGH (26 檔):  ← 持平 30-40%（橫盤較多）
      Down: 25.0% | Neutral: 35.2% | Up: 39.8%

⚠️  極端案例:
   持平 < 5%: 8 檔（強趨勢股）
      2330: Down 42.1% | Neutral 3.2% | Up 54.7%
      2317: Down 45.8% | Neutral 4.1% | Up 50.1%

   持平 > 50%: 3 檔（橫盤股）
      1234: Down 20.1% | Neutral 58.3% | Up 21.6%
```

#### 關鍵洞察

從這個分析可以看出：

1. **整體不平衡嚴重**：持平類只有 15%，上漲類高達 54%
2. **股票特性多樣**：有強趨勢股（持平 < 10%）、橫盤股（持平 > 30%）
3. **需要選擇性組合**：如果直接混合所有股票，訓練集會極度不平衡

---

### 步驟 3：生成股票選取建議（推薦模式）

根據你的訓練目標，生成股票選取策略：

```bash
# 目標：訓練平衡模型（Down 30% | Neutral 40% | Up 30%）
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode recommend \
    --target-dist "0.30,0.40,0.30" \
    --output stock_selection.json
```

#### 輸出範例

```
================================================================================
股票選取建議
================================================================================

目標分布: Down 30.0% | Neutral 40.0% | Up 30.0%
當前分布: Down 31.3% | Neutral 15.0% | Up 53.7%

缺口分析:
   Down gap:     -1.3%  ← 略多，可接受
   Neutral gap: +25.0%  ← 嚴重不足！需要補償
   Up gap:      -23.7%  ← 嚴重過多！需要抑制

推薦策略:

   策略 1: boost_neutral
      理由: 持平類不足 (25.0%)，選取持平比例 > 25% 的股票
      股票數: 111 檔
      股票代碼: 2454, 2881, 3008, 5876, 6505, 2882, 2886, 1301, ...

   策略 2: 組合建議
      - 選取 70% 來自 HIGH 組（持平 30-40%）
      - 選取 30% 來自 MEDIUM 組（持平 20-30%）
      預期達到: Down 27% | Neutral 32% | Up 41% ← 更接近目標！
```

**`stock_selection.json` 內容**：

```json
{
  "target_dist": {
    "down": 0.30,
    "neutral": 0.40,
    "up": 0.30
  },
  "current_dist": {
    "down": 0.3128,
    "neutral": 0.1498,
    "up": 0.5374
  },
  "gap_analysis": {
    "down_gap": -0.0128,
    "neutral_gap": 0.2502,
    "up_gap": -0.2374
  },
  "recommendations": [
    {
      "strategy": "boost_neutral",
      "reason": "持平類不足 (25.0%)，選取持平比例 > 25% 的股票",
      "stocks": ["2454", "2881", "3008", ...],
      "count": 111
    }
  ]
}
```

---

### 步驟 4：根據建議配置數據提取

根據 `stock_selection.json` 的建議，修改 `config_pro_v5_ml_optimal.yaml`：

#### 方案 A：在 config 中設定過濾條件

```yaml
# 數據配置
data:
  # ... 其他配置 ...

  # 🆕 股票選取策略（基於標籤預覽）
  stock_selection:
    enabled: true  # 是否啟用選取邏輯

    # 基於分析結果的選取規則
    rules:
      # 規則 1：選取持平比例較高的股票（補償持平類）
      - name: "boost_neutral"
        enabled: true
        min_neutral_pct: 0.25  # 持平 ≥ 25%
        max_neutral_pct: 0.45  # 持平 ≤ 45%
        min_total_labels: 1000  # 至少 1000 標籤
        weight: 0.7  # 佔總樣本的 70%

      # 規則 2：選取平衡股票（避免極端）
      - name: "balanced_stocks"
        enabled: true
        min_neutral_pct: 0.15
        max_neutral_pct: 0.25
        min_down_pct: 0.25
        min_up_pct: 0.25
        weight: 0.3  # 佔總樣本的 30%
```

#### 方案 B：直接指定股票清單（更精確）

```yaml
data:
  # ... 其他配置 ...

  # 直接指定要使用的股票（從 stock_selection.json 複製）
  selected_stocks:
    enabled: true
    stock_list: ["2454", "2881", "3008", "5876", "6505", ...]  # 從推薦結果複製
```

---

### 步驟 5：提取訓練數據

使用修改後的配置提取數據：

```bash
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5 \
    --output-dir data/processed_v6_balanced \
    --config configs/config_pro_v5_ml_optimal.yaml
```

**預期結果**：

```
載入數據完成:
  訓練集: 2,345,678 樣本
  驗證集: 456,789 樣本
  測試集: 789,123 樣本

標籤分布:
  訓練集: Down 28.5% | Neutral 35.2% | Up 36.3%  ← 更平衡！
  驗證集: Down 29.1% | Neutral 34.8% | Up 36.1%
  測試集: Down 28.8% | Neutral 35.5% | Up 35.7%
```

---

## 實際應用場景

### 場景 1：訓練「平衡」模型

**目標**：三類標籤儘量均衡（30/40/30）

**步驟**：

```bash
# 1. 分析當前分布
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze

# 2. 生成建議
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode recommend \
    --target-dist "0.30,0.40,0.30" \
    --output stock_selection_balanced.json

# 3. 根據建議修改 config，然後提取
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5 \
    --output-dir data/processed_v6_balanced \
    --config configs/config_balanced.yaml
```

**結果**：
- 訓練集標籤分布：Down 28% | Neutral 37% | Up 35%
- ✅ 避免被「持平類只有 5%」的極端股票破壞平衡

---

### 場景 2：訓練「趨勢捕捉」模型

**目標**：專注於明確的上漲/下跌信號，減少持平噪音

**步驟**：

```bash
# 生成建議（目標：持平類較少）
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode recommend \
    --target-dist "0.40,0.20,0.40" \
    --output stock_selection_trending.json
```

**config 設定**：

```yaml
data:
  stock_selection:
    enabled: true
    rules:
      - name: "trending_stocks"
        enabled: true
        max_neutral_pct: 0.25  # 持平 ≤ 25%（強趨勢股）
        min_total_labels: 1500
```

**結果**：
- 訓練集標籤分布：Down 42% | Neutral 18% | Up 40%
- ✅ 模型學會識別明確的方向性信號

---

### 場景 3：訓練「橫盤識別」模型

**目標**：提高對持平狀態的識別能力

**步驟**：

```bash
# 生成建議（目標：持平類較多）
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode recommend \
    --target-dist "0.25,0.50,0.25" \
    --output stock_selection_sideways.json
```

**config 設定**：

```yaml
data:
  stock_selection:
    enabled: true
    rules:
      - name: "sideways_stocks"
        enabled: true
        min_neutral_pct: 0.35  # 持平 ≥ 35%（橫盤股）
        min_total_labels: 1000
```

**結果**：
- 訓練集標籤分布：Down 22% | Neutral 48% | Up 30%
- ✅ 模型學會識別橫盤狀態

---

### 場景 4：組合策略（最靈活）⭐⭐⭐

**目標**：根據實際觀察動態調整股票組合

**步驟**：

```bash
# 1. 先分析
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze
```

**觀察結果**：
```
整體分布:
  Down: 47% | Neutral: 13% | Up: 40%  ← 不平衡！

分組統計:
  MEDIUM 組 (85 檔): Down 28% | Neutral 25% | Up 47%
  HIGH 組 (26 檔): Down 25% | Neutral 35% | Up 40%
```

**組合策略**：

```yaml
data:
  stock_selection:
    enabled: true
    rules:
      # 70% 來自 HIGH 組（補償持平類）
      - name: "high_neutral"
        enabled: true
        min_neutral_pct: 0.30
        max_neutral_pct: 0.45
        weight: 0.7

      # 30% 來自 MEDIUM 組（平衡）
      - name: "medium_neutral"
        enabled: true
        min_neutral_pct: 0.20
        max_neutral_pct: 0.30
        weight: 0.3
```

**預期結果**：
- 最終訓練集：Down 27% | Neutral 32% | Up 41% ✅

---

## 測試驗證

### 快速驗證標籤預覽

```bash
python scripts/test_label_preview.py
```

**預期輸出**：

```
======================================================================
測試 NPZ 標籤預覽功能
======================================================================

檔案: 2330.npz
✅ 標籤預覽:
   總標籤數: 14,278
   Down:      4,512 ( 31.62%)
   Neutral:   2,145 ( 15.02%)
   Up:        7,621 ( 53.37%)
   ✓ 總和驗證通過

======================================================================
示範：基於標籤分布的股票篩選
======================================================================

過濾條件:
   持平比例: 10% - 50%
   最少標籤數: 1,000

結果:
   通過: 198 檔
   未通過: 13 檔

✅ 通過的股票（前 10 檔）:
   2330: Down 31.6% | Neutral 15.0% | Up 53.4%
   2317: Down 28.9% | Neutral 18.2% | Up 52.9%
```

---

## 進階用法

### 自定義分析腳本

根據標籤分布排序並選取股票：

```python
import glob
import json
import numpy as np

# 讀取所有 NPZ
npz_files = glob.glob("data/preprocessed_v5/daily/20250901/*.npz")
stocks_info = []

for npz_file in npz_files:
    data = np.load(npz_file, allow_pickle=True)
    metadata = json.loads(str(data['metadata']))

    if 'label_preview' in metadata and metadata['label_preview'] is not None:
        lp = metadata['label_preview']
        stocks_info.append({
            'symbol': metadata['symbol'],
            'neutral_pct': lp['neutral_pct'],
            'down_pct': lp['down_pct'],
            'up_pct': lp['up_pct'],
            'total_labels': lp['total_labels']
        })

# 按持平比例排序
stocks_sorted = sorted(stocks_info, key=lambda x: x['neutral_pct'])

# 選取持平比例 20-30% 的股票（適合訓練平衡模型）
balanced_stocks = [s for s in stocks_sorted if 0.20 <= s['neutral_pct'] <= 0.30]

print(f"找到 {len(balanced_stocks)} 檔平衡股票")
for stock in balanced_stocks[:10]:
    print(f"  {stock['symbol']}: Down {stock['down_pct']:.1%} | "
          f"Neutral {stock['neutral_pct']:.1%} | Up {stock['up_pct']:.1%}")

# 保存選取結果
selected_symbols = [s['symbol'] for s in balanced_stocks]
with open('selected_stocks.json', 'w') as f:
    json.dump(selected_symbols, f)
```

---

## 常見問題

### Q1: 標籤預覽和實際 extract_v6 的標籤會一樣嗎？

**A**: 幾乎一樣，但可能有微小差異（< 1%）：

- **預覽階段**：使用完整的 `mids` 序列計算（14400 個點）
- **提取階段**：可能因為滑窗切分或數據驗證而略有不同

差異通常可以忽略，放心使用。

---

### Q2: 如果某個股票沒有 `label_preview`？

**A**: 可能原因：

1. 未通過波動率過濾（`pass_filter=false`）
2. Triple-Barrier 計算失敗（數據質量問題）
3. 使用舊版本的 `preprocess_single_day.py`（未更新）

**解決方法**：
- 重新執行 `batch_preprocess.bat`（使用最新版本）
- 無標籤預覽的股票會在 extract_v6 時自動跳過

---

### Q3: 如何調整標籤選取策略？

**A**: 根據 `analyze_label_distribution.py` 的分析結果調整 config：

**範例 1：當前持平類太少（15%），目標 40%**
```yaml
data:
  stock_selection:
    rules:
      - min_neutral_pct: 0.30  # 選持平類較多的股票
        max_neutral_pct: 0.50
        weight: 0.8  # 佔 80%
```

**範例 2：當前持平類適中（35%），維持平衡**
```yaml
data:
  stock_selection:
    rules:
      - min_neutral_pct: 0.25
        max_neutral_pct: 0.45
        weight: 1.0  # 全部使用
```

---

### Q4: 標籤預覽會影響預處理速度嗎？

**A**: 會略微增加，但影響很小：

- 單檔股票：+2-5 秒
- 批次預處理（100+ 檔）：+3-8 分鐘
- **總體影響 < 10%**

**優點**：提前知道標籤分布，避免 extract_v6 時浪費時間處理不合適的股票。

---

### Q5: 為什麼不直接硬性過濾？

**A**: 因為不同訓練目標需要不同的標籤分布：

| 訓練目標 | 理想分布 | 選取策略 |
|---------|---------|---------|
| 平衡模型 | Down 30% \| Neutral 40% \| Up 30% | 選持平 25-45% 股票 |
| 趨勢捕捉 | Down 40% \| Neutral 20% \| Up 40% | 選持平 < 25% 股票 |
| 橫盤識別 | Down 25% \| Neutral 50% \| Up 25% | 選持平 > 35% 股票 |

**硬性過濾**無法滿足這些不同需求，**靈活選擇**才是王道。

---

## 工具一覽表

| 工具 | 用途 | 命令 |
|------|------|------|
| `preprocess_single_day.py` | 生成標籤預覽 | `scripts\batch_preprocess.bat` |
| `test_label_preview.py` | 驗證標籤預覽 | `python scripts/test_label_preview.py` |
| `analyze_label_distribution.py` | 分析並推薦選取策略 | `python scripts/analyze_label_distribution.py --mode analyze` |
| `extract_tw_stock_data_v6.py` | 根據策略提取數據 | `python scripts/extract_tw_stock_data_v6.py ...` |

---

## 總結

✅ **核心優勢**:

1. **全局視野**：看到每個股票的標籤分布特性
2. **靈活組合**：根據訓練目標主動挑選股票
3. **避免極端**：不會被少數極端股票主導訓練集
4. **針對性補償**：缺什麼類別就選什麼特性的股票

⚠️ **關鍵理念**:

> 標籤預覽是「知情選擇」而非「硬性過濾」。每個股票都有價值，關鍵是如何組合它們來達到你的訓練目標。

🎯 **推薦工作流程**:

1. 執行 `batch_preprocess.bat` 生成標籤預覽
2. 使用 `analyze_label_distribution.py` 分析當前分布
3. 根據分析結果制定選取策略
4. 修改 config 並執行 `extract_tw_stock_data_v6.py`
5. 檢查最終訓練集的標籤分布是否符合預期

---

**最後更新**: 2025-10-23
**版本**: v2.0（新增 analyze_label_distribution.py 工具）
