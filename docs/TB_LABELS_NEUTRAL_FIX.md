# Triple-Barrier 標籤「平」比例過低問題分析與解決方案

## 📊 問題診斷

### 當前狀況
根據你的配置 `config_pro_v5_ml_optimal.yaml`：
```yaml
triple_barrier:
  pt_multiplier: 3.5
  sl_multiplier: 3.5
  max_holding: 40
  min_return: 0.0015  # 0.15%
```

### 為什麼「平」(label=0) 被吃掉？

**原始邏輯（extract_tw_stock_data_v5.py 第 306-316 行）**：
```python
if trigger_why == 'time':
    if np.abs(ret) < min_return:  # ← 只有這裡會產生 label=0
        label = 0
    else:
        label = int(np.sign(ret))
else:  # PT/SL 觸發
    label = int(np.sign(ret))  # ← 一律標為 ±1
```

**問題鏈條**：
1. **PT/SL 觸發占大多數** → 台股日內波動大，3.5σ 經常被觸及
2. **PT/SL 一律標 ±1** → 即使實際收益很小（如 0.05%），也被標為上漲/下跌
3. **time trigger 也常超過閾值** → `min_return=0.0015` 雖然看似小，但在 40 bars 後價格常有 >0.15% 變動
4. **結果** → 只有「時間到 + 收益極小」才是 0 → **比例 <10%**

### 數據驗證（示意）
假設一天 1000 個樣本：
- PT 觸發：350 個 → 全標 +1（即使有些只漲 0.08%）
- SL 觸發：320 個 → 全標 -1（即使有些只跌 0.06%）
- Time 觸發：330 個
  - 其中 280 個 |ret| > 0.15% → 標 ±1
  - **只有 50 個 |ret| < 0.15% → 標 0**
- **最終分布**：下跌 35% / 持平 5% / 上漲 35% / 其他 25%

---

## 🎯 解決方案（3 種策略）

### 方案 A：PT/SL 也檢查 min_return（推薦）
**理念**：雖然觸及 barrier，但如果實際收益太小，視為「雜訊」標為 0

**修改邏輯**：
```python
if trigger_why == 'time':
    if np.abs(ret) < min_return:
        label = 0
    else:
        label = int(np.sign(ret))
else:  # PT/SL 觸發
    # ✅ 新增：也檢查 min_return
    if np.abs(ret) < min_return:
        label = 0  # 雖然觸及 barrier，但收益太小
    else:
        label = int(np.sign(ret))
```

**優點**：
- 符合「日沖交易」邏輯：微小波動不值得交易
- 增加「平」標籤比例（預期 20-40%）
- 保留 Triple-Barrier 的路徑特性

**缺點**：
- 可能與「先觸及」哲學稍有衝突（但實務更合理）

**建議參數**：
```yaml
min_return: 0.002  # 提高到 0.2%（約 2 個 tick）
pt_sl_check_min_return: true
```

---

### 方案 B：放寬 barrier（增加 time trigger）
**理念**：讓 PT/SL 更難觸及，更多樣本由時間到期決定

**修改參數**：
```yaml
triple_barrier:
  pt_multiplier: 5.0  # 3.5 → 5.0（更寬）
  sl_multiplier: 5.0
  max_holding: 40
  min_return: 0.0015
```

**優點**：
- 保持原始邏輯不變
- 增加 time trigger 比例

**缺點**：
- 可能錯過真實的止盈/止損信號
- 「平」的增加有限（因為 time trigger 也常超過 min_return）

---

### 方案 C：提高 min_return 閾值
**理念**：把「不夠清晰」的漲跌都視為平

**修改參數**：
```yaml
triple_barrier:
  pt_multiplier: 3.5
  sl_multiplier: 3.5
  max_holding: 40
  min_return: 0.005  # 提高到 0.5%（5 個 tick）
```

**優點**：
- 邏輯簡單清晰
- 「平」標籤增加明顯

**缺點**：
- 0.5% 可能太大，會錯失真實的小趨勢
- 不符合高頻交易特性

---

## ✅ 推薦組合方案（A + 參數微調）

### 使用改進版函數
已提供 `scripts/tb_labels_improved.py` 中的 `tb_labels_v2()`

### 建議配置
```yaml
triple_barrier:
  pt_multiplier: 4.0      # 稍微放寬（3.5 → 4.0）
  sl_multiplier: 4.0
  max_holding: 50         # 稍微延長（40 → 50）
  min_return: 0.0025      # 提高閾值（0.15% → 0.25%）
  pt_sl_check_min_return: true  # ← 關鍵新增
```

### 預期效果
- **下跌** (label=0)：28-32%
- **持平** (label=1)：35-45% ✅
- **上漲** (label=2)：28-32%

---

## 🔧 實施步驟

### Step 1：測試改進版函數
```bash
python scripts/test_tb_comparison.py --config configs/config_pro_v5_ml_optimal.yaml
```
這會比較 3 種方法的標籤分布。

### Step 2：選擇最佳參數
根據測試結果，調整配置文件中的：
- `min_return`
- `pt_multiplier` / `sl_multiplier`
- 是否啟用 `pt_sl_check_min_return`

### Step 3：應用到生產流程
修改 `extract_tw_stock_data_v5.py` 中的 `tb_labels` 函數（或直接替換為 `tb_labels_v2`）

### Step 4：重新生成數據
```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_balanced \
    --config configs/config_pro_v5_ml_optimal_v2.yaml
```

---

## 📈 驗證清單

生成新數據後，檢查：
1. ✅ 標籤分布：`label_0` 30-45%
2. ✅ 觸發原因：time/up/down 分布合理
3. ✅ 隨機抽查：畫出價格走勢 + barrier，確認標籤合理
4. ✅ 訓練穩定性：模型訓練時 loss 下降正常，不偏向某類

---

## 🎯 哲學選擇

### 對於「日沖交易」
**推薦：方案 A（路徑導向 + PT/SL 檢查 min_return）**

理由：
1. 實際交易會設止盈止損（符合 Triple-Barrier）
2. 但微小波動不值得執行（符合 min_return 檢查）
3. 訓練數據應反映「值得交易的信號」而非「所有波動」
4. 增加「平」標籤 = 告訴模型「這種情況不要出手」→ 降低過度交易

### 如果想純粹預測價格方向
可考慮**終點導向**（方案 B 在我提供的文檔中）：
- 只看固定視窗後的漲跌
- 更簡單，但失去止盈止損資訊

---

## 📝 配置文件範例

已為你準備：`configs/config_pro_v5_balanced_neutral.yaml`
```yaml
triple_barrier:
  pt_multiplier: 4.0
  sl_multiplier: 4.0
  max_holding: 50
  min_return: 0.0025
  pt_sl_check_min_return: true  # 新增參數
```

使用：
```bash
python scripts/extract_tw_stock_data_v5.py \
    --config configs/config_pro_v5_balanced_neutral.yaml \
    --output-dir ./data/processed_v5_neutral_fix
```

