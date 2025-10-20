# 跨日污染修復報告

**版本**: 5.0.3
**日期**: 2025-10-20
**問題編號**: #001

---

## 問題摘要

### 核心問題

原始 V5 版本的數據處理管線存在**跨日污染**（Cross-Day Contamination）問題，導致：

1. **Class 1（持平）幾乎消失**：持平類樣本佔比接近 0%，嚴重偏離預期的 30-45%
2. **標籤分布崩潰**：幾乎只有 Class 0（下跌）和 Class 2（上漲），無法學習「不交易」策略
3. **資料洩漏**：模型偷看到隔夜跳空效應，泛化能力受損

### 根本原因

[extract_tw_stock_data_v5.py](../scripts/extract_tw_stock_data_v5.py) 中的 `sliding_windows_v5` 函數：

1. **時間序列拼接錯誤**（第 816-834 行）：
   - 把同一檔股票的**所有日期**用 `np.concatenate` 串成一條長序列
   - 昨天尾盤 + 今天開盤被視為「連續時間」

2. **波動率跨日累積**（第 924-934 行）：
   - EWMA/GARCH 在整條序列上計算，隔夜跳空被當作「日內波動」
   - 波動率估計嚴重扭曲

3. **Triple-Barrier 越日觸發**（第 279 行）：
   - `end_idx = min(i + max_holding, n)` 沒有日界限制
   - vertical barrier 可以延伸到隔夜，容易撞 PT/SL
   - 原本應該「時間到→持平」的樣本被改判成上/下

4. **min_return 應用範圍過寬**（第 305-309 行）：
   - 對**所有樣本**（包含 PT/SL 觸發）都檢查 `min_return`
   - 標準做法：只有 vertical（時間到）才用 `min_return` 判斷
   - 造成標籤分布進一步扭曲

---

## 修復方案

### 修改 1: `tb_labels` 函數（triple-barrier 標籤生成）

**檔案**: [scripts/extract_tw_stock_data_v5.py:240-350](../scripts/extract_tw_stock_data_v5.py)

#### 新增參數

```python
def tb_labels(..., day_end_idx: Optional[int] = None):
```

- **作用**: 限制 vertical barrier 不得超過 `day_end_idx`（當日最後一個索引）

#### 修正 vertical barrier 計算

**修改前**:
```python
end_idx = min(i + max_holding, n)
```

**修改後**:
```python
if day_end_idx is not None:
    end_idx = min(i + max_holding, day_end_idx + 1, n)
else:
    end_idx = min(i + max_holding, n)
```

#### 修正 min_return 應用邏輯

**修改前** (所有樣本都檢查):
```python
if np.abs(ret) < min_return:
    label = 0
else:
    label = int(np.sign(ret))
```

**修改後** (只在 vertical 時檢查):
```python
if trigger_why == 'time':
    # 時間到期：檢查最小報酬閾值
    if np.abs(ret) < min_return:
        label = 0
    else:
        label = int(np.sign(ret))
else:
    # PT/SL 觸發：固定標籤為 ±1
    label = int(np.sign(ret))
```

---

### 修改 2: `sliding_windows_v5` 函數（滑窗流程重構）

**檔案**: [scripts/extract_tw_stock_data_v5.py:802-1114](../scripts/extract_tw_stock_data_v5.py)

#### 核心改動：按日處理架構

**修改前** (跨日拼接):
```python
# 步驟 1: 以股票為單位串接資料
for date, sym, Xd, mids in days_points:
    stock_data[sym]['dates'].append(date)
    stock_data[sym]['X'].append(Xd)
    stock_data[sym]['mids'].append(mids)

# 串接成長序列
X_concat = np.concatenate([...])  # 跨日拼接
mids_concat = np.concatenate([...])  # 跨日拼接
```

**修改後** (保留日期結構):
```python
# 步驟 1: 重組資料（保留日期結構）
stock_data[sym]['day_data'].append((date, Xd, mids))

# 不拼接，保留 [(date, X, mids)] 列表
```

#### 逐日獨立處理

**修改前** (一次處理整條序列):
```python
# Z-score 正規化
Xn = zscore_apply(Xd, mu, sd)  # Xd 是跨日拼接的

# 計算波動率（跨日）
vol = ewma_vol(close, halflife=60)

# 生成標籤（跨日）
tb_df = tb_labels(close, vol, ...)
```

**修改後** (逐日獨立處理):
```python
for sym, n_points, day_data_sorted in stock_list:
    for date, Xd, mids in day_data_sorted:  # 逐日迴圈
        # 1. Z-score 正規化（每日獨立）
        Xn = zscore_apply(Xd, mu, sd)

        # 2. 構建 DataFrame
        close = pd.Series(mids, name='close')

        # 3. 計算波動率（每日重置狀態）
        vol = ewma_vol(close, halflife=60)

        # 4. 生成 Triple-Barrier 標籤（限制在當日內）
        day_end_idx = len(close) - 1 if respect_day_boundary else None
        tb_df = tb_labels(close, vol, ..., day_end_idx=day_end_idx)

        # 5. 產生滑窗樣本（禁止跨日）
        for t in range(SEQ_LEN - 1, max_t):
            window_start = t - SEQ_LEN + 1
            if respect_day_boundary and window_start < 0:
                cross_day_filtered += 1
                continue
            ...
```

---

### 修改 3: 配置文件更新

**檔案**: [configs/config_pro_v5_ml_optimal.yaml](../configs/config_pro_v5_ml_optimal.yaml)

#### 新增參數

```yaml
# 核心修正：防止跨日污染
respect_day_boundary: true  # 強制啟用日界線保護（預設 true）
```

#### 調整 triple-barrier 參數

**修改原因**: 原參數（5.9σ, min_return=0.215%）是為了「壓制持平」而極限調優，修正後不再需要

```yaml
triple_barrier:
  pt_multiplier: 3.5   # 止盈：3.5σ（修正後建議值）
  sl_multiplier: 3.5   # 止損：3.5σ
  max_holding: 40      # 最大持有：40 bars（考慮日內限制）
  min_return: 0.0015   # 閾值：0.15%（只用於 vertical）
```

---

## 修復效果

### 預期改善

| 指標 | 修復前 | 修復後（預期） |
|------|--------|----------------|
| Class 1 比例 | 0-5% | 35-45% |
| 觸發原因 'time' | <5% | 30-50% |
| 跨日污染 | 嚴重 | 無 |
| 標籤噪音 | 高 | 顯著降低 |

### 技術保證

1. ✅ **波動率每日重置**：EWMA/GARCH 狀態不跨夜
2. ✅ **vertical barrier 日內限制**：不會越日觸發
3. ✅ **滑窗禁止跨日**：昨天 + 今天不視為連續
4. ✅ **min_return 只影響 vertical**：PT/SL 固定標 ±1

---

## 驗證方法

### 快速驗證腳本

```bash
# 使用少量數據測試修復
python scripts/verify_fix_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/test_fix \
    --max-files 5
```

### 檢查清單

1. **標籤分布**：
   - 打開 `./data/test_fix/npz/normalization_meta.json`
   - 查看 `data_split.results.train.label_dist`
   - 確認 Class 1 比例在 35-45%

2. **觸發原因**：
   - 檢查日誌中的「触发原因分布」
   - 確認 `time` 觸發佔比 > 30%

3. **跨日過濾**：
   - 檢查日誌中的「跨日过滤」統計
   - 啟用 `respect_day_boundary` 時應有過濾紀錄

### 完整數據驗證

```bash
# 使用完整數據重新生成
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_fixed \
    --config ./configs/config_pro_v5_ml_optimal.yaml
```

---

## 相關文件

- 配置文件: [configs/config_pro_v5_ml_optimal.yaml](../configs/config_pro_v5_ml_optimal.yaml)
- 主程式: [scripts/extract_tw_stock_data_v5.py](../scripts/extract_tw_stock_data_v5.py)
- 驗證腳本: [scripts/verify_fix_v5.py](../scripts/verify_fix_v5.py)
- 專案說明: [CLAUDE.md](../CLAUDE.md)

---

## 後續建議

### 立即行動

1. ✅ 使用 [verify_fix_v5.py](../scripts/verify_fix_v5.py) 快速驗證修復
2. ⏳ 使用完整數據重新生成訓練集
3. ⏳ 比較修復前後的模型性能

### 參數微調

修復後可能需要重新調整：

1. **triple-barrier 參數**：
   - 若 Class 1 過多（>50%）：降低 `pt/sl_multiplier` 或增加 `max_holding`
   - 若 Class 1 仍少（<30%）：提高 `pt/sl_multiplier` 或降低 `min_return`

2. **波動率參數**：
   - EWMA `halflife` 可能需要調整（目前 60 bars）
   - 考慮改用 `garch` 方法以獲得更穩健的估計

3. **訓練策略**：
   - 根據實際分布選擇 loss function（Focal Loss / CrossEntropy）
   - 調整 sample weights 參數（`tau`, `return_scaling`）

---

## 技術細節

### 為什麼 min_return 要區分 vertical 和 PT/SL？

**標準 triple-barrier 邏輯**：

- **PT/SL 觸發**：表示價格有明確趨勢（超過 nσ 邊界）→ 直接標 ±1
- **時間到期**：表示價格橫盤（沒觸及邊界）→ 用 `min_return` 判斷是否為噪音

**原代碼問題**：

- 對所有樣本都用 `min_return` 判斷
- PT/SL 觸發但 `|ret| < min_return` → 改標 0（錯誤）
- 時間到期但 `|ret| > min_return` → 改標 ±1（可接受但過於敏感）

**修復後**：

- PT/SL → 固定 ±1（不受 `min_return` 影響）
- vertical → 用 `min_return` 過濾噪音（符合原始設計）

### 為什麼要按日處理？

**金融時間序列特性**：

1. **隔夜跳空不連續**：收盤價 ≠ 次日開盤價（新聞、外盤影響）
2. **波動率日內特徵**：開盤波動大，尾盤波動小（有日內模式）
3. **風險管理日界**：實務上不會讓 intraday 策略跨夜持倉

**跨日拼接後果**：

- 波動率被隔夜跳空扭曲
- 標籤被跨日趨勢污染
- 模型學到「假連續性」，無法泛化

---

**最後更新**: 2025-10-20
**版本**: v5.0.3
**狀態**: 已修復，待驗證
