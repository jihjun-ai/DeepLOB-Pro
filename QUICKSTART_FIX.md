# 跨日污染修復 - 快速開始指南

**版本**: 5.0.3-fixed
**更新日期**: 2025-10-20

---

## 修復摘要

已修復 V5 版本的**跨日污染**問題，主要改動：

1. ✅ **按日處理架構**：每天獨立做標準化→波動率→triple-barrier→滑窗
2. ✅ **波動率每日重置**：EWMA/GARCH 狀態不跨夜累積
3. ✅ **vertical barrier 日內限制**：`max_holding` 不得越過當日最後一根
4. ✅ **min_return 只影響 vertical**：PT/SL 觸發固定標 ±1
5. ✅ **滑窗禁止跨日**：昨天尾盤 + 今天開盤不視為連續

**預期效果**：Class 1（持平）比例從 0-5% 提升至 35-45%

---

## 使用流程

### 步驟 1: 快速驗證（5-10 分鐘）

使用少量數據測試修復是否有效：

```bash
# 啟動環境
conda activate deeplob-pro

# 快速驗證（處理 5 個檔案）
python scripts/verify_fix_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/test_fix \
    --max-files 5
```

**檢查結果**：

1. 打開 `./data/test_fix/npz/normalization_meta.json`
2. 查看 `data_split.results.train.label_dist`
3. **成功標準**：Class 1 比例應在 **35-45%**

**範例輸出**：

```json
{
  "data_split": {
    "results": {
      "train": {
        "label_dist": [1200, 1800, 1250],  // [下跌, 持平, 上漲]
        // 持平佔比: 1800 / (1200+1800+1250) = 42.3% ✅
      }
    }
  }
}
```

### 步驟 2: 完整數據重新生成（1-2 小時）

驗證成功後，使用完整數據重新生成訓練集：

```bash
# 使用修復後的配置
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_fixed \
    --config ./configs/config_pro_v5_ml_optimal.yaml
```

**關鍵配置參數**：

```yaml
# config_pro_v5_ml_optimal.yaml
respect_day_boundary: true  # 啟用日界線保護

triple_barrier:
  pt_multiplier: 3.5   # 止盈：3.5σ（修正後建議值）
  sl_multiplier: 3.5   # 止損：3.5σ
  max_holding: 40      # 最大持有：40 bars
  min_return: 0.0015   # 閾值：0.15%（只用於 vertical）
```

### 步驟 3: 檢查標籤分布

查看生成的報告：

```bash
# 查看 metadata
cat ./data/processed_v5_fixed/npz/normalization_meta.json | grep -A 10 "label_dist"

# 查看震盪統計
cat ./data/processed_v5_fixed/volatility_summary.json
```

**預期分布**：

| Split | Class 0 (下跌) | Class 1 (持平) | Class 2 (上漲) |
|-------|----------------|----------------|----------------|
| Train | 25-35% | 35-45% | 25-35% |
| Val   | 25-35% | 35-45% | 25-35% |
| Test  | 25-35% | 35-45% | 25-35% |

### 步驟 4: 訓練模型

使用修復後的數據訓練 DeepLOB：

```bash
# 使用新數據訓練
python scripts/train_deeplob_v5.py \
    --data-dir ./data/processed_v5_fixed/npz \
    --output-dir ./checkpoints/v5_fixed \
    --config ./configs/config_pro_v5_ml_optimal.yaml \
    --epochs 50 \
    --batch-size 128
```

**關鍵指標**：

- 訓練時監控 **per-class F1 score**
- Class 1（持平）的 Recall 應 > 60%（避免橫盤誤交易）
- Class 0/2 的 Precision 應 > 70%（買賣信號可靠）

---

## 常見問題

### Q1: Class 1 比例仍然很低（<20%）？

**可能原因**：

1. `pt_multiplier/sl_multiplier` 過低（太容易觸發 PT/SL）
2. `max_holding` 過長（日內大多會觸發邊界）
3. `min_return` 過高（把太多 vertical 改標成 ±1）

**解決方案**：

調整 `config_pro_v5_ml_optimal.yaml`：

```yaml
triple_barrier:
  pt_multiplier: 4.5   # 提高到 4.5σ（更難觸發）
  sl_multiplier: 4.5
  max_holding: 30      # 縮短到 30 bars
  min_return: 0.001    # 降低到 0.1%
```

### Q2: Class 1 比例過高（>60%）？

**可能原因**：

1. `pt_multiplier/sl_multiplier` 過高（PT/SL 觸發太少）
2. `max_holding` 過短（大多是 vertical 到期）

**解決方案**：

```yaml
triple_barrier:
  pt_multiplier: 2.5   # 降低到 2.5σ
  sl_multiplier: 2.5
  max_holding: 50      # 延長到 50 bars
  min_return: 0.002    # 提高到 0.2%
```

### Q3: 如何確認沒有跨日污染？

**檢查清單**：

1. 查看日誌中的「跨日过滤」統計：

```
跨日过滤: 1234 个滑窗被移除
```

2. 檢查 metadata 中的 `respect_day_boundary`：

```json
{
  "respect_day_boundary": true
}
```

3. 比較觸發原因分布（修復後 'time' 應佔 30-50%）：

```
触发原因分布: {'up': 1200, 'down': 1150, 'time': 1800}
```

---

## 參數調整指南

### 目標：Class 1 = 40%

| 當前 Class 1 | 調整方向 | 建議參數 |
|--------------|----------|----------|
| < 20% | 增加持平 | pt/sl ↑ 到 4-5σ, max_holding ↓ 到 30 |
| 20-30% | 微調 | pt/sl ↑ 到 3.5-4σ |
| 30-50% | ✅ 理想 | 保持當前參數 |
| 50-60% | 微調 | pt/sl ↓ 到 2.5-3σ |
| > 60% | 減少持平 | pt/sl ↓ 到 2σ, max_holding ↑ 到 60 |

### 快速迭代流程

```bash
# 1. 修改 config_pro_v5_ml_optimal.yaml
vim ./configs/config_pro_v5_ml_optimal.yaml

# 2. 快速驗證（只處理 5 個檔案）
python scripts/verify_fix_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/test_iter \
    --max-files 5

# 3. 檢查標籤分布
cat ./data/test_iter/npz/normalization_meta.json | grep "label_dist"

# 4. 滿意後用完整數據重新生成
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_fixed \
    --config ./configs/config_pro_v5_ml_optimal.yaml
```

---

## 修復效果對比

### 修復前（V5.0.2）

```
標籤分布:
  Class 0 (下跌): 2,800,000 (49.5%)
  Class 1 (持平):   120,000 ( 2.1%) ❌
  Class 2 (上漲): 2,730,000 (48.4%)

觸發原因:
  {'up': 2730000, 'down': 2800000, 'time': 120000}
  time 佔比: 2.1% ❌
```

### 修復後（V5.0.3 預期）

```
標籤分布:
  Class 0 (下跌): 1,750,000 (31.0%)
  Class 1 (持平): 2,250,000 (40.0%) ✅
  Class 2 (上漲): 1,650,000 (29.0%)

觸發原因:
  {'up': 1650000, 'down': 1750000, 'time': 2250000}
  time 佔比: 40.0% ✅
```

---

## 相關文件

- **修復詳情**: [docs/FIX_CROSS_DAY_CONTAMINATION.md](docs/FIX_CROSS_DAY_CONTAMINATION.md)
- **配置文件**: [configs/config_pro_v5_ml_optimal.yaml](configs/config_pro_v5_ml_optimal.yaml)
- **驗證腳本**: [scripts/verify_fix_v5.py](scripts/verify_fix_v5.py)
- **專案說明**: [CLAUDE.md](CLAUDE.md)

---

## 下一步

1. ✅ 執行快速驗證（verify_fix_v5.py）
2. ⏳ 檢查 Class 1 比例是否符合預期
3. ⏳ 使用完整數據重新生成訓練集
4. ⏳ 訓練模型並比較性能
5. ⏳ 根據實際分布微調參數

**預期提升**：

- ✅ Class 1（持平）回歸正常比例
- ✅ 模型學會「不交易」策略
- ✅ 泛化能力提升（無跨日資料洩漏）
- ✅ 實盤表現更穩健（風險控制更準確）

---

**最後更新**: 2025-10-20
**版本**: v5.0.3
**狀態**: 已修復，待驗證
