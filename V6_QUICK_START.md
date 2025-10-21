# V6 快速上手指南

**3 步驟開始使用 V6 雙階段資料處理**

---

## ⚡ 最快上手（5 分鐘）

### 前置需求

✅ Conda 環境: `deeplob-pro`
✅ 原始數據: `data/temp/*.txt`
✅ 配置文件: `configs/config_pro_v5_ml_optimal.yaml`

### 執行步驟

```bash
# 1. 啟動環境
conda activate deeplob-pro

# 2. 批次預處理（首次，約 30 分鐘）
scripts\batch_preprocess.bat

# 3. 生成訓練數據（約 8 分鐘）
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

**完成！** 訓練數據已保存至 `data/processed_v6/npz/`

---

## 📊 檢查結果

### 查看標籤分布

```bash
type data\processed_v6\npz\normalization_meta.json
```

**尋找**:
```json
{
  "data_split": {
    "results": {
      "train": {
        "label_dist": [1678365, 2233111, 1673077]
      }
    }
  }
}
```

**計算比例**:
- Down: 1678365 / 總數 = 30.1% ✅
- Neutral: 2233111 / 總數 = 40.0% ✅
- Up: 1673077 / 總數 = 29.9% ✅

### 查看整體報告

```bash
type data\preprocessed_v5\reports\overall_summary.json
```

**關鍵指標**:
- `total_statistics.overall_pass_rate`: 通過過濾的比例
- `predicted_label_distribution`: 預測的標籤分布
- `filter_threshold_distribution.methods`: 閾值選擇方法分布

---

## 🧪 測試單檔（開發用）

```bash
# 測試 20250901.txt
scripts\test_preprocess.bat

# 查看當天摘要
type data\preprocessed_v5_test\daily\20250901\summary.json
```

**摘要內容**:
```json
{
  "date": "20250901",
  "total_symbols": 195,
  "passed_filter": 156,
  "filtered_out": 39,
  "filter_threshold": 0.0050,
  "filter_method": "adaptive_P25",
  "predicted_label_dist": {
    "down": 0.32,
    "neutral": 0.38,
    "up": 0.30
  }
}
```

---

## 🎯 調整 Triple-Barrier 參數（快速）

### 場景：持平標籤過多（>50%）

**步驟 1**: 複製配置文件

```bash
copy configs\config_pro_v5_ml_optimal.yaml configs\config_test.yaml
```

**步驟 2**: 修改參數

```yaml
# configs/config_test.yaml
triple_barrier:
  pt_multiplier: 3.0      # 從 3.5 降到 3.0（更容易觸發）
  sl_multiplier: 3.0
  max_holding: 40
  min_return: 0.002       # 從 0.0015 提高（更嚴格）
```

**步驟 3**: 重新生成（僅需 5-10 分鐘）

```bash
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6_test ^
    --config .\configs\config_test.yaml
```

**步驟 4**: 檢查新分布

```bash
type data\processed_v6_test\npz\normalization_meta.json
```

---

## 🔄 新增數據（增量處理）

### 場景：新增 20250913.txt

**步驟 1**: 僅預處理新日期（約 3 分鐘）

```bash
python scripts\preprocess_single_day.py ^
    --input .\data\temp\20250913.txt ^
    --output-dir .\data\preprocessed_v5 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

**步驟 2**: 重新生成訓練數據（約 10 分鐘）

```bash
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

**完成！** 新數據已自動整合

---

## 🚀 開始訓練

### 使用 V6 數據訓練 DeepLOB

```bash
python scripts\train_deeplob_generic.py ^
    --data-dir .\data\processed_v6\npz ^
    --output-dir .\checkpoints\v6 ^
    --config .\configs\deeplob_config.yaml ^
    --epochs 50
```

**完全兼容 V5 訓練代碼！**

---

## ❓ 常見問題

### Q: 預處理太慢？

**A**: 首次需處理所有歷史數據（約 30 分鐘）。之後：
- 調整 TB 參數：僅需 5-10 分鐘
- 新增 1 天數據：僅需 4 分鐘

### Q: 標籤分布不理想？

**A**: 檢查報告並調整：
```bash
type data\preprocessed_v5\reports\overall_summary.json
```

| 問題 | 解決方案 |
|------|---------|
| Neutral > 50% | 提高 `min_return` 或降低 `pt_multiplier` |
| Neutral < 20% | 降低 `min_return` 或提高 `pt_multiplier` |

### Q: 如何查看每天的過濾決策？

**A**: 查看過濾決策記錄：
```bash
type data\preprocessed_v5\reports\filter_decisions.csv
```

### Q: V6 與 V5 兼容嗎？

**A**: 完全兼容！輸出格式相同，訓練代碼無需修改。

---

## 📚 進階文檔

- **完整指南**: [docs/V6_TWO_STAGE_PIPELINE_GUIDE.md](docs/V6_TWO_STAGE_PIPELINE_GUIDE.md)
- **實作摘要**: [docs/V6_IMPLEMENTATION_SUMMARY.md](docs/V6_IMPLEMENTATION_SUMMARY.md)
- **專案配置**: [CLAUDE.md](CLAUDE.md)

---

## 🎉 核心優勢

✅ **動態過濾**: 每天自動調整閾值
✅ **效率提升**: 參數調整快 82%
✅ **穩定標籤**: 維持 30/40/30 分布
✅ **完全兼容**: 無需修改訓練代碼

---

**版本**: 6.0.0
**更新**: 2025-10-21
**狀態**: ✅ 可用
