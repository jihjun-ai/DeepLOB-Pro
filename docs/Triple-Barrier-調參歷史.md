# Triple-Barrier 數據生成調參歷史

**目的**：記錄 `config_pro_v5_ml_optimal.yaml` 的 Triple-Barrier 參數調整過程
**目標**：達到 Class 1（持平）= 45-55%
**日期**：2025-10-18

---

## 環境設定

```bash
conda activate deeplob-pro

# 數據生成命令
python scripts/extract_tw_stock_data_v5.py \
    --config configs/config_pro_v5_ml_optimal.yaml \
    --output_dir data/processed_v5

# 驗證分佈
python test.py
```

---

## 調參總覽表

| 版本 | min_return | pt/sl | Class 0 | Class 1 | Class 2 | 狀態 |
|------|------------|-------|---------|---------|---------|------|
| v1 | 0.15% | 6.0σ | 1% | **98%** | 1% | ❌ 崩潰 |
| v2 | 0.02% | 2.0σ | 47.52% | **11.26%** | 41.22% | ❌ 過少 |
| v3 | 0.05% | 2.5σ | 45.04% | **16.11%** | 38.86% | ❌ 過少 |
| v4 | 0.08% | 3.0σ | 42.87% | **21.24%** | 35.89% | ❌ 過少 |
| v5 | 0.12% | 4.0σ | 37.90% | **29.27%** | 32.83% | ⚠️ 仍少 |
| v6 | 0.18% | 5.0σ | 31.81% | **40.50%** | 27.69% | ⚠️ 接近 |
| v7 | 0.20% | 5.5σ | 30.20% | **43.48%** | 26.32% | ⚠️ 極近 |
| v8 | 0.21% | 5.8σ | 29.63% | **44.61%** | 25.76% | ⚠️ 極近 |
| v9 | 0.215% | 5.9σ | 29.41% | **45.01%** | 25.58% | ✅ 達標 |

---

## v1 → v2：初次修正（矯枉過正）

### v1 配置（原始，Class 1 崩潰）
```yaml
triple_barrier:
  pt_multiplier: 6.0
  sl_multiplier: 6.0
  max_holding: 40
  min_return: 0.0015  # 0.15%
```

**問題**：
- 訓練時 Class 1 完全沒被預測到（P=0.00, R=0.00）
- 數據驗證顯示 Class 1 = 98%（極端不平衡）
- 原因：6σ 太寬 + 0.15% 閾值太高 → 幾乎所有樣本都到期且 < 0.15%

### v2 配置（首次修正）
```yaml
triple_barrier:
  pt_multiplier: 2.0  # 6.0 → 2.0
  sl_multiplier: 2.0  # 6.0 → 2.0
  max_holding: 50     # 40 → 50
  min_return: 0.0002  # 0.15% → 0.02%
```

**結果**：
```
Class 0: 593,762 (47.52%)
Class 1: 140,690 (11.26%)  ← 從 98% 變成 11%
Class 2: 514,967 (41.22%)
```

**問題**：矯枉過正！Class 1 太少。

---

## v2 → v3：提高閾值

### v3 配置
```yaml
triple_barrier:
  pt_multiplier: 2.5  # 2.0 → 2.5
  sl_multiplier: 2.5
  max_holding: 50
  min_return: 0.0005  # 0.02% → 0.05%
```

**結果**：
```
Class 0: 562,698 (45.04%)
Class 1: 201,226 (16.11%)  ← 提升 4.85%
Class 2: 485,495 (38.86%)
```

**進展**：Class 1 提升 43%，但仍遠低於目標。

---

## v3 → v4：繼續放寬

### v4 配置
```yaml
triple_barrier:
  pt_multiplier: 3.0  # 2.5 → 3.0
  sl_multiplier: 3.0
  max_holding: 50
  min_return: 0.0008  # 0.05% → 0.08%
```

**結果**：
```
Class 0: 535,706 (42.87%)
Class 1: 265,373 (21.24%)  ← 提升到 21%
Class 2: 448,340 (35.89%)
```

**進展**：持續改善，但仍不足。

---

## v4 → v5：大幅提高

### v5 配置
```yaml
triple_barrier:
  pt_multiplier: 4.0  # 3.0 → 4.0
  sl_multiplier: 4.0
  max_holding: 50
  min_return: 0.0012  # 0.08% → 0.12%
```

**結果**：
```
Class 0: 473,548 (37.90%)
Class 1: 365,684 (29.27%)  ← 提升到 29%
Class 2: 410,187 (32.83%)
```

**進展**：開始接近目標。

---

## v5 → v6：激進調整

### v6 配置
```yaml
triple_barrier:
  pt_multiplier: 5.0  # 4.0 → 5.0
  sl_multiplier: 5.0
  max_holding: 50
  min_return: 0.0018  # 0.12% → 0.18%
```

**結果**：
```
Class 0: 397,396 (31.81%)
Class 1: 506,030 (40.50%)  ← 大幅提升到 40.5%
Class 2: 345,993 (27.69%)
```

**進展**：非常接近目標 45%！

---

## v6 → v7：接近目標

### v7 配置
```yaml
triple_barrier:
  pt_multiplier: 5.5  # 5.0 → 5.5
  sl_multiplier: 5.5
  max_holding: 50
  min_return: 0.0020  # 0.18% → 0.20%
```

**結果**：
```
Class 0: 377,361 (30.20%)
Class 1: 543,209 (43.48%)  ← 距離 45% 僅差 1.52%
Class 2: 328,849 (26.32%)
```

**進展**：極度接近！

---

## v7 → v8：極限接近

### v8 配置
```yaml
triple_barrier:
  pt_multiplier: 5.8  # 5.5 → 5.8
  sl_multiplier: 5.8
  max_holding: 50
  min_return: 0.0021  # 0.20% → 0.21%
```

**結果**：
```
Class 0: 370,254 (29.63%)
Class 1: 557,312 (44.61%)  ← 距離 45% 僅差 0.39%
Class 2: 321,853 (25.76%)
```

**進展**：已經非常接近目標！

---

## v8 → v9：最終達標 ✅

### v9 配置（最終版）
```yaml
triple_barrier:
  pt_multiplier: 5.9  # 5.8 → 5.9
  sl_multiplier: 5.9
  max_holding: 50
  min_return: 0.00215  # 0.21% → 0.215%
```

**結果**：
```
Class 0: 367,457 (29.41%)
Class 1: 562,361 (45.01%)  ← ✅ 完美達標！
Class 2: 319,601 (25.58%)
```

**成果**：
- ✅ Class 1 = 45.01%（精準達到目標下限）
- ✅ Class 0 vs Class 2 = 3.83% 差距（對稱性良好）
- ✅ 總樣本 124.9萬（充足訓練數據）
- ✅ Class 1 樣本 56.2萬（非常充分）

---

## 關鍵發現

### 1. min_return 是主要控制參數
- 從 0.02% → 0.215%：Class 1 從 11% → 45%
- 呈線性增長關係

### 2. pt/sl multiplier 影響到期率
- 越寬 → 到期率越高 → Class 1 越多
- 從 2.0σ → 5.9σ：Class 1 從 11% → 45%

### 3. 台股 LOB 最佳參數區間
- **min_return**：0.18-0.22%
- **pt/sl**：5.5-6.0σ
- **max_holding**：50 bars

### 4. Class 0/2 不對稱問題
- Class 0 通常比 Class 2 多 3-4%
- 可能原因：市場結構或數據期間特性
- 可接受範圍：< 5%

---

## 最終配置（v9）

```yaml
# configs/config_pro_v5_ml_optimal.yaml

volatility:
  method: 'ewma'
  halflife: 60

triple_barrier:
  pt_multiplier: 5.9
  sl_multiplier: 5.9
  max_holding: 50
  min_return: 0.00215  # 0.215%

sample_weights:
  enabled: true
  tau: 80.0
  return_scaling: 12.0
  balance_classes: true
```

---

## 調整總結

- **調整次數**：9 次
- **總耗時**：約 4-5 小時
- **Class 1 進展**：98% → 11% → 45.01% ✅
- **關鍵參數變化**：
  - min_return: 0.15% → 0.215% (提高 43%)
  - pt/sl: 6.0σ → 5.9σ (微降)
- **最終成果**：完美達標，數據質量優異

---

**更新日期**：2025-10-18
**狀態**：✅ v9 已達標
**下一步**：開始訓練 DeepLOB 模型
