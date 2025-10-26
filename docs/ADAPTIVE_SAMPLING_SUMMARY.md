# TensorBoard 自適應採樣功能總結

## 功能概述

`analyze_tensorboard.py` 腳本現在支持**自適應採樣**，會根據實際數據量自動調整採樣密度，確保輸出始終有意義。

## 自適應採樣規則

### 段數自動調整

| 數據點數 | 採樣段數 | 說明 |
|---------|---------|------|
| < 10 點 | 全部返回 | 數據太少，返回所有點 |
| 10-50 點 | 5 段 | 小規模數據 |
| 50-200 點 | 10 段 | 中等規模數據 |
| 200-1000 點 | 20 段 | 正常規模數據 |
| 1000-5000 點 | 50 段 | 大規模數據 |
| > 5000 點 | 100 段 | 超大規模數據 |

### 用戶指定段數

如果用戶通過 `--segments` 參數指定段數，則使用用戶指定的值（但不超過實際數據點數）。

```bash
# 指定使用 30 段（如果數據足夠）
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/analysis \
    --segments 30
```

## 實際測試結果

### 測試場景：極短訓練（10K steps）

**訓練參數**:
- 總步數: 10,000 steps
- 訓練時長: 43 秒
- 記錄點數: 僅 2 個數據點（非常稀疏）

**自適應行為**:
```
數據點數: 2, 採樣段數: 2
```

**輸出結果**:
```json
{
  "timeseries": {
    "segment_samples": [
      {
        "step": 5120,
        "value": -40.34382629394531,
        "min": -40.34382629394531,
        "max": -40.34382629394531,
        "std": 0.0
      },
      {
        "step": 10240,
        "value": -39.516258239746094,
        "min": -39.516258239746094,
        "max": -39.516258239746094,
        "std": 0.0
      }
    ],
    "inflection_samples": [
      {
        "step": 5120,
        "value": -40.34382629394531
      },
      {
        "step": 10240,
        "value": -39.516258239746094
      }
    ]
  }
}
```

## 優勢

### 1. 自動適應數據規模 ✅
- 極短測試（2 點）→ 返回所有點
- 正常訓練（200-1000 點）→ 20 段
- 超長訓練（5000+ 點）→ 100 段

### 2. 避免過度採樣 ✅
- 不會嘗試從 2 個點中創建 20 個段（會導致重複）
- 始終返回有意義的樣本數量

### 3. 數據完整性 ✅
- 小數據集返回所有點（無信息丟失）
- 大數據集智能壓縮（保持趨勢可見）

### 4. 靈活控制 ✅
- 默認自動調整（推薦）
- 支持用戶手動指定段數

## 壓縮效果示例

### 實際訓練場景（1M steps）

假設 1M steps 訓練產生約 5000 個 TensorBoard 數據點：

**原始數據**:
- 5000 個數據點
- JSON 大小: ~500 KB（含完整時間序列）

**自適應採樣後**:
- 段數採樣: 100 個段（每段統計信息）
- 轉折點採樣: ~50-80 個關鍵點
- JSON 大小: ~15 KB
- **壓縮率**: 97% (500 KB → 15 KB)
- **信息保留**: 趨勢、峰值、轉折點完整保留

## 使用建議

### 快速測試（推薦默認）
```bash
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/test_analysis \
    --format both \
    --verbose
```

**自動行為**:
- 10K steps 測試 → 返回所有點（2-5 點）
- 100K steps 訓練 → 10-20 段
- 1M steps 訓練 → 50-100 段

### 手動控制（特殊需求）
```bash
# 要求更高精度（更多採樣點）
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/detailed_analysis \
    --segments 50

# 要求更緊湊（更少採樣點）
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/compact_analysis \
    --segments 10
```

## Verbose 輸出解讀

啟用 `--verbose` 後，會顯示自適應決策：

```
[LOAD] 載入日誌: logs\sb3_deeplob\PPO_1
[OK] 找到 14 個指標

[ANALYZE] 分析數據...
  數據點數: 2, 採樣段數: 2     ← 自適應決策
  數據點數: 2, 採樣段數: 2
  數據點數: 5, 採樣段數: 5
  數據點數: 100, 採樣段數: 10
  數據點數: 5000, 採樣段數: 100
...
```

每一行顯示：
- **數據點數**: TensorBoard 實際記錄的點數
- **採樣段數**: 自適應決定使用的段數

## 技術實現

### 核心邏輯（`_segment_sampling` 方法）

```python
def _segment_sampling(self, steps: List[int], values: List[float]) -> List[Dict[str, float]]:
    """分段採樣：根據數據量自動調整段數"""
    n = len(steps)

    # 根據數據量自動確定段數
    if n < 10:
        actual_segments = n  # 返回所有點
    elif n < 50:
        actual_segments = min(5, n)
    elif n < 200:
        actual_segments = min(10, n)
    elif n < 1000:
        actual_segments = min(20, n)
    elif n < 5000:
        actual_segments = min(50, n)
    else:
        actual_segments = min(100, n)

    # 如果用戶指定了段數，使用用戶指定的
    if self.num_segments > 0:
        actual_segments = min(self.num_segments, n)

    if self.verbose:
        print(f"  數據點數: {n}, 採樣段數: {actual_segments}")

    # 如果段數 >= 數據點數，返回所有點（含完整字段）
    if actual_segments >= n:
        return [{
            'step': int(s),
            'value': float(v),
            'min': float(v),
            'max': float(v),
            'std': 0.0
        } for s, v in zip(steps, values)]

    # 否則進行分段採樣...
    # （計算每段的 min/max/mean/std）
```

## 對比：V6 vs 自適應版本

| 特性 | V6 固定段數 | 自適應版本 |
|-----|-----------|-----------|
| 短訓練（2 點） | 返回 2 點，段數標記為 20 | 返回 2 點，段數標記為 2 ✅ |
| 中訓練（100 點） | 返回 20 段 | 返回 10 段（更合適） ✅ |
| 長訓練（5000 點） | 返回 20 段（可能不夠） | 返回 100 段（更精細） ✅ |
| 用戶控制 | 僅默認 20 段 | 支持手動指定 ✅ |
| Verbose 提示 | 無 | 顯示自適應決策 ✅ |

## 總結

**核心改進**:
1. ✅ **智能自適應**: 根據數據量自動調整，無需手動干預
2. ✅ **避免過度採樣**: 不會從 2 個點中生成 20 個段
3. ✅ **最佳壓縮率**: 大數據集使用更多段，小數據集返回全部
4. ✅ **用戶友好**: Verbose 模式顯示決策過程
5. ✅ **靈活控制**: 支持手動覆蓋自動決策

**解決問題**:
- 用戶原始請求: "可以一時間長短自動切分,不用固定段落"
- 實現效果: 完全符合需求，根據數據量智能調整段數

---

**最後更新**: 2025-10-26
**腳本版本**: analyze_tensorboard.py v2.0 (自適應採樣版)
