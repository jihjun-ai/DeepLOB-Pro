# 趨勢標籤實現說明

**更新日期**: 2025-10-23
**版本**: v1.0
**狀態**: ✅ 已完成實現

---

## 概述

針對**日內波段交易**需求，新增**趨勢標籤**（Trend Labeling）支持，作為 Triple-Barrier 標籤的替代方案。

### 為什麼需要趨勢標籤？

**用戶需求**:
- 日內波段交易（每天 1-2 次交易）
- 最少 1% 利潤（扣除手續費後仍有利）
- 及時抓住趨勢（不是每個小波動都交易）

**Triple-Barrier 的問題**:
- ❌ **短視**: 持倉時間太短（40 bars ≈ 4-8 分鐘）
- ❌ **高頻**: 產生過多交易信號（不符合日沖需求）
- ❌ **噪音**: 小震盪產生錯誤標籤（無法識別整體趨勢）

### 解決方案對比

| 特性 | Triple-Barrier | 趨勢標籤 |
|------|---------------|---------|
| 時間尺度 | 40 bars (4-8 分鐘) | 150 bars (1.5-3 小時) |
| 利潤閾值 | 0.25% | 自適應（基於波動率） |
| 交易頻率 | 高頻（10-20 次/天） | 波段（1-2 次/天） |
| 趨勢識別 | ❌ 無法識別 | ✅ 正確識別 |
| 適用場景 | 超短線/高頻 | 日內波段 |

---

## 實現細節

### 1. 核心函數

**位置**: `src/utils/financial_engineering.py`

```python
def trend_labels_adaptive(
    close: pd.Series,
    volatility: pd.Series,
    lookforward: int = 100,
    vol_multiplier: float = 2.0
) -> pd.Series:
    """
    自適應趨勢標籤（閾值基於波動率）

    Args:
        close: 收盤價序列
        volatility: 波動率序列
        lookforward: 往前看的時間點數
        vol_multiplier: 波動率倍數（趨勢閾值 = vol × multiplier）

    Returns:
        Series with labels: -1 (下跌), 0 (持平), 1 (上漲)
    """
```

**特點**:
- 波動率自適應: 高波動期需要更大變化才算趨勢
- 長期視角: 看未來 150 bars（1.5-3 小時）
- 穩定性: 過濾小震盪噪音

### 2. 數據處理腳本修改

#### preprocess_single_day.py

**修改位置**: 行 591-620

```python
def trend_labels(close: pd.Series,
                 vol: pd.Series,
                 lookforward: int = 150,
                 vol_multiplier: float = 2.0) -> pd.Series:
    """趨勢標籤生成（專業版包裝函數）"""
    return trend_labels_adaptive(
        close=close,
        volatility=vol,
        lookforward=lookforward,
        vol_multiplier=vol_multiplier
    )
```

**compute_label_preview() 函數增強**:
- 支持標籤方法選擇（`labeling_method` 參數）
- 自動路由到 Triple-Barrier 或趨勢標籤
- 返回結果包含 `labeling_method` 字段

#### extract_tw_stock_data_v6.py

**修改位置**: 行 234-261, 783-844

- 新增 `trend_labels()` 函數
- 修改主處理邏輯支持方法選擇
- 趨勢標籤生成兼容的 DataFrame 格式

### 3. 配置文件

**新增**: `configs/config_intraday_swing.yaml`

```yaml
# 日內波段交易配置
labeling_method: "trend_adaptive"  # 使用趨勢標籤

trend_labeling:
  lookforward: 150           # 往前看 150 bars ≈ 1.5-3 小時
  vol_multiplier: 2.0        # 波動率倍數
```

**向後兼容**:
- 未指定 `labeling_method` 時預設為 `triple_barrier`
- 現有配置無需修改即可繼續使用

---

## 使用方式

### 選項 A: 使用趨勢標籤（推薦日內波段）

```bash
# 使用趨勢標籤配置
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_swing \
    --config ./configs/config_intraday_swing.yaml
```

### 選項 B: 繼續使用 Triple-Barrier（高頻交易）

```bash
# 使用原有配置（Triple-Barrier）
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_v6 \
    --config ./configs/config_pro_v5_ml_optimal.yaml
```

---

## 預期效果

### 標籤分布

**趨勢標籤**:
- 下跌: 20-30%
- 持平: 40-60% ← 小震盪全部歸為持平
- 上漲: 20-30%

**Triple-Barrier** (對比):
- 下跌: 30%
- 持平: 40%
- 上漲: 30%

### 交易特性

**趨勢標籤**:
- 交易頻率: 1-3 次/股/天
- 平均持倉: 1-3 小時
- 預期利潤: ≥1% (符合日沖需求)
- 勝率目標: 55-65%

**Triple-Barrier** (對比):
- 交易頻率: 10-20 次/股/天
- 平均持倉: 4-8 分鐘
- 預期利潤: 0.25-0.5%
- 勝率目標: 52-55%

---

## 測試驗證

### 測試腳本

**位置**: `scripts/test_trend_label_config.py`

```bash
# 快速測試趨勢標籤功能
python scripts/test_trend_label_config.py
```

**測試內容**:
1. 載入配置文件
2. 生成模擬數據（上漲/橫盤/下跌）
3. 計算趨勢標籤
4. 分析各階段識別率

**驗收標準**:
- ✅ 上漲階段識別率 > 50%
- ✅ 下跌階段識別率 > 50%
- ✅ 橫盤階段持平率 > 50%

---

## 參數調整指南

### vol_multiplier 調整

**當前值**: 2.0

| 場景 | 建議值 | 效果 |
|------|-------|------|
| 趨勢識別不足 | 1.5-1.8 | 更敏感，捕捉更多趨勢 |
| 噪音過多 | 2.5-3.0 | 更保守，過濾小震盪 |
| 平衡 | 2.0 | 預設值（推薦） |

### lookforward 調整

**當前值**: 150 bars

| 場景 | 建議值 | 效果 |
|------|-------|------|
| 短期波段 | 100-120 | 1-2 小時持倉 |
| 日內波段 | 150-200 | 2-3 小時持倉（推薦） |
| 長期趨勢 | 250-300 | 3-5 小時持倉 |

---

## 文件清單

### 修改的文件

1. ✅ `src/utils/financial_engineering.py`
   - 已實現 `trend_labels_adaptive()` 函數

2. ✅ `scripts/preprocess_single_day.py`
   - 新增 `trend_labels()` 包裝函數
   - 修改 `compute_label_preview()` 支持方法選擇

3. ✅ `scripts/extract_tw_stock_data_v6.py`
   - 新增 `trend_labels()` 包裝函數
   - 修改標籤生成邏輯支持趨勢標籤

### 新增的文件

4. ✅ `configs/config_intraday_swing.yaml`
   - 日內波段交易配置

5. ✅ `scripts/test_trend_label_config.py`
   - 趨勢標籤功能測試腳本

6. ✅ `docs/TREND_LABELING_IMPLEMENTATION.md`
   - 本文檔

---

## 後續建議

### 立即可做

1. **測試不同參數組合**
   ```bash
   # 修改 config_intraday_swing.yaml 中的參數
   # 重新生成數據並比較標籤分布
   ```

2. **實際數據驗證**
   ```bash
   # 使用真實台股數據生成訓練集
   python scripts/extract_tw_stock_data_v6.py \
       --config ./configs/config_intraday_swing.yaml
   ```

3. **分析標籤質量**
   ```bash
   # 檢查生成的標籤分布是否符合預期
   type data/processed_swing/npz/normalization_meta.json
   ```

### 進階優化

1. **多時間尺度組合**
   - 實現短期 (50) + 中期 (150) + 長期 (300) 趨勢判斷
   - 只在多尺度趨勢一致時交易

2. **趨勢強度量化**
   - 不僅判斷方向，還量化趨勢強度
   - 用於風險管理和倉位調整

3. **動態參數調整**
   - 根據市場波動率自動調整 `vol_multiplier`
   - 根據時段調整 `lookforward`（開盤/尾盤不同）

---

## 常見問題

### Q1: 趨勢標籤會影響現有 Triple-Barrier 數據嗎？

**答**: 不會。兩種方法完全獨立：
- 趨勢標籤使用新配置文件 `config_intraday_swing.yaml`
- Triple-Barrier 使用原配置 `config_pro_v5_ml_optimal.yaml`
- 輸出到不同目錄

### Q2: 如何選擇使用哪種標籤？

**答**: 根據交易策略選擇：
- **日內波段**（1-2 次/天，≥1% 利潤）→ 趨勢標籤
- **高頻交易**（10-20 次/天，0.3-0.5% 利潤）→ Triple-Barrier

### Q3: 可以混合使用嗎？

**答**: 可以，有兩種方式：
1. **訓練兩個模型**: 一個用趨勢標籤，一個用 TB，根據市場狀態切換
2. **多任務學習**: 同時預測趨勢和短期波動（需要修改模型架構）

### Q4: 趨勢標籤的計算成本如何？

**答**: 比 Triple-Barrier 更快：
- TB: O(N × max_holding) - 需要搜索未來價格
- 趨勢: O(N) - 向量化計算，一次性完成

### Q5: 如何驗證趨勢標籤的效果？

**答**: 三種方式：
1. **可視化**: 使用 `label_viewer/` 工具查看標籤
2. **回測**: 根據標籤進行模擬交易
3. **訓練模型**: 看準確率和實際交易表現

---

**最後更新**: 2025-10-23
**實現者**: Claude (Anthropic)
**狀態**: ✅ 核心功能完成，可投入使用
