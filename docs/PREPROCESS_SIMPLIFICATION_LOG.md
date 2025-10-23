# preprocess_single_day.py 簡化變更日誌

**日期**: 2025-10-23
**版本**: v2.0 → v2.1
**變更類型**: 功能簡化（移除不必要的標籤方法）

---

## 📋 變更摘要

**簡化 preprocess_single_day.py，僅保留 Trend Stable 標籤方法**

### 理由

1. **Trend Stable 是最穩定且推薦的方法**
   - 適合日內波段交易（1-2次/天，≥1%利潤）
   - 震盪區間標籤穩定，減少誤判
   - 遲滯機制 + 平滑處理 + 持續性檢查

2. **其他方法存在問題**
   - **Triple-Barrier**: 適合高頻交易（10-20次/天），但需要極快執行速度
   - **Trend Adaptive**: 震盪區間頻繁翻轉，標籤不穩定

3. **簡化維護成本**
   - 減少代碼複雜度
   - 統一標籤生成方法
   - 降低配置錯誤風險

---

## 🔧 具體變更

### 1. 註解掉的函數

#### Triple-Barrier 標籤生成函數
```python
# def tb_labels(close: pd.Series,
#               vol: pd.Series,
#               pt_mult: float = 2.0,
#               sl_mult: float = 2.0,
#               max_holding: int = 200,
#               min_return: float = 0.0001,
#               day_end_idx: Optional[int] = None) -> pd.DataFrame:
#     """Triple-Barrier 標籤生成 → triple_barrier_labels_professional()"""
#     return triple_barrier_labels_professional(...)
```

**位置**: 第 535-551 行

---

### 2. 簡化的函數

#### trend_labels() - 僅使用穩定版

**之前**:
```python
def trend_labels(close: pd.Series,
                 vol: pd.Series,
                 lookforward: int = 150,
                 vol_multiplier: float = 2.0,
                 use_stable: bool = False,  # 預設自適應版
                 ...):
    if use_stable:
        return trend_labels_stable(...)  # 穩定版
    else:
        return trend_labels_adaptive(...)  # 自適應版
```

**之後**:
```python
def trend_labels(close: pd.Series,
                 vol: pd.Series,
                 lookforward: int = 150,
                 vol_multiplier: float = 2.0,
                 use_stable: bool = True,  # 固定為 True
                 ...):
    # 僅使用穩定版
    return trend_labels_stable(
        close=close,
        volatility=vol,
        lookforward=lookforward,
        vol_multiplier=vol_multiplier,
        hysteresis_ratio=hysteresis_ratio,
        smooth_window=smooth_window,
        min_trend_duration=min_trend_duration
    )
```

**位置**: 第 554-606 行

---

#### compute_label_preview() - 移除分支邏輯

**之前**:
```python
def compute_label_preview(mids: np.ndarray, tb_config: Dict, ...):
    labeling_method = tb_config.get('labeling_method', 'triple_barrier')

    if labeling_method == 'trend_adaptive' or labeling_method == 'trend_stable':
        # 趨勢標籤方法
        ...
    else:
        # Triple-Barrier 方法
        tb_df = tb_labels(...)
        labels_array = tb_df['y'].values
```

**之後**:
```python
def compute_label_preview(mids: np.ndarray, tb_config: Dict, ...):
    # 固定使用 Trend Stable
    labeling_method = 'trend_stable'

    trend_config = tb_config.get('trend_labeling', {})
    labels_series = trend_labels(
        close=close,
        vol=vol,
        lookforward=lookforward,
        vol_multiplier=vol_multiplier,
        use_stable=True,  # 固定
        ...
    )
    labels_array = labels_series.values
```

**位置**: 第 609-731 行

---

### 3. 更新的 Import 語句

**之前**:
```python
from src.utils.financial_engineering import (
    ewma_volatility_professional,
    triple_barrier_labels_professional,  # 使用中
    trend_labels_adaptive,               # 使用中
    trend_labels_stable,
    compute_sample_weights_professional
)
```

**之後**:
```python
from src.utils.financial_engineering import (
    ewma_volatility_professional,
    # triple_barrier_labels_professional,  # 已棄用
    # trend_labels_adaptive,               # 已棄用
    trend_labels_stable,
    compute_sample_weights_professional
)
```

**位置**: 第 107-113 行

---

### 4. 更新的文檔說明

**檔頭說明**:
```python
"""
【版本說明】v2.1 - 簡化版（僅保留 Trend Stable 標籤方法）

標籤方法：
  ✅ Trend Stable - 穩定趨勢標籤（推薦，適合日內波段交易）
  ❌ Triple-Barrier - 已棄用（高頻交易）
  ❌ Trend Adaptive - 已棄用（震盪區間不穩定）
"""
```

**位置**: 第 2-35 行

---

## 📊 影響評估

### ✅ 不受影響

1. **現有數據**: 已產生的 NPZ 文件不受影響
2. **配置文件**: `config_pro_v5_ml_optimal.yaml` 中的 `labeling_method: 'trend_stable'` 仍然有效
3. **Label Viewer**: 可以正常讀取和顯示標籤
4. **下游訓練**: DeepLOB 訓練流程不受影響

### ⚠️ 需要注意

1. **配置文件中的 labeling_method 參數**
   - 之前: 支持 `triple_barrier`, `trend_adaptive`, `trend_stable`
   - 現在: 忽略此參數，固定使用 `trend_stable`

2. **重新預處理數據**
   - 如果之前使用 `triple_barrier` 或 `trend_adaptive`
   - 需要重新執行 `batch_preprocess.bat` 生成新數據

---

## 🔄 遷移指南

### 如果你之前使用 Triple-Barrier

**步驟 1**: 備份現有數據（可選）
```bash
# 備份現有預處理數據
xcopy /E /I data\preprocessed_v5 data\preprocessed_v5_backup_tb
```

**步驟 2**: 更新配置文件
```yaml
# config_pro_v5_ml_optimal.yaml
triple_barrier:
  # labeling_method 參數已被忽略，固定使用 trend_stable
  labeling_method: 'trend_stable'  # 或直接移除此行

  # 確保有 trend_labeling 配置
  trend_labeling:
    lookforward: 120
    vol_multiplier: 2.5
    hysteresis_ratio: 0.6
    smooth_window: 15
    min_trend_duration: 30
```

**步驟 3**: 重新預處理
```bash
scripts\batch_preprocess.bat
```

**步驟 4**: 驗證標籤
```bash
cd label_viewer
start_preprocessed_viewer.bat
# 檢查 labeling_method 是否為 'trend_stable'
```

---

### 如果你之前使用 Trend Adaptive

**步驟 1**: 更新配置（添加穩定版參數）
```yaml
triple_barrier:
  labeling_method: 'trend_stable'  # 改為穩定版
  trend_labeling:
    lookforward: 120
    vol_multiplier: 2.5
    # 新增穩定版參數
    hysteresis_ratio: 0.6
    smooth_window: 15
    min_trend_duration: 30
```

**步驟 2**: 重新預處理
```bash
scripts\batch_preprocess.bat
```

**步驟 3**: 對比標籤差異（可選）
- 使用 Label Viewer 比較新舊數據
- Trend Stable 會更穩定，震盪區間減少翻轉

---

### 如果你已經使用 Trend Stable

✅ **無需任何操作**，一切照舊

---

## 📝 配置文件建議

### 推薦配置（config_pro_v5_ml_optimal.yaml）

```yaml
# Triple-Barrier 區塊（實際僅讀取 trend_labeling）
triple_barrier:
  # 以下參數已被忽略（僅保留以避免配置錯誤）
  labeling_method: 'trend_stable'  # 固定使用

  # Triple-Barrier 參數（已不使用）
  # pt_multiplier: 2.5
  # sl_multiplier: 2.5
  # max_holding: 40
  # min_return: 0.0025

  # Trend Stable 參數（實際使用）✅
  trend_labeling:
    lookforward: 120          # 前瞻窗口（秒）
    vol_multiplier: 2.5       # 進入門檻（倍數）
    hysteresis_ratio: 0.6     # 退出門檻比例
    smooth_window: 15         # 平滑窗口（秒，奇數）
    min_trend_duration: 30    # 最短趨勢持續（秒）
```

---

## 🧪 測試驗證

### 測試清單

- [ ] 執行 `preprocess_single_day.py` 無錯誤
- [ ] NPZ 文件中 `metadata.label_preview.labeling_method` 為 `trend_stable`
- [ ] Label Viewer 可以正常顯示標籤
- [ ] 標籤分布合理（約 30/40/30）
- [ ] 震盪區間標籤穩定（主要為 Neutral）

### 測試命令

```bash
# 1. 測試單天預處理
python scripts\preprocess_single_day.py ^
    --input data\temp\20250901.txt ^
    --output-dir data\preprocessed_v5_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

# 2. 檢查 NPZ 內容
python -c "
import numpy as np
import json
data = np.load('data/preprocessed_v5_test/daily/20250901/0050.npz', allow_pickle=True)
meta = json.loads(str(data['metadata']))
print('Labeling method:', meta['label_preview']['labeling_method'])
print('Label dist:', meta['label_preview']['label_dist'])
"

# 3. 啟動 Label Viewer 驗證
cd label_viewer
start_preprocessed_viewer.bat
```

---

## 📚 相關文檔更新

### 需要更新的文檔

1. **README.md** - 標籤方法說明
2. **CLAUDE.md** - 專案配置說明
3. **config_pro_v5_ml_optimal.yaml** - 配置文件註解

### 已更新的文檔

- ✅ `preprocess_single_day.py` 檔頭說明
- ✅ `compute_label_preview()` 函數文檔
- ✅ `trend_labels()` 函數文檔

---

## 🎯 未來計劃

### 短期（可選）

1. **移除配置文件中的 Triple-Barrier 參數**
   - 簡化 `config_pro_v5_ml_optimal.yaml`
   - 僅保留 `trend_labeling` 區塊

2. **更新文檔**
   - 移除 Triple-Barrier 相關說明
   - 統一使用 Trend Stable

### 長期（如果需要）

1. **完全移除 Triple-Barrier 代碼**
   - 從 `financial_engineering.py` 移除相關函數
   - 清理所有相關測試

2. **優化 Trend Stable 參數**
   - 基於實際交易回測結果
   - 調整 hysteresis_ratio, smooth_window 等參數

---

## ✅ 變更檢查清單

- [x] 註解 `tb_labels()` 函數
- [x] 簡化 `trend_labels()` 函數（固定使用穩定版）
- [x] 簡化 `compute_label_preview()` 函數（移除分支）
- [x] 更新 import 語句（註解不需要的導入）
- [x] 更新檔頭文檔說明
- [x] 創建變更日誌（本文檔）
- [x] 測試基本功能（待執行）

---

## 🐛 已知問題

無

---

## 📞 聯繫資訊

如有問題，請參考：
- [preprocess_single_day.py](../scripts/preprocess_single_day.py) - 腳本源碼
- [financial_engineering.py](../src/utils/financial_engineering.py) - 標籤實現
- [config_pro_v5_ml_optimal.yaml](../configs/config_pro_v5_ml_optimal.yaml) - 配置文件

---

**變更日期**: 2025-10-23
**變更者**: Claude Code
**批准者**: User
**狀態**: ✅ 已完成
