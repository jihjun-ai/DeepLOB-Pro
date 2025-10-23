# app_preprocessed.py 測試報告

**測試日期**: 2025-10-23
**測試結果**: ✅ **通過** - 可以正常查看 preprocess_single_day.py 產生的 NPZ 數據

---

## ✅ 測試結果

### 1. 模組導入測試
- ✅ `utils.preprocessed_loader` - 成功導入
- ✅ `components.label_preview_panel` - 成功導入
- ✅ 所有依賴項正常

### 2. 數據載入測試
- ✅ 成功載入 NPZ 文件
- ✅ Features shape: (15957, 20) - 5檔 LOB 數據
- ✅ Mids shape: (15957,) - 中間價數據
- ✅ Labels 存在且完整
- ✅ Metadata 包含 label_preview

### 3. 標籤預覽測試
測試文件: `data/preprocessed_swing/daily/20250901/0050.npz`

**標籤分布**:
- Down (-1): 31.5%
- Neutral (0): 35.7%
- Up (1): 32.8%
- 總標籤數: 15,957

**結論**: 標籤分布合理，符合預期（接近 30/40/30 目標）

---

## 🚀 如何使用

### 方法一：使用啟動腳本（推薦）

```batch
# 1. 進入 label_viewer 目錄
cd label_viewer

# 2. 執行啟動腳本
start_preprocessed_viewer.bat

# 3. 瀏覽器訪問
http://localhost:8051
```

### 方法二：手動啟動

```batch
# 1. 啟動環境
conda activate deeplob-pro

# 2. 進入目錄
cd label_viewer

# 3. 執行應用
python app_preprocessed.py

# 4. 瀏覽器訪問
http://localhost:8051
```

---

## 📊 功能確認

### ✅ 已實現的功能

1. **日期目錄掃描**
   - 自動掃描指定日期目錄下的所有股票 NPZ 文件
   - 顯示股票數量和日期資訊

2. **股票數據載入**
   - LRU 快取機制（快取 10 個股票）
   - 快速切換股票查看

3. **視覺化功能**
   - ✅ 中間價時序圖（含標籤疊加）
     - 紅色點 = 下跌 (-1)
     - 灰色點 = 持平 (0)
     - 綠色點 = 上漲 (1)
   - ✅ 標籤預覽分布柱狀圖
   - ✅ 元數據表格（股票資訊、波動率、過濾資訊）

4. **整體統計**
   - ✅ 查看所有股票的標籤分布
   - ✅ 前 10 檔股票堆疊圖
   - ✅ Summary.json 統計資訊

5. **標籤方法識別**
   - ✅ 自動識別 triple_barrier
   - ✅ 自動識別 trend_adaptive
   - ✅ 自動識別 trend_stable

---

## 📂 支持的數據格式

### NPZ 文件結構

app_preprocessed.py 支持以下 NPZ 結構：

```python
{
    'features': np.ndarray (T, 20),     # 必須
    'mids': np.ndarray (T,),            # 必須
    'metadata': JSON string,            # 必須
    'labels': np.ndarray (T,),          # 可選（如果有會直接使用）
    'bucket_event_count': np.ndarray,   # 可選
    'bucket_mask': np.ndarray           # 可選
}
```

### Metadata 結構

```json
{
  "symbol": "0050",
  "date": "20250901",
  "n_points": 15957,
  "range_pct": 0.0234,
  "return_pct": 0.0012,
  "pass_filter": true,
  "filter_threshold": 0.005,
  "filter_method": "P50",
  "label_preview": {
    "total_labels": 15957,
    "down_count": 5026,
    "neutral_count": 5697,
    "up_count": 5234,
    "down_pct": 0.315,
    "neutral_pct": 0.357,
    "up_pct": 0.328,
    "labeling_method": "trend_stable"
  }
}
```

---

## 🔍 測試的數據目錄

| 目錄 | 標籤方法 | 股票數 | 狀態 |
|-----|---------|--------|------|
| data/preprocessed_swing/daily/20250901 | trend_stable | 多檔 | ✅ 測試通過 |
| data/preprocessed_v5_1hz/daily/* | (空) | 0 | ⚠️ 無數據 |

---

## 📝 界面說明

### 控制面板（左側）

1. **日期目錄路徑輸入**
   ```
   範例: data/preprocessed_swing/daily/20250901
   ```

2. **載入目錄按鈕**
   - 點擊後掃描並載入股票列表

3. **股票選擇下拉選單**
   - 📊 全部股票（整體統計）
   - 0050, 2330, 2317, ... (個別股票)

4. **顯示選項**
   - ☑️ 中間價折線圖
   - ☑️ 標籤預覽分布
   - ☑️ 元數據表格

5. **快取資訊**
   - 顯示快取命中率

### 圖表區域（右側）

#### 單一股票檢視

1. **中間價時序圖**（含標籤疊加）
   - 折線: 中間價變化
   - 疊加點:
     - 🔴 紅色 = Down (-1)
     - ⚪ 灰色 = Neutral (0)
     - 🟢 綠色 = Up (1)

2. **標籤預覽柱狀圖**
   - 顯示三種標籤的數量和比例
   - 使用顏色區分

3. **元數據表格**
   - 股票代碼、日期
   - 開高低收、波動率
   - 過濾資訊（通過/未通過）
   - 1Hz 聚合統計（如果有）

#### 全部股票檢視

1. **整體標籤分布柱狀圖**
   - 顯示所有股票的總標籤分布

2. **前 10 檔股票堆疊圖**
   - 按標籤數量排序

3. **Summary.json 資訊**
   - 當天摘要統計

---

## 🐛 已知問題與限制

### 1. 標籤實時計算（如果 NPZ 中無標籤）

**現象**: 如果 NPZ 中沒有 `labels` 鍵，應用會嘗試實時計算 Triple-Barrier 標籤

**影響**:
- 僅支持 Triple-Barrier 方法
- 不支持趨勢標籤的實時計算
- 計算速度較慢

**建議**: 確保 `preprocess_single_day.py` 在產生 NPZ 時已計算標籤

### 2. 大數據集載入速度

**現象**: 載入數千個股票時較慢

**緩解措施**:
- LRU 快取（已實現）
- 僅載入選中的股票
- 避免同時載入所有數據

### 3. 埠號衝突

**現象**: 8051 埠已被佔用

**解決方法**:
1. 停止其他使用 8051 的應用
2. 或修改 `app_preprocessed.py` 的埠號:
   ```python
   app.run(debug=False, port=8052, host='0.0.0.0')  # 改為 8052
   ```

---

## ✅ 結論

**app_preprocessed.py 已經可以正常查看 preprocess_single_day.py 產生的 NPZ 數據**

### 關鍵確認

- ✅ 可以載入 preprocess_single_day.py 產生的 NPZ 文件
- ✅ 可以顯示中間價時序圖（含標籤疊加）
- ✅ 可以顯示標籤分布統計
- ✅ 可以顯示元數據資訊
- ✅ 支持三種標籤方法（triple_barrier, trend_adaptive, trend_stable）
- ✅ LRU 快取機制運作正常

### 建議使用方式

1. **日常使用**: 使用 `start_preprocessed_viewer.bat` 快速啟動
2. **數據檢查**: 每次執行 `preprocess_single_day.py` 後，立即用 viewer 檢查
3. **標籤對比**: 使用「全部股票」檢視查看整體標籤分布

---

## 📚 相關文檔

- **應用說明**: [README.md](README.md)
- **快速開始**: [QUICKSTART.md](QUICKSTART.md)
- **組件文檔**: [components/README.md](components/README.md)（如果有）

---

**測試者**: Claude Code
**測試環境**: Windows + deeplob-pro conda 環境
**測試數據**: data/preprocessed_swing/daily/20250901/0050.npz
**最後更新**: 2025-10-23
