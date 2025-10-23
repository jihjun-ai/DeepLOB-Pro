# Label Viewer - 快速參考卡片

## 🚀 快速啟動

```bash
# 啟動應用
cd label_viewer
python app_preprocessed.py

# 或使用測試腳本
scripts\test_app_preprocessed.bat

# 瀏覽器訪問
http://localhost:8051
```

---

## 📊 支援的圖表類型（8 種）

| # | 圖表名稱 | NPZ 欄位 | 用途 |
|---|---------|---------|------|
| 1 | 中間價折線圖 | `mids` | 價格走勢 + 標籤疊加 |
| 2 | LOB 特徵矩陣熱圖 | `features` | 5 檔買賣價量分布 |
| 3 | 標籤陣列圖 | `labels` | 標籤時間序列 |
| 4 | 事件數量圖 | `bucket_event_count` | 流動性評估 |
| 5 | 時間桶遮罩圖 | `bucket_mask` | 數據完整性 |
| 6 | 標籤預覽分布 | `metadata.label_preview` | 快速統計 |
| 7 | 權重策略對比 | `metadata.weight_strategies` | 11 種權重策略對比 |
| 8 | 元數據表格 | `metadata` | 完整元數據 |

---

## 🎯 常用操作

### 載入數據
1. 輸入日期目錄: `data/preprocessed_v5_1hz/daily/20250901`
2. 點擊「載入目錄」
3. 選擇股票（個股或全部股票）

### 選擇圖表
- ☑ 勾選要顯示的圖表
- ☐ 取消勾選不需要的圖表
- 默認：中間價 + 標籤預覽 + 元數據

### 圖表互動
- **縮放**: 滾輪或拖拽選區
- **懸停**: 顯示詳細資訊
- **圖例點擊**: 隱藏/顯示數據系列
- **重置**: 雙擊圖表

---

## 🔍 關鍵指標

### 數據質量
- ✅ 平均事件數 > 5
- ✅ 有效比例 > 95%
- ✅ 缺失比例 < 5%
- ✅ 填充比例 < 10%

### 標籤分布
- ✅ Down: 25-35%
- ✅ Neutral: 35-45%
- ✅ Up: 25-35%

---

## 🛠️ 故障排除

### 問題：載入失敗
```bash
# 檢查路徑
dir data\preprocessed_v5_1hz\daily\20250901\*.npz
```

### 問題：圖表不顯示
- 檢查 NPZ 是否包含該欄位
- 舊版 NPZ 可能缺少 labels, bucket_event_count 等

### 問題：瀏覽器無法訪問
```bash
# 檢查端口佔用
netstat -ano | findstr 8051
```

---

## 📚 文檔

- **完整指南**: [APP_PREPROCESSED_GUIDE.md](APP_PREPROCESSED_GUIDE.md)
- **更新報告**: [APP_PREPROCESSED_V4_UPDATE.md](../docs/APP_PREPROCESSED_V4_UPDATE.md)
- **數據規格**: [PREPROCESSED_DATA_SPECIFICATION.md](../docs/PREPROCESSED_DATA_SPECIFICATION.md)

---

## 📞 快速聯絡

- **版本**: v4.1 完整版（含權重策略）
- **最後更新**: 2025-10-23
- **狀態**: ✅ 可用
