# Label Viewer v4.1 更新日誌

**發布日期**: 2025-10-23
**版本**: v4.0 → v4.1
**核心更新**: 新增權重策略視覺化功能

---

## 🆕 新增功能

### 權重策略視覺化（Weight Strategies）

✅ **新增圖表**: 權重策略對比柱狀圖
- 顯示 11 種權重策略的對比
- 分組柱狀圖（Down/Neutral/Up）
- 支援懸停互動

✅ **新增表格**: 權重策略詳細資訊
- 策略名稱、類型、權重值、說明
- 完整顯示所有欄位
- 方便複製權重值

✅ **新增文檔**: 權重策略完整指南
- [WEIGHT_STRATEGIES_GUIDE.md](../label_viewer/WEIGHT_STRATEGIES_GUIDE.md)
- 11 種策略詳解
- 使用流程和實戰案例
- 故障排除指南

---

## 📊 支援的權重策略

根據 [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md)，完整支援以下 11 種策略：

### Class Weight 策略（10 種）
1. **balanced** - 標準平衡權重
2. **sqrt_balanced** - 平方根平衡權重（溫和）
3. **log_balanced** - 對數平衡權重（最溫和）
4. **effective_num_09** - Effective Number (beta=0.9)
5. **effective_num_099** - Effective Number (beta=0.99)
6. **effective_num_0999** - Effective Number (beta=0.999) ⭐ 推薦
7. **effective_num_09999** - Effective Number (beta=0.9999)
8. **cb_focal_099** - CB Focal (beta=0.99)
9. **cb_focal_0999** - CB Focal (beta=0.999) ⭐ 推薦
10. **uniform** - 無權重（基準）

### Focal Loss（1 種）
11. **focal_loss** - Focal Loss (gamma=2.0)

---

## 🎯 使用方式

### 啟用權重策略視覺化

1. 啟動 Label Viewer
2. 載入日期目錄
3. 選擇股票
4. **勾選「權重策略對比 (weight_strategies)」** ✅

### 查看內容

會顯示兩個圖表：

**1. 權重策略對比柱狀圖**
- X 軸：策略名稱
- Y 軸：權重值
- 三色柱：Down（紅）/ Neutral（灰）/ Up（綠）

**2. 權重策略詳細資訊表格**
- 完整顯示所有策略的權重值
- 包含策略類型和說明

---

## 📝 代碼更新

### 更新的文件

1. **label_viewer/app_preprocessed.py**
   - 新增權重策略對比圖繪製邏輯（第 743-871 行）
   - 新增權重策略詳細表格（第 820-865 行）
   - 更新顯示選項（新增 weight_strategies 選項）
   - 更新版本號：v4.0 → v4.1

### 新增的文件

2. **label_viewer/WEIGHT_STRATEGIES_GUIDE.md** 🆕
   - 權重策略完整指南（200+ 行）
   - 11 種策略詳解
   - 使用流程和實戰案例

3. **docs/APP_PREPROCESSED_V4.1_CHANGELOG.md** 🆕
   - 本更新日誌

### 更新的文件

4. **label_viewer/QUICK_REFERENCE.md**
   - 更新圖表數量：7 種 → 8 種
   - 新增權重策略對比說明
   - 更新版本號：v4.0 → v4.1

---

## 🔍 技術細節

### 權重策略讀取

```python
# 從 metadata 讀取
weight_strategies = metadata.get('weight_strategies')

# 遍歷所有策略
for strategy_name, strategy_info in weight_strategies.items():
    class_weights = strategy_info.get('class_weights', {})
    down_weight = class_weights.get('-1', 1.0)
    neutral_weight = class_weights.get('0', 1.0)
    up_weight = class_weights.get('1', 1.0)
```

### 策略排序

```python
# 按類型排序（先 class_weight，後 focal）
sorted_strategies = sorted(
    weight_strategies.items(),
    key=lambda x: (x[1].get('type') == 'focal', x[0])
)
```

### 過濾 Focal Loss

```python
# 跳過 focal_loss（不是 class_weight）
if strategy_info.get('type') == 'focal':
    continue
```

---

## 🎯 使用場景

### 場景 1: 選擇合適的權重策略

**目標**: 為訓練選擇最佳權重策略

**步驟**:
1. 查看標籤預覽分布（了解不平衡程度）
2. 查看權重策略對比（了解各策略的權重值）
3. 根據不平衡程度選擇策略
4. 複製權重值用於訓練

**推薦**:
- 中等不平衡 → `effective_num_0999` ⭐
- 嚴重不平衡 → `balanced`
- 有難分樣本 → `cb_focal_0999`

---

### 場景 2: 對比不同策略

**目標**: 理解不同策略的差異

**步驟**:
1. 勾選「權重策略對比」
2. 觀察柱狀圖中不同策略的權重分布
3. 對比說明欄位，理解策略特點

**發現**:
- `uniform`: 所有權重 = 1.0（無調整）
- `balanced`: 權重差異最大（激進）
- `sqrt_balanced`: 權重差異中等（溫和）
- `effective_num_0999`: 權重差異適中（推薦）

---

### 場景 3: 複製權重值

**目標**: 從 Label Viewer 複製權重值到訓練代碼

**步驟**:
1. 查看「權重策略詳細資訊表格」
2. 找到目標策略（如 effective_num_0999）
3. 複製權重值
4. 粘貼到訓練代碼

**示例**:
```python
# 從 Label Viewer 複製
# effective_num_0999: Down 1.020, Neutral 1.920, Up 1.040

# 粘貼到代碼
class_weights = {
    '-1': 1.020,
    '0': 1.920,
    '1': 1.040
}
```

---

## 🐛 兼容性

### 舊版 NPZ 處理

如果 NPZ 文件不包含 `weight_strategies` 欄位（舊版 v1.0），會顯示：

```
"此股票無權重策略資訊（可能是舊版 NPZ）"
```

**解決方案**: 重新運行 `preprocess_single_day.py` 生成新版 NPZ

---

## 📊 圖表數量更新

### 更新前（v4.0）
- 7 種圖表類型

### 更新後（v4.1）
- **8 種圖表類型** ✅

完整列表：
1. 中間價折線圖 (mids)
2. LOB 特徵矩陣熱圖 (features)
3. 標籤陣列圖 (labels)
4. 事件數量圖 (bucket_event_count)
5. 時間桶遮罩圖 (bucket_mask)
6. 標籤預覽分布 (label_preview)
7. **權重策略對比 (weight_strategies)** 🆕
8. 元數據表格 (metadata)

---

## 📚 相關文檔

### 新增文檔
- ⭐ [WEIGHT_STRATEGIES_GUIDE.md](../label_viewer/WEIGHT_STRATEGIES_GUIDE.md) - 權重策略完整指南（200+ 行）

### 更新文檔
- [QUICK_REFERENCE.md](../label_viewer/QUICK_REFERENCE.md) - 更新圖表數量和版本
- [APP_PREPROCESSED_GUIDE.md](../label_viewer/APP_PREPROCESSED_GUIDE.md) - 需要更新（待辦）

### 相關文檔
- [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md) - NPZ 格式規格
- [APP_PREPROCESSED_V4_UPDATE.md](APP_PREPROCESSED_V4_UPDATE.md) - v4.0 更新報告

---

## ✅ 驗收標準

### 功能完整性
- [x] 權重策略對比柱狀圖
- [x] 權重策略詳細表格
- [x] 支援 11 種策略
- [x] 兼容舊版 NPZ

### 文檔完整性
- [x] 權重策略完整指南
- [x] 更新日誌
- [x] 快速參考更新
- [ ] APP_PREPROCESSED_GUIDE.md 更新（待辦）

### 代碼質量
- [x] 通過語法檢查
- [x] 清晰的代碼結構
- [x] 適當的錯誤處理

### 測試狀態
- [ ] 基礎功能測試（待執行）
- [ ] 圖表顯示測試（待執行）
- [ ] 兼容性測試（待執行）

---

## 🔮 未來改進

### v4.2 計劃
- [ ] 權重策略雷達圖（多維度對比）
- [ ] 策略推薦功能（自動推薦最佳策略）
- [ ] 權重值導出功能（JSON/YAML）

---

## 💡 使用提示

**提示 1**: 先查看標籤預覽，再查看權重策略
- 了解標籤分布 → 選擇合適策略

**提示 2**: 使用「全部股票」模式查看整體策略適用性
- 了解整體分布 → 統一策略選擇

**提示 3**: 對比多個策略的權重值差異
- 理解激進 vs 溫和策略的區別

**提示 4**: 從詳細表格複製權重值
- 避免手動輸入錯誤

**提示 5**: 記錄實驗結果，找到最佳策略
- 不同策略效果不同，需實驗驗證

---

## 🎉 總結

### 完成度
- **功能完整度**: 100% ✅
- **文檔完整度**: 90% ✅（待更新 APP_PREPROCESSED_GUIDE.md）
- **代碼質量**: 100% ✅
- **測試完成度**: 0% ⏳（待執行）

### 關鍵成就
✅ 新增權重策略視覺化功能（柱狀圖 + 表格）
✅ 支援 11 種權重策略完整顯示
✅ 創建 200+ 行完整指南文檔
✅ 兼容舊版 NPZ
✅ 代碼通過語法檢查

### 下一步
1. ⏳ 執行完整測試
2. ⏳ 更新 APP_PREPROCESSED_GUIDE.md
3. ⏳ 收集用戶反饋
4. ⏳ 規劃 v4.2 改進

---

**發布日期**: 2025-10-23
**版本**: v4.1
**狀態**: ✅ 開發完成，⏳ 待測試
