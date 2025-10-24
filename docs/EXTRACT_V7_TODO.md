# extract_tw_stock_data_v7.py 實施任務清單（簡化版）

**版本**: v7.0.0-simplified
**創建日期**: 2025-10-23
**預計總時長**: 3-4 小時
**狀態**: ⏳ 待啟動

---

## 📋 核心理念

### ❌ V7 不做的事（已在預處理完成）

- ❌ 標籤計算（預處理已完成）
- ❌ 波動率計算（不需要，標籤已生成）
- ❌ 權重計算（預處理已完成 11 種策略）
- ❌ 數據清洗（預處理已完成）
- ❌ 參數匹配檢查（不需要重算）

### ✅ V7 只做的事

1. **讀取預處理 NPZ**：直接使用 features, labels
2. **數據選擇**：支持 dataset_selection.json 或配置過濾
3. **滑動窗口**：生成 (100, 20) 時間序列
4. **按股票劃分**：train/val/test (70/15/15)
5. **輸出 NPZ**：與 V6 格式兼容

---

## 總體進度

| Phase         | 任務數    | 預計時長      | 狀態    | 完成度    |
| ------------- | ------ | --------- | ----- | ------ |
| Phase 1: 準備工作 | 3      | 30分鐘      | ⏳ 待啟動 | 0%     |
| Phase 2: 核心實現 | 4      | 1.5小時     | ⏳ 待啟動 | 0%     |
| Phase 3: 測試驗證 | 3      | 1小時       | ⏳ 待啟動 | 0%     |
| Phase 4: 文檔更新 | 2      | 30分鐘      | ⏳ 待啟動 | 0%     |
| **總計**        | **12** | **3.5小時** | ⏳     | **0%** |

---

# 

# 

---

### 任務 1.2: 創建 V7 腳本骨架 ✅

**優先級**: P0（關鍵）
**預計時長**: 15分鐘

**步驟**:

- 複製 V6 到 V7 文件名
- 更新文件頭部文檔（版本 v7.0，簡化版）
- 移除所有標籤/波動率計算函數（ewma_vol, tb_labels, trend_labels）
- 設置版本號常量為 "7.0.0"

**核心改進點**:

1. 不重新計算標籤（直接使用 NPZ 的 labels）
2. 不重新計算波動率（不需要）
3. 不重新計算權重（直接使用 metadata 的 weight_strategies）
4. 支持 dataset_selection.json 數據選擇

**完成標準**:

- [ ] V7 腳本創建
- [ ] 文件頭部更新
- [ ] 移除不需要的函數
- [ ] 版本號設置為 7.0.0

---

### 任務 1.3: 更新全局統計變量 ✅

**優先級**: P1（重要）
**預計時長**: 10分鐘

**修改內容**:

- 保留：loaded_npz_files, symbols_passed_filter, symbols_filtered_out, valid_windows
- 移除：tb_success, data_quality_errors（不需要）
- 新增：json_filtered_files（從 dataset_selection.json 過濾的文件數）

**完成標準**:

- [ ] 全局統計更新
- [ ] 移除不需要的統計項

---

## Phase 2: 核心實現

**目標**: 實現簡化的數據處理流程
**預計時長**: 1.5小時
**依賴**: Phase 1 完成

### 任務 2.1: 實現 dataset_selection.json 讀取 ⭐⭐⭐⭐⭐

**優先級**: P0（關鍵）
**預計時長**: 20分鐘
**複雜度**: 低

**函數設計**: `read_dataset_selection_json()`

**輸入參數**:

- json_path: str - JSON 文件路徑

**返回值**:

- Dict - 包含 file_list, dates, symbols 等信息

**功能**:

1. 讀取 JSON 文件
2. 驗證必要字段存在（file_list）
3. 提取 file_list（每個元素包含 date, symbol, file_path）
4. 返回完整字典供後續使用

**完成標準**:

- [ ] 函數實現完成
- [ ] JSON 格式驗證
- [ ] 錯誤處理完善
- [ ] 日誌輸出清晰

---

### 任務 2.2: 簡化 load_preprocessed_npz() ⭐⭐⭐⭐⭐

**優先級**: P0（關鍵）
**預計時長**: 20分鐘
**複雜度**: 低

**修改目標**:

- V6 返回：(features, mids, bucket_mask, meta)
- V7 返回：(features, labels, meta)

**主要變更**:

1. 直接讀取 NPZ 的 labels 字段
2. 不再返回 mids（不需要計算標籤）
3. 不再返回 bucket_mask（不需要）
4. 強制檢查 labels 字段存在（V7 要求預處理 v2.0+）

**完成標準**:

- [ ] 修改返回值（新增 labels，移除 mids, bucket_mask）
- [ ] 添加版本檢查（強制 labels 字段存在）
- [ ] 更新 docstring

---

### 任務 2.3: 簡化 sliding_windows 函數 ⭐⭐⭐⭐⭐

**優先級**: P0（關鍵）
**預計時長**: 40分鐘
**複雜度**: 中

**重命名**: `sliding_windows_v6()` → `sliding_windows_v7()`

**V7 簡化流程**:

1. **讀取預處理數據**:
   
   - features (T, 20) - 已標準化
   - labels (T,) - 值為 {-1, 0, 1}
   - metadata - 包含 weight_strategies

2. **滑動窗口切片**:
   
   - 生成 (X, y) 窗口
   - X: (N, 100, 20)
   - y: (N,) - 轉換為 {0, 1, 2}

3. **讀取權重**（從 metadata）:
   
   - 使用配置指定的策略（如 "effective_num_0999"）
   - 從 metadata['weight_strategies'][strategy_name] 讀取

4. **按股票劃分**:
   
   - train/val/test = 70/15/15

**移除的邏輯**:

- ❌ Z-Score 標準化（預處理已完成）
- ❌ 波動率計算（不需要）
- ❌ 標籤生成（預處理已完成）
- ❌ 權重計算（預處理已完成）

**完成標準**:

- [ ] 函數重命名 v6 → v7
- [ ] 移除標籤/波動率計算邏輯
- [ ] 直接使用預處理的 labels
- [ ] 從 metadata 讀取權重
- [ ] 更新 docstring

---

### 任務 2.4: 實現數據選擇過濾 ⭐⭐⭐⭐

**優先級**: P1（重要）
**預計時長**: 30分鐘
**複雜度**: 中

**函數設計**: `filter_data_by_selection()`

**輸入參數**:

- all_data: List[Tuple] - 全部預處理數據
- config: Dict - 配置字典

**過濾邏輯**:

**優先級 1: dataset_selection.json**

- 如果配置中有 `data_selection.json_file`
- 讀取 JSON，使用 file_list 過濾

**優先級 2: 配置過濾**

- 如果沒有 JSON，使用配置過濾：
  - start_date, end_date, num_days
  - symbols
  - sample_ratio

**完成標準**:

- [ ] JSON 過濾實現（優先級最高）
- [ ] 配置過濾實現（後備方案）
- [ ] 日誌輸出清晰
- [ ] 統計信息記錄

---

## Phase 3: 測試驗證

**目標**: 確保功能正確性
**預計時長**: 1小時
**依賴**: Phase 2 完成

### 任務 3.1: 小數據集測試 ⭐⭐⭐⭐⭐

**優先級**: P0（關鍵）
**預計時長**: 20分鐘

**測試步驟**:

1. **準備測試配置**: 創建 `configs/config_v7_test.yaml`
   
   - 設置 data_selection.start_date = "20250901"
   - 設置 data_selection.num_days = 2

2. **運行 V7 腳本**

3. **檢查輸出文件**:
   
   - 驗證生成 train/val/test.npz
   - 檢查標籤分布

**完成標準**:

- [ ] 程序正常運行
- [ ] 生成 train/val/test.npz
- [ ] 標籤分布正確

---

### 任務 3.2: 對比 V6 vs V7 輸出一致性 ⭐⭐⭐⭐⭐

**優先級**: P0（關鍵）
**預計時長**: 20分鐘

**驗證邏輯**:

1. 使用相同的預處理數據
2. 使用相同的配置運行 V6 和 V7
3. 對比輸出的標籤分布（應該一致，因為都用預處理的 labels）
4. 對比樣本數量

**完成標準**:

- [ ] 標籤分布一致（誤差 < 1%）
- [ ] 樣本數量一致
- [ ] 特徵維度一致

---

### 任務 3.3: dataset_selection.json 測試 ⭐⭐⭐⭐

**優先級**: P0（關鍵）
**預計時長**: 20分鐘

**測試步驟**:

1. **使用 analyze_label_distribution.py 生成 JSON**

2. **運行 V7 使用 JSON 過濾**:
   
   - 配置 data_selection.json_file 指向 JSON

3. **驗證**:
   
   - 只處理 JSON 中指定的文件
   - 標籤分布符合 JSON 預期

**完成標準**:

- [ ] JSON 過濾正確
- [ ] 標籤分布符合預期
- [ ] 日誌輸出正確

---

## Phase 4: 文檔更新

**目標**: 更新文檔說明 V7 簡化版
**預計時長**: 30分鐘
**依賴**: Phase 3 完成

### 任務 4.1: 創建 V7 簡化版說明 ⭐⭐⭐

**優先級**: P1（重要）
**預計時長**: 20分鐘

**文件**: `docs/EXTRACT_V7_SIMPLIFIED_GUIDE.md`

**內容結構**:

**1. 簡化理念**:

- 預處理階段已完成所有數據處理
- V7 只做：讀取 → 滑窗 → 劃分 → 輸出

**2. 與 V6 的差異**:

- 不重新計算標籤
- 不重新計算波動率
- 支持 dataset_selection.json

**3. 使用方式**:

- 標準使用（配置過濾）
- 進階使用（JSON 過濾）

**4. 性能提升**:

- 處理時間減少（無需標籤計算）
- 代碼更簡單可靠

**完成標準**:

- [ ] 文檔創建完成
- [ ] 說明清晰易懂
- [ ] 包含使用範例

---

### 任務 4.2: 更新 CLAUDE.md ⭐⭐

**優先級**: P2（一般）
**預計時長**: 10分鐘

**更新文件**: `CLAUDE.md` 中的訓練流程章節

**更新內容**:

**階段二：訓練數據生成（V7 簡化版）**:

- 目標：從預處理 NPZ 直接生成訓練數據
- 核心理念：預處理已完成，V7 只做數據組織
- 新增功能：dataset_selection.json 支持

**完成標準**:

- [ ] CLAUDE.md 更新完成
- [ ] 說明 V7 簡化理念

---

## 總結

### 完成標準

- [ ] **Phase 1**: 準備工作完成
- [ ] **Phase 2**: 核心功能實現
- [ ] **Phase 3**: 測試驗證通過
- [ ] **Phase 4**: 文檔更新完成

### V7 簡化版核心優勢

**1. 代碼更簡單**:

- 移除 500+ 行標籤/波動率計算代碼
- 核心邏輯：讀取 → 滑窗 → 輸出

**2. 更可靠**:

- 不重複計算（避免不一致）
- 直接使用預處理驗證過的數據

**3. 更快**:

- 無需標籤計算（節省 50-70% 時間）
- 無需波動率計算（節省 10-15% 時間）

**4. 更靈活**:

- 支持 dataset_selection.json 精確控制數據集

### 預期交付物

1. **代碼**:
   
   - `scripts/extract_tw_stock_data_v7.py`（簡化版）

2. **配置**:
   
   - `configs/config_pro_v7_optimal.yaml`

3. **文檔**:
   
   - `docs/EXTRACT_V7_SIMPLIFIED_GUIDE.md`（新增）
   - 更新的 `CLAUDE.md`

4. **Git**:
   
   - Commit: "feat: implement extract_tw_stock_data_v7 simplified version"
   - Tag: v7.0.0

---

**文檔版本**: v7.0.0-simplified
**最後更新**: 2025-10-23
**狀態**: ✅ 計劃完成，待執行
**預計完成日期**: 2025-10-23（當天完成）
