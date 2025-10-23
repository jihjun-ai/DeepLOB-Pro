# DeepLOB-Pro 技術文檔索引

**更新日期**: 2025-10-23
**用途**: 快速查找相關技術文檔

---

## 📚 文檔分類索引

### 🎯 【必讀】核心文檔

#### 1. 專業金融工程套件
- **[PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md)**
  - **用途**: 專業金融工程套件遷移指南
  - **包含**: 性能對比、API 使用、測試結果
  - **適用**: 了解本專案使用的專業套件（pandas、numpy、sklearn）
  - **相關腳本**: `preprocess_single_day.py`, `src/utils/financial_engineering.py`

#### 2. 數據處理流程
- **[V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md)**
  - **用途**: V6 雙階段資料處理流程指南
  - **包含**: 階段1預處理 + 階段2訓練數據生成
  - **適用**: 了解完整數據處理流程
  - **相關腳本**: `preprocess_single_day.py`, `extract_tw_stock_data_v6.py`

#### 3. 數據格式規範
- **[PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md)**
  - **用途**: 預處理數據格式規範
  - **包含**: NPZ 檔案結構、metadata 說明
  - **適用**: 了解預處理後的數據格式
  - **相關**: `data/preprocessed_v5/daily/{date}/{symbol}.npz`

---

### 📖 【參考】進階文檔

#### 4. 標籤處理
- **[LABEL_PREVIEW_GUIDE.md](LABEL_PREVIEW_GUIDE.md)**
  - **用途**: 標籤預覽功能使用指南
  - **包含**: Triple-Barrier 參數調整、標籤分布查看
  - **適用**: 調整標籤參數、查看標籤分布
  - **相關工具**: `label_viewer/app_preprocessed.py`

- **[PREPROCESSED_DATA_QUICK_REF.md](PREPROCESSED_DATA_QUICK_REF.md)**
  - **用途**: 預處理數據快速參考
  - **包含**: 常見問題、快速查詢
  - **適用**: 快速查找預處理數據相關資訊

#### 5. 技術規範
- **[V5_Pro_NoMLFinLab_Guide.md](V5_Pro_NoMLFinLab_Guide.md)**
  - **用途**: V5 專業版技術規範（原始設計文檔）
  - **包含**: 完整的技術架構、設計理念
  - **適用**: 了解系統設計原理
  - **歷史**: 早期設計文檔，部分已被專業套件替代

- **[V5_Technical_Specification.md](V5_Technical_Specification.md)**
  - **用途**: V5 技術規格書
  - **包含**: 詳細的技術參數、公式推導
  - **適用**: 深入了解技術細節

#### 6. 權重策略
- **[WEIGHT_STRATEGY_IMPLEMENTATION_SUMMARY.md](WEIGHT_STRATEGY_IMPLEMENTATION_SUMMARY.md)**
  - **用途**: 權重策略實現總結
  - **包含**: 多種權重策略的實現與對比
  - **適用**: 了解樣本權重計算方法

- **[WEIGHT_STRATEGY_ANALYSIS.md](WEIGHT_STRATEGY_ANALYSIS.md)**
  - **用途**: 權重策略分析報告
  - **包含**: 各種權重策略的效果分析
  - **適用**: 選擇合適的權重策略

- **[WHY_WEIGHT_STILL_MATTERS.md](WHY_WEIGHT_STILL_MATTERS.md)**
  - **用途**: 為何權重仍然重要
  - **包含**: 權重在不平衡數據中的作用
  - **適用**: 理解權重的重要性

---

### 🔧 【工具】使用指南

#### 7. 數據視覺化
- **標籤查看器**: `label_viewer/app_preprocessed.py`
  - **用途**: 預處理數據視覺化工具
  - **文檔**: 見 `label_viewer/` 目錄
  - **適用**: 查看預處理後的標籤分布、價格走勢

#### 8. 快速測試
- **[PREPROCESSED_DATA_QUICK_REF.md](PREPROCESSED_DATA_QUICK_REF.md)**
  - **用途**: 快速測試與驗證
  - **包含**: 常用命令、快速檢查腳本
  - **適用**: 日常開發與測試

---

### 📊 【配置】參數說明

#### 9. 配置文件
- **[Triple-Barrier-調參歷史.md](Triple-Barrier-調參歷史.md)**
  - **用途**: Triple-Barrier 參數調整歷史
  - **包含**: 歷次調參記錄與效果
  - **適用**: 了解參數調整經驗

- **[20251018-調參歷史.md](20251018-調參歷史.md)**
  - **用途**: 完整調參歷史記錄
  - **包含**: 所有調參實驗與結果
  - **適用**: 參考歷史調參經驗

---

### 🐛 【故障排除】問題診斷

#### 10. 常見問題
- **[BUG_FIX_MIDS_ZERO.md](BUG_FIX_MIDS_ZERO.md)**
  - **用途**: mids=0 問題修復
  - **包含**: 問題原因、解決方案
  - **適用**: 遇到 mids=0 錯誤時查閱

- **[DATA_QUALITY_IMPROVEMENT_GUIDE.md](DATA_QUALITY_IMPROVEMENT_GUIDE.md)**
  - **用途**: 數據質量改進指南
  - **包含**: 數據清洗、質量檢查
  - **適用**: 提升數據質量

- **[深度診斷報告：batch_preprocess.bat + extract_tw_stock_data_v6.py.md](深度診斷報告：batch_preprocess.bat + extract_tw_stock_data_v6.py.md)**
  - **用途**: 批次預處理問題診斷
  - **包含**: 詳細的問題分析與解決方案
  - **適用**: 批次處理遇到問題時查閱

---

## 🗂️ 按使用場景分類

### 場景 1: 第一次使用專案
**推薦閱讀順序**:
1. [V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md) - 了解整體流程
2. [PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md) - 了解使用的套件
3. [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md) - 了解數據格式

### 場景 2: 調整 Triple-Barrier 參數
**推薦閱讀順序**:
1. [LABEL_PREVIEW_GUIDE.md](LABEL_PREVIEW_GUIDE.md) - 了解標籤預覽功能
2. [Triple-Barrier-調參歷史.md](Triple-Barrier-調參歷史.md) - 參考歷史經驗
3. 使用 `label_viewer/app_preprocessed.py` 查看效果

### 場景 3: 遇到數據質量問題
**推薦閱讀順序**:
1. [DATA_QUALITY_IMPROVEMENT_GUIDE.md](DATA_QUALITY_IMPROVEMENT_GUIDE.md) - 了解質量檢查
2. [BUG_FIX_MIDS_ZERO.md](BUG_FIX_MIDS_ZERO.md) - 常見問題修復
3. [深度診斷報告：batch_preprocess.bat + extract_tw_stock_data_v6.py.md](深度診斷報告：batch_preprocess.bat + extract_tw_stock_data_v6.py.md) - 詳細診斷

### 場景 4: 優化性能
**推薦閱讀順序**:
1. [PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md) - 了解性能優化
2. [V5_Pro_NoMLFinLab_Guide.md](V5_Pro_NoMLFinLab_Guide.md) - 了解架構設計

### 場景 5: 理解權重策略
**推薦閱讀順序**:
1. [WHY_WEIGHT_STILL_MATTERS.md](WHY_WEIGHT_STILL_MATTERS.md) - 了解重要性
2. [WEIGHT_STRATEGY_IMPLEMENTATION_SUMMARY.md](WEIGHT_STRATEGY_IMPLEMENTATION_SUMMARY.md) - 了解實現
3. [WEIGHT_STRATEGY_ANALYSIS.md](WEIGHT_STRATEGY_ANALYSIS.md) - 了解效果

---

## 📁 腳本與文檔對應關係

| 腳本 | 主要文檔 | 輔助文檔 |
|------|---------|---------|
| `scripts/preprocess_single_day.py` | [V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md)<br>[PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md) | [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md) |
| `scripts/extract_tw_stock_data_v6.py` | [V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md) | [PREPROCESSED_DATA_QUICK_REF.md](PREPROCESSED_DATA_QUICK_REF.md) |
| `src/utils/financial_engineering.py` | [PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md) | [V5_Pro_NoMLFinLab_Guide.md](V5_Pro_NoMLFinLab_Guide.md) |
| `label_viewer/app_preprocessed.py` | [LABEL_PREVIEW_GUIDE.md](LABEL_PREVIEW_GUIDE.md) | - |
| `scripts/batch_preprocess.bat` | [V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md) | [深度診斷報告](深度診斷報告：batch_preprocess.bat + extract_tw_stock_data_v6.py.md) |

---

## 🔍 快速查找

### 關鍵字索引

- **EWMA**: [PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md)
- **Triple-Barrier**: [LABEL_PREVIEW_GUIDE.md](LABEL_PREVIEW_GUIDE.md), [Triple-Barrier-調參歷史.md](Triple-Barrier-調參歷史.md)
- **權重策略**: [WEIGHT_STRATEGY_IMPLEMENTATION_SUMMARY.md](WEIGHT_STRATEGY_IMPLEMENTATION_SUMMARY.md)
- **數據格式**: [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md)
- **性能優化**: [PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md)
- **數據質量**: [DATA_QUALITY_IMPROVEMENT_GUIDE.md](DATA_QUALITY_IMPROVEMENT_GUIDE.md)
- **mids=0 錯誤**: [BUG_FIX_MIDS_ZERO.md](BUG_FIX_MIDS_ZERO.md)

---

## 📝 文檔更新記錄

| 日期 | 文檔 | 變更 |
|------|------|------|
| 2025-10-23 | [PROFESSIONAL_PACKAGES_MIGRATION.md](PROFESSIONAL_PACKAGES_MIGRATION.md) | 新增專業套件遷移指南 |
| 2025-10-23 | [README_DOCS_INDEX.md](README_DOCS_INDEX.md) | 新增文檔索引（本文件） |
| 2025-10-21 | [V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md) | V6 雙階段流程指南 |

---

**最後更新**: 2025-10-23
**維護者**: DeepLOB-Pro Team
**建議**: 將本文件加入書籤，方便日後快速查閱
