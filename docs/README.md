# DeepLOB-Pro 文檔索引

## 📚 文檔總覽

本目錄包含 DeepLOB-Pro 專案的完整技術文檔。

**最後更新**：2025-10-19

---

## 🎯 快速導航

### 新手入門

| 文檔 | 說明 | 優先級 |
|------|------|--------|
| [../CLAUDE.md](../CLAUDE.md) | **專案總覽**（必讀） | ⭐⭐⭐⭐⭐ |
| [../QUICK_START_VOLATILITY.md](../QUICK_START_VOLATILITY.md) | 震盪分析快速開始 | ⭐⭐⭐⭐ |
| [SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md) | SB3 實作報告 | ⭐⭐⭐⭐ |

### 數據處理

| 文檔 | 說明 | 狀態 |
|------|------|------|
| [extract_tw_stock_data_v5_usage.md](extract_tw_stock_data_v5_usage.md) | **V5 數據處理工具使用說明** ⭐ NEW | ✅ 完成 |
| [V5_Pro_NoMLFinLab_Guide.md](V5_Pro_NoMLFinLab_Guide.md) | V5 Pro 設計指南 | ✅ 完成 |
| [../VOLATILITY_ANALYSIS_GUIDE.md](../VOLATILITY_ANALYSIS_GUIDE.md) | 震盪分析完整指南 | ✅ 完成 |
| [FI2010_Dataset_Specification.md](FI2010_Dataset_Specification.md) | FI-2010 數據集規格 | ✅ 完成 |

### 模型訓練

| 文檔 | 說明 | 狀態 |
|------|------|------|
| [1.DeepLOB 台股模型訓練最終報告.md](1.DeepLOB%20台股模型訓練最終報告.md) | DeepLOB 訓練報告（準確率 72.98%） | ✅ 完成 |
| [SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md) | SB3 強化學習實作報告 | ✅ 完成 |

### 其他資源

| 文檔 | 說明 |
|------|------|
| [20251018-調參歷史.md](20251018-調參歷史.md) | 參數調優歷史記錄 |

---

## 📖 文檔詳細說明

### 1. 專案總覽

#### [CLAUDE.md](../CLAUDE.md) ⭐⭐⭐⭐⭐
**必讀文檔**

**內容**：
- 專案架構與技術棧
- DeepLOB + SB3 雙層學習架構
- 開發指導原則
- 訓練流程（階段 1-7）
- 當前開發狀態

**適合對象**：所有開發者、新成員

---

### 2. 數據處理工具

#### [extract_tw_stock_data_v5_usage.md](extract_tw_stock_data_v5_usage.md) ⭐ NEW
**V5 數據處理完整指南**

**內容**：
- V5 相較 V4 的升級說明
- 輸入/輸出格式詳解
- 基本使用方法
- 配置文件詳解
- **震盪統計功能**（新增）
- 進階使用案例
- 常見問題解決

**適合對象**：
- 需要處理台股數據的開發者
- 調整數據處理參數的研究者
- 理解 Triple-Barrier 標籤機制的學習者

**快速開始**：
```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5 \
    --config configs/config_pro_v5.yaml
```

---

#### [V5_Pro_NoMLFinLab_Guide.md](V5_Pro_NoMLFinLab_Guide.md)
**V5 Pro 設計文檔**

**內容**：
- V5 設計理念
- Triple-Barrier 標籤原理
- 波動率估計方法（EWMA/GARCH）
- 樣本權重策略

**適合對象**：深度理解 V5 設計的研究者

---

#### [VOLATILITY_ANALYSIS_GUIDE.md](../VOLATILITY_ANALYSIS_GUIDE.md)
**震盪分析完整指南**

**內容**：
- 震盪幅度 vs 漲跌幅
- 為什麼需要震盪篩選
- 階段 1-5 完整實驗流程
- 閾值決策矩陣
- 實驗對照組設計

**適合對象**：
- 進行數據篩選實驗的研究者
- 優化訓練數據質量的開發者

---

#### [QUICK_START_VOLATILITY.md](../QUICK_START_VOLATILITY.md)
**震盪分析快速開始**

**內容**：
- 5 步快速執行指南
- 統計報告解讀
- 閾值決策建議

**適合對象**：快速上手震盪分析的用戶

---

#### [FI2010_Dataset_Specification.md](FI2010_Dataset_Specification.md)
**FI-2010 數據集規格**

**內容**：
- FI-2010 原始數據格式
- 特徵說明
- 標籤定義

**適合對象**：理解原始數據結構的開發者

---

### 3. 模型訓練報告

#### [1.DeepLOB 台股模型訓練最終報告.md](1.DeepLOB%20台股模型訓練最終報告.md)
**DeepLOB 訓練成果報告**

**關鍵成果**：
- ✅ 測試準確率：**72.98%**
- ✅ F1 Score：**73.24%**
- ✅ 訓練樣本數：5,584,553
- ✅ 股票覆蓋：195 檔台股

**內容**：
- 模型架構詳解
- 訓練超參數
- 評估指標分析
- 混淆矩陣
- 下一步優化方向

---

#### [SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md)
**SB3 強化學習實作報告**

**內容**：
- 階段 3-7 實作總結
- PPO + DeepLOB 整合架構
- 環境設計（TaiwanLOBTradingEnv）
- 訓練腳本使用說明
- 評估系統

**適合對象**：進行強化學習訓練的開發者

---

### 4. 其他文檔

#### [20251018-調參歷史.md](20251018-調參歷史.md)
**參數調優記錄**

記錄了歷次參數調整的實驗結果。

---

## 🔍 按使用場景查找文檔

### 場景 1: 我是新成員，想了解專案

**閱讀順序**：
1. [CLAUDE.md](../CLAUDE.md) - 專案總覽
2. [1.DeepLOB 台股模型訓練最終報告.md](1.DeepLOB%20台股模型訓練最終報告.md) - 了解當前成果
3. [SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md) - 了解強化學習部分

---

### 場景 2: 我要處理台股數據

**閱讀順序**：
1. [extract_tw_stock_data_v5_usage.md](extract_tw_stock_data_v5_usage.md) - 使用說明 ⭐
2. [QUICK_START_VOLATILITY.md](../QUICK_START_VOLATILITY.md) - 震盪分析
3. [V5_Pro_NoMLFinLab_Guide.md](V5_Pro_NoMLFinLab_Guide.md) - 深入理解

**快速指令**：
```bash
# 查看使用說明
cat docs/extract_tw_stock_data_v5_usage.md

# 執行數據處理
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5
```

---

### 場景 3: 我要進行震盪篩選實驗

**閱讀順序**：
1. [QUICK_START_VOLATILITY.md](../QUICK_START_VOLATILITY.md) - 快速開始
2. [VOLATILITY_ANALYSIS_GUIDE.md](../VOLATILITY_ANALYSIS_GUIDE.md) - 完整指南
3. [extract_tw_stock_data_v5_usage.md](extract_tw_stock_data_v5_usage.md) - 工具使用

**快速指令**：
```bash
# 步驟 1: 快速驗證
python scripts/quick_verify_volatility.py

# 步驟 2: 執行統計
python scripts/extract_tw_stock_data_v5.py \
    --output-dir ./data/processed_v5_stats

# 步驟 3: 查看報告
cat data/processed_v5_stats/volatility_summary.json
```

---

### 場景 4: 我要訓練 DeepLOB 模型

**閱讀順序**：
1. [CLAUDE.md](../CLAUDE.md) - 階段 2 說明
2. [1.DeepLOB 台股模型訓練最終報告.md](1.DeepLOB%20台股模型訓練最終報告.md) - 參考成果

**快速指令**：
```bash
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v5/npz \
    --checkpoint-dir ./checkpoints/deeplob
```

---

### 場景 5: 我要訓練強化學習模型

**閱讀順序**：
1. [SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md) - 實作報告
2. [CLAUDE.md](../CLAUDE.md) - 階段 5-7 說明

**快速指令**：
```bash
# 完整訓練（PPO + DeepLOB）
python scripts/train_sb3_deeplob.py --timesteps 1000000
```

---

## 📝 文檔貢獻指南

### 新增文檔時

1. 在本目錄（`docs/`）創建 Markdown 文件
2. 在本文件（README.md）中更新索引
3. 在 [CLAUDE.md](../CLAUDE.md) 中更新相關章節（如需要）

### 文檔命名規範

- 使用小寫英文 + 底線：`extract_tw_stock_data_v5_usage.md`
- 中文文檔可使用中文：`調參歷史.md`
- 重要文檔加入日期前綴：`20251018-調參歷史.md`

---

## 🔗 外部資源

### 官方文檔

- [Stable-Baselines3 文檔](https://stable-baselines3.readthedocs.io/)
- [PyTorch 文檔](https://pytorch.org/docs/stable/index.html)
- [Gymnasium 文檔](https://gymnasium.farama.org/)

### 核心論文

1. **DeepLOB**: "Deep Convolutional Neural Networks for Limit Order Books" (Zhang et al., 2019)
2. **Triple-Barrier**: "Advances in Financial Machine Learning" (López de Prado, 2018)
3. **PPO**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

---

**文檔索引版本**：v1.0
**最後更新**：2025-10-19
**維護者**：DeepLOB-Pro Team
