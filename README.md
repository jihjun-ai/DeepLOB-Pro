# RLlib-DeepLOB: Deep Reinforcement Learning for High-Frequency Trading

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Ray 2.40](https://img.shields.io/badge/Ray-2.40-orange.svg)](https://www.ray.io/)

使用 **DeepLOB** 深度學習模型結合 **RLlib RecurrentPPO** 強化學習算法，基於 FI-2010 限價單簿數據集訓練高頻交易策略。

## 🎯 專案目標

- **價格預測準確率**: > 65% (DeepLOB)
- **策略 Sharpe Ratio**: > 2.0 (RL Agent)
- **GPU 利用率**: > 85% (RTX 5090)
- **推理延遲**: < 10ms

## 📍 專案階段劃分

> **專案分為兩大階段，需依序完成**

### **第一大階段：基礎架構建立** ✅ 已完成
包含階段一到階段三：
- ✅ 環境配置與專案結構
- ✅ 數據處理與模型架構
- ✅ 訓練腳本與工具

**狀態**：架構完成，所有核心模組已實作（含完整繁體中文註釋）

### **等待點：DeepLOB 模型訓練** ⏸️ 待執行
- ⏳ 訓練 DeepLOB 模型（目標準確率 > 65%）
- ⏳ 驗證模型性能
- ⏳ 保存最佳檢查點

### **第二大階段：強化學習訓練** 🔒 鎖定中
包含階段四到階段八（需等待 DeepLOB 完成）：
- 🔒 RL 環境整合與基礎訓練
- 🔒 評估與基準對比
- 🔒 超參數調優
- 🔒 模型優化與部署準備
- 🔒 文檔與交付

**解鎖條件**：DeepLOB 訓練完成並驗收通過

詳細階段規劃請參考 [PROJECT_PHASES.md](PROJECT_PHASES.md)

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                     FI-2010 Dataset                         │
│              (Limit Order Book Time Series)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Preprocessing                         │
│         (Z-score Normalization + Windowing)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   DeepLOB Model                             │
│    CNN (Conv1→Conv2→Conv3→Inception) + LSTM                │
│          Output: Price Movement Prediction                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              LOB Trading Environment                        │
│   Observation: LOB Features + DeepLOB Prediction            │
│   Action: {Hold, Buy, Sell}                                 │
│   Reward: PnL - Cost - Inventory Penalty                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            RecurrentPPO Agent (RLlib)                       │
│     Policy: LSTM(256) → Actor/Critic Heads                 │
│          Optimization: Proximal Policy Optimization         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Trading Strategy                             │
│         (Evaluation: Sharpe, MDD, Win Rate)                 │
└─────────────────────────────────────────────────────────────┘
```

## 📦 專案結構

```
RLlib-DeepLOB/
├── configs/              # YAML 配置文件
│   ├── deeplob_config.yaml
│   ├── rl_config.yaml
│   └── data_config.yaml
├── src/
│   ├── data/            # 數據處理模組
│   │   ├── fi2010_loader.py
│   │   └── preprocessor.py
│   ├── models/          # 模型定義
│   │   ├── deeplob.py
│   │   ├── deeplob_feature_extractor.py
│   │   └── rl_policy_network.py
│   ├── envs/            # 強化學習環境
│   │   ├── lob_trading_env.py
│   │   └── reward_shaper.py
│   ├── training/        # 訓練邏輯
│   │   ├── deeplob_trainer.py
│   │   ├── rl_trainer.py
│   │   └── hyperparameter_search.py
│   ├── callbacks/       # 訓練回調
│   │   └── training_monitor.py
│   ├── evaluation/      # 評估模組
│   │   └── evaluator.py
│   └── utils/           # 工具函數
├── scripts/             # 執行腳本
│   ├── train_deeplob.py
│   ├── train_rl.py
│   └── run_tune.py
├── tests/               # 單元測試
├── notebooks/           # Jupyter 實驗
├── data/                # 數據目錄
├── checkpoints/         # 模型檢查點
├── logs/                # 訓練日誌
└── results/             # 實驗結果
```

## 🚀 快速開始

### 1. 環境安裝

```bash
# 創建 Conda 環境
conda env create -f environment.yml
conda activate deeplob

# 驗證安裝
python src/tests/test_environment.py
```

詳細安裝指南請參考 [INSTALL_CUDA129.md](INSTALL_CUDA129.md)

### 2. 數據準備

```bash
# 下載 FI-2010 數據集
# 放置於 data/raw/

# 預處理數據
python scripts/prepare_data.py
```

### 3. 訓練 DeepLOB 模型

```bash
# 訓練價格預測模型
python scripts/train_deeplob.py --config configs/deeplob_config.yaml

# 監控訓練（另一個終端）
tensorboard --logdir logs/deeplob
```

### 4. 訓練 RL 策略

```bash
# 訓練強化學習交易策略
python scripts/train_rl.py --config configs/rl_config.yaml

# 查看 Ray Dashboard
# 訪問 http://localhost:8265
```

### 5. 評估與回測

```bash
# 評估訓練好的策略
python scripts/evaluate.py --checkpoint checkpoints/best_policy.pt

# 生成評估報告
jupyter notebook notebooks/evaluation_analysis.ipynb
```

## 📊 性能指標

| 指標 | 目標 | 當前 | 狀態 |
|------|------|------|------|
| DeepLOB 準確率 | > 65% | TBD | ⏳ |
| Sharpe Ratio | > 2.0 | TBD | ⏳ |
| 最大回撤 (MDD) | < 15% | TBD | ⏳ |
| 勝率 | > 55% | TBD | ⏳ |
| GPU 利用率 | > 85% | TBD | ⏳ |
| 推理延遲 | < 10ms | TBD | ⏳ |

## 🔬 技術細節

### DeepLOB 架構

- **輸入**: (100, 40) - 100 時間步 × 40 維 LOB 特徵
- **卷積層**:
  - Conv1: (1,2) kernel - 捕捉檔位間關係
  - Conv2/Conv3: (4,1) kernel - 捕捉時序模式
- **Inception Module**: 多尺度特徵融合
- **LSTM**: 64 hidden units
- **輸出**: 3 類別（上漲/穩定/下跌）

### RecurrentPPO 配置

```yaml
algorithm: PPO
model:
  use_lstm: true
  lstm_cell_size: 256
  max_seq_len: 100
lr: 3e-4
gamma: 0.99
lambda: 0.95
clip_param: 0.2
train_batch_size: 4096
```

### 獎勵函數

```python
reward = PnL - transaction_cost - inventory_penalty - risk_penalty
```

## 📚 文檔

### 核心文檔
- [PROJECT_PHASES.md](PROJECT_PHASES.md) - **專案階段規劃**（兩大階段說明）
- [PROGRESS.md](PROGRESS.md) - 當前進度報告
- [todolist_rllib_stock.md](todolist_rllib_stock.md) - 詳細任務清單

### 訓練指南
- [docs/DEEPLOB_TRAINING_GUIDE.md](docs/DEEPLOB_TRAINING_GUIDE.md) - **DeepLOB 訓練完整指南** ⭐

### 數據集文檔
- [docs/FI2010_Dataset_Specification.md](docs/FI2010_Dataset_Specification.md) - **FI-2010 數據集完整規格** ✅
  - 數據格式詳解（已驗證）
  - LOB 特徵排列方式
  - 標籤定義與轉換
  - 常見錯誤與陷阱
  - 完整程式碼範例

### 技術文檔
- [CLAUDE.md](CLAUDE.md) - 專案配置與開發指導
- [INSTALL_CUDA129.md](INSTALL_CUDA129.md) - CUDA 12.9 安裝指南

## 🧪 測試

```bash
# 運行所有測試
pytest tests/ -v

# 測試覆蓋率
pytest tests/ --cov=src --cov-report=html
```

## 📈 實驗追蹤

- **TensorBoard**: `tensorboard --logdir logs/`
- **Ray Dashboard**: `http://localhost:8265`
- **Jupyter Notebooks**: `notebooks/`

## 🛠️ 開發狀態

**當前階段**: 第一大階段已完成 ✅，等待 DeepLOB 訓練 ⏸️

**完成項目**：
- ✅ 完整專案架構
- ✅ 所有核心模組（7個，含完整繁體中文註釋）
- ✅ DeepLOB 訓練腳本
- ✅ RLlib RL 訓練腳本
- ✅ 環境工廠和數據整合

**下一步**：
1. 準備 FI-2010 數據集
2. 訓練 DeepLOB 模型
3. 等待訓練完成後進入第二大階段

詳細進度請參考：
- [PROJECT_PHASES.md](PROJECT_PHASES.md) - 階段規劃
- [PROGRESS.md](PROGRESS.md) - 進度報告
- [todolist_rllib_stock.md](todolist_rllib_stock.md) - 任務清單

## 📖 參考文獻

1. **DeepLOB**: Zhang et al. (2019) - "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
2. **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
3. **FI-2010**: Ntakaris et al. (2018) - "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data"

## 📝 引用

```bibtex
@misc{deeplob-rllib-2025,
  title={RLlib-DeepLOB: Deep Reinforcement Learning for High-Frequency Trading},
  author={DeepLOB Project Team},
  year={2025},
  howpublished={\url{https://github.com/yourusername/RLlib-DeepLOB}}
}
```

## 📄 授權

MIT License

## 👥 貢獻

歡迎提交 Issue 和 Pull Request！

---

**最後更新**: 2025-01-09
**Python**: 3.11 | **PyTorch**: 2.8.0+cu129 | **Ray**: 2.40.0 | **GPU**: RTX 5090
