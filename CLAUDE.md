# SB3-DeepLOB 專案配置

## 專案環境

```bash
conda activate deeplob-pro
```

## 專案概述

這是一個結合 **DeepLOB** 深度學習模型與 **Stable-Baselines3** 強化學習框架的**高頻交易系統**專案。專案目標是實現基於台股 LOB (Limit Order Book) 數據的實際高頻交易策略。

系統使用雙層學習架構：

- **第一層**: DeepLOB CNN-LSTM 模型學習價格變動預測 (已達成 72.98% 準確率)
- **第二層**: PPO (Stable-Baselines3) 強化學習算法學習最優交易策略 (待實作)

**專案最終目標**: 實現台股高頻交易 (High-Frequency Trading for Taiwan Stocks)

**目標硬體**: NVIDIA RTX 5090 (32GB VRAM) + CUDA 12.9
**核心技術**: PyTorch 2.5 + Stable-Baselines3 + Gymnasium + Taiwan Stock LOB Data
**完成標準**:

- ✅ DeepLOB 價格預測準確率 > 65% (已達成 72.98%)
- ⏳ 強化學習策略 Sharpe Ratio > 2.0 (待訓練驗證)
- ⏳ GPU 利用率 > 85% (待訓練驗證)

## 為什麼選擇 Stable-Baselines3？

### 從 RLlib 遷移的原因

1. **簡單性** ⭐⭐⭐⭐⭐
   - 5 行代碼就能訓練
   - API 清晰直觀
   - 學習曲線平緩

2. **穩定性** ⭐⭐⭐⭐⭐
   - LSTM + PPO 完全支持
   - 無 API 兼容性問題
   - 算法實現可靠

3. **調試友好** ⭐⭐⭐⭐⭐
   - 錯誤訊息清晰
   - 邏輯透明
   - 社群活躍

4. **適合單機訓練** ⭐⭐⭐⭐⭐
   - RTX 5090 單卡足夠
   - 無需分散式配置
   - GPU 利用率高

### 對比 RLlib

| 特性 | RLlib | Stable-Baselines3 |
|------|-------|-------------------|
| 簡單性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| LSTM 支持 | ⭐⭐⭐ (有問題) | ⭐⭐⭐⭐⭐ |
| 調試難度 | ⚠️ 困難 | ✅ 容易 |
| 文檔質量 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 社群支持 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 單機性能 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 分散式訓練 | ⭐⭐⭐⭐⭐ | ⭐⭐ (需額外設置) |

---

## 專案結構

```
SB3-DeepLOB/
├── configs/              # 配置文件
│   ├── deeplob_config.yaml
│   ├── sb3_config.yaml   # SB3 配置
│   └── data_config.yaml
├── src/
│   ├── data/            # 數據處理模組
│   │   ├── fi2010_loader.py
│   │   └── preprocessor.py
│   ├── models/          # 模型定義
│   │   ├── deeplob.py
│   │   ├── deeplob_feature_extractor.py  # ✅ SB3 特徵提取器
│   │   └── sb3_lstm_policy.py            # ⏳ 待實作 (可選)
│   ├── envs/            # 強化學習環境
│   │   ├── tw_lob_trading_env.py         # ✅ 台股交易環境
│   │   ├── reward_shaper.py              # ✅ 獎勵塑形模組
│   │   └── __init__.py
│   ├── training/        # 訓練邏輯
│   │   ├── deeplob_trainer.py
│   │   └── sb3_trainer.py                # ⏳ 待實作
│   ├── callbacks/       # 訓練回調
│   │   └── sb3_callbacks.py              # ⏳ 待實作
│   ├── evaluation/      # 評估模組
│   │   ├── evaluator.py
│   │   └── baseline_strategies.py
│   └── utils/           # 工具函數
├── data/
│   ├── raw/             # FI-2010 原始數據
│   ├── processed/       # 預處理後數據
│   └── cache/
├── scripts/             # 執行腳本
│   ├── extract_tw_stock_data_v3.py     # ✅ 台股數據抽取
│   ├── train_deeplob_generic.py        # ✅ DeepLOB 訓練
│   ├── verify_env.py                   # ✅ 環境驗證
│   ├── train_sb3.py                    # ✅ SB3 基礎訓練
│   ├── train_sb3_deeplob.py            # ✅ SB3+DeepLOB 完整訓練
│   └── evaluate_sb3.py                 # ✅ 評估腳本
├── tests/               # 單元測試
├── notebooks/           # Jupyter 實驗筆記
├── checkpoints/         # 模型檢查點
│   ├── deeplob/         # DeepLOB 檢查點
│   └── sb3/             # SB3 檢查點
├── logs/                # 訓練日誌
│   ├── deeplob/
│   └── sb3/
└── results/             # 實驗結果
```

---

## 核心技術要點

### 1. DeepLOB 模型架構（台股版本）⭐

- **輸入**: (batch, 100, 20) - 100 時間步 × 20 維 LOB 特徵（5檔 LOB）
- **卷積層**:
  - Conv1: 24 filters - 捕捉檔位間關係
  - Conv2: 24 filters - 捕捉時序模式
  - Conv3: 24 filters - 深層特徵提取
- **LSTM**: 64 hidden units, 單層 - 長期時序建模
- **全連接層**: 64 hidden units
- **Dropout**: 0.7 - 強正則化（防止過擬合）
- **輸出**: 3 類別（價格下跌/持平/上漲）預測概率
- **參數量**: ~250K（輕量級）

**訓練成果**:

- ✅ 測試準確率: **72.98%** (1,878,358 樣本)
- ✅ F1 Score: **73.24%** (Macro)
- ✅ 持平類召回率: **93.7%** (避免橫盤誤交易)
- ✅ 上漲類精確率: **75.6%** (買入信號可靠)
- ✅ 超越 FI-2010 基準: **+12.98%**

**數據規模**:

- 訓練集: 5,584,553 樣本
- 驗證集: 828,011 樣本
- 測試集: 1,878,358 樣本
- 股票覆蓋: 195 檔台股

**參考文檔**: [docs/1.DeepLOB 台股模型訓練最終報告.md](docs/1.DeepLOB 台股模型訓練最終報告.md)

### 2. 強化學習環境（台股 LOB Trading Env）⭐

**環境類別**: `TaiwanLOBTradingEnv` (專為台股 5檔 LOB 設計)

- **觀測空間**: **28維** (LOB 20 + DeepLOB 3 + 狀態 5)

  - LOB 原始特徵 (20維): 5檔買賣價量
  - DeepLOB 預測 (3維): 下跌/持平/上漲概率
  - 交易狀態 (5維): 標準化持倉、庫存、成本、時間、上次動作

- **動作空間**: Discrete(3) - {0: Hold, 1: Buy, 2: Sell}

- **獎勵函數**（多組件設計）:

  - ✅ 基礎 PnL 獎勵: 價格變動 × 持倉
  - ✅ 交易成本懲罰: -0.1% 手續費
  - ✅ 庫存懲罰: 避免長時間持倉
  - ✅ 風險調整項: 波動率 × 持倉懲罰

- **數據來源**: `stock_embedding_*.npz` (台股預處理數據)

- **支援模式**: train/val/test 三種數據切換

- **Episode 長度**: 500 步（可配置）

**文件位置**: [src/envs/tw_lob_trading_env.py](src/envs/tw_lob_trading_env.py)

### 3. Stable-Baselines3 PPO 配置⭐

**基礎配置**:

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpLstmPolicy",  # 內建 LSTM 支持
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/sb3/",
    device="cuda"
)
```

**整合 DeepLOB**:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DeepLOBExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.deeplob = load_pretrained_deeplob()
        self.deeplob.eval()
        for param in self.deeplob.parameters():
            param.requires_grad = False

policy_kwargs = dict(
    features_extractor_class=DeepLOBExtractor,
    lstm_hidden_size=256,
    n_lstm_layers=1,
)

model = PPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
```

### 4. RTX 5090 優化策略

- 混合精度訓練 (FP16/TF32)
- 大 batch size 充分利用 32GB 顯存
- 向量化環境並行採樣
- TensorBoard 實時監控

---

## 開發指導原則

### 編碼規範

1. **命名規範**:
   - 類別使用 PascalCase: `DeepLOBModel`, `LOBTradingEnv`
   - 函數使用 snake_case: `load_data()`, `calculate_reward()`
   - 常數使用 UPPER_CASE: `MAX_POSITION`, `TRANSACTION_COST`

2. **文檔字串**:
   - 所有公開類別和函數必須有 docstring
   - 使用 Google 風格或 NumPy 風格
   - 包含參數說明、返回值、範例用法

3. **類型提示**:
   - 使用 Python 3.11+ 的 typing 模組
   - 函數參數和返回值加上型別標註
   - 複雜類型使用 TypeAlias

---

## 訓練流程（使用 Stable-Baselines3）

### 階段一：數據預處理與驗證（V5 → V6 演進）

#### V5 單階段處理 ✅ (已完成，保留)

**腳本**: `scripts/extract_tw_stock_data_v5.py`

- 載入原始 TXT
- 標準化（Z-score）
- Triple-Barrier 標籤生成
- 創建時間序列窗口（100 timesteps）
- 時間序列劃分（70/15/15）

**限制**：
- ❌ 固定過濾閾值
- ❌ 調整參數需重跑全部（45 分鐘）
- ❌ 標籤分布不穩定

#### V6 雙階段處理 ✅ (最新，推薦) ⭐⭐⭐⭐⭐

**文檔**: [docs/V6_TWO_STAGE_PIPELINE_GUIDE.md](docs/V6_TWO_STAGE_PIPELINE_GUIDE.md)

**階段1**: 預處理（逐檔處理，動態過濾）
```bash
# 批次預處理所有歷史數據
conda activate deeplob-pro
scripts\batch_preprocess.bat
```

**腳本**: `scripts/preprocess_single_day.py`
- 讀取單一 TXT → 清洗 → 聚合
- **動態決定當天過濾閾值**（基於目標標籤分布 30/40/30）
- 保存中間格式 NPZ
- 生成每日摘要報告

**階段2**: 訓練數據生成（快速參數測試）
```bash
# 從預處理 NPZ 生成訓練數據（僅需 5-10 分鐘）
python scripts\extract_tw_stock_data_v6.py \
    --preprocessed-dir .\data\preprocessed_v5 \
    --output-dir .\data\processed_v6 \
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

**腳本**: `scripts/extract_tw_stock_data_v6.py`
- 讀取預處理 NPZ → Z-Score → 波動率 → **標籤生成**
- **支持兩種標籤方法**: Triple-Barrier（高頻）/ 趨勢標籤（波段）
- 滑窗生成樣本
- 按股票切分 70/15/15

**標籤方法選擇** ⭐⭐⭐ (2025-10-23 新增):
- **Triple-Barrier**: 適合超短線/高頻交易（10-20次/天，0.3-0.5%利潤）
- **趨勢標籤**: 適合日內波段交易（1-2次/天，≥1%利潤）
- 詳見: [docs/TREND_LABELING_IMPLEMENTATION.md](docs/TREND_LABELING_IMPLEMENTATION.md)

**核心改進**：
- ✅ **動態過濾**: 每天自動調整閾值（維持目標分布）
- ✅ **效率提升**: 參數調整快 **82%** (45 min → 8 min)
- ✅ **增量處理**: 新增一天僅需 4 分鐘
- ✅ **穩定標籤**: Down 30% / Neutral 40% / Up 30%
- ✅ **完全兼容**: 輸出格式與 V5 相同

**快速開始**：
```bash
# 步驟 1: 預處理（首次，30 分鐘）
conda activate deeplob-pro
scripts\batch_preprocess.bat

# 步驟 2: 生成訓練數據（8 分鐘）
python scripts\extract_tw_stock_data_v6.py \
    --preprocessed-dir .\data\preprocessed_v5 \
    --output-dir .\data\processed_v6 \
    --config .\configs\config_pro_v5_ml_optimal.yaml

# 步驟 3: 檢查標籤分布
type data\processed_v6\npz\normalization_meta.json
```

**測試不同參數**（僅需修改配置，無需重跑階段1）：
```bash
# 修改 config 後直接重新執行階段2（5-10 分鐘）
python scripts\extract_tw_stock_data_v6.py \
    --preprocessed-dir .\data\preprocessed_v5 \
    --output-dir .\data\processed_v6_test \
    --config .\configs\config_test.yaml
```

### 階段二：DeepLOB 獨立訓練 ✅ (已完成)

- Batch size: 128
- Epochs: 50-100
- 使用 ReduceLROnPlateau 學習率調整
- Early stopping (patience=10)
- **結果**: 測試集準確率 72.98%

### 階段三：環境驗證 ✅ (已完成)

**目標**: 驗證環境符合 Gymnasium 標準

**新增文件**: `scripts/verify_env.py`

**驗證結果**: 🎉 完全通過

```bash
conda activate sb3-stock
python scripts/verify_env.py
```

**驗證項目**:
- ✅ 環境創建成功
- ✅ 觀測空間和動作空間正確 (28 維觀測, 3 動作)
- ✅ reset() 方法正常
- ✅ step() 方法正常
- ✅ 完整 episode 運行成功
- ✅ SB3 check_env() 驗證通過

### 階段四：SB3 基礎訓練 ✅ (已完成)

**目標**: 實現基礎 PPO 訓練

**新增文件**: `scripts/train_sb3.py`

**使用範例**:
```bash
# 快速測試（10K steps）
python scripts/train_sb3.py --timesteps 10000 --test

# 完整訓練（500K steps）
python scripts/train_sb3.py --timesteps 500000

# 監控訓練
tensorboard --logdir logs/sb3/
```

**功能特色**:
- MlpPolicy（簡化訓練）
- 向量化環境支持
- CheckpointCallback + EvalCallback
- TensorBoard 日誌
- 可配置超參數

### 階段五：整合 DeepLOB ✅ (已完成)

**目標**: 將預訓練 DeepLOB 整合到 SB3

**新增文件**:
1. `src/models/deeplob_feature_extractor.py` - DeepLOB 特徵提取器
2. `scripts/train_sb3_deeplob.py` - 完整訓練腳本

**使用範例**:
```bash
# 完整訓練（1M steps，推薦）
python scripts/train_sb3_deeplob.py --timesteps 1000000

# 快速測試
python scripts/train_sb3_deeplob.py --timesteps 50000 --test

# 監控訓練
tensorboard --logdir logs/sb3_deeplob/
```

**核心設計**:
- DeepLOB 作為凍結的特徵提取器
- PPO 策略學習最優交易決策
- 雙層學習架構（DeepLOB 預測 + RL 決策）

**驗收標準**:
- ✅ DeepLOB 正確載入
- ✅ 特徵提取無錯誤
- ✅ 訓練管線完整

### 階段六：超參數優化 ⏳ (待訓練後實作)

**優化方向**:
- 學習率調整 (1e-4 ~ 1e-3)
- Gamma 調整 (0.95 ~ 0.995)
- Entropy係數 (0.001 ~ 0.05)
- 獎勵函數權重微調
- 網絡架構優化

**實作建議**:
- 使用 Optuna 或網格搜索
- 目標: Sharpe Ratio > 2.5

### 階段七：評估與部署 ✅ (評估完成，部署待實作)

**新增文件**: `scripts/evaluate_sb3.py`

**使用範例**:
```bash
# 評估最佳模型
python scripts/evaluate_sb3.py --model checkpoints/sb3/ppo_deeplob/best_model

# 評估並保存報告
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/ppo_deeplob_final \
    --save_report \
    --n_episodes 20
```

**評估指標** (7 大類):
- 收益指標: 總收益、收益率、Sharpe Ratio
- 風險指標: 最大回撤、勝率
- 交易統計: 交易次數、成本
- 持倉統計: 平均倉位、利用率

**待實作**:
- 推理優化（TorchScript/ONNX）
- 回測系統整合
- 模型壓縮與部署

---

## Git 提交規範

使用語義化提交訊息:

- `feat:` 新功能
- `fix:` 錯誤修復
- `refactor:` 代碼重構
- `test:` 添加測試
- `docs:` 文檔更新
- `perf:` 性能優化
- `chore:` 構建/工具變動

範例: `feat: implement SB3 PPO training script`

---

## 參考資源

### 官方文檔

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **Gymnasium**: https://gymnasium.farama.org/

### 教程與範例

- **SB3 教程**: https://github.com/araffin/rl-tutorial-jnrr19
- **金融交易範例 (FinRL)**: https://github.com/AI4Finance-Foundation/FinRL
- **中文教程**: https://zhuanlan.zhihu.com/p/374023700

### 專案文檔

- [docs/1.DeepLOB 台股模型訓練最終報告.md](docs/1.DeepLOB 台股模型訓練最終報告.md) ⭐⭐⭐ - 階段二完成報告
- [docs/SB3_IMPLEMENTATION_REPORT.md](docs/SB3_IMPLEMENTATION_REPORT.md) ⭐⭐⭐⭐⭐ - 階段 3-7 完成報告 (NEW)
- [docs/FI2010_Dataset_Specification.md](docs/FI2010_Dataset_Specification.md) ⭐

### 核心論文

1. **DeepLOB**: "Deep Convolutional Neural Networks for Limit Order Books" (Zhang et al., 2019)
2. **PPO**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
3. **Stable-Baselines3**: "Stable-Baselines3: Reliable Reinforcement Learning Implementations" (Raffin et al., 2021)

---

## 當前開發狀態

**當前階段**: 核心實作完成，準備訓練 🎉

**最新進度** (截至 2025-10-16):

### 專案總覽

| 階段 | 完成度 | 狀態 |
|------|--------|------|
| 階段一：數據處理 | 100% | ✅ |
| 階段二：DeepLOB 訓練 | 100% | ✅ |
| 階段三：環境驗證 | 100% | ✅ |
| 階段四：基礎訓練 | 100% | ✅ |
| 階段五：整合 DeepLOB | 100% | ✅ |
| 階段六：超參數優化 | 0% | ⏳ |
| 階段七：評估與部署 | 90% | ✅ |
| **總體完成度** | **85%** | **🎉** |

### 新增文件（2025-10-16）

**核心腳本**:
- ✅ `scripts/verify_env.py` - 環境驗證腳本
- ✅ `scripts/train_sb3.py` - 基礎 PPO 訓練
- ✅ `scripts/train_sb3_deeplob.py` - PPO + DeepLOB 完整訓練
- ✅ `scripts/evaluate_sb3.py` - 模型評估腳本

**模型模組**:
- ✅ `src/models/deeplob_feature_extractor.py` - SB3 特徵提取器
  - DeepLOBExtractor 類別
  - 凍結 DeepLOB 權重
  - 特徵融合網絡

**文檔**:
- ✅ `docs/SB3_IMPLEMENTATION_REPORT.md` - 階段 3-7 完整實作報告

### 關鍵成就

1. ✅ **DeepLOB 準確率達標**: 72.98% (目標 > 65%)
2. ✅ **環境完全驗證通過**: Gymnasium 標準
3. ✅ **雙層學習架構完成**: DeepLOB + PPO
4. ✅ **完整訓練管線**: 從數據到評估
5. ✅ **評估系統**: 7 大績效指標

---

## 如何開始

### 1. 環境設置

```bash
# 創建 sb3-stock 環境
conda create -n sb3-stock python=3.11
conda activate sb3-stock

# 安裝核心依賴
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install stable-baselines3[extra]
pip install gymnasium pandas numpy scipy matplotlib seaborn tensorboard pytest

# 驗證安裝
python -c "import stable_baselines3; print(f'SB3 版本: {stable_baselines3.__version__}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

### 2. 驗證環境

```bash
# 運行環境驗證腳本
python scripts/verify_env.py
```

### 3. 快速訓練（測試流程）

```bash
# 基礎 PPO 測試（10K steps，5-10 分鐘）
python scripts/train_sb3.py --timesteps 10000 --test

# PPO + DeepLOB 測試（10K steps，5-10 分鐘）
python scripts/train_sb3_deeplob.py --timesteps 10000 --test

# 評估測試模型
python scripts/evaluate_sb3.py --model checkpoints/sb3/ppo_basic/ppo_basic_final
```

### 4. 完整訓練

```bash
# 基礎 PPO 訓練（500K steps，2-4 小時）
python scripts/train_sb3.py --timesteps 500000

# PPO + DeepLOB 訓練（1M steps，4-8 小時，推薦）
python scripts/train_sb3_deeplob.py --timesteps 1000000

# 監控訓練
tensorboard --logdir logs/sb3_deeplob/

# 評估最佳模型
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 20 \
    --save_report
```

---

## 常見問題與解決方案

### 1. 環境驗證失敗

- **檢查**: 觀測空間維度是否正確
- **檢查**: 動作空間是否為 Discrete(3)
- **檢查**: reset() 返回格式是否符合 Gymnasium

### 2. DeepLOB 載入失敗

- **檢查**: 檢查點路徑是否正確
- **檢查**: PyTorch 版本是否兼容
- **嘗試**: 使用 map_location='cpu' 載入後再移到 GPU

### 3. GPU 利用率低

- **增加**: batch_size
- **增加**: n_steps (rollout buffer size)
- **使用**: 向量化環境 (DummyVecEnv 或 SubprocVecEnv)

### 4. 訓練不穩定

- **降低**: learning_rate (從 3e-4 到 1e-4)
- **增加**: n_epochs (從 10 到 20)
- **調整**: clip_range (從 0.2 到 0.1)

---

**最後更新**: 2025-10-16
**版本**: v4.0
**更新內容**:

- **核心實作完成**: 階段 3-7 所有腳本和模組已完成
- **新增 4 個腳本**: verify_env.py, train_sb3.py, train_sb3_deeplob.py, evaluate_sb3.py
- **新增 DeepLOBExtractor**: SB3 特徵提取器模組
- **新增完整報告**: docs/SB3_IMPLEMENTATION_REPORT.md（階段 3-7 詳細文檔）
- **專案完成度**: 85% (核心代碼完成，待訓練和優化)
- **下一步**: 開始實際訓練並評估模型性能

### 快速開始指令

```bash
# 1. 驗證環境（1 分鐘）
python scripts/verify_env.py

# 2. 快速測試（10 分鐘）
python scripts/train_sb3_deeplob.py --timesteps 10000 --test

# 3. 完整訓練（4-8 小時，RTX 5090）
python scripts/train_sb3_deeplob.py --timesteps 1000000

# 4. 監控訓練
tensorboard --logdir logs/sb3_deeplob/

# 5. 評估模型
python scripts/evaluate_sb3.py --model checkpoints/sb3/ppo_deeplob/best_model --save_report
```

詳細說明請參閱: [docs/SB3_IMPLEMENTATION_REPORT.md](docs/SB3_IMPLEMENTATION_REPORT.md)
