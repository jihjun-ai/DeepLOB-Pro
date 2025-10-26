# SB3-DeepLOB 訓練調教完整指南

## 文件信息

- **建立日期**: 2025-10-26
- **版本**: v1.0
- **核心腳本**: `scripts/train_sb3_deeplob.py`
- **配置文件**: `configs/sb3_deeplob_config.yaml`
- **目標**: 實現 Sharpe Ratio > 2.0 的高頻交易策略

---

## 目錄

1. [快速開始](#一快速開始)
2. [訓練指令詳解](#二訓練指令詳解)
3. [配置文件結構](#三配置文件結構)
4. [超參數調整指南](#四超參數調整指南)
5. [訓練監控](#五訓練監控)
6. [實驗設計](#六實驗設計)
7. [問題診斷](#七問題診斷)
8. [進階技巧](#八進階技巧)
9. [完整訓練流程](#九完整訓練流程)

---

## 一、快速開始

### 1.1 環境準備

```bash
# 激活環境
conda activate deeplob-pro

# 驗證安裝
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 檢查 GPU
nvidia-smi
```

### 1.2 首次訓練（快速測試 10 分鐘）

```bash
# 使用測試模式（10K steps）
python scripts/train_sb3_deeplob.py --test

# 或使用指令別名（Windows）
scripts\quick_train.bat
```

**預期輸出**:
```
======================================================================
SB3-DeepLOB 訓練配置
======================================================================
測試模式: True
總訓練步數: 10,000

環境配置:
  最大倉位: 1
  初始資金: 100,000 TWD
  Episode 長度: 500
  數據採樣比: 10.0%

獎勵配置:
  PnL 權重: 1.0
  成本懲罰: 1.0
  庫存懲罰: 0.01
  風險懲罰: 0.005

PPO 配置:
  學習率: 0.0003
  Batch size: 64
  N steps: 2048
  ...

開始訓練...
```

### 1.3 完整訓練（4-8 小時）

```bash
# 使用預設配置訓練 1M steps
python scripts/train_sb3_deeplob.py

# 自訂訓練步數
python scripts/train_sb3_deeplob.py --timesteps 500000

# 使用自訂配置文件
python scripts/train_sb3_deeplob.py --config configs/my_config.yaml
```

---

## 二、訓練指令詳解

### 2.1 基礎指令

#### 指令格式
```bash
python scripts/train_sb3_deeplob.py [OPTIONS]
```

#### 可用參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--config` | str | `configs/sb3_deeplob_config.yaml` | 配置文件路徑 |
| `--timesteps` | int | `1000000` | 總訓練步數 |
| `--test` | flag | `False` | 測試模式（10K steps） |
| `--device` | str | `auto` | 訓練設備 (`cuda`/`cpu`/`auto`) |

#### 範例

```bash
# 1. 快速測試（10K steps，約 5-10 分鐘）
python scripts/train_sb3_deeplob.py --test

# 2. 短期訓練（50K steps，約 30-60 分鐘）
python scripts/train_sb3_deeplob.py --timesteps 50000

# 3. 中期訓練（500K steps，約 2-4 小時）
python scripts/train_sb3_deeplob.py --timesteps 500000

# 4. 長期訓練（1M steps，約 4-8 小時，推薦）
python scripts/train_sb3_deeplob.py --timesteps 1000000

# 5. 使用自訂配置
python scripts/train_sb3_deeplob.py \
    --config configs/my_experiment.yaml \
    --timesteps 1000000

# 6. 強制使用 CPU（調試用）
python scripts/train_sb3_deeplob.py --device cpu --test
```

### 2.2 測試模式詳解

**啟用方式**:
```bash
python scripts/train_sb3_deeplob.py --test
```

**測試模式修改**:
- ✅ 訓練步數：1M → 10K
- ✅ 數據採樣比：100% → 10%
- ✅ Episode 長度：500 → 100
- ✅ Checkpoint 頻率：50K → 5K
- ✅ 評估頻率：10K → 2K

**用途**:
- 快速驗證配置正確性
- 測試新的超參數組合
- 驗證代碼修改
- Debug 錯誤

**不適用於**:
- 評估最終性能
- 超參數最終選擇
- 實盤部署

---

## 三、配置文件結構

### 3.1 完整配置範例

```yaml
# ===== 數據配置 =====
data:
  base_dir: "./data/processed_v7"
  mode: "train"              # train/val/test
  data_sample_ratio: 1.0     # 數據採樣比例 (0.0-1.0)

# ===== 環境配置 =====
env_config:
  max_position: 1            # 最大倉位（張）
  initial_balance: 100000    # 初始資金（TWD）
  max_steps: 500             # 每個 episode 最大步數

  # 獎勵配置
  reward:
    pnl_scale: 1.0           # PnL 權重
    cost_penalty: 1.0        # 成本懲罰權重
    inventory_penalty: 0.01  # 庫存懲罰權重
    risk_penalty: 0.005      # 風險懲罰權重

  # 交易成本配置
  transaction_cost:
    shares_per_lot: 1000     # 每張股數
    commission:
      base_rate: 0.001425    # 手續費基準
      discount: 0.3          # 折扣
      min_fee: 20.0          # 最低手續費
    securities_tax:
      rate: 0.0015           # 交易稅率（當沖）
    slippage:
      enabled: false
      rate: 0.0001

# ===== DeepLOB 模型配置 =====
deeplob:
  checkpoint_path: "./checkpoints/deeplob/best_deeplob_tw_pro_v5.pth"
  freeze_weights: true       # 是否凍結權重

  feature_extractor:
    features_dim: 64         # 特徵維度
    dropout: 0.0             # Dropout 率

# ===== PPO 配置 =====
ppo:
  learning_rate: 0.0003      # 學習率
  n_steps: 2048              # Rollout buffer 大小
  batch_size: 64             # Mini-batch 大小
  n_epochs: 10               # 每次更新的 epoch 數
  gamma: 0.99                # 折扣因子
  gae_lambda: 0.95           # GAE lambda
  clip_range: 0.2            # PPO clip 範圍
  ent_coef: 0.01             # 熵係數
  vf_coef: 0.5               # 價值函數係數
  max_grad_norm: 0.5         # 梯度裁剪

  policy_kwargs:
    lstm_hidden_size: 256    # LSTM 隱藏層大小
    n_lstm_layers: 1         # LSTM 層數
    shared_lstm: false       # 是否共享 LSTM
    enable_critic_lstm: true # Critic 是否使用 LSTM

# ===== 訓練配置 =====
training:
  total_timesteps: 1000000   # 總訓練步數
  n_eval_episodes: 10        # 評估 episode 數
  eval_freq: 10000           # 評估頻率
  save_freq: 50000           # Checkpoint 保存頻率
  log_interval: 10           # 日誌間隔
  verbose: 1                 # 詳細程度 (0/1/2)

# ===== 輸出配置 =====
output:
  log_dir: "./logs/sb3_deeplob"
  checkpoint_dir: "./checkpoints/sb3/ppo_deeplob"
  experiment_name: "ppo_deeplob"
```

### 3.2 配置文件修改

#### 方法 1: 直接編輯 YAML（推薦）

```bash
# 使用文本編輯器
notepad configs/sb3_deeplob_config.yaml

# 或使用 VSCode
code configs/sb3_deeplob_config.yaml
```

#### 方法 2: 複製並修改

```bash
# 複製預設配置
copy configs\sb3_deeplob_config.yaml configs\my_experiment.yaml

# 編輯新配置
notepad configs\my_experiment.yaml

# 使用新配置訓練
python scripts/train_sb3_deeplob.py --config configs/my_experiment.yaml
```

#### 方法 3: Python 腳本修改

```python
from src.utils.yaml_manager import YAMLManager

# 載入配置
yaml_mgr = YAMLManager('configs/sb3_deeplob_config.yaml')

# 修改參數
yaml_mgr.set('ppo.learning_rate', 0.0001)
yaml_mgr.set('env_config.reward.pnl_scale', 2.0)

# 保存新配置
yaml_mgr.save('configs/my_experiment.yaml')
```

---

## 四、超參數調整指南

### 4.1 獎勵函數參數 ⭐⭐⭐⭐⭐（最重要）

#### `pnl_scale` - PnL 權重

**作用**: 控制盈利在總獎勵中的重要性

**調整方向**:
```yaml
# 激進策略（追求高收益）
env_config.reward.pnl_scale: 2.0

# 平衡策略（預設）
env_config.reward.pnl_scale: 1.0

# 保守策略（重視風險）
env_config.reward.pnl_scale: 0.5
```

**觀察指標**:
- `sharpe_ratio`: 應該提高
- `total_profit`: 可能增加
- `max_drawdown`: 可能增加（風險）

**調整建議**:
- 初始值: 1.0
- 如果策略過於保守（很少交易）→ 增加到 1.5-2.0
- 如果風險過高（大幅回撤）→ 降低到 0.5-0.8

---

#### `cost_penalty` - 成本懲罰權重

**作用**: 控制交易成本的懲罰強度

**調整方向**:
```yaml
# 減少交易頻率
env_config.reward.cost_penalty: 2.0

# 平衡（預設）
env_config.reward.cost_penalty: 1.0

# 允許頻繁交易
env_config.reward.cost_penalty: 0.5
```

**觀察指標**:
- `n_trades`: 交易次數
- `transaction_cost`: 總交易成本
- `profit_after_cost`: 扣除成本後利潤

**調整建議**:
- 如果過度交易（>200 次/episode）→ 增加到 1.5-2.0
- 如果幾乎不交易（<10 次/episode）→ 降低到 0.3-0.5
- 目標: 50-100 次/episode（高頻策略）

---

#### `inventory_penalty` - 庫存懲罰權重

**作用**: 控制持倉時間的懲罰

**調整方向**:
```yaml
# 極快進出（日內高頻）
env_config.reward.inventory_penalty: 0.05

# 平衡（預設）
env_config.reward.inventory_penalty: 0.01

# 允許持倉（波段）
env_config.reward.inventory_penalty: 0.001
```

**觀察指標**:
- `avg_holding_time`: 平均持倉時間
- `position_utilization`: 倉位利用率

**調整建議**:
- 高頻策略: 0.02-0.05（快速進出）
- 日內策略: 0.01（預設）
- 波段策略: 0.001-0.005（允許持倉）

---

#### `risk_penalty` - 風險懲罰權重

**作用**: 基於波動率的風險懲罰

**調整方向**:
```yaml
# 極度保守
env_config.reward.risk_penalty: 0.02

# 平衡（預設）
env_config.reward.risk_penalty: 0.005

# 激進
env_config.reward.risk_penalty: 0.001
```

**觀察指標**:
- `max_drawdown`: 最大回撤
- `volatility`: 策略波動率

**調整建議**:
- 實盤策略: 0.01-0.02（保守）
- 回測優化: 0.005（預設）
- 高風險高收益: 0.001-0.003

---

### 4.2 PPO 核心參數 ⭐⭐⭐⭐

#### `learning_rate` - 學習率

**作用**: 控制參數更新的步長

**調整方向**:
```yaml
# 快速學習（可能不穩定）
ppo.learning_rate: 0.001

# 平衡（預設）
ppo.learning_rate: 0.0003

# 穩定學習（較慢）
ppo.learning_rate: 0.0001
```

**觀察指標**:
- `train/loss`: 損失曲線
- `train/explained_variance`: 解釋方差
- 訓練穩定性

**調整建議**:
- 如果訓練不穩定（loss 震盪）→ 降低到 1e-4
- 如果學習太慢（性能無提升）→ 增加到 5e-4 或 1e-3
- 建議: 從 3e-4 開始

**進階技巧**: 使用學習率調度
```yaml
# 線性衰減
ppo.learning_rate: 0.0003
# 在訓練過程中自動衰減
```

---

#### `n_steps` - Rollout Buffer 大小

**作用**: 每次更新前收集的步數

**調整方向**:
```yaml
# 小 buffer（更頻繁更新）
ppo.n_steps: 1024

# 平衡（預設）
ppo.n_steps: 2048

# 大 buffer（更穩定）
ppo.n_steps: 4096
```

**觀察指標**:
- 訓練速度
- GPU 利用率
- 訓練穩定性

**調整建議**:
- GPU 記憶體充足: 4096（RTX 5090 推薦）
- GPU 記憶體有限: 1024-2048
- 影響訓練速度和穩定性的平衡

---

#### `batch_size` - Mini-batch 大小

**作用**: 每次梯度更新的樣本數

**調整方向**:
```yaml
# 小 batch（噪聲大，可能更好探索）
ppo.batch_size: 32

# 平衡（預設）
ppo.batch_size: 64

# 大 batch（穩定，GPU 利用率高）
ppo.batch_size: 128
```

**約束條件**:
- `batch_size` 必須是 `n_steps` 的因數
- 例如: n_steps=2048 → batch_size 可以是 32/64/128/256

**調整建議**:
- RTX 5090 (32GB): 128-256（充分利用 GPU）
- RTX 4090 (24GB): 64-128
- RTX 3090 (24GB): 64

---

#### `n_epochs` - 更新 Epoch 數

**作用**: 每批數據重複訓練的次數

**調整方向**:
```yaml
# 快速更新
ppo.n_epochs: 5

# 平衡（預設）
ppo.n_epochs: 10

# 充分學習
ppo.n_epochs: 20
```

**觀察指標**:
- `train/approx_kl`: KL 散度（應保持較小）
- 訓練時間

**調整建議**:
- 如果 KL 散度過大（>0.02）→ 減少 epoch
- 如果學習不充分 → 增加到 15-20
- 預設 10 通常足夠

---

#### `gamma` - 折扣因子

**作用**: 控制未來獎勵的重要性

**調整方向**:
```yaml
# 短視（重視即時獎勵）
ppo.gamma: 0.95

# 平衡（預設）
ppo.gamma: 0.99

# 遠視（重視長期獎勵）
ppo.gamma: 0.995
```

**適用場景**:
- 高頻交易: 0.95-0.98（短期）
- 日內策略: 0.99（預設）
- 波段策略: 0.995-0.999（長期）

---

#### `clip_range` - PPO Clip 範圍

**作用**: 限制策略更新幅度

**調整方向**:
```yaml
# 保守更新
ppo.clip_range: 0.1

# 平衡（預設）
ppo.clip_range: 0.2

# 激進更新
ppo.clip_range: 0.3
```

**調整建議**:
- 訓練不穩定 → 降低到 0.1
- 學習太慢 → 增加到 0.3
- 預設 0.2 通常最佳

---

#### `ent_coef` - 熵係數

**作用**: 鼓勵探索（策略隨機性）

**調整方向**:
```yaml
# 低探索（快速收斂）
ppo.ent_coef: 0.001

# 平衡（預設）
ppo.ent_coef: 0.01

# 高探索（避免早期收斂）
ppo.ent_coef: 0.05
```

**觀察指標**:
- `train/entropy_loss`: 熵損失
- 策略多樣性

**調整建議**:
- 訓練初期: 0.01-0.05（鼓勵探索）
- 訓練後期: 0.001-0.01（收斂）
- 可以使用衰減策略

---

### 4.3 環境參數 ⭐⭐⭐

#### `max_position` - 最大倉位

**作用**: 限制最大持倉張數

**調整方向**:
```yaml
# 保守（小倉位）
env_config.max_position: 1

# 中等
env_config.max_position: 2

# 激進（大倉位）
env_config.max_position: 5
```

**風險考量**:
- 倉位越大，風險越高
- 建議: 從 1 開始，逐步增加
- 實盤: 根據資金規模決定

---

#### `max_steps` - Episode 長度

**作用**: 每個 episode 的最大步數

**調整方向**:
```yaml
# 短 episode
env_config.max_steps: 100

# 平衡（預設）
env_config.max_steps: 500

# 長 episode
env_config.max_steps: 1000
```

**影響**:
- 短 episode: 學習快速決策
- 長 episode: 學習長期策略
- 訓練時間: 長 episode 更慢

---

#### `data_sample_ratio` - 數據採樣比例

**作用**: 使用多少比例的數據

**調整方向**:
```yaml
# 快速測試
data.data_sample_ratio: 0.1  # 10%

# 中等
data.data_sample_ratio: 0.5  # 50%

# 完整數據（預設）
data.data_sample_ratio: 1.0  # 100%
```

**用途**:
- 測試階段: 0.1-0.3
- 正式訓練: 1.0

---

### 4.4 DeepLOB 參數 ⭐⭐

#### `freeze_weights` - 凍結權重

**作用**: 是否凍結 DeepLOB 模型

**選項**:
```yaml
# 凍結（預設，推薦）
deeplob.freeze_weights: true

# 解凍（微調 DeepLOB）
deeplob.freeze_weights: false
```

**建議**:
- 初始訓練: true（凍結）
- 進階優化: false（微調）
- 風險: 解凍可能導致過擬合

---

#### `features_dim` - 特徵維度

**作用**: DeepLOB 輸出特徵的維度

**調整方向**:
```yaml
# 小模型（快速）
deeplob.feature_extractor.features_dim: 32

# 平衡（預設）
deeplob.feature_extractor.features_dim: 64

# 大模型（表達力強）
deeplob.feature_extractor.features_dim: 128
```

**建議**:
- 預設 64 通常足夠
- 增加維度會增加訓練時間

---

### 4.5 LSTM 參數 ⭐⭐⭐

#### `lstm_hidden_size` - LSTM 隱藏層大小

**作用**: LSTM 的記憶容量

**調整方向**:
```yaml
# 小模型
ppo.policy_kwargs.lstm_hidden_size: 128

# 平衡（預設）
ppo.policy_kwargs.lstm_hidden_size: 256

# 大模型
ppo.policy_kwargs.lstm_hidden_size: 512
```

**建議**:
- RTX 5090: 512（大模型）
- RTX 4090: 256-384
- 記憶體有限: 128-256

---

## 五、訓練監控

### 5.1 TensorBoard 監控

#### 啟動 TensorBoard

```bash
# 監控訓練日誌
tensorboard --logdir logs/sb3_deeplob/

# 指定端口
tensorboard --logdir logs/sb3_deeplob/ --port 6006

# 瀏覽器訪問
# http://localhost:6006
```

#### 關鍵指標

**1. Rollout 指標**（訓練性能）

| 指標 | 說明 | 目標值 |
|------|------|--------|
| `rollout/ep_len_mean` | 平均 episode 長度 | 接近 max_steps |
| `rollout/ep_rew_mean` | 平均 episode 獎勵 | 持續上升 |
| `rollout/success_rate` | 成功率 | > 0.5 |

**2. Train 指標**（訓練質量）

| 指標 | 說明 | 目標值 |
|------|------|--------|
| `train/loss` | 總損失 | 逐漸下降 |
| `train/policy_loss` | 策略損失 | 穩定 |
| `train/value_loss` | 價值損失 | 逐漸下降 |
| `train/entropy_loss` | 熵損失 | 逐漸下降 |
| `train/approx_kl` | KL 散度 | < 0.02 |
| `train/explained_variance` | 解釋方差 | > 0.7 |
| `train/learning_rate` | 當前學習率 | 監控衰減 |

**3. Eval 指標**（評估性能）

| 指標 | 說明 | 目標值 |
|------|------|--------|
| `eval/mean_reward` | 評估平均獎勵 | 持續上升 |
| `eval/mean_ep_length` | 評估 episode 長度 | 穩定 |

**4. Custom 指標**（自定義）

| 指標 | 說明 | 目標值 |
|------|------|--------|
| `custom/sharpe_ratio` | Sharpe Ratio | > 2.0 |
| `custom/total_profit` | 總利潤 | > 0 |
| `custom/max_drawdown` | 最大回撤 | < 10% |
| `custom/win_rate` | 勝率 | > 55% |
| `custom/n_trades` | 交易次數 | 50-100 |

### 5.2 終端監控

訓練過程中，終端會定期輸出：

```
-----------------------------------------
| rollout/              |              |
|    ep_len_mean        | 500          |
|    ep_rew_mean        | 1234.56      |
| time/                 |              |
|    fps                | 2048         |
|    iterations         | 10           |
|    time_elapsed       | 600          |
|    total_timesteps    | 20480        |
| train/                |              |
|    approx_kl          | 0.0123       |
|    clip_fraction      | 0.156        |
|    explained_variance | 0.876        |
|    learning_rate      | 0.0003       |
|    loss               | 12.34        |
|    policy_loss        | -0.0123      |
|    value_loss         | 34.56        |
-----------------------------------------
```

**重點關注**:
- `ep_rew_mean`: 是否上升
- `explained_variance`: 是否 > 0.7
- `approx_kl`: 是否 < 0.02

### 5.3 Checkpoint 監控

#### 自動保存的文件

```
checkpoints/sb3/ppo_deeplob/
├── ppo_deeplob_50000_steps.zip     # 定期 checkpoint
├── ppo_deeplob_100000_steps.zip
├── ...
├── best_model.zip                  # 最佳模型
└── ppo_deeplob_final.zip           # 最終模型
```

#### 手動檢查 Checkpoint

```python
from stable_baselines3 import PPO

# 載入模型
model = PPO.load("checkpoints/sb3/ppo_deeplob/best_model")

# 檢查參數
print(f"Learning rate: {model.learning_rate}")
print(f"Total timesteps: {model.num_timesteps}")
```

---

## 六、實驗設計

### 6.1 基礎實驗（驗證配置）

**目標**: 驗證系統正常運行

**配置**: 預設配置

**步驟**:
```bash
# 1. 快速測試（10K steps）
python scripts/train_sb3_deeplob.py --test

# 2. 檢查 TensorBoard
tensorboard --logdir logs/sb3_deeplob/

# 3. 確認以下指標正常：
#    - ep_rew_mean 不為 NaN
#    - loss 逐漸下降
#    - 無錯誤訊息
```

**驗收標準**:
- ✅ 訓練完成無錯誤
- ✅ TensorBoard 可視化正常
- ✅ 模型保存成功

---

### 6.2 實驗一：獎勵權重對比 ⭐⭐⭐

**目標**: 找到最佳獎勵權重組合

**實驗設計**:

| 實驗 | pnl_scale | cost_penalty | inventory_penalty | 策略類型 |
|------|-----------|--------------|-------------------|----------|
| Exp1 | 2.0 | 0.5 | 0.001 | 激進 |
| Exp2 | 1.0 | 1.0 | 0.01 | 平衡（基準） |
| Exp3 | 1.0 | 2.0 | 0.05 | 保守 |
| Exp4 | 1.5 | 1.0 | 0.02 | 中激進 |

**執行步驟**:

```bash
# 1. 創建配置文件
copy configs\sb3_deeplob_config.yaml configs\exp1_aggressive.yaml
copy configs\sb3_deeplob_config.yaml configs\exp2_balanced.yaml
copy configs\sb3_deeplob_config.yaml configs\exp3_conservative.yaml
copy configs\sb3_deeplob_config.yaml configs\exp4_moderate.yaml

# 2. 修改各配置文件的獎勵參數（手動編輯）

# 3. 並行訓練（需要多個 GPU 或分時運行）
# 終端 1
python scripts/train_sb3_deeplob.py --config configs/exp1_aggressive.yaml

# 終端 2
python scripts/train_sb3_deeplob.py --config configs/exp2_balanced.yaml

# 終端 3
python scripts/train_sb3_deeplob.py --config configs/exp3_conservative.yaml

# 終端 4
python scripts/train_sb3_deeplob.py --config configs/exp4_moderate.yaml
```

**評估指標**:
- Sharpe Ratio
- 總收益
- 最大回撤
- 勝率
- 交易次數

**選擇最佳**:
- 優先: Sharpe Ratio > 2.0
- 其次: 最大回撤 < 10%
- 最後: 總收益最高

---

### 6.3 實驗二：學習率對比 ⭐⭐

**目標**: 找到最佳學習率

**實驗設計**:

| 實驗 | learning_rate | 預期效果 |
|------|---------------|----------|
| Exp1 | 1e-4 | 穩定但慢 |
| Exp2 | 3e-4 | 平衡（基準） |
| Exp3 | 5e-4 | 較快 |
| Exp4 | 1e-3 | 快但可能不穩定 |

**執行步驟**:

```bash
# 修改配置文件的 ppo.learning_rate
# 其他參數保持不變

python scripts/train_sb3_deeplob.py --config configs/lr_1e4.yaml
python scripts/train_sb3_deeplob.py --config configs/lr_3e4.yaml
python scripts/train_sb3_deeplob.py --config configs/lr_5e4.yaml
python scripts/train_sb3_deeplob.py --config configs/lr_1e3.yaml
```

**觀察**:
- 訓練曲線穩定性
- 收斂速度
- 最終性能

---

### 6.4 實驗三：LSTM 大小對比 ⭐⭐

**目標**: 找到最佳 LSTM 隱藏層大小

**實驗設計**:

| 實驗 | lstm_hidden_size | 模型複雜度 |
|------|------------------|-----------|
| Exp1 | 128 | 小 |
| Exp2 | 256 | 中（基準） |
| Exp3 | 384 | 中大 |
| Exp4 | 512 | 大 |

**執行步驟**:

```bash
# 修改配置文件的 ppo.policy_kwargs.lstm_hidden_size

python scripts/train_sb3_deeplob.py --config configs/lstm_128.yaml
python scripts/train_sb3_deeplob.py --config configs/lstm_256.yaml
python scripts/train_sb3_deeplob.py --config configs/lstm_384.yaml
python scripts/train_sb3_deeplob.py --config configs/lstm_512.yaml
```

**觀察**:
- 訓練時間
- GPU 記憶體使用
- 最終性能
- 過擬合風險

---

### 6.5 實驗四：Batch Size 對比 ⭐

**目標**: 找到最佳 batch size

**實驗設計**:

| 實驗 | n_steps | batch_size | 說明 |
|------|---------|------------|------|
| Exp1 | 2048 | 32 | 小 batch |
| Exp2 | 2048 | 64 | 中 batch（基準） |
| Exp3 | 2048 | 128 | 大 batch |
| Exp4 | 4096 | 128 | 大 buffer + 大 batch |

**執行步驟**:

```bash
# 修改配置文件的 ppo.n_steps 和 ppo.batch_size

python scripts/train_sb3_deeplob.py --config configs/batch_32.yaml
python scripts/train_sb3_deeplob.py --config configs/batch_64.yaml
python scripts/train_sb3_deeplob.py --config configs/batch_128.yaml
python scripts/train_sb3_deeplob.py --config configs/batch_large.yaml
```

**觀察**:
- GPU 利用率
- 訓練速度（FPS）
- 訓練穩定性

---

### 6.6 網格搜索實驗 ⭐⭐⭐⭐

**目標**: 系統性搜索最佳超參數組合

**配置空間**:
```python
param_grid = {
    'pnl_scale': [0.5, 1.0, 1.5, 2.0],
    'cost_penalty': [0.5, 1.0, 1.5, 2.0],
    'learning_rate': [1e-4, 3e-4, 5e-4],
    'lstm_hidden_size': [128, 256, 384],
}

# 總共 4 × 4 × 3 × 3 = 144 種組合
```

**執行腳本**（示例）:

```python
# scripts/grid_search.py

import itertools
import subprocess
from src.utils.yaml_manager import YAMLManager

# 定義參數網格
param_grid = {
    'env_config.reward.pnl_scale': [0.5, 1.0, 1.5, 2.0],
    'env_config.reward.cost_penalty': [0.5, 1.0, 1.5, 2.0],
    'ppo.learning_rate': [1e-4, 3e-4, 5e-4],
    'ppo.policy_kwargs.lstm_hidden_size': [128, 256, 384],
}

# 生成所有組合
keys = list(param_grid.keys())
values = list(param_grid.values())

for i, combination in enumerate(itertools.product(*values)):
    # 創建配置
    yaml_mgr = YAMLManager('configs/sb3_deeplob_config.yaml')

    for key, value in zip(keys, combination):
        yaml_mgr.set(key, value)

    # 保存配置
    config_path = f'configs/grid_search/exp_{i:03d}.yaml'
    yaml_mgr.save(config_path)

    # 訓練
    cmd = f'python scripts/train_sb3_deeplob.py --config {config_path} --timesteps 50000'
    print(f'Running: {cmd}')
    subprocess.run(cmd, shell=True)
```

**警告**: 網格搜索非常耗時，建議：
- 減少參數範圍
- 使用較少訓練步數（50K）
- 使用多 GPU 並行
- 或使用 Optuna 等自動優化工具

---

## 七、問題診斷

### 7.1 訓練不收斂

**症狀**:
- `ep_rew_mean` 不上升或震盪
- `loss` 不下降
- 訓練 100K+ steps 仍無改善

**可能原因與解決方法**:

#### 原因 1: 學習率過高
```yaml
# 解決方法：降低學習率
ppo.learning_rate: 0.0001  # 從 0.0003 降低
```

#### 原因 2: 獎勵函數設計不當
```yaml
# 檢查獎勵是否合理
# - PnL 獎勵是否正常
# - 懲罰項是否過大

# 調整建議
env_config.reward.cost_penalty: 0.5  # 降低懲罰
env_config.reward.pnl_scale: 1.5     # 增加 PnL 權重
```

#### 原因 3: 數據質量問題
```bash
# 檢查數據標籤分布
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze_all

# 確保標籤分布合理（30/40/30）
```

#### 原因 4: 網絡容量不足
```yaml
# 增加網絡容量
ppo.policy_kwargs.lstm_hidden_size: 384  # 從 256 增加
deeplob.feature_extractor.features_dim: 128  # 從 64 增加
```

---

### 7.2 智能體只執行 Hold 動作

**症狀**:
- 90%+ 的動作都是 Hold (action=1)
- 很少 Buy/Sell
- 獎勵接近 0

**可能原因與解決方法**:

#### 原因 1: 交易成本懲罰過高
```yaml
# 解決方法：降低成本懲罰
env_config.reward.cost_penalty: 0.3  # 從 1.0 降低
```

#### 原因 2: PnL 獎勵過低
```yaml
# 解決方法：增加 PnL 權重
env_config.reward.pnl_scale: 2.0  # 從 1.0 增加
```

#### 原因 3: 探索不足
```yaml
# 解決方法：增加熵係數
ppo.ent_coef: 0.05  # 從 0.01 增加
```

#### 原因 4: 環境設計問題
```python
# 檢查動作空間定義
# 確保 Hold/Buy/Sell 都能獲得合理獎勵
```

---

### 7.3 智能體過度交易

**症狀**:
- 每個 episode 交易 200+ 次
- 交易成本過高
- 扣除成本後虧損

**可能原因與解決方法**:

#### 原因 1: 交易成本懲罰過低
```yaml
# 解決方法：增加成本懲罰
env_config.reward.cost_penalty: 2.0  # 從 1.0 增加
```

#### 原因 2: 庫存懲罰過高
```yaml
# 解決方法：降低庫存懲罰
env_config.reward.inventory_penalty: 0.005  # 從 0.01 降低
```

#### 原因 3: 獎勵塑形問題
```python
# 檢查獎勵計算是否正確
# 確保交易成本正確計入獎勵
```

---

### 7.4 訓練過程中出現 NaN

**症狀**:
- `loss` 變為 NaN
- `ep_rew_mean` 變為 NaN
- 訓練中斷

**可能原因與解決方法**:

#### 原因 1: 學習率過高
```yaml
# 解決方法：降低學習率
ppo.learning_rate: 0.0001
```

#### 原因 2: 梯度爆炸
```yaml
# 解決方法：降低梯度裁剪閾值
ppo.max_grad_norm: 0.3  # 從 0.5 降低
```

#### 原因 3: 數值不穩定
```python
# 檢查獎勵計算
# 確保沒有除以零或 log(0) 等操作
```

#### 原因 4: 批標準化問題
```yaml
# 使用更穩定的配置
ppo.clip_range: 0.1  # 從 0.2 降低
```

---

### 7.5 GPU 利用率低

**症狀**:
- GPU 利用率 < 50%
- 訓練速度慢（FPS < 1000）

**可能原因與解決方法**:

#### 原因 1: Batch size 過小
```yaml
# 解決方法：增加 batch size
ppo.batch_size: 128  # 從 64 增加
ppo.n_steps: 4096    # 從 2048 增加
```

#### 原因 2: 數據載入瓶頸
```python
# 檢查數據是否已載入到記憶體
# 避免每步從硬碟讀取
```

#### 原因 3: CPU 瓶頸
```bash
# 使用多進程環境
# SubprocVecEnv 而非 DummyVecEnv
```

---

### 7.6 記憶體不足（OOM）

**症狀**:
- `CUDA out of memory` 錯誤
- 訓練中斷

**解決方法**:

#### 方法 1: 減少 batch size
```yaml
ppo.batch_size: 32   # 從 64 減少
ppo.n_steps: 1024    # 從 2048 減少
```

#### 方法 2: 減少模型大小
```yaml
ppo.policy_kwargs.lstm_hidden_size: 128  # 從 256 減少
deeplob.feature_extractor.features_dim: 32  # 從 64 減少
```

#### 方法 3: 使用混合精度訓練
```python
# 在訓練腳本中啟用 FP16
# 需要修改代碼支持
```

#### 方法 4: 減少數據量
```yaml
data.data_sample_ratio: 0.5  # 使用 50% 數據
```

---

## 八、進階技巧

### 8.1 課程學習（Curriculum Learning）

**概念**: 從簡單任務逐步過渡到複雜任務

**實施方法**:

#### 階段 1: 簡單環境（50K steps）
```yaml
env_config.max_steps: 100           # 短 episode
data.data_sample_ratio: 0.3         # 少量數據
env_config.reward.pnl_scale: 2.0    # 高 PnL 權重
```

#### 階段 2: 中等環境（200K steps）
```yaml
env_config.max_steps: 300
data.data_sample_ratio: 0.6
env_config.reward.pnl_scale: 1.5
```

#### 階段 3: 完整環境（500K steps）
```yaml
env_config.max_steps: 500
data.data_sample_ratio: 1.0
env_config.reward.pnl_scale: 1.0
```

**執行腳本**:
```bash
# 階段 1
python scripts/train_sb3_deeplob.py \
    --config configs/curriculum_stage1.yaml \
    --timesteps 50000

# 階段 2（載入階段 1 的模型）
python scripts/train_sb3_deeplob.py \
    --config configs/curriculum_stage2.yaml \
    --timesteps 200000 \
    --load-model checkpoints/sb3/stage1/best_model.zip

# 階段 3
python scripts/train_sb3_deeplob.py \
    --config configs/curriculum_stage3.yaml \
    --timesteps 500000 \
    --load-model checkpoints/sb3/stage2/best_model.zip
```

---

### 8.2 自適應獎勵權重

**使用 AdaptiveRewardShaper**:

```python
# 在 train_sb3_deeplob.py 中修改
from src.envs.reward_shaper import AdaptiveRewardShaper

# 創建自適應獎勵塑形器
reward_config = {
    'pnl_scale': 2.0,          # 初始高 PnL 權重
    'min_pnl_scale': 0.5,      # 最終低 PnL 權重
    'scale_decay': 0.9999,     # 衰減率
    'cost_penalty': 1.0,
    'inventory_penalty': 0.01,
    'risk_penalty': 0.005,
}

reward_shaper = AdaptiveRewardShaper(reward_config)
```

**配置**:
```yaml
env_config.reward:
  adaptive: true              # 啟用自適應
  pnl_scale: 2.0              # 初始值
  min_pnl_scale: 0.5          # 最小值
  scale_decay: 0.9999         # 衰減率
```

**效果**:
- 訓練初期: 高 PnL 權重 → 鼓勵探索盈利機會
- 訓練後期: 低 PnL 權重 → 強化風險管理

---

### 8.3 學習率調度

**線性衰減**:

```python
# 在 train_sb3_deeplob.py 中修改
from stable_baselines3.common.utils import linear_schedule

# 創建學習率調度
lr_schedule = linear_schedule(
    initial_value=3e-4,  # 初始學習率
    final_value=1e-5     # 最終學習率
)

model = PPO(
    ...,
    learning_rate=lr_schedule,  # 使用調度而非固定值
    ...
)
```

**配置**:
```yaml
ppo.learning_rate:
  initial: 0.0003
  final: 0.00001
  schedule: "linear"  # linear/exponential/constant
```

---

### 8.4 遷移學習

**從其他模型繼續訓練**:

```bash
# 載入預訓練模型
python scripts/train_sb3_deeplob.py \
    --load-model checkpoints/sb3/ppo_deeplob/best_model.zip \
    --timesteps 500000
```

**應用場景**:
- 在新數據上微調
- 更改獎勵函數後繼續訓練
- 延長訓練時間

---

### 8.5 多環境並行訓練

**使用向量化環境**:

```python
# 在 train_sb3_deeplob.py 中修改
from stable_baselines3.common.vec_env import SubprocVecEnv

# 創建多個環境
n_envs = 4  # 並行 4 個環境

def make_env():
    def _init():
        return TaiwanLOBTradingEnv(config)
    return _init

envs = SubprocVecEnv([make_env() for _ in range(n_envs)])

model = PPO("MlpLstmPolicy", envs, ...)
```

**優勢**:
- 加快數據收集
- 提高 GPU 利用率
- 更穩定的訓練

**注意**:
- 需要足夠的 CPU 核心
- 記憶體需求增加

---

### 8.6 超參數優化（Optuna）

**自動搜索最佳超參數**:

```python
# scripts/optuna_optimization.py

import optuna
from stable_baselines3 import PPO

def objective(trial):
    # 定義超參數搜索空間
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    pnl_scale = trial.suggest_float("pnl_scale", 0.5, 2.0)
    cost_penalty = trial.suggest_float("cost_penalty", 0.3, 2.0)
    lstm_size = trial.suggest_categorical("lstm_hidden_size", [128, 256, 384, 512])

    # 創建環境和模型
    env = TaiwanLOBTradingEnv(config)
    model = PPO(..., learning_rate=lr, ...)

    # 訓練
    model.learn(total_timesteps=50000)

    # 評估
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward

# 創建 Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 輸出最佳參數
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

**執行**:
```bash
python scripts/optuna_optimization.py
```

**優勢**:
- 自動搜索
- 智能採樣（非窮舉）
- 可視化結果

---

## 九、完整訓練流程

### 9.1 階段一：環境驗證（1 天）

**目標**: 確保所有組件正常工作

**步驟**:

```bash
# 1. 驗證數據
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze_all

# 2. 驗證環境
python scripts/verify_env.py

# 3. 快速訓練測試
python scripts/train_sb3_deeplob.py --test

# 4. 檢查輸出
tensorboard --logdir logs/sb3_deeplob/
```

**驗收標準**:
- ✅ 數據載入正常
- ✅ 環境通過 SB3 驗證
- ✅ 訓練無錯誤
- ✅ TensorBoard 可視化正常

---

### 9.2 階段二：基準測試（2-3 天）

**目標**: 建立性能基準

**步驟**:

```bash
# 1. 使用預設配置訓練
python scripts/train_sb3_deeplob.py --timesteps 100000

# 2. 評估模型
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 20 \
    --save_report

# 3. 記錄基準性能
#    - Sharpe Ratio: ?
#    - Total Profit: ?
#    - Max Drawdown: ?
#    - Win Rate: ?
```

**記錄結果**:
```yaml
# results/baseline_performance.yaml
baseline:
  config: configs/sb3_deeplob_config.yaml
  timesteps: 100000
  sharpe_ratio: 1.23
  total_profit: 5678.90
  max_drawdown: 0.15
  win_rate: 0.52
  n_trades: 87
```

---

### 9.3 階段三：超參數優化（1-2 週）

**目標**: 找到最佳超參數組合

**方案 A: 手動實驗**

```bash
# 實驗 1: 獎勵權重
python scripts/train_sb3_deeplob.py --config configs/exp_reward_aggressive.yaml
python scripts/train_sb3_deeplob.py --config configs/exp_reward_conservative.yaml

# 實驗 2: 學習率
python scripts/train_sb3_deeplob.py --config configs/exp_lr_low.yaml
python scripts/train_sb3_deeplob.py --config configs/exp_lr_high.yaml

# 實驗 3: LSTM 大小
python scripts/train_sb3_deeplob.py --config configs/exp_lstm_256.yaml
python scripts/train_sb3_deeplob.py --config configs/exp_lstm_512.yaml
```

**方案 B: 自動優化（推薦）**

```bash
# 使用 Optuna
python scripts/optuna_optimization.py --n_trials 50
```

**選擇最佳配置**:
- 比較所有實驗的 Sharpe Ratio
- 選擇 Sharpe Ratio 最高且穩定的配置

---

### 9.4 階段四：長期訓練（3-5 天）

**目標**: 使用最佳配置進行長期訓練

**步驟**:

```bash
# 1. 使用最佳配置訓練 1M steps
python scripts/train_sb3_deeplob.py \
    --config configs/best_config.yaml \
    --timesteps 1000000

# 2. 持續監控
tensorboard --logdir logs/sb3_deeplob/

# 3. 定期檢查 checkpoints
ls -lh checkpoints/sb3/ppo_deeplob/
```

**監控重點**:
- 每 10K steps 檢查一次 TensorBoard
- 確保 `ep_rew_mean` 持續上升
- 確保 `approx_kl` < 0.02（訓練穩定）
- 確保無 NaN 或錯誤

---

### 9.5 階段五：評估與回測（2-3 天）

**目標**: 全面評估最終模型

**步驟**:

```bash
# 1. 訓練集評估
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --data_mode train \
    --n_episodes 50 \
    --save_report

# 2. 驗證集評估
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --data_mode val \
    --n_episodes 50 \
    --save_report

# 3. 測試集評估（最終性能）
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --data_mode test \
    --n_episodes 50 \
    --save_report

# 4. 詳細回測分析
# TODO: 實作回測系統
```

**驗收標準**:
- ✅ 測試集 Sharpe Ratio > 2.0
- ✅ 最大回撤 < 10%
- ✅ 勝率 > 55%
- ✅ 訓練/驗證/測試性能一致（無過擬合）

---

### 9.6 階段六：部署準備（1-2 天）

**目標**: 準備模型部署

**步驟**:

```bash
# 1. 模型轉換（TorchScript/ONNX）
# TODO: 實作模型轉換

# 2. 推理速度測試
# TODO: 實作推理基準測試

# 3. 實盤模擬測試
# TODO: 實作紙上交易系統

# 4. 文檔整理
# - 訓練日誌
# - 最佳配置
# - 性能報告
# - 部署指南
```

---

## 十、最佳實踐

### 10.1 訓練前檢查清單

- [ ] 數據已正確預處理（V7 pipeline）
- [ ] 標籤分布合理（30/40/30）
- [ ] DeepLOB 模型已訓練（準確率 > 65%）
- [ ] 環境通過驗證（`verify_env.py`）
- [ ] 配置文件語法正確
- [ ] TensorBoard 已啟動
- [ ] 有足夠的硬碟空間（>10GB）
- [ ] GPU 可用且記憶體充足

### 10.2 訓練中檢查清單

- [ ] 定期檢查 TensorBoard（每 1 小時）
- [ ] 監控 GPU 利用率（應 > 80%）
- [ ] 檢查 `approx_kl` < 0.02
- [ ] 檢查 `explained_variance` > 0.7
- [ ] 確保無 NaN 或錯誤訊息
- [ ] 定期備份 checkpoints

### 10.3 訓練後檢查清單

- [ ] 評估訓練集性能
- [ ] 評估驗證集性能
- [ ] 評估測試集性能
- [ ] 檢查過擬合（訓練/測試差距）
- [ ] 保存最佳配置
- [ ] 撰寫實驗報告
- [ ] 備份所有結果

---

## 十一、常用指令速查

### 訓練指令

```bash
# 快速測試（10K steps）
python scripts/train_sb3_deeplob.py --test

# 短期訓練（100K steps）
python scripts/train_sb3_deeplob.py --timesteps 100000

# 長期訓練（1M steps）
python scripts/train_sb3_deeplob.py --timesteps 1000000

# 使用自訂配置
python scripts/train_sb3_deeplob.py --config configs/my_config.yaml
```

### 監控指令

```bash
# TensorBoard
tensorboard --logdir logs/sb3_deeplob/ --port 6006

# GPU 監控
nvidia-smi -l 1

# 檢查訓練進度
tail -f logs/sb3_deeplob/train.log
```

### 評估指令

```bash
# 評估最佳模型
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 20 \
    --save_report

# 評估測試集
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --data_mode test \
    --n_episodes 50
```

### 數據檢查指令

```bash
# 分析標籤分布
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode analyze_all

# 驗證環境
python scripts/verify_env.py
```

---

## 十二、參考資源

### 文檔

- **獎勵計算指南**: [docs/REWARD_CALCULATION_GUIDE.md](REWARD_CALCULATION_GUIDE.md)
- **交易成本配置**: [docs/TAIWAN_STOCK_TRANSACTION_COSTS.md](TAIWAN_STOCK_TRANSACTION_COSTS.md)
- **DeepLOB 訓練報告**: [docs/1.DeepLOB 台股模型訓練最終報告.md](1.DeepLOB%20台股模型訓練最終報告.md)
- **SB3 實作報告**: [docs/SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md)

### 官方文檔

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **PPO 論文**: https://arxiv.org/abs/1707.06347
- **Gymnasium**: https://gymnasium.farama.org/

### 相關專案

- **FinRL**: https://github.com/AI4Finance-Foundation/FinRL
- **SB3 教程**: https://github.com/araffin/rl-tutorial-jnrr19

---

## 附錄

### A. 推薦配置組合

#### A.1 高頻策略（快速進出）

```yaml
env_config:
  reward:
    pnl_scale: 1.0
    cost_penalty: 0.3          # 低成本懲罰
    inventory_penalty: 0.05    # 高庫存懲罰
    risk_penalty: 0.01

ppo:
  learning_rate: 0.0003
  batch_size: 128
  lstm_hidden_size: 256
```

#### A.2 波段策略（允許持倉）

```yaml
env_config:
  reward:
    pnl_scale: 1.5
    cost_penalty: 1.5          # 中高成本懲罰
    inventory_penalty: 0.001   # 低庫存懲罰
    risk_penalty: 0.005

ppo:
  learning_rate: 0.0003
  batch_size: 64
  lstm_hidden_size: 384
  gamma: 0.995               # 高折扣因子（長期）
```

#### A.3 保守策略（風險優先）

```yaml
env_config:
  reward:
    pnl_scale: 1.0
    cost_penalty: 2.0          # 高成本懲罰
    inventory_penalty: 0.02    # 高庫存懲罰
    risk_penalty: 0.02         # 高風險懲罰

ppo:
  learning_rate: 0.0001        # 低學習率（穩定）
  batch_size: 64
  lstm_hidden_size: 256
  clip_range: 0.1              # 保守更新
```

---

**文件版本**: v1.0
**最後更新**: 2025-10-26
**作者**: SB3-DeepLOB Team
**下次更新計劃**: 加入 Optuna 自動優化腳本、回測系統整合
