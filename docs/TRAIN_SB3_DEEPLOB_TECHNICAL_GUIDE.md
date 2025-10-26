# train_sb3_deeplob.py 技術指南

## 文檔概述

本文檔詳細說明 `train_sb3_deeplob.py` 的訓練流程、技術架構、自定義函數及模型學習機制。

**文檔版本**: v1.0
**更新日期**: 2025-10-26
**適用版本**: SB3-DeepLOB v4.0

---

## 目錄

1. [專案背景與目標](#1-專案背景與目標)
2. [整體架構](#2-整體架構)
3. [訓練流程詳解](#3-訓練流程詳解)
4. [核心模組技術說明](#4-核心模組技術說明)
5. [自定義函數說明](#5-自定義函數說明)
6. [模型學習機制](#6-模型學習機制)
7. [配置參數詳解](#7-配置參數詳解)
8. [使用範例](#8-使用範例)
9. [常見問題與調試](#9-常見問題與調試)

---

## 1. 專案背景與目標

### 1.1 專案目標

實現基於台股 LOB (Limit Order Book) 數據的高頻交易系統，使用雙層學習架構：

- **第一層 (DeepLOB)**: CNN-LSTM 模型學習價格變動預測
- **第二層 (PPO)**: 強化學習算法學習最優交易策略

### 1.2 技術選型

| 組件 | 技術選擇 | 原因 |
|------|---------|------|
| 深度學習框架 | PyTorch 2.5 | 靈活性、GPU 支持 |
| 強化學習框架 | Stable-Baselines3 | 簡單穩定、LSTM 支持完善 |
| 環境標準 | Gymnasium | OpenAI Gym 官方升級版 |
| 硬體 | NVIDIA RTX 5090 (32GB) | 大批量訓練、混合精度 |

### 1.3 性能指標

- DeepLOB 準確率: **72.98%** ✅ (已達成)
- 目標 Sharpe Ratio: **> 2.0** (待驗證)
- GPU 利用率: **> 85%** (待驗證)

---

## 2. 整體架構

### 2.1 系統架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                     train_sb3_deeplob.py                        │
│                     主訓練腳本 (Orchestrator)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 初始化並協調
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        三大核心模組                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────┐         │
│  │  1. DeepLOBExtractor (特徵提取器)                 │         │
│  │  [deeplob_feature_extractor.py]                   │         │
│  │                                                    │         │
│  │  - 載入預訓練 DeepLOB 模型                        │         │
│  │  - 凍結權重 (不參與 PPO 訓練)                     │         │
│  │  - 提取深層特徵 (LSTM hidden / 預測概率)          │         │
│  │  - 特徵融合 MLP (28維 → features_dim)            │         │
│  └───────────────────────────────────────────────────┘         │
│                              │                                  │
│                              │ 輸出特徵                         │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────┐         │
│  │  2. TaiwanLOBTradingEnv (交易環境)                │         │
│  │  [tw_lob_trading_env.py]                          │         │
│  │                                                    │         │
│  │  觀測空間: 28 維                                   │         │
│  │    - LOB 原始特徵 (20維): 5檔買賣價量             │         │
│  │    - DeepLOB 預測 (3維): 下跌/持平/上漲概率       │         │
│  │    - 交易狀態 (5維): 持倉/庫存/成本/時間/動作     │         │
│  │                                                    │         │
│  │  動作空間: Discrete(3)                             │         │
│  │    - 0: Hold (持有)                                │         │
│  │    - 1: Buy (買入)                                 │         │
│  │    - 2: Sell (賣出)                                │         │
│  │                                                    │         │
│  │  獎勵函數: 多組件設計                              │         │
│  │    - PnL 獎勵 (價格變動 × 持倉)                    │         │
│  │    - 交易成本懲罰 (-0.1% 手續費)                   │         │
│  │    - 庫存懲罰 (避免長期持倉)                       │         │
│  │    - 風險調整項 (波動率懲罰)                       │         │
│  └───────────────────────────────────────────────────┘         │
│                              │                                  │
│                              │ 狀態 & 獎勵                      │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────┐         │
│  │  3. PPO (強化學習算法)                             │         │
│  │  [Stable-Baselines3]                              │         │
│  │                                                    │         │
│  │  Policy: MlpPolicy + DeepLOBExtractor              │         │
│  │    - Actor (π): [256, 128] → 動作概率              │         │
│  │    - Critic (V): [256, 128] → 狀態價值             │         │
│  │                                                    │         │
│  │  核心超參數:                                        │         │
│  │    - Learning Rate: 3e-4                           │         │
│  │    - Gamma (折扣因子): 0.99                        │         │
│  │    - N Steps (rollout): 2048                       │         │
│  │    - Batch Size: 64                                │         │
│  │    - Clip Range: 0.2                               │         │
│  │    - Entropy Coef: 0.01                            │         │
│  └───────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 訓練循環
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       訓練回調 (Callbacks)                       │
├─────────────────────────────────────────────────────────────────┤
│  - CheckpointCallback: 每 50K steps 保存模型                     │
│  - EvalCallback: 每 10K steps 在驗證集評估                       │
│  - 自動保存最佳模型 (based on mean reward)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 數據流向

```
台股數據 (NPZ)
     │
     ├─ LOB 時序數據 (100 timesteps × 20 features)
     ├─ 標籤 (0/1/2: 下跌/持平/上漲)
     └─ 價格數據 (中價)
     │
     ▼
TaiwanLOBTradingEnv
     │
     ├─ DeepLOB 預測 (3維概率)
     ├─ LOB 當前特徵 (20維)
     └─ 交易狀態 (5維)
     │
     ▼ 觀測 (28維)
DeepLOBExtractor
     │
     ├─ 特徵融合 MLP
     └─ 輸出 features_dim (128維)
     │
     ▼
PPO Policy (Actor-Critic)
     │
     ├─ Actor → 動作概率 P(a|s)
     └─ Critic → 狀態價值 V(s)
     │
     ▼
Action (0/1/2)
     │
     ▼
Environment Step
     │
     ├─ 執行交易
     ├─ 計算獎勵
     └─ 更新狀態
     │
     ▼
PPO 學習
     │
     ├─ 收集 rollout (n_steps=2048)
     ├─ 計算 Advantage
     └─ 更新策略 (clip gradient)
```

---

## 3. 訓練流程詳解

### 3.1 主流程 (main 函數)

```python
def main():
    """訓練流程（9個步驟）"""

    # 1. 載入配置文件
    config = load_config(args.config)

    # 2. 驗證 DeepLOB 檢查點
    verify_deeplob_checkpoint(args.deeplob_checkpoint)

    # 3. 創建訓練環境（向量化）
    env = create_vec_env(config, n_envs=args.n_envs)

    # 4. 創建評估環境（驗證集）
    eval_env = create_eval_env(config)

    # 5. 設置訓練回調
    callbacks = create_callbacks(config, eval_env)

    # 6. 創建 PPO + DeepLOB 模型
    model = create_ppo_deeplob_model(env, config, deeplob_checkpoint)

    # 7. 開始訓練
    model = train_model(model, total_timesteps, callbacks)

    # 8. 保存最終模型
    save_path = save_final_model(model, config)

    # 9. 輸出總結報告
    logger.info(f"最佳模型: {save_path}")
```

### 3.2 訓練階段細節

#### 階段 A: 初始化 (0-5 分鐘)

1. **檢查 GPU 可用性**
   ```python
   if torch.cuda.is_available():
       device = 'cuda'
       logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

2. **載入 DeepLOB 模型**
   ```python
   checkpoint = torch.load(checkpoint_path)
   deeplob_model = DeepLOB(**model_config)
   deeplob_model.load_state_dict(checkpoint['model_state_dict'])
   deeplob_model.eval()  # 設為評估模式
   ```

3. **凍結 DeepLOB 權重**
   ```python
   for param in deeplob_model.parameters():
       param.requires_grad = False
   ```

#### 階段 B: 環境創建 (5-10 分鐘)

1. **載入台股數據**
   - 訓練集: 5,584,553 樣本
   - 驗證集: 828,011 樣本
   - 支持數據採樣 (預設 10%) 以減少記憶體

2. **創建向量化環境**
   ```python
   # 單環境
   env = DummyVecEnv([lambda: TaiwanLOBTradingEnv(config)])

   # 多環境 (n_envs=4)
   env_fns = [make_env(config, i) for i in range(4)]
   env = SubprocVecEnv(env_fns)
   ```

#### 階段 C: 訓練循環 (主要時間)

**PPO 訓練循環**（每 2048 steps 一次更新）：

```
For timestep in [0, total_timesteps]:
    # 1. 收集 Rollout (n_steps=2048)
    For step in [0, n_steps]:
        action = policy.predict(observation)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, value, log_prob)

    # 2. 計算 Advantage & Returns
    advantages = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
    returns = advantages + values

    # 3. 策略更新 (n_epochs=10)
    For epoch in [0, n_epochs]:
        For batch in get_batches(buffer, batch_size=64):
            # 3.1 計算策略損失
            log_prob_new, entropy = policy.evaluate_actions(batch.obs, batch.actions)
            ratio = torch.exp(log_prob_new - batch.log_probs)

            # PPO Clip
            surr1 = ratio * batch.advantages
            surr2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * batch.advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 3.2 計算價值損失
            value_pred = policy.predict_values(batch.obs)
            value_loss = F.mse_loss(value_pred, batch.returns)

            # 3.3 計算熵損失（鼓勵探索）
            entropy_loss = -entropy.mean()

            # 3.4 總損失
            loss = policy_loss + vf_coef*value_loss + ent_coef*entropy_loss

            # 3.5 梯度更新
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

    # 4. Callbacks
    if timestep % eval_freq == 0:
        evaluate_policy(eval_env)
        save_best_model()

    if timestep % checkpoint_freq == 0:
        save_checkpoint()
```

#### 階段 D: 評估與保存

1. **驗證集評估** (每 10K steps)
   ```python
   mean_reward, std_reward = evaluate_policy(
       model, eval_env,
       n_eval_episodes=10
   )
   ```

2. **保存檢查點** (每 50K steps)
   ```python
   model.save(f"checkpoints/ppo_model_{timestep}_steps")
   ```

3. **保存最佳模型**
   ```python
   if mean_reward > best_mean_reward:
       model.save("checkpoints/best_model")
   ```

---

## 4. 核心模組技術說明

### 4.1 DeepLOBExtractor (特徵提取器)

**文件**: [src/models/deeplob_feature_extractor.py](src/models/deeplob_feature_extractor.py:54)

#### 核心功能

將預訓練 DeepLOB 模型整合到 SB3 策略中，作為凍結的特徵提取器。

#### 類別繼承關係

```python
class DeepLOBExtractor(BaseFeaturesExtractor):
    """
    BaseFeaturesExtractor (SB3 抽象基類)
        │
        ├─ 定義接口: forward(observations) -> features
        └─ 必須實現: __init__, forward
    """
```

#### 初始化參數

```python
DeepLOBExtractor(
    observation_space: gym.spaces.Box,  # 觀測空間 (28維)
    features_dim: int = 128,             # 輸出特徵維度
    deeplob_checkpoint: str = None,      # DeepLOB 模型路徑
    use_lstm_hidden: bool = True,        # 是否使用 LSTM 隱藏層
    freeze_deeplob: bool = True,         # 是否凍結 DeepLOB
    extractor_net_arch: list = None      # MLP 網絡架構
)
```

#### 架構設計

**模式選擇**:

1. **Mode 1 (簡單模式)** - `use_lstm_hidden=False`
   ```
   觀測 (28維) → MLP → features_dim

   輸入: [LOB(20) + DeepLOB_Pred(3) + State(5)]
   網絡: Linear(28 → 256) → ReLU → Linear(256 → 128) → ReLU
         → Linear(128 → features_dim) → ReLU
   輸出: features (features_dim 維)
   ```

2. **Mode 2 (特徵提取模式)** - `use_lstm_hidden=True`
   ```
   LOB → DeepLOB.LSTM → Hidden(64) ┐
   State (5維)                      ├→ Concat → MLP → features_dim
                                    ┘

   理論輸入: [LOB(20) + State(5) + LSTM_Hidden(64)] = 89維
   實際: 簡化為 Mode 1（當前實現）
   ```

   **註**: Mode 2 完整實現需要修改環境以傳遞 LOB 時序數據。

#### 權重凍結機制

```python
def _load_deeplob(self, checkpoint_path: str):
    """載入並凍結 DeepLOB"""
    # 1. 載入檢查點
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # 2. 提取配置
    model_config = checkpoint['config']['model']

    # 3. 初始化模型
    model = DeepLOB(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4. 凍結權重
    for param in model.parameters():
        param.requires_grad = False

    # 5. 評估模式
    model.eval()

    return model
```

#### Forward 傳播

```python
def forward(self, observations: torch.Tensor) -> torch.Tensor:
    """特徵提取

    輸入: observations (batch_size, 28)
    輸出: features (batch_size, features_dim)
    """
    # Mode 1: 使用完整觀測
    combined_features = observations  # (batch, 28)

    # 通過 MLP 提取器
    features = self.extractor_net(combined_features)  # (batch, features_dim)

    return features
```

---

### 4.2 TaiwanLOBTradingEnv (交易環境)

**文件**: [src/envs/tw_lob_trading_env.py](src/envs/tw_lob_trading_env.py:31)

#### 環境規格

| 屬性 | 規格 | 說明 |
|------|------|------|
| 觀測空間 | Box(28,) | 連續向量 |
| 動作空間 | Discrete(3) | {0: Sell, 1: Hold, 2: Buy} |
| 獎勵範圍 | [-∞, +∞] | 無界（PnL based） |
| Episode 長度 | 500 steps | 可配置 |

#### 觀測組成 (28維)

```python
observation = [
    # ===== LOB 原始特徵 (20維) =====
    # 5檔買賣價量（交錯式）
    best_bid_price,    # 最佳買價
    best_bid_volume,   # 最佳買量
    bid_2_price,       # 第2檔買價
    bid_2_volume,      # 第2檔買量
    ...,               # 買方 3-5 檔
    best_ask_price,    # 最佳賣價
    best_ask_volume,   # 最佳賣量
    ask_2_price,       # 第2檔賣價
    ask_2_volume,      # 第2檔賣量
    ...,               # 賣方 3-5 檔

    # ===== DeepLOB 預測 (3維) =====
    prob_down,         # 下跌概率
    prob_stationary,   # 持平概率
    prob_up,           # 上漲概率

    # ===== 交易狀態 (5維) =====
    normalized_position,      # 持倉 / max_position
    normalized_inventory,     # 庫存 / initial_balance
    normalized_cost,          # 成本 / initial_balance
    time_progress,            # current_step / max_steps
    prev_action_normalized    # prev_action / 2.0
]
```

#### 動作空間

```python
action_space = Discrete(3)

# 動作映射
0 → Sell (賣出 1 單位)
1 → Hold (持有，不動)
2 → Buy (買入 1 單位)

# 持倉限制
position ∈ [-max_position, +max_position]  # 預設 [-1, +1]
```

#### 獎勵函數 (多組件設計)

**獎勵計算由 `RewardShaper` 執行**，主要組件：

1. **基礎 PnL 獎勵**
   ```python
   pnl_reward = position * (next_price - prev_price)
   ```
   - 持多倉：價格上漲 → 正獎勵
   - 持空倉：價格下跌 → 正獎勵

2. **交易成本懲罰**
   ```python
   cost_penalty = -transaction_cost
   transaction_cost = |position_change| * price * 0.001
   ```
   - 手續費率: 0.1% (可配置)

3. **庫存懲罰**
   ```python
   inventory_penalty = -abs(inventory) * inventory_weight
   ```
   - 避免長時間持倉
   - 鼓勵快速平倉

4. **風險調整項**
   ```python
   risk_penalty = -abs(position) * volatility * risk_weight
   ```
   - 高波動率 → 增加懲罰
   - 鼓勵風險控制

**總獎勵**:
```python
total_reward = pnl_reward + cost_penalty + inventory_penalty + risk_penalty
```

#### 核心方法

**1. reset() - 重置環境**

```python
def reset(self, seed=None, options=None) -> Tuple[obs, info]:
    """重置環境到初始狀態"""
    # 1. 重置狀態變數
    self.current_step = 0
    self.position = 0
    self.entry_price = 0.0
    self.balance = self.initial_balance
    self.inventory = 0.0
    self.total_cost = 0.0
    self.prev_action = 0

    # 2. 隨機選擇起始樣本
    max_start = self.data_length - self.max_steps
    self.current_data_idx = np.random.randint(0, max_start)

    # 3. 初始化 LOB 歷史 (100 timesteps)
    self.lob_history = self.lob_data[self.current_data_idx].tolist()

    # 4. 生成觀測
    obs = self._get_observation()
    info = self._get_info()

    return obs, info
```

**2. step() - 執行動作**

```python
def step(self, action: int) -> Tuple[obs, reward, terminated, truncated, info]:
    """執行交易動作並更新環境"""
    # 1. 獲取當前價格
    current_price = self.prices[self.current_data_idx]

    # 2. 執行交易邏輯
    if action == 0:  # Sell
        self.position = max(-self.max_position, self.position - 1)
    elif action == 2:  # Buy
        self.position = min(self.max_position, self.position + 1)
    # action == 1: Hold (不動)

    # 3. 計算交易成本
    if action != prev_position + 1:
        position_change = abs(action - (prev_position + 1))
        transaction_cost = position_change * current_price * 0.001
        self.total_cost += transaction_cost

    # 4. 更新庫存
    if self.position != 0:
        self.inventory = self.position * (current_price - self.entry_price)

    # 5. 推進時間
    self.current_step += 1
    self.current_data_idx += 1
    next_price = self.prices[self.current_data_idx]

    # 6. 計算獎勵
    reward = self.reward_shaper.calculate_reward(
        prev_state, action, new_state, transaction_cost
    )

    # 7. 檢查終止
    terminated = False  # 無提前終止條件
    truncated = (self.current_step >= self.max_steps)

    # 8. 生成新觀測
    obs = self._get_observation()
    info = self._get_info()

    return obs, reward, terminated, truncated, info
```

**3. _get_observation() - 生成觀測**

```python
def _get_observation(self) -> np.ndarray:
    """生成 28 維觀測"""
    # 1. 當前 LOB 特徵（最後一個時間步）
    current_lob = np.array(self.lob_history[-1], dtype=np.float32)  # (20,)

    # 2. DeepLOB 預測
    if self.deeplob_model is not None:
        with torch.no_grad():
            lob_seq = torch.FloatTensor(self.lob_history).unsqueeze(0)  # (1, 100, 20)
            deeplob_probs = self.deeplob_model.predict_proba(lob_seq)[0].numpy()  # (3,)
    else:
        deeplob_probs = np.random.rand(3).astype(np.float32)
        deeplob_probs /= deeplob_probs.sum()

    # 3. 交易狀態
    state_features = np.array([
        self.position / self.max_position,
        self.inventory / self.initial_balance,
        self.total_cost / self.initial_balance,
        self.current_step / self.max_steps,
        self.prev_action / 2.0
    ], dtype=np.float32)  # (5,)

    # 4. 串接所有特徵
    obs = np.concatenate([current_lob, deeplob_probs, state_features])  # (28,)

    return obs
```

---

### 4.3 PPO (Proximal Policy Optimization)

**框架**: Stable-Baselines3

#### 算法原理

PPO 是一種 **on-policy** 強化學習算法，通過限制策略更新幅度來保證訓練穩定性。

**核心公式**:

1. **策略梯度目標（有約束）**
   ```
   L^CLIP(θ) = E_t[ min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t) ]

   其中:
   r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  # 重要性採樣比
   A_t = 優勢函數 (Advantage)
   ε = clip_range (預設 0.2)
   ```

2. **優勢函數（GAE）**
   ```
   A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2·δ_{t+2} + ...

   其中:
   δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  # TD 誤差
   γ = 折扣因子 (0.99)
   λ = GAE lambda (0.95)
   ```

3. **價值函數損失**
   ```
   L^VF(θ) = E_t[ (V_θ(s_t) - V_t^target)^2 ]

   V_t^target = A_t + V(s_t)  # Returns
   ```

4. **熵損失（鼓勵探索）**
   ```
   L^ENT(θ) = E_t[ -H(π_θ(·|s_t)) ]

   H(π) = -Σ π(a|s)·log(π(a|s))  # 策略熵
   ```

5. **總損失**
   ```
   L(θ) = -L^CLIP(θ) + c1·L^VF(θ) - c2·L^ENT(θ)

   c1 = vf_coef (0.5)
   c2 = ent_coef (0.01)
   ```

#### PPO 配置

```python
model = PPO(
    "MlpPolicy",           # 策略類型（多層感知器）
    env,                   # 環境

    # ===== 學習率配置 =====
    learning_rate=3e-4,    # 優化器學習率

    # ===== Rollout 配置 =====
    n_steps=2048,          # Rollout buffer 大小（每次收集 2048 步）
    batch_size=64,         # Mini-batch 大小
    n_epochs=10,           # 每次 rollout 更新 10 次

    # ===== 折扣與優勢估計 =====
    gamma=0.99,            # 折扣因子
    gae_lambda=0.95,       # GAE λ

    # ===== PPO Clip =====
    clip_range=0.2,        # 策略更新裁剪範圍 [0.8, 1.2]

    # ===== 損失係數 =====
    ent_coef=0.01,         # 熵係數（鼓勵探索）
    vf_coef=0.5,           # 價值函數係數
    max_grad_norm=0.5,     # 梯度裁剪

    # ===== 策略網絡配置 =====
    policy_kwargs=dict(
        features_extractor_class=DeepLOBExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            deeplob_checkpoint="checkpoints/deeplob_v5_best.pth"
        ),
        net_arch=dict(
            pi=[256, 128],  # Actor: 128 → 256 → 128 → 3
            vf=[256, 128]   # Critic: 128 → 256 → 128 → 1
        )
    ),

    # ===== 日誌與設備 =====
    tensorboard_log="logs/sb3_deeplob/",
    verbose=1,
    device="cuda"
)
```

#### Actor-Critic 架構

```
觀測 (28維)
    │
    ▼
DeepLOBExtractor (凍結 DeepLOB + MLP)
    │
    ├─ DeepLOB (凍結) → LOB 深層特徵
    ├─ MLP: 28 → 256 → 128
    └─ 輸出: features (128維)
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
Actor (π)     Critic (V)     (共享特徵)
    │              │
Linear(128→256) Linear(128→256)
ReLU            ReLU
Linear(256→128) Linear(256→128)
ReLU            ReLU
Linear(128→3)   Linear(128→1)
Softmax         -
    │              │
    ▼              ▼
動作概率        狀態價值
P(a|s)          V(s)
```

---

## 5. 自定義函數說明

### 5.1 配置與驗證函數

#### load_config()

**位置**: [train_sb3_deeplob.py:76](train_sb3_deeplob.py:76)

```python
def load_config(config_path: str) -> dict:
    """載入 YAML 配置文件

    參數:
        config_path: 配置文件路徑 (如 'configs/sb3_config.yaml')

    返回:
        config: 配置字典

    配置結構:
        {
            'env_config': {...},           # 環境配置
            'ppo': {...},                  # PPO 超參數
            'deeplob_extractor': {...},    # DeepLOB 特徵提取器配置
            'training': {...},             # 訓練配置
            'callbacks': {...},            # 回調配置
            'test_mode': {...}             # 測試模式配置
        }
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
```

#### verify_deeplob_checkpoint()

**位置**: [train_sb3_deeplob.py:83](train_sb3_deeplob.py:83)

```python
def verify_deeplob_checkpoint(checkpoint_path: str):
    """驗證 DeepLOB 檢查點存在且可載入

    功能:
        1. 檢查文件存在性
        2. 嘗試載入檢查點
        3. 顯示模型信息（epoch, 準確率等）

    參數:
        checkpoint_path: DeepLOB 檢查點路徑

    拋出:
        FileNotFoundError: 文件不存在
        RuntimeError: 檢查點載入失敗

    檢查點格式:
        {
            'epoch': int,
            'model_state_dict': OrderedDict,
            'optimizer_state_dict': OrderedDict,
            'val_acc': float,
            'test_acc': float,
            'config': {
                'model': {
                    'input_shape': [100, 20],
                    'num_classes': 3,
                    'conv1_filters': 32,
                    ...
                }
            }
        }
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DeepLOB 檢查點不存在: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info(f"✅ DeepLOB 檢查點驗證成功: {checkpoint_path}")

        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                logger.info(f"  - Epoch: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                logger.info(f"  - 驗證準確率: {checkpoint['val_acc']:.4f}")

    except Exception as e:
        raise RuntimeError(f"DeepLOB 檢查點載入失敗: {e}")
```

---

### 5.2 環境創建函數

#### make_env()

**位置**: [train_sb3_deeplob.py:109](train_sb3_deeplob.py:109)

```python
def make_env(env_config: dict, rank: int = 0):
    """環境工廠函數（用於向量化環境）

    參數:
        env_config: 環境配置字典
        rank: 環境編號（用於多進程環境）

    返回:
        _init: 環境初始化函數

    使用場景:
        - 創建多個並行環境（SubprocVecEnv）
        - 每個環境有獨立的隨機種子

    範例:
        env_fns = [make_env(config, i) for i in range(4)]
        env = SubprocVecEnv(env_fns)
    """
    def _init():
        env = TaiwanLOBTradingEnv(env_config)
        env = Monitor(env)  # 包裝 Monitor（記錄 episode 統計）
        return env
    return _init
```

#### create_vec_env()

**位置**: [train_sb3_deeplob.py:118](train_sb3_deeplob.py:118)

```python
def create_vec_env(config: dict, n_envs: int = 1, vec_type: str = "dummy"):
    """創建向量化環境

    參數:
        config: 完整配置字典
        n_envs: 並行環境數量
        vec_type: 向量化類型
            - "dummy": DummyVecEnv（單進程，適合調試）
            - "subproc": SubprocVecEnv（多進程，適合訓練）

    返回:
        env: 向量化環境

    向量化環境類型對比:

        DummyVecEnv:
            - 單進程順序執行
            - 記憶體共享
            - 適合調試、小規模訓練
            - 性能: ⭐⭐

        SubprocVecEnv:
            - 多進程並行執行
            - 每個環境獨立進程
            - 適合大規模訓練（充分利用多核 CPU）
            - 性能: ⭐⭐⭐⭐

    範例:
        # 單環境
        env = create_vec_env(config, n_envs=1)

        # 4 個並行環境（多進程）
        env = create_vec_env(config, n_envs=4, vec_type="subproc")
    """
    env_config = config['env_config']

    if n_envs == 1:
        env = TaiwanLOBTradingEnv(env_config)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        logger.info("✅ 創建單一環境")
    else:
        env_fns = [make_env(env_config, i) for i in range(n_envs)]

        if vec_type == "subproc":
            env = SubprocVecEnv(env_fns)
            logger.info(f"✅ 創建 SubprocVecEnv ({n_envs} 個環境)")
        else:
            env = DummyVecEnv(env_fns)
            logger.info(f"✅ 創建 DummyVecEnv ({n_envs} 個環境)")

    return env
```

#### create_eval_env()

**位置**: [train_sb3_deeplob.py:140](train_sb3_deeplob.py:140)

```python
def create_eval_env(config: dict):
    """創建評估環境（使用驗證集）

    參數:
        config: 完整配置字典

    返回:
        env: 評估環境（單個，非向量化）

    配置覆蓋:
        eval_env_config:
            data_mode: 'val'          # 使用驗證集
            data_sample_ratio: 0.1    # 10% 採樣（減少評估時間）

    注意:
        - 評估環境不向量化（不需要並行）
        - 使用驗證集數據（避免過擬合測試集）
    """
    eval_config = config.get('evaluation', {}).get('eval_env_config', {})
    env_config = config['env_config'].copy()
    env_config.update(eval_config)

    env = TaiwanLOBTradingEnv(env_config)
    env = Monitor(env)
    logger.info("✅ 創建評估環境（驗證集）")

    return env
```

---

### 5.3 回調創建函數

#### create_callbacks()

**位置**: [train_sb3_deeplob.py:153](train_sb3_deeplob.py:153)

```python
def create_callbacks(config: dict, eval_env):
    """創建訓練回調

    參數:
        config: 完整配置字典
        eval_env: 評估環境

    返回:
        CallbackList: 回調列表（或 None）

    包含回調:
        1. CheckpointCallback: 定期保存模型
        2. EvalCallback: 定期評估並保存最佳模型

    配置範例:
        callbacks:
            checkpoint:
                enabled: true
                save_freq: 50000        # 每 50K steps
                save_path: 'checkpoints/sb3/ppo_deeplob'
                name_prefix: 'ppo_model'

            eval:
                enabled: true
                eval_freq: 10000        # 每 10K steps
                n_eval_episodes: 10     # 評估 10 個 episodes
                best_model_save_path: 'checkpoints/sb3/ppo_deeplob'
                deterministic: true     # 評估時使用確定性策略
    """
    callbacks = []

    # CheckpointCallback
    checkpoint_config = config.get('callbacks', {}).get('checkpoint', {})
    if checkpoint_config.get('enabled', True):
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_config.get('save_freq', 50000),
            save_path=checkpoint_config.get('save_path', 'checkpoints/sb3/ppo_deeplob'),
            name_prefix=checkpoint_config.get('name_prefix', 'ppo_model'),
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

    # EvalCallback
    eval_config = config.get('callbacks', {}).get('eval', {})
    if eval_config.get('enabled', True) and eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=eval_config.get('eval_freq', 10000),
            n_eval_episodes=eval_config.get('n_eval_episodes', 10),
            best_model_save_path=eval_config.get('best_model_save_path', 'checkpoints/sb3/ppo_deeplob'),
            log_path=eval_config.get('log_path', 'logs/sb3_eval'),
            deterministic=eval_config.get('deterministic', True),
            render=False,
        )
        callbacks.append(eval_callback)

    return CallbackList(callbacks) if len(callbacks) > 0 else None
```

---

### 5.4 模型創建與訓練函數

#### create_ppo_deeplob_model()

**位置**: [train_sb3_deeplob.py:191](train_sb3_deeplob.py:191)

```python
def create_ppo_deeplob_model(env, config: dict, deeplob_checkpoint: str, device: str = "cuda"):
    """創建整合 DeepLOB 的 PPO 模型

    參數:
        env: 向量化環境
        config: 完整配置字典
        deeplob_checkpoint: DeepLOB 檢查點路徑
        device: 設備 ('cuda' / 'cpu')

    返回:
        model: PPO 模型實例

    核心流程:
        1. 創建 DeepLOB 特徵提取器配置
        2. 配置 PPO 超參數
        3. 初始化 PPO 模型

    Policy 架構:
        觀測 (28) → DeepLOBExtractor (128) → Actor/Critic

        Actor: 128 → 256 → 128 → 3 (動作概率)
        Critic: 128 → 256 → 128 → 1 (狀態價值)
    """
    ppo_config = config.get('ppo', {})
    deeplob_config = config.get('deeplob_extractor', {})

    # 創建 policy_kwargs
    if deeplob_config.get('use_deeplob', True):
        policy_kwargs = make_deeplob_policy_kwargs(
            deeplob_checkpoint=deeplob_checkpoint,
            features_dim=deeplob_config.get('features_dim', 128),
            use_lstm_hidden=deeplob_config.get('use_lstm_hidden', False),
            net_arch=ppo_config.get('net_arch', dict(pi=[256, 128], vf=[256, 128]))
        )
    else:
        # 回退到基礎 MlpPolicy
        policy_kwargs = dict(
            net_arch=ppo_config.get('net_arch', dict(pi=[256, 128], vf=[256, 128]))
        )

    # 創建 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.get('training', {}).get('tensorboard_log', 'logs/sb3_deeplob'),
        verbose=ppo_config.get('verbose', 1),
        seed=ppo_config.get('seed', 42),
        device=device
    )

    logger.info("✅ PPO + DeepLOB 模型創建成功")
    return model
```

#### train_model()

**位置**: [train_sb3_deeplob.py:253](train_sb3_deeplob.py:253)

```python
def train_model(model, total_timesteps: int, callbacks=None, log_interval: int = 10):
    """訓練模型

    參數:
        model: PPO 模型實例
        total_timesteps: 總訓練步數
        callbacks: 回調列表
        log_interval: 日誌輸出間隔

    返回:
        model: 訓練後的模型

    訓練統計:
        - 訓練時間
        - 訓練速度 (steps/sec)
        - TensorBoard 日誌

    日誌內容:
        - rollout/ep_rew_mean: Episode 平均獎勵
        - rollout/ep_len_mean: Episode 平均長度
        - train/policy_loss: 策略損失
        - train/value_loss: 價值損失
        - train/entropy_loss: 熵損失
        - train/clip_fraction: Clip 比例（策略更新被裁剪的比例）
    """
    logger.info("=" * 60)
    logger.info("🚀 開始訓練 (PPO + DeepLOB)")
    logger.info("=" * 60)
    logger.info(f"總步數: {total_timesteps:,}")

    start_time = datetime.now()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
        progress_bar=True
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("✅ 訓練完成")
    logger.info(f"訓練時間: {duration/60:.2f} 分鐘 ({duration/3600:.2f} 小時)")
    logger.info(f"訓練速度: {total_timesteps/duration:.1f} steps/sec")

    return model
```

#### save_final_model()

**位置**: [train_sb3_deeplob.py:281](train_sb3_deeplob.py:281)

```python
def save_final_model(model, config: dict):
    """保存最終模型

    參數:
        model: 訓練後的 PPO 模型
        config: 完整配置字典

    返回:
        save_path: 保存路徑

    保存內容:
        - 模型權重 (policy + value network)
        - 優化器狀態
        - 環境標準化參數（如果有）

    文件格式:
        .zip 壓縮文件（SB3 標準格式）

    範例:
        checkpoints/sb3/ppo_deeplob/ppo_deeplob_final.zip
    """
    save_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints/sb3/ppo_deeplob')
    model_name = config.get('training', {}).get('final_model_name', 'ppo_deeplob_final')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    model.save(save_path)
    logger.info(f"✅ 最終模型已保存: {save_path}.zip")

    return save_path
```

---

## 6. 模型學習機制

### 6.1 雙層學習架構

#### 第一層：DeepLOB (凍結，不學習)

**功能**: 特徵提取器（Feature Extractor）

**訓練狀態**: 凍結（`requires_grad=False`）

**作用**:
- 輸入: LOB 時序數據 (100 × 20)
- 輸出: 價格預測概率 (3維) 或 LSTM 隱藏層 (64維)
- 提供: 深層 LOB 特徵表示

**為什麼凍結？**
1. **已經訓練好**: DeepLOB 在 560 萬樣本上訓練，準確率 72.98%
2. **避免災難性遺忘**: 如果繼續訓練，可能破壞已學習的價格預測能力
3. **提高效率**: 減少可訓練參數，加快訓練速度

#### 第二層：PPO (學習)

**功能**: 策略學習器（Policy Learner）

**訓練狀態**: 學習中（`requires_grad=True`）

**學習內容**:
1. **Actor (策略網絡)**: 學習最優交易策略
   - 輸入: DeepLOB 特徵 + 交易狀態
   - 輸出: 動作概率 P(Buy|s), P(Hold|s), P(Sell|s)
   - 目標: 最大化期望回報

2. **Critic (價值網絡)**: 學習狀態價值評估
   - 輸入: DeepLOB 特徵 + 交易狀態
   - 輸出: 狀態價值 V(s)
   - 目標: 準確估計未來回報

---

### 6.2 PPO 學習過程詳解

#### 步驟 1: 數據收集（Rollout）

**n_steps = 2048**（每 2048 步更新一次策略）

```python
for step in range(n_steps):
    # 1. 策略採樣動作
    action, value, log_prob = policy.predict(observation)

    # 2. 執行動作
    next_obs, reward, done, info = env.step(action)

    # 3. 存入 Rollout Buffer
    buffer.add(
        obs=observation,
        action=action,
        reward=reward,
        value=value,
        log_prob=log_prob
    )

    observation = next_obs
```

**Rollout Buffer 結構**:
```
buffer = {
    'observations': (2048, 28),      # 觀測序列
    'actions': (2048,),              # 動作序列
    'rewards': (2048,),              # 獎勵序列
    'values': (2048,),               # 價值估計
    'log_probs': (2048,),            # 動作對數概率
    'dones': (2048,)                 # 終止標誌
}
```

---

#### 步驟 2: 優勢估計（Advantage Calculation）

**使用 GAE (Generalized Advantage Estimation)**:

```python
def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    """計算 GAE 優勢函數"""
    advantages = np.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        # TD 誤差
        if t == len(rewards) - 1:
            next_value = 0  # 最後一步
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]

        # GAE
        advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae

    # Returns（目標價值）
    returns = advantages + values

    # 標準化優勢（提高穩定性）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns
```

**GAE 優勢**:
- 平衡偏差 (bias) 與方差 (variance)
- `λ=0`: 低方差，高偏差 (TD(0))
- `λ=1`: 高方差，低偏差 (蒙特卡洛)
- `λ=0.95`: 平衡點（常用值）

---

#### 步驟 3: 策略更新（Policy Optimization）

**n_epochs = 10**（每個 rollout 更新 10 次）
**batch_size = 64**（每次更新使用 64 樣本）

```python
for epoch in range(n_epochs):
    # 打亂數據
    indices = np.random.permutation(n_steps)

    # Mini-batch 更新
    for start in range(0, n_steps, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        # 獲取 batch 數據
        obs_batch = buffer.observations[batch_indices]
        actions_batch = buffer.actions[batch_indices]
        old_log_probs_batch = buffer.log_probs[batch_indices]
        advantages_batch = buffer.advantages[batch_indices]
        returns_batch = buffer.returns[batch_indices]

        # ===== Forward Pass =====
        # 1. 重新評估動作
        new_log_probs, entropy = policy.evaluate_actions(obs_batch, actions_batch)

        # 2. 重新評估價值
        new_values = policy.predict_values(obs_batch)

        # ===== Loss Calculation =====
        # 1. 策略損失（PPO Clip）
        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()

        # 2. 價值損失（MSE）
        value_loss = F.mse_loss(new_values, returns_batch)

        # 3. 熵損失（鼓勵探索）
        entropy_loss = -entropy.mean()

        # 4. 總損失
        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        # ===== Backward Pass =====
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

        optimizer.step()
```

---

#### 步驟 4: 評估與保存

**EvalCallback** (每 10K steps):

```python
def evaluate_policy(policy, eval_env, n_eval_episodes=10):
    """評估策略性能"""
    episode_rewards = []

    for i in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            # 確定性策略（取最大概率動作）
            action, _states = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # 保存最佳模型
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        policy.save('checkpoints/best_model')

    return mean_reward, std_reward
```

---

### 6.3 梯度流向圖

```
觀測 (28)
    │
    ▼
DeepLOBExtractor (凍結 ❌)
    │
    │ [梯度不回傳]
    │
    ├─ DeepLOB (凍結)
    └─ MLP (凍結)
    │
    ▼ features (128)
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
Actor MLP      Critic MLP     (可訓練 ✅)
    │              │
    │ [梯度回傳]   │ [梯度回傳]
    │              │
    ▼              ▼
Policy Loss    Value Loss
    │              │
    └──────┬───────┘
           │
           ▼
    Total Loss
           │
           ▼
    Backward Pass
           │
           ▼
    Update Weights (只更新 Actor & Critic)
```

**可訓練參數統計**:
```
DeepLOB:              ~250K (凍結)
DeepLOBExtractor MLP: ~50K  (凍結)
Actor Network:        ~100K (訓練)
Critic Network:       ~100K (訓練)
────────────────────────────
總參數:               ~500K
可訓練參數:           ~200K (40%)
```

---

### 6.4 學習曲線監控

**TensorBoard 指標**:

1. **rollout/ep_rew_mean** (Episode 平均獎勵)
   - 上升 → 策略改進 ✅
   - 下降 → 策略退化 ❌
   - 目標: 持續上升至收斂

2. **train/policy_loss** (策略損失)
   - 初期: 較高（策略不穩定）
   - 後期: 降低（策略收斂）

3. **train/value_loss** (價值損失)
   - 應逐步降低（價值估計越來越準確）

4. **train/entropy** (策略熵)
   - 初期: 高（探索多）
   - 後期: 降低（開發多）
   - 過低: 可能陷入局部最優

5. **train/clip_fraction** (Clip 比例)
   - 0.1-0.3: 正常（適度更新）
   - > 0.5: 更新過激（可能不穩定）

6. **eval/mean_reward** (驗證集獎勵)
   - 與訓練集獎勵對比
   - 差距過大 → 過擬合

---

## 7. 配置參數詳解

### 7.1 配置文件結構

**文件**: [configs/sb3_config.yaml](configs/sb3_config.yaml)

```yaml
# ===== 環境配置 =====
env_config:
  data_dir: 'data/processed_v5/npz'
  max_steps: 500                      # Episode 長度
  initial_balance: 10000.0
  transaction_cost_rate: 0.001        # 0.1% 手續費
  max_position: 1                     # 最大持倉
  data_mode: 'train'                  # train/val/test
  data_sample_ratio: 0.1              # 10% 採樣（減少記憶體）

  deeplob_checkpoint: 'checkpoints/v5/deeplob_v5_best.pth'

  reward_config:
    pnl_weight: 1.0
    cost_weight: 1.0
    inventory_weight: 0.1
    risk_weight: 0.05

# ===== PPO 超參數 =====
ppo:
  learning_rate: 0.0003               # 3e-4
  n_steps: 2048                       # Rollout buffer size
  batch_size: 64
  n_epochs: 10
  gamma: 0.99                         # 折扣因子
  gae_lambda: 0.95
  clip_range: 0.2                     # PPO clip [0.8, 1.2]
  ent_coef: 0.01                      # 熵係數
  vf_coef: 0.5                        # 價值函數係數
  max_grad_norm: 0.5

  net_arch:
    pi: [256, 128]                    # Actor
    vf: [256, 128]                    # Critic

  verbose: 1
  seed: 42

# ===== DeepLOB 特徵提取器 =====
deeplob_extractor:
  use_deeplob: true
  features_dim: 128                   # 輸出特徵維度
  use_lstm_hidden: false              # 簡單模式
  freeze_deeplob: true                # 凍結權重

# ===== 訓練配置 =====
training:
  total_timesteps: 1000000            # 1M steps
  tensorboard_log: 'logs/sb3_deeplob/'
  checkpoint_dir: 'checkpoints/sb3/ppo_deeplob'
  final_model_name: 'ppo_deeplob_final'
  log_interval: 10

# ===== 回調配置 =====
callbacks:
  checkpoint:
    enabled: true
    save_freq: 50000                  # 每 50K steps
    save_path: 'checkpoints/sb3/ppo_deeplob'
    name_prefix: 'ppo_model'

  eval:
    enabled: true
    eval_freq: 10000                  # 每 10K steps
    n_eval_episodes: 10
    best_model_save_path: 'checkpoints/sb3/ppo_deeplob'
    log_path: 'logs/sb3_eval'
    deterministic: true

# ===== 評估環境配置 =====
evaluation:
  eval_env_config:
    data_mode: 'val'                  # 使用驗證集
    data_sample_ratio: 0.1

# ===== 測試模式 =====
test_mode:
  total_timesteps: 10000              # 快速測試
  save_freq: 5000
  eval_freq: 5000
  n_steps: 512
  batch_size: 32
```

### 7.2 關鍵參數調優建議

#### PPO 超參數

| 參數 | 預設值 | 調優範圍 | 影響 |
|------|--------|---------|------|
| learning_rate | 3e-4 | [1e-4, 1e-3] | 學習速度 |
| gamma | 0.99 | [0.95, 0.999] | 長期 vs 短期回報 |
| n_steps | 2048 | [512, 4096] | Rollout 長度 |
| batch_size | 64 | [32, 256] | 更新穩定性 |
| n_epochs | 10 | [5, 20] | 數據利用率 |
| clip_range | 0.2 | [0.1, 0.3] | 策略更新幅度 |
| ent_coef | 0.01 | [0.001, 0.05] | 探索 vs 開發 |

**調優策略**:

1. **學習率** (learning_rate)
   - 過高: 訓練不穩定，策略震盪
   - 過低: 訓練過慢
   - 建議: 從 3e-4 開始，觀察 policy_loss 曲線

2. **折扣因子** (gamma)
   - 接近 1: 重視長期回報（適合持倉策略）
   - 遠離 1: 重視短期回報（適合高頻交易）
   - 建議: 0.99 (高頻) 或 0.995 (波段)

3. **Rollout 步數** (n_steps)
   - 更大: 更穩定，但記憶體消耗高
   - 更小: 更新頻繁，但方差大
   - 建議: 2048 (平衡點)

4. **熵係數** (ent_coef)
   - 更大: 更多探索（初期）
   - 更小: 更多開發（後期）
   - 建議: 0.01 (初期) → 0.001 (後期)，可使用 Schedule

---

## 8. 使用範例

### 8.1 快速測試（10 分鐘）

```bash
conda activate deeplob-pro

# 測試流程（10K steps）
python scripts/train_sb3_deeplob.py \
    --config configs/sb3_config.yaml \
    --deeplob-checkpoint checkpoints/v5/deeplob_v5_best.pth \
    --timesteps 10000 \
    --test \
    --device cuda

# 預期輸出
# ✅ DeepLOB 檢查點驗證成功
# ✅ 創建單一環境
# ✅ PPO + DeepLOB 模型創建成功
# 🚀 開始訓練 (10000 steps)
# [進度條]
# ✅ 訓練完成 (約 10 分鐘)
```

### 8.2 完整訓練（4-8 小時）

```bash
# 1M steps 完整訓練（推薦）
python scripts/train_sb3_deeplob.py \
    --config configs/sb3_config.yaml \
    --timesteps 1000000 \
    --device cuda

# 監控訓練（另一個終端）
tensorboard --logdir logs/sb3_deeplob/

# 訪問 http://localhost:6006
```

### 8.3 高性能訓練（多環境並行）

```bash
# 4 個並行環境（加速訓練）
python scripts/train_sb3_deeplob.py \
    --config configs/sb3_config.yaml \
    --timesteps 2000000 \
    --n-envs 4 \
    --vec-type subproc \
    --device cuda

# 性能提升
# - 單環境: ~1500 steps/sec
# - 4 環境: ~4500 steps/sec (3x 加速)
```

### 8.4 評估訓練模型

```bash
# 評估最佳模型
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 20 \
    --save_report

# 輸出報告
# results/evaluation_report_YYYYMMDD_HHMMSS.json
```

---

## 9. 常見問題與調試

### 9.1 記憶體不足

**問題**: `CUDA out of memory`

**解決方案**:
1. 降低數據採樣比例
   ```yaml
   env_config:
     data_sample_ratio: 0.05  # 5% 採樣
   ```

2. 減少 batch_size
   ```yaml
   ppo:
     batch_size: 32  # 從 64 降到 32
   ```

3. 減少並行環境
   ```bash
   --n-envs 1  # 單環境
   ```

### 9.2 訓練不穩定

**問題**: `ep_rew_mean` 劇烈震盪

**解決方案**:
1. 降低學習率
   ```yaml
   ppo:
     learning_rate: 0.0001  # 從 3e-4 降到 1e-4
   ```

2. 減少 clip_range
   ```yaml
   ppo:
     clip_range: 0.1  # 從 0.2 降到 0.1
   ```

3. 增加 n_steps
   ```yaml
   ppo:
     n_steps: 4096  # 從 2048 增加到 4096
   ```

### 9.3 獎勵始終為負

**問題**: 策略無法獲得正獎勵

**檢查項目**:
1. 獎勵函數權重
   ```yaml
   reward_config:
     pnl_weight: 1.0      # 確保 PnL 權重足夠
     cost_weight: 0.5     # 降低成本懲罰
   ```

2. 交易成本率
   ```yaml
   env_config:
     transaction_cost_rate: 0.0005  # 降低手續費
   ```

3. 檢查價格數據
   ```python
   # 驗證價格變動是否合理
   prices = env.prices
   returns = np.diff(prices) / prices[:-1]
   print(f"日收益率: mean={returns.mean():.4f}, std={returns.std():.4f}")
   ```

### 9.4 GPU 利用率低

**問題**: GPU 利用率 < 50%

**解決方案**:
1. 增加 batch_size
   ```yaml
   ppo:
     batch_size: 128  # 從 64 增加到 128
   ```

2. 增加並行環境（CPU 並行採樣）
   ```bash
   --n-envs 8 --vec-type subproc
   ```

3. 混合精度訓練（需修改代碼）
   ```python
   # 在 create_ppo_deeplob_model() 中添加
   policy_kwargs['use_amp'] = True  # 自動混合精度
   ```

### 9.5 DeepLOB 載入失敗

**問題**: `DeepLOB 檢查點載入失敗`

**解決方案**:
1. 檢查檢查點格式
   ```python
   checkpoint = torch.load('checkpoints/v5/deeplob_v5_best.pth', weights_only=False)
   print(checkpoint.keys())
   # 應包含: model_state_dict, config, epoch, val_acc
   ```

2. 確認配置完整
   ```python
   if 'config' in checkpoint and 'model' in checkpoint['config']:
       print(checkpoint['config']['model'])
   ```

3. 使用 CPU 載入
   ```python
   checkpoint = torch.load(path, map_location='cpu')
   ```

---

## 10. 總結

### 10.1 核心要點

1. **雙層學習架構**
   - DeepLOB (凍結): 特徵提取
   - PPO (學習): 策略優化

2. **環境設計**
   - 觀測: 28 維（LOB + DeepLOB + State）
   - 動作: 3 個（Buy/Hold/Sell）
   - 獎勵: 多組件（PnL + Cost + Inventory + Risk）

3. **訓練流程**
   - Rollout → Advantage → Policy Update → Evaluate

4. **自定義函數**
   - 環境創建: `create_vec_env`, `create_eval_env`
   - 模型創建: `create_ppo_deeplob_model`
   - 訓練執行: `train_model`

### 10.2 下一步

1. **完整訓練** (1M steps)
2. **超參數優化** (Optuna)
3. **性能評估** (Sharpe Ratio, Max Drawdown)
4. **模型部署** (推理優化)

---

## 參考資源

- **Stable-Baselines3 文檔**: https://stable-baselines3.readthedocs.io/
- **PPO 論文**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **DeepLOB 論文**: Zhang et al., "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books" (2019)
- **專案文檔**: [docs/SB3_IMPLEMENTATION_REPORT.md](docs/SB3_IMPLEMENTATION_REPORT.md)

---

**文檔結束**
