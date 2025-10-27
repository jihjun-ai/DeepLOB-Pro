# ChatGPT 調參建議評估報告

**評估日期**: 2025-10-27
**評估對象**: ChatGPT 提供的 11 點 PPO 調參建議
**評估標準**: 基於 PPO_5 → PPO_14 的實際訓練經驗

---

## 📊 總體評估

**整體評分**: ⭐⭐⭐⭐ (8.0/10)

**優點**:
- ✅ 系統性分析框架完整（樣本效率、探索、獎勵設計、監控）
- ✅ 參數範圍建議合理（基於業界經驗）
- ✅ 強調行為監控（buy_ratio 等）與我們的發現一致
- ✅ A/B 實驗設計科學（單變量控制）

**缺點**:
- ⚠️ 未考慮台股特殊性（Long Only、當沖規則）
- ⚠️ 部分建議與當前項目狀態不完全匹配
- ⚠️ 過於強調向量環境（單卡 RTX 5090 可能不是瓶頸）

---

## 📝 逐項評估（11 點分析）

### 1️⃣ 訓練預算與學習率衰減 ⭐⭐⭐⭐⭐

**ChatGPT 建議**:
```yaml
total_timesteps: 1e6 (區間 6e5–2e6)
lr_schedule: linear
initial: 1e-4 → final: 3e-5
```

**評估**: ✅ **強烈同意**

**理由**:
- PPO_13 在 200K 時仍有上升趨勢（R²=0.930）
- 固定 LR=1e-4 已驗證穩定（KL=0.0036）
- 線性衰減有助於後期精細調整

**項目實際情況**:
- ✅ PPO_9 (500K) 達成盈利 (0.00)
- ✅ PPO_11 (200K) 證明有盈利空間 (+24.15)
- 🎯 建議: PPO_14 先測試 200K，成功後延長至 1M

**優先級**: 🔥🔥🔥🔥🔥 (極高)

**實施建議**:
```yaml
# 第一階段 (PPO_14): 保持 200K 驗證方向
total_timesteps: 200000

# 第二階段 (PPO_15): 延長訓練 + LR 衰減
total_timesteps: 1000000
advanced:
  lr_schedule: "linear"
  lr_linear:
    initial: 1e-4
    final: 3e-5
```

**風險**: 低 | 已有成功經驗（PPO_9 500K）

---

### 2️⃣ 向量環境與大 Batch ⭐⭐⭐

**ChatGPT 建議**:
```yaml
n_envs: 8 (區間 4–16)
n_steps: 1024 (區間 512–2048)
batch_size: 2048 (區間 1024–4096)
```

**評估**: ⚠️ **部分同意，需謹慎**

**理由**:
- ✅ 理論上增加樣本效率正確
- ❌ 您的項目已證明 `n_envs=1, batch=64` 可穩定訓練（PPO_11/13）
- ⚠️ RTX 5090 單卡，GPU 利用率可能不是瓶頸
- ⚠️ 環境向量化可能引入新 bug（資料同步、隨機性）

**項目實際情況**:
- 當前配置: `n_envs=1, n_steps=2048, batch_size=64`
- 訓練速度: **119 steps/sec** (PPO_13)
- KL 散度: **0.0036** (極度穩定)

**優先級**: 🔥🔥 (中) | 非緊急問題

**實施建議**:
```yaml
# 保守方案 (優先推薦)
n_envs: 1          # 保持單環境，已驗證穩定
batch_size: 128    # 適度增加 (64 → 128)
n_steps: 2048      # 保持不變

# 激進方案 (GPU 利用率 < 50% 時)
n_envs: 4          # 謹慎增加
batch_size: 512    # n_envs * n_steps / 16 = 512
n_steps: 2048
```

**風險**: 中 | 可能影響訓練穩定性

**驗證方式**:
```bash
# 訓練 50K steps 對比
A: n_envs=1, batch=64  (當前)
B: n_envs=1, batch=128
C: n_envs=4, batch=512

# 比較: KL 散度、Buy 比例、訓練速度
```

---

### 3️⃣ 探索係數（熵係數）⭐⭐⭐⭐

**ChatGPT 建議**:
```yaml
ent_coef: 0.005 (區間 0.002–0.01)
```

**評估**: ✅ **同意，值得嘗試**

**理由**:
- 當前 `ent_coef=0.01` 可能略高
- PPO_13 過度保守 (Buy 2%) 可能與探索不足有關
- 降低熵係數 → 策略更確定性

**項目實際情況**:
- PPO_11: `ent_coef=0.01` → Buy 99.4% (過度探索？)
- PPO_13: `ent_coef=0.01` → Buy 2% (探索不足？)
- 🤔 可能是獎勵函數主導，而非熵係數

**優先級**: 🔥🔥🔥 (中高) | 低風險調整

**實施建議**:
```yaml
# PPO_14: 先保持 0.01 驗證獎勵函數修正
ent_coef: 0.01

# PPO_15: 若 PPO_14 成功，微調熵係數
ent_coef: 0.005  # 降低 50%
```

**A/B 測試**:
```
A: ent_coef=0.01 (當前)
B: ent_coef=0.005
C: ent_coef=0.002

比較: Buy 比例、勝率、策略多樣性
```

---

### 4️⃣ Target KL 與剪裁 ⭐⭐⭐⭐⭐

**ChatGPT 建議**:
```yaml
clip_range: 0.10 (區間 0.08–0.2)
target_kl: 0.02 (區間 0.01–0.05)
```

**評估**: ✅ **完全同意**

**理由**:
- 當前 `clip_range=0.1` 已證明穩定
- KL 散度 0.0036–0.0183 非常健康
- 添加 `target_kl` 是良好的保險機制

**項目實際情況**:
- PPO_11/13: `clip_range=0.1` → KL < 0.02 ✅
- 當前未設置 `target_kl` (SB3 預設 None)

**優先級**: 🔥🔥🔥🔥 (高) | 零風險改進

**實施建議**:
```yaml
ppo:
  clip_range: 0.10    # 保持不變
  target_kl: 0.02     # 新增保險機制
```

**驗證**: 若訓練中 KL > 0.02，PPO 自動提前停止該 epoch

---

### 5️⃣ 獎勵結構改進 ⭐⭐⭐⭐⭐ **最重要！**

**ChatGPT 建議**:
```yaml
# 持倉激勵衰減
holding_bonus.start: 0.02
holding_bonus.end: 0.005
holding_bonus.warmup_steps: 50_000
holding_bonus.schedule: linear

# 未平倉懲罰
terminal_unclosed_penalty.coef: 2.0

# 冷卻期懲罰
trade_cooldown_penalty.enabled: true
trade_cooldown_penalty.steps: 2
trade_cooldown_penalty.per_step: 0.005
```

**評估**: ✅ **極度同意，這是核心問題**

**理由**:
- 🎯 完全契合項目經驗:
  - PPO_11: 固定 +0.05 → 99.4% Buy (過度)
  - PPO_13: 固定 -10 → 2% Buy (保守)
  - **需要動態平衡！**

**項目實際情況**:
- PPO_14: 已將 -10 降至 -2 (手動調整)
- 但仍是**固定值**，缺乏自適應

**優先級**: 🔥🔥🔥🔥🔥 (極高) | **這是關鍵突破點**

**實施計劃**:

**階段 1 (PPO_14)**: 驗證固定 -2 是否有效
```python
# tw_lob_trading_env.py (當前)
unclosed_penalty = -2.0 * self.position
```

**階段 2 (PPO_15)**: 實施持倉激勵衰減
```python
# reward_shaper.py (新增)
class DecayingHoldingBonus:
    def __init__(self, start=0.02, end=0.005, warmup_steps=50000):
        self.start = start
        self.end = end
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_bonus(self, position):
        # 線性衰減
        progress = min(1.0, self.current_step / self.warmup_steps)
        current_bonus = self.start - (self.start - self.end) * progress
        return current_bonus * position if position > 0 else 0.0

    def step(self):
        self.current_step += 1
```

**階段 3 (PPO_16)**: 實施交易冷卻懲罰
```python
# 防止頻繁交易
if action_changed and self.steps_since_last_trade < cooldown_steps:
    penalty = -0.005 * (cooldown_steps - self.steps_since_last_trade)
    reward += penalty
```

**風險**: 低-中 | 需要修改代碼，但邏輯清晰

---

### 6️⃣ 行為監控 ⭐⭐⭐⭐⭐

**ChatGPT 建議**:
```python
buy_ratio, trades_per_ep, avg_holding_steps
目標: buy_ratio in [0.15, 0.40]
```

**評估**: ✅ **完全同意，已在實施**

**理由**:
- 您已有 `check_trading_behavior.py` ✅
- PPO_13 正是靠這個發現 Buy 2% 問題
- 需整合到 TensorBoard 實時監控

**項目實際情況**:
- ✅ 已有離線檢查: `check_trading_behavior.py`
- ❌ 缺乏訓練中實時監控

**優先級**: 🔥🔥🔥🔥🔥 (極高) | 防止浪費訓練時間

**實施建議**:

```python
# callbacks/behaviour_callback.py (新增)
class TradingBehaviorCallback(BaseCallback):
    def __init__(self, check_freq=10000, target_buy_ratio=(0.15, 0.40)):
        super().__init__()
        self.check_freq = check_freq
        self.target_buy_ratio = target_buy_ratio
        self.action_counts = {0: 0, 1: 0}  # Hold/Sell, Buy

    def _on_step(self):
        # 記錄動作
        action = self.locals['actions'][0]
        self.action_counts[action] += 1

        if self.n_calls % self.check_freq == 0:
            # 計算 Buy 比例
            total = sum(self.action_counts.values())
            buy_ratio = self.action_counts[1] / total if total > 0 else 0.0

            # 記錄到 TensorBoard
            self.logger.record("behaviour/buy_ratio", buy_ratio)
            self.logger.record("behaviour/hold_ratio", self.action_counts[0] / total)

            # 警告
            if buy_ratio < self.target_buy_ratio[0]:
                print(f"[WARN] Buy 比例過低: {buy_ratio:.1%} < {self.target_buy_ratio[0]:.1%}")
            elif buy_ratio > self.target_buy_ratio[1]:
                print(f"[WARN] Buy 比例過高: {buy_ratio:.1%} > {self.target_buy_ratio[1]:.1%}")

            # 重置計數
            self.action_counts = {0: 0, 1: 0}

        return True
```

**風險**: 極低 | 純監控，不影響訓練

---

### 7️⃣ 動作遮罩 ⭐⭐

**ChatGPT 建議**:
```yaml
action_masking.enabled: true
```

**評估**: ⚠️ **理論正確，但項目中優先級低**

**理由**:
- Long Only 策略下，動作空間已簡化 (Discrete(2))
- 滿倉時仍允許 Buy 的確是無效動作
- 但當前問題是「不交易」(2%)，而非「過度交易」

**項目實際情況**:
- 當前: `position ∈ [0, max_position]`
- 動作: `{0: Hold/Sell, 1: Buy}`
- **問題**: Buy 比例 2%（需鼓勵交易，而非限制）

**優先級**: 🔥 (低) | 非當前瓶頸

**實施建議**: **暫不實施**，待 Buy 比例正常後再考慮

**若未來實施**:
```python
# 在 policy forward 前
def get_action_mask(self):
    mask = np.ones(self.action_space.n, dtype=bool)
    if self.position >= self.max_position:
        mask[1] = False  # 禁止 Buy
    if self.position <= 0:
        mask[0] = False  # 禁止 Sell (已在 Long Only 中處理)
    return mask
```

---

### 8️⃣ 評估頻率與一致性 ⭐⭐⭐⭐

**ChatGPT 建議**:
```yaml
eval_freq: 10000 (區間 5k–20k)
n_eval_episodes: 10
確保 train/eval 配置一致
```

**評估**: ✅ **同意**

**理由**:
- 當前 `eval_freq=10000` 已合理
- 需確認 train/eval 成本一致性

**項目實際情況**:
```yaml
evaluation:
  eval_freq: 10000          # ✅ 已設置
  n_eval_episodes: 10       # ✅ 已設置
  eval_env_config:
    data_mode: "val"        # ✅ 使用驗證集
```

**優先級**: 🔥🔥🔥 (中高) | 需驗證一致性

**驗證清單**:
```python
# 檢查 train/eval 配置是否一致
✅ 交易成本 (commission, tax, slippage)
✅ 標準化參數 (normalization_meta)
✅ Episode 長度 (max_steps=500)
✅ 數據來源 (train vs val NPZ)
```

**實施**: 創建配置驗證腳本
```python
# scripts/verify_train_eval_consistency.py
def verify_consistency(train_env, eval_env):
    assert train_env.transaction_cost == eval_env.transaction_cost
    assert train_env.max_steps == eval_env.max_steps
    # ... 其他檢查
```

---

### 9️⃣ 交易成本落地 ⭐⭐⭐⭐⭐

**ChatGPT 建議**:
```yaml
transaction_cost_bps: 2 (區間 1–5)
slippage_bps: 1 (區間 0.5–2)
```

**評估**: ✅ **完全同意，當前配置正確**

**理由**:
- 台股成本已實施 (commission + tax)
- 配置符合實際交易環境

**項目實際情況**:
```yaml
transaction_cost:
  commission:
    base_rate: 0.001425      # 0.1425%
    discount: 0.3            # 3折 → 實際 0.04275%
  securities_tax:
    rate: 0.0015             # 當沖 0.15%
  slippage:
    enabled: false           # 當前未啟用
```

**實際成本計算**:
```
買入: 0.04275%
賣出: 0.04275% + 0.15% = 0.19275%
單次來回: 0.04275% + 0.19275% = 0.2355% ≈ 2.4 bps ✅
```

**優先級**: 🔥🔥 (中) | 當前已正確，可選增加滑點

**實施建議**:
```yaml
# PPO_15+: 增加滑點模擬（更真實）
slippage:
  enabled: true
  rate: 0.0001    # 1 bps
```

---

### 🔟 網絡結構 ⭐⭐⭐

**ChatGPT 建議**:
```yaml
net_arch: [512, 256] (區間 [256,256]~[512,512])
activation_fn: ReLU
features_extractor frozen: true
```

**評估**: ✅ **已正確配置**

**項目實際情況**:
```yaml
ppo:
  net_arch:
    pi: [512, 256]    # ✅ 已設置
    vf: [512, 256]    # ✅ 已設置
  activation_fn: "ReLU"  # ✅ 已設置

deeplob_extractor:
  freeze_deeplob: true  # ✅ 已凍結
```

**優先級**: 🔥 (低) | 已優化，無需調整

**驗證**: 解釋方差 0.966 (PPO_13) 證明網絡容量充足

---

### 1️⃣1️⃣ 穩定性細節 ⭐⭐⭐⭐

**ChatGPT 建議**:
```yaml
max_grad_norm: 0.5 (區間 0.3–1.0)
save_freq: 50_000
reset_timesteps: false
```

**評估**: ✅ **已正確配置**

**項目實際情況**:
```yaml
ppo:
  max_grad_norm: 0.5        # ✅ 已設置

callbacks:
  checkpoint:
    save_freq: 50000        # ✅ 已設置

training:
  resume:
    reset_timesteps: false  # ✅ 已設置
```

**優先級**: 🔥 (低) | 已優化

---

## 🎯 最小變更清單（優先級排序）

根據 ChatGPT 建議 + 項目實際情況，**推薦優先順序**：

### 🔥🔥🔥🔥🔥 極高優先級（立即實施）

1. **獎勵結構動態調整** (建議 5)
   - PPO_14: 驗證固定 -2
   - PPO_15: 實施持倉激勵衰減 (0.02→0.005)
   - PPO_16: 實施交易冷卻懲罰

2. **行為監控回調** (建議 6)
   - 創建 `TradingBehaviorCallback`
   - 實時監控 buy_ratio, trades_per_ep
   - 目標: 15%-40% Buy

3. **Target KL 保險** (建議 4)
   - 添加 `target_kl: 0.02`
   - 零風險改進

### 🔥🔥🔥🔥 高優先級（PPO_14 成功後）

4. **延長訓練 + LR 衰減** (建議 1)
   - 200K → 1M steps
   - 線性衰減 1e-4 → 3e-5

5. **成本一致性驗證** (建議 8)
   - 創建 `verify_train_eval_consistency.py`
   - 確保 train/eval 配置一致

### 🔥🔥🔥 中高優先級（可選優化）

6. **微調熵係數** (建議 3)
   - A/B 測試: 0.01 vs 0.005 vs 0.002

7. **增加滑點** (建議 9)
   - 啟用 1 bps 滑點模擬

### 🔥🔥 中優先級（非瓶頸）

8. **向量環境 + 大 Batch** (建議 2)
   - 謹慎測試: n_envs=4, batch=512
   - 驗證不影響穩定性

### 🔥 低優先級（暫不實施）

9. **動作遮罩** (建議 7)
   - 當前問題是「不交易」，而非「過度交易」
   - 待 Buy 比例正常後再考慮

---

## 🔬 A/B 實驗設計（推薦）

### 實驗 E1: 獎勵結構對比 ⭐⭐⭐⭐⭐

**目的**: 驗證動態獎勵是否優於固定獎勵

```yaml
A (PPO_14): 固定未平倉懲罰 -2
B (PPO_15): 持倉激勵衰減 0.02→0.005 + 未平倉 -2
C (PPO_16): B + 交易冷卻懲罰

訓練: 各 200K steps
比較: Buy 比例、Episode 獎勵、交易次數
```

### 實驗 E2: 學習率衰減 ⭐⭐⭐⭐

**目的**: 驗證 LR 衰減是否提升後期穩定性

```yaml
A: 固定 1e-4
B: 線性 1e-4 → 3e-5

訓練: 各 1M steps
比較: 600K 之後的獎勵方差、KL 穩定性
```

### 實驗 E3: 向量環境 ⭐⭐

**目的**: 驗證向量化是否提升樣本效率

```yaml
A: n_envs=1, batch=64
B: n_envs=1, batch=128
C: n_envs=4, batch=512

訓練: 各 200K steps
比較: 訓練速度、KL 散度、Buy 比例
```

---

## 💡 關鍵洞察

### ChatGPT 分析的精華部分 ⭐⭐⭐⭐⭐

1. **動態獎勵結構** (建議 5)
   > 完全契合項目痛點（PPO_11 過度 → PPO_13 保守）

2. **行為監控至關重要** (建議 6)
   > PPO_13 正是靠離線檢查發現 Buy 2% 問題

3. **訓練預算不足** (建議 1)
   > PPO_13 在 200K 時仍有上升趨勢（R²=0.930）

### ChatGPT 分析的待商榷部分

1. **向量環境優先級過高** (建議 2)
   > 當前 n_envs=1 已穩定訓練（KL=0.0036）
   > 向量化可能引入新 bug

2. **動作遮罩優先級過高** (建議 7)
   > 當前問題是「不交易」，而非「過度交易」

3. **未考慮台股特殊性**
   > Long Only、當沖規則未提及

---

## 📋 實施路線圖（推薦）

### 短期（本週）- PPO_14

```
✅ 1. 驗證固定未平倉懲罰 -2 (已完成)
⏳ 2. 添加 TradingBehaviorCallback (30 分鐘)
⏳ 3. 添加 target_kl=0.02 (5 分鐘)
⏳ 4. 訓練 200K steps (28 分鐘)
⏳ 5. 驗證結果 (10 分鐘)
```

### 中期（下週）- PPO_15/16

```
1. 實施持倉激勵衰減 (2 小時)
2. 實施交易冷卻懲罰 (1 小時)
3. A/B 測試 3 組獎勵配置 (各 28 分鐘)
4. 選擇最佳配置
5. 延長訓練至 1M steps + LR 衰減 (1.8 小時)
```

### 長期（本月）- PPO_17+

```
1. 配置一致性驗證
2. 向量環境測試（可選）
3. 熵係數微調（可選）
4. 最終模型評估與部署
```

---

## ✅ 結論

**ChatGPT 分析質量**: ⭐⭐⭐⭐ (8.0/10)

**最有價值的建議**:
1. ✅ 動態獎勵結構 (建議 5) - **核心突破點**
2. ✅ 行為監控回調 (建議 6) - **防止浪費時間**
3. ✅ 延長訓練 + LR 衰減 (建議 1) - **提升性能**

**需調整的建議**:
1. ⚠️ 向量環境優先級降低（當前非瓶頸）
2. ⚠️ 動作遮罩暫不實施（問題方向不符）

**總體評價**:
> ChatGPT 提供了系統性、專業的分析框架，80% 的建議與項目實際需求高度契合。
> 建議優先實施動態獎勵結構 + 行為監控，這是解決 PPO_11/13 擺盪問題的關鍵。

**行動建議**:
1. 立即實施 3 個極高優先級項目
2. PPO_14 成功後，進入中期路線圖
3. 持續監控 buy_ratio，目標 15%-40%

---

**最後更新**: 2025-10-27
**評估者**: Claude (基於 PPO_5–PPO_14 實際訓練經驗)
**下一步**: 實施 TradingBehaviorCallback + target_kl → 訓練 PPO_14
