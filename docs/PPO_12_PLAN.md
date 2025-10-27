# PPO_12 實驗計劃 - 當日收盤未平倉懲罰

## 實驗信息

- **日期**: 2025-10-27
- **版本**: PPO_12
- **策略**: 當日收盤未平倉懲罰（台股當沖規則）
- **前置實驗**: PPO_11（持倉激勵導致過度交易）

---

## 一、問題分析

### PPO_11 結果回顧

| 指標 | PPO_9 | PPO_11 | 問題 |
|------|-------|--------|------|
| **Buy 動作比例** | 0% | 99.4% | 從「永不交易」到「永遠持倉」 |
| **實際交易次數** | 0 次 | 3.8 次/Ep | 幾乎不離場 |
| **Episode 獎勵** | 0.00 | 24.15 | 盈利，但策略極端 |
| **交易行為** | 永不動 | 一直持倉 | 兩個極端 |

### 根本問題

1. **持倉激勵過強**: `holding_bonus = 0.05` × 500 步 = 25 元獎勵
2. **缺乏離場動機**: 沒有機制鼓勵平倉
3. **不符合實際**: 台股當沖必須當日平倉

---

## 二、PPO_12 策略設計

### 核心理念：當日收盤未平倉懲罰

```
策略轉變:
  移除: 持倉激勵 (每步 +0.05)
  新增: 未平倉懲罰 (收盤時 -10.0 × position)

效果:
  - 鼓勵日內持倉（可獲利）
  - 強制收盤平倉（避免隔夜）
  - 符合台股當沖規則
```

### 實施方式

**1. 移除持倉激勵** (src/envs/reward_shaper.py)

```python
# Line 206-209
reward_components['holding_bonus'] = 0.0  # 移除
```

**2. 新增未平倉懲罰** (src/envs/tw_lob_trading_env.py)

```python
# Line 376-385
if truncated and self.position > 0:
    unclosed_penalty = -10.0 * self.position
    reward += unclosed_penalty
    reward_info['unclosed_position_penalty'] = unclosed_penalty
```

### 懲罰強度設計

| 持倉大小 | 懲罰值 | 佔平均獎勵比例 | 影響 |
|----------|--------|---------------|------|
| 0.5 單位 | -5.0 元 | 21% | 輕度 |
| 1.0 單位 | -10.0 元 | 41% | 中度 ⭐ |
| 1.5 單位 | -15.0 元 | 62% | 重度 |

**選擇 -10.0 的原因**:
1. 足夠強迫模型平倉（佔獎勵 41%）
2. 不會完全抵消盈利（仍有 59% 獎勵）
3. 平衡日內交易靈活性與收盤平倉需求

---

## 三、預期效果

### 目標指標

| 指標 | PPO_11 | PPO_12 目標 | 理由 |
|------|--------|------------|------|
| **Buy 比例** | 99.4% | 20-40% | 適度交易 |
| **交易次數** | 3.8 次/Ep | 5-20 次/Ep | 正常進出場 |
| **收盤持倉** | ~100% | < 10% | 學會平倉 |
| **Episode 獎勵** | 24.15 | 15-25 | 維持盈利 |

### 預期交易行為

```
理想的 Episode (500 步):

步數 1-50:     Hold (空倉，觀望)
步數 51-100:   Buy (進場)
步數 101-200:  Hold (持倉中，持倉獎勵已移除)
步數 201-450:  持續持倉或部分平倉
步數 451-500:  Sell (收盤前平倉，避免懲罰)

關鍵: 最後 10-50 步應出現 Sell 動作
```

---

## 四、訓練配置

### 參數設置

```yaml
# configs/sb3_deeplob_config.yaml

ppo:
  learning_rate: 0.0001    # 保持 PPO_11
  clip_range: 0.1          # 保持 PPO_11
  vf_coef: 1.0             # 保持 PPO_11
  net_arch:
    pi: [512, 256]         # 保持 PPO_11
    vf: [512, 256]         # 保持 PPO_11

training:
  total_timesteps: 200000  # 快速測試（21 分鐘）

env_config:
  reward_config:
    pnl_scale: 1.0
    cost_penalty: 1.0
    # 持倉激勵已在 reward_shaper.py 中移除
    # 未平倉懲罰在 tw_lob_trading_env.py 中實施
```

### 訓練步驟

```bash
# Step 1: 確認代碼已修改
# - src/envs/reward_shaper.py: holding_bonus = 0.0
# - src/envs/tw_lob_trading_env.py: 添加未平倉懲罰

# Step 2: 訓練
python scripts/train_sb3_deeplob.py --config configs/sb3_deeplob_config.yaml

# Step 3: 驗證
python scripts/check_trading_behavior.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 5

# Step 4: 分析
python scripts/analyze_tensorboard.py
```

---

## 五、驗證指標

### 關鍵檢查點

1. **unclosed_position_penalty** (TensorBoard):
   - 初期: 接近 -10.0（模型尚未學會平倉）
   - 後期: 接近 0.0（模型學會平倉）

2. **Episode 最後 50 步動作分布**:
   - Sell (action=0) 比例應增加
   - 表示模型學習收盤前平倉

3. **整體交易行為**:
   - Buy 比例: 20-40%
   - 交易次數: 5-20 次/Episode
   - 收盤持倉率: < 10%

### 成功標準

| 指標 | 最低要求 | 理想目標 |
|------|---------|---------|
| **Buy 比例** | 10-50% | 20-40% |
| **收盤持倉率** | < 20% | < 10% |
| **Episode 獎勵** | > 10 | > 20 |
| **KL 散度** | < 0.03 | < 0.02 |
| **解釋方差** | > 0.5 | > 0.8 |

---

## 六、可能問題與對策

### 問題1: Buy 比例仍過高 (> 50%)

**原因**: 懲罰不夠強
**解決**: 提高懲罰係數

```python
unclosed_penalty = -20.0 * self.position  # 從 -10.0 → -20.0
```

### 問題2: Buy 比例過低 (< 10%)

**原因**: 懲罰過強或回到「永不交易」
**解決**: 降低懲罰或增加小額持倉激勵

```python
# 方案1: 降低懲罰
unclosed_penalty = -5.0 * self.position

# 方案2: 增加小額持倉激勵
holding_bonus = 0.01  # 從 0.05 → 0.01
```

### 問題3: 模型仍不平倉

**原因**: 持倉獲利 > 未平倉懲罰
**解決**: 提高懲罰或使用漸進式懲罰

```python
# 方案1: 大幅提高懲罰
unclosed_penalty = -50.0 * self.position

# 方案2: 漸進式懲罰（最後 N 步逐漸增加）
steps_to_end = self.max_steps - self.current_step
if steps_to_end <= 50:
    penalty_multiplier = (51 - steps_to_end) / 5.0  # 0.2 → 10.0
    unclosed_penalty = -penalty_multiplier * self.position
```

---

## 七、與其他方案對比

| 方案 | 優點 | 缺點 | 複雜度 | 推薦度 |
|------|------|------|--------|--------|
| **方案B (PPO_12)** | 符合當沖規則<br>簡單有效 | 固定懲罰<br>不考慮市況 | ⭐ | ⭐⭐⭐⭐⭐ |
| 方案A | 數值可調 | 難找最佳值 | ⭐ | ⭐⭐⭐ |
| 方案C | 動作平衡 | 設計複雜 | ⭐⭐⭐ | ⭐⭐ |
| 方案D | 利用 DeepLOB | 實施複雜 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**選擇方案B的原因**:
1. 最符合台股當沖規則
2. 實施簡單（只需修改一處）
3. 可調整性高（懲罰係數可配置）
4. 解決「永遠持倉」問題

---

## 八、預期時間表

| 階段 | 任務 | 預計時間 |
|------|------|---------|
| **階段1** | 代碼實施與驗證 | ✅ 已完成 |
| **階段2** | PPO_12 訓練 (200K) | 21 分鐘 |
| **階段3** | 交易行為驗證 | 5 分鐘 |
| **階段4** | 結果分析與調整 | 10 分鐘 |
| **階段5** | 文檔更新 | ✅ 已完成 |

**總計**: ~36 分鐘

---

## 九、文檔索引

相關文檔:
- [UNCLOSED_POSITION_PENALTY.md](UNCLOSED_POSITION_PENALTY.md): 詳細策略說明
- [20251026-sb3調參歷史.md](20251026-sb3調參歷史.md): 實驗記錄
- [REWARD_CALCULATION_GUIDE.md](REWARD_CALCULATION_GUIDE.md): 獎勵函數指南

修改文件:
- [src/envs/tw_lob_trading_env.py](../src/envs/tw_lob_trading_env.py): Line 376-385
- [src/envs/reward_shaper.py](../src/envs/reward_shaper.py): Line 206-209

---

## 十、總結

PPO_12 採用**當日收盤未平倉懲罰**策略，旨在：

1. ✅ 解決 PPO_11 的「永遠持倉」問題
2. ✅ 符合台股當沖交易規則
3. ✅ 鼓勵模型學習完整進出場策略
4. ✅ 維持訓練穩定性與盈利能力

**核心機制**:
```
日內: 自由交易，可持倉獲利
收盤: position > 0 → 懲罰 -10.0 × position
結果: 模型被迫學習收盤前平倉
```

**預期成果**:
- Buy 比例: 20-40%（適度）
- 收盤持倉: < 10%（學會平倉）
- Episode 獎勵: 15-25（維持盈利）
- 符合台股當沖規則 ✅

---

**最後更新**: 2025-10-27
**狀態**: 準備訓練
**下一步**: 運行 PPO_12 訓練並驗證結果
