# PPO_14 訓練計劃 - 降低未平倉懲罰

**日期**: 2025-10-27
**版本**: v14.0
**策略**: Long Only + 降低未平倉懲罰

---

## 🎯 目標

解決 PPO_13 過度保守問題（Buy 比例僅 2%），找到「鼓勵交易」與「避免隔夜」的平衡點。

---

## 📊 PPO_13 問題診斷

### 訓練指標 (看似良好)

- ✅ **KL 散度**: 0.0036 (極度穩定)
- ✅ **解釋方差**: 0.966 (價值函數擬合優秀)
- ✅ **趨勢 R²**: 0.930 (強上升趨勢)
- ✅ **健康度**: 90/100 (極佳)

### 實際交易行為 (嚴重問題)

- ❌ **Buy 比例**: 2.0% (目標: 15%-40%)
- ❌ **Hold 比例**: 98.0% (幾乎不交易)
- ❌ **Episode 獎勵**: -2.91 ± 4.06 (仍為負)
- ⚠️ **交易次數**: 5.8/Episode (數量合理，但多為空倉)

### 根本原因

**未平倉懲罰過於嚴苛**: `-10 元/倉位`

```python
# tw_lob_trading_env.py:380 (PPO_13)
unclosed_penalty = -10.0 * self.position
```

**懲罰強度分析**:
- PPO_11 持倉 500 步獎勵: `+0.05 × 500 = +25 元`
- PPO_13 未平倉懲罰: `-10 元` (相當於損失 **40%** 獎勵)
- **結果**: 模型學到「盡量不持倉」以避免懲罰

---

## 🔧 PPO_14 調整方案

### 核心修改

**降低未平倉懲罰 80%**: `-10 元 → -2 元/倉位`

```python
# tw_lob_trading_env.py:380 (PPO_14)
unclosed_penalty = -2.0 * self.position
```

### 理由

1. **-2 元仍能鼓勵平倉**:
   - 持倉 500 步需盈利 > `2 / 500 / 報價 ≈ 0.4%` 才划算
   - 仍保留當沖規則激勵

2. **不會過度抑制交易**:
   - `-2 元 = 8%` PPO_11 持倉獎勵（vs. PPO_13 的 40%）
   - 溫和懲罰，讓策略有空間探索

3. **參考 PPO_11 成功經驗**:
   - PPO_11 平均獎勵: `+24.15` (證明有盈利空間)
   - PPO_14 目標: 保留盈利能力，避免過度交易

---

## 🎯 訓練目標

| 指標 | PPO_13 (當前) | PPO_14 (目標) | 評估標準 |
|------|--------------|--------------|----------|
| **Buy 比例** | 2.0% ❌ | 15%-40% | 適度交易 |
| **Episode 獎勵** | -2.91 ❌ | > 0 | 盈利 |
| **交易次數** | 5.8/Ep | 5-10/Ep | 保持合理 |
| **KL 散度** | 0.0036 ✅ | < 0.02 | 保持穩定 |
| **解釋方差** | 0.966 ✅ | > 0.7 | 保持良好 |

---

## 📋 訓練配置

### 超參數 (保持 PPO_13)

```yaml
ppo:
  learning_rate: 1e-4       # ✅ 穩定訓練
  clip_range: 0.1           # ✅ KL 散度良好
  vf_coef: 1.0              # ✅ 解釋方差 0.966
  net_arch:
    pi: [512, 256]          # ✅ 容量充足
    vf: [512, 256]

training:
  total_timesteps: 200000   # 快速測試
```

### 獎勵函數

```python
# reward_shaper.py (無變化)
總獎勵 = PnL - 交易成本

# tw_lob_trading_env.py (唯一變化)
if Episode 結束 and 有持倉:
    總獎勵 -= 2.0 * position  # PPO_14: -2 | PPO_13: -10
```

---

## 🚀 執行步驟

### 1. 驗證修改

```bash
# 檢查環境代碼
type src\envs\tw_lob_trading_env.py | findstr "unclosed_penalty"
# 應顯示: unclosed_penalty = -2.0 * self.position

# 檢查配置文件
type configs\sb3_deeplob_config.yaml | findstr "total_timesteps"
# 應顯示: total_timesteps: 200000
```

### 2. 開始訓練 (200K steps, ~28 分鐘)

```bash
# 激活環境
conda activate deeplob-pro

# 開始訓練
python scripts/train_sb3_deeplob.py --config configs/sb3_deeplob_config.yaml

# 監控訓練
tensorboard --logdir logs/sb3_deeplob/
```

### 3. 訓練後驗證

```bash
# 3.1 分析 TensorBoard 日誌
python scripts/analyze_tensorboard.py --log-dir logs/sb3_deeplob/PPO_14

# 3.2 檢查交易行為
python scripts/check_trading_behavior.py --model checkpoints/sb3/ppo_deeplob/best_model --n_episodes 5

# 3.3 評估模型性能
python scripts/evaluate_sb3.py --model checkpoints/sb3/ppo_deeplob/best_model --n_episodes 20
```

---

## 📈 預期結果

### 樂觀預期 (⭐⭐⭐⭐⭐)

- Buy 比例: **25%-35%** (適度交易)
- Episode 獎勵: **+5~+15** (實際盈利)
- 交易次數: **6-8/Ep** (合理頻率)
- KL 散度: **< 0.01** (保持穩定)

### 保守預期 (⭐⭐⭐)

- Buy 比例: **10%-20%** (仍偏保守)
- Episode 獎勵: **-1~+3** (接近盈虧平衡)
- 需進一步調整 unclosed_penalty (-2 → -1 或 -0.5)

### 失敗情況 (❌)

- Buy 比例: **< 5%** (仍過度保守)
- Episode 獎勵: **< -2** (仍為負)
- 需改用其他方案（見下文備選方案）

---

## 🔄 備選方案 (如 PPO_14 失敗)

### 方案 B: 漸進式懲罰 ⭐⭐⭐⭐

```python
# 根據持倉時間調整懲罰強度
if holding_duration < 200:
    unclosed_penalty = 0            # 短期持倉無懲罰
elif holding_duration < 400:
    unclosed_penalty = -1 * position  # 中期持倉輕懲罰
else:
    unclosed_penalty = -5 * position  # 長期持倉重懲罰
```

**優點**: 不懲罰短期持倉，只懲罰長期持倉
**缺點**: 需修改代碼，增加持倉時間記錄

### 方案 C: 混合策略 ⭐⭐⭐

```python
# 持倉激勵 (鼓勵交易)
holding_bonus = +0.01 * position

# 未平倉懲罰 (避免隔夜)
unclosed_penalty = -5.0 * position

# 500 步淨效果: +0.01 × 500 - 5 = 0 元 (需盈利 > 0.1% 才划算)
```

**優點**: 平衡鼓勵與懲罰
**缺點**: 參數調整複雜（需同時平衡兩個數值）

### 方案 D: 回到 PPO_11 + 降低激勵 ⭐⭐

```python
# 僅降低持倉激勵，移除未平倉懲罰
holding_bonus = +0.01 * position  # 降低 80% (0.05 → 0.01)
unclosed_penalty = 0              # 移除懲罰
```

**優點**: PPO_11 已證明可獲利 (+24.15)
**缺點**: 可能仍有過度交易風險

---

## 📝 成功標準

**必須達成**:
1. ✅ Buy 比例 > 5% (避免過度保守)
2. ✅ Episode 獎勵 > -1 (接近盈利)
3. ✅ KL 散度 < 0.02 (訓練穩定)

**理想目標**:
1. 🎯 Buy 比例 15%-40% (適度交易)
2. 🎯 Episode 獎勵 > +5 (實際盈利)
3. 🎯 交易次數 5-10/Ep (合理頻率)

---

## 📚 參考文檔

- [調參歷史](20251026-sb3調參歷史.md) - PPO_11/PPO_13 詳細記錄
- [獎勵計算指南](REWARD_CALCULATION_GUIDE.md) - 獎勵函數設計
- [未平倉懲罰策略](UNCLOSED_POSITION_PENALTY.md) - PPO_12 策略文檔

---

**最後更新**: 2025-10-27
**狀態**: ⏳ 待訓練
**下一步**: 執行訓練 → 驗證結果 → 更新記錄
