# PPO_14 調參總結報告

**生成時間**: 2025-10-27
**分析版本**: v14.0
**數據來源**: PPO_13 TensorBoard + 交易行為驗證

---

## 📊 PPO_13 分析結果

### 訓練指標 (看似優秀)

```
✅ Episode 獎勵: -25.48 → -7.69 (提升 17.79)
✅ KL 散度: 0.0036 (極度穩定)
✅ 解釋方差: 0.966 (價值函數擬合優秀)
✅ 趨勢 R²: 0.930 (強上升趨勢)
✅ 健康度: 90/100 (極佳)
```

### 交易行為 (實際問題)

```
❌ Buy 比例: 2.0% (目標: 15%-40%)
❌ Hold 比例: 98.0% (幾乎不交易)
❌ Episode 獎勵: -2.91 ± 4.06 (仍為負)
⚠️ 交易次數: 5.8/Episode (合理但多為空倉)
```

### 關鍵發現

**未平倉懲罰過於嚴苛**: `-10 元/倉位`

```python
# tw_lob_trading_env.py:380 (PPO_13)
unclosed_penalty = -10.0 * self.position
```

**懲罰強度分析**:
- PPO_11 持倉 500 步獎勵: `+0.05 × 500 = +25 元`
- PPO_13 未平倉懲罰: `-10 元` = **40%** 獎勵損失
- **結果**: 模型學到「盡量不持倉」策略

---

## 🎯 PPO_14 調整方案

### 核心修改

**降低未平倉懲罰 80%**:

```python
# tw_lob_trading_env.py:380
unclosed_penalty = -2.0 * self.position  # PPO_14: -2 | PPO_13: -10
```

### 調整理由

| 項目 | PPO_13 (-10 元) | PPO_14 (-2 元) |
|------|----------------|---------------|
| **懲罰比例** | 40% 持倉獎勵 | 8% 持倉獎勵 |
| **Buy 比例** | 2% (過低) | 預期 15%-40% |
| **策略** | 過度保守 | 適度交易 |

---

## 📈 三代策略演進

| 指標 | PPO_11 (持倉激勵) | PPO_13 (未平倉懲罰) | PPO_14 (降低懲罰) |
|------|------------------|---------------------|-------------------|
| **獎勵策略** | 持倉 +0.05/步 | Episode末 -10/倉位 | Episode末 -2/倉位 |
| **Buy 比例** | 99.4% 🔴 | 2.0% 🔴 | 預期 15-40% ✅ |
| **Episode 獎勵** | +24.15 ✅ | -2.91 ❌ | 預期 > 0 🎯 |
| **交易次數** | 3.8/Ep | 5.8/Ep | 預期 5-10/Ep |
| **KL 散度** | 0.0076 ✅ | 0.0036 ✅ | 預期 < 0.02 ✅ |
| **解釋方差** | 0.827 ✅ | 0.966 ✅ | 預期 > 0.7 ✅ |
| **問題** | 過度交易 | **過度保守** | **平衡點** 🎯 |

---

## 🚀 執行計劃

### 1. 快速啟動

```bash
# 方式 1: 使用批次腳本 (推薦)
run_ppo14.bat

# 方式 2: 手動執行
conda activate deeplob-pro
python scripts/train_sb3_deeplob.py --config configs/sb3_deeplob_config.yaml
```

### 2. 監控訓練

```bash
# 打開 TensorBoard
tensorboard --logdir logs/sb3_deeplob/

# 瀏覽器訪問
http://localhost:6006/
```

### 3. 訓練後驗證 (3 步驟)

```bash
# 步驟 1: 分析訓練日誌
python scripts/analyze_tensorboard.py --log-dir logs/sb3_deeplob/PPO_14

# 步驟 2: 檢查交易行為
python scripts/check_trading_behavior.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 5

# 步驟 3: 完整評估
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 20 \
    --save_report
```

---

## 🎯 成功標準

### 必須達成 (Basic)

- [ ] Buy 比例 > 5% (避免過度保守)
- [ ] Episode 獎勵 > -1 (接近盈利)
- [ ] KL 散度 < 0.02 (訓練穩定)
- [ ] 解釋方差 > 0.7 (價值函數良好)

### 理想目標 (Target)

- [ ] Buy 比例: 15%-40% (適度交易)
- [ ] Episode 獎勵: > +5 (實際盈利)
- [ ] 交易次數: 5-10/Episode (合理頻率)
- [ ] Sharpe Ratio: > 1.0 (風險調整後收益)

### 卓越表現 (Stretch)

- [ ] Buy 比例: 25%-35% (最佳平衡)
- [ ] Episode 獎勵: > +15 (高盈利)
- [ ] 勝率: > 55% (多數交易盈利)
- [ ] 最大回撤: < 10% (風險可控)

---

## 🔄 備選方案 (如失敗)

### 方案 B: 漸進式懲罰 ⭐⭐⭐⭐

```python
# 根據持倉時間調整懲罰
持倉 < 200 步: 無懲罰
持倉 200-400 步: -1 元/倉位
持倉 > 400 步: -5 元/倉位
```

**何時使用**: PPO_14 仍過度保守 (Buy < 5%)

### 方案 C: 混合策略 ⭐⭐⭐

```python
holding_bonus = +0.01 * position      # 鼓勵交易
unclosed_penalty = -5.0 * position    # 避免隔夜
```

**何時使用**: 需要更精細的平衡控制

### 方案 D: 回到 PPO_11 ⭐⭐

```python
holding_bonus = +0.01 * position      # 降低 80%
unclosed_penalty = 0                  # 移除懲罰
```

**何時使用**: PPO_14 效果不佳，回退至已驗證的方案

---

## 📝 文件修改記錄

### 修改文件

1. **[src/envs/tw_lob_trading_env.py](../src/envs/tw_lob_trading_env.py#L380)**
   - Line 380: `unclosed_penalty = -10.0` → `-2.0`

2. **[configs/sb3_deeplob_config.yaml](../configs/sb3_deeplob_config.yaml)**
   - Line 71-72: 更新註釋說明 PPO_14 配置

3. **[docs/20251026-sb3調參歷史.md](../docs/20251026-sb3調參歷史.md)**
   - 新增 PPO_13 完整分析
   - 新增 PPO_14 計劃

### 新增文件

1. **[docs/PPO_14_PLAN.md](PPO_14_PLAN.md)** - 詳細訓練計劃
2. **[run_ppo14.bat](../run_ppo14.bat)** - 一鍵訓練腳本
3. **[results/PPO_14_SUMMARY.md](PPO_14_SUMMARY.md)** - 本文檔

---

## 🔍 核心洞察

### 為什麼 PPO_13 失敗？

1. **獎勵信號衝突**:
   - PnL 鼓勵盈利（需持倉）
   - 未平倉懲罰抑制持倉（-10 元過重）
   - 結果: 模型選擇「不動」最安全

2. **懲罰過度放大**:
   - -10 元 = 40% 持倉獎勵
   - 模型寧可放棄盈利機會，也要避免懲罰

3. **訓練指標誤導**:
   - KL/解釋方差優秀（模型收斂穩定）
   - 但收斂到錯誤策略（不交易）
   - **教訓**: 訓練指標 ≠ 實際性能

### PPO_14 為何可能成功？

1. **適度懲罰**: -2 元 = 8% 獎勵（vs. 40%）
2. **保留激勵**: 仍鼓勵當日平倉，但不過度抑制
3. **參考成功經驗**: PPO_11 證明 +24.15 盈利可達成

---

## 📚 參考資料

### 專案文檔

- [CLAUDE.md](../CLAUDE.md) - 專案總覽
- [調參歷史](../docs/20251026-sb3調參歷史.md) - PPO_5 到 PPO_14 完整記錄
- [獎勵計算指南](../docs/REWARD_CALCULATION_GUIDE.md) - 獎勵函數設計
- [未平倉懲罰策略](../docs/UNCLOSED_POSITION_PENALTY.md) - PPO_12 原始設計

### 訓練結果

- PPO_11 報告: `logs/sb3_deeplob/PPO_11/` (持倉激勵成功)
- PPO_13 報告: `logs/sb3_deeplob/PPO_13/` (未平倉懲罰過度)
- PPO_14 報告: `logs/sb3_deeplob/PPO_14/` (待生成)

---

## ⏱️ 時間預估

| 步驟 | 時間 | 說明 |
|------|------|------|
| 訓練 | ~28 分鐘 | 200K steps @ 119 steps/sec |
| 分析 | ~2 分鐘 | TensorBoard + 交易行為 |
| 評估 | ~5 分鐘 | 20 Episodes 完整評估 |
| **總計** | **~35 分鐘** | 完整測試循環 |

---

## 📞 問題排查

### Q1: 訓練卡住不動？

```bash
# 檢查 GPU 利用率
nvidia-smi

# 檢查進程
tasklist | findstr python

# 強制停止
taskkill /IM python.exe /F
```

### Q2: 內存不足？

```yaml
# 降低 batch_size
ppo:
  batch_size: 32  # 預設 64
```

### Q3: Buy 比例仍 < 5%？

**立即切換方案 B (漸進式懲罰)**:
- 修改 [tw_lob_trading_env.py](../src/envs/tw_lob_trading_env.py#L376-L385)
- 實施持倉時間追蹤
- 分階段懲罰

---

**最後更新**: 2025-10-27
**狀態**: ⏳ 待訓練
**下一步**: 執行 `run_ppo14.bat` 開始訓練
