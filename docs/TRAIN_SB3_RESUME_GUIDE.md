# train_sb3_deeplob.py 續訓功能使用指南

## 版本資訊

- **腳本版本**: v2.1
- **更新日期**: 2025-10-26
- **新增功能**: 支持續訓（Resume Training）

---

## 功能概述

`train_sb3_deeplob.py` 現在支持以下兩種訓練模式：

1. **從頭訓練** (Train from Scratch) - 創建新的 PPO 模型
2. **續訓模式** (Resume Training) - 從已保存的檢查點繼續訓練

---

## 續訓功能特性

### ✅ 支持的功能

- 從任意 PPO 檢查點繼續訓練
- 保留訓練狀態（優化器狀態、學習率調度等）
- 自動累加訓練步數（或選擇重置）
- 支持命令行參數和 YAML 配置兩種方式
- 完整的檢查點驗證

### 🎯 適用場景

1. **訓練中斷恢復**: 意外中斷後繼續訓練
2. **分階段訓練**: 先訓練 500K steps，評估後再訓練 500K steps
3. **超參數微調**: 載入模型後調整學習率繼續訓練
4. **長時間訓練**: 分多個階段完成長時間訓練任務

---

## 使用方法

### 方法 1: 命令行參數（推薦）

#### 基礎續訓（步數累加）

```bash
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_500000_steps.zip \
    --timesteps 1000000
```

**說明**:
- 從 500K steps 檢查點繼續訓練
- 再訓練 1M steps（總計 1.5M steps）
- 步數計數器繼續累加

#### 續訓並重置步數

```bash
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_500000_steps.zip \
    --reset-timesteps \
    --timesteps 500000
```

**說明**:
- 從檢查點載入模型
- 步數計數器重置為 0
- 訓練 500K steps（計數器從 0 → 500K）

#### 完整參數範例

```bash
python scripts/train_sb3_deeplob.py \
    --config configs/sb3_deeplob_config.yaml \
    --resume checkpoints/sb3/ppo_deeplob/best_model.zip \
    --timesteps 1000000 \
    --n-envs 4 \
    --device cuda
```

---

### 方法 2: YAML 配置文件

#### 1. 編輯配置文件

修改 `configs/sb3_deeplob_config.yaml`:

```yaml
# ===== 訓練配置 =====
training:
  # 訓練步數
  total_timesteps: 1000000      # 續訓的步數

  # 續訓配置
  resume:
    enabled: true               # 啟用續訓模式
    checkpoint_path: "checkpoints/sb3/ppo_deeplob/ppo_model_500000_steps.zip"
    reset_timesteps: false      # 繼續累加步數
```

#### 2. 運行訓練

```bash
python scripts/train_sb3_deeplob.py --config configs/sb3_deeplob_config.yaml
```

---

## 參數說明

### 命令行參數

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `--resume` | str | PPO 檢查點路徑（.zip 文件） | None |
| `--reset-timesteps` | flag | 重置時間步數計數器 | False |
| `--timesteps` | int | 訓練步數 | 1000000 |
| `--config` | str | 配置文件路徑 | configs/sb3_deeplob_config.yaml |

### YAML 配置

```yaml
training:
  resume:
    enabled: false              # 是否啟用續訓模式
    checkpoint_path: null       # PPO 模型路徑
    reset_timesteps: false      # 是否重置步數
```

---

## 檢查點文件說明

### 自動保存的檢查點

訓練過程中會自動保存以下檢查點：

```
checkpoints/sb3/ppo_deeplob/
├── ppo_model_50000_steps.zip     # 每 50K steps 保存
├── ppo_model_100000_steps.zip
├── ppo_model_150000_steps.zip
├── ...
├── best_model.zip                # 最佳模型（評估分數最高）
└── ppo_deeplob_final.zip         # 最終模型（訓練完成）
```

### 推薦續訓檢查點

1. **best_model.zip** - 最佳性能模型（推薦）
2. **ppo_deeplob_final.zip** - 最新訓練完成的模型
3. **ppo_model_XXXXX_steps.zip** - 特定步數的檢查點

---

## 實際使用場景

### 場景 1: 訓練中斷恢復

**問題**: 訓練到 300K steps 時電腦意外重啟

**解決方案**:
```bash
# 找到最近的檢查點
ls checkpoints/sb3/ppo_deeplob/ppo_model_*steps.zip

# 從最近的檢查點（250K）繼續訓練
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_250000_steps.zip \
    --timesteps 1000000  # 繼續訓練到 1.25M
```

### 場景 2: 分階段訓練

**目標**: 先訓練 500K，評估結果後決定是否繼續

**第一階段**:
```bash
python scripts/train_sb3_deeplob.py --timesteps 500000
```

**評估**:
```bash
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model
```

**第二階段**（如果結果良好）:
```bash
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/best_model.zip \
    --timesteps 500000  # 再訓練 500K（總計 1M）
```

### 場景 3: 超參數微調

**目標**: 訓練到 500K 後降低學習率繼續訓練

**第一階段**（正常訓練）:
```bash
python scripts/train_sb3_deeplob.py --timesteps 500000
```

**第二階段**（降低學習率）:

1. 修改 `configs/sb3_deeplob_config.yaml`:
```yaml
ppo:
  learning_rate: 0.0001  # 從 3e-4 降低到 1e-4
```

2. 續訓:
```bash
python scripts/train_sb3_deeplob.py \
    --config configs/sb3_deeplob_config.yaml \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_500000_steps.zip \
    --timesteps 500000
```

### 場景 4: 長時間訓練分段執行

**目標**: 訓練 5M steps，每次訓練 1M steps

```bash
# 階段 1: 0 → 1M
python scripts/train_sb3_deeplob.py --timesteps 1000000

# 階段 2: 1M → 2M
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_1000000_steps.zip \
    --timesteps 1000000

# 階段 3: 2M → 3M
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_2000000_steps.zip \
    --timesteps 1000000

# 階段 4: 3M → 4M
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_3000000_steps.zip \
    --timesteps 1000000

# 階段 5: 4M → 5M
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/ppo_model_4000000_steps.zip \
    --timesteps 1000000
```

---

## 常見問題 (FAQ)

### Q1: 續訓時會保留哪些訓練狀態？

**A**: 續訓會保留：
- ✅ PPO 策略網絡權重（Actor）
- ✅ 價值網絡權重（Critic）
- ✅ 優化器狀態（Adam 動量等）
- ✅ 學習率調度器狀態
- ✅ 訓練步數計數器（除非使用 --reset-timesteps）

**不會保留**:
- ❌ Rollout buffer（經驗緩衝）- 從新環境重新收集
- ❌ 環境狀態 - 重新初始化

### Q2: reset_timesteps 參數有什麼用？

**A**: 控制步數計數器的行為：

- `reset_timesteps=False`（預設）:
  - 步數繼續累加
  - 範例: 從 500K 檢查點訓練 500K steps → 最終 1M steps
  - **推薦用於**: 正常續訓、分階段訓練

- `reset_timesteps=True`:
  - 步數從 0 重新計數
  - 範例: 從 500K 檢查點訓練 500K steps → 計數器顯示 0 → 500K
  - **推薦用於**: 超參數實驗、獨立訓練階段

### Q3: 續訓時可以修改哪些超參數？

**A**: 理論上可以修改所有超參數，但建議只修改：

**安全修改**:
- ✅ `learning_rate` - 學習率
- ✅ `total_timesteps` - 訓練步數
- ✅ `n_envs` - 環境數量
- ✅ `save_freq` / `eval_freq` - 保存/評估頻率

**不建議修改**:
- ⚠️ `n_steps` / `batch_size` - 可能影響訓練穩定性
- ⚠️ `gamma` / `gae_lambda` - 改變獎勵計算邏輯
- ❌ `net_arch` - 網絡架構（會導致載入失敗）

### Q4: 如何檢查續訓是否成功？

**A**: 查看日誌輸出：

```
🔄 續訓模式: 載入已訓練的 PPO 模型
  - 檢查點: checkpoints/sb3/ppo_deeplob/ppo_model_500000_steps.zip
  - 重置步數: False
✅ PPO 模型載入成功
  - 當前步數: 500,000
  - Learning Rate: 0.0003
  - Device: cuda
```

關鍵指標:
- `當前步數` 應該等於檢查點的步數
- 訓練應該從該步數繼續

### Q5: 續訓時 DeepLOB 特徵提取器會怎樣？

**A**: DeepLOB 特徵提取器在續訓時：

- ✅ 會重新載入（從配置或命令行指定的檢查點）
- ✅ 權重仍然保持凍結（freeze_deeplob: true）
- ✅ 不會被 PPO 檢查點覆蓋

**注意**: 確保續訓時使用相同的 DeepLOB 檢查點，否則可能導致性能下降。

### Q6: 續訓失敗如何排查？

**常見錯誤及解決方案**:

1. **檢查點文件不存在**:
   ```
   FileNotFoundError: 續訓檢查點不存在: xxx.zip
   ```
   - 檢查路徑是否正確
   - 確認文件是否存在

2. **檢查點損壞**:
   ```
   RuntimeError: 無法載入續訓檢查點
   ```
   - 使用其他檢查點（如 best_model.zip）
   - 檢查磁碟空間是否充足

3. **環境不匹配**:
   ```
   ValueError: observation_space mismatch
   ```
   - 確保使用相同的數據配置
   - 檢查 env_config 設置是否一致

---

## 最佳實踐

### ✅ 推薦做法

1. **定期保存檢查點**: 設置合理的 `save_freq`（推薦 50K steps）
2. **保留多個檢查點**: 不要只保留最新檢查點，保留最近 3-5 個
3. **使用 best_model**: 續訓時優先使用評估性能最佳的模型
4. **記錄訓練日誌**: 使用 TensorBoard 監控訓練進度
5. **驗證續訓效果**: 續訓前後對比評估指標

### ❌ 避免的做法

1. **頻繁更改超參數**: 每次續訓都改變超參數會導致訓練不穩定
2. **跨數據集續訓**: 不要在不同數據集之間續訓
3. **忽略檢查點版本**: 確保 SB3 版本一致
4. **過度續訓**: 注意過擬合風險

---

## 監控續訓進度

### 使用 TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir logs/sb3_deeplob/

# 訪問 http://localhost:6006
```

**關鍵指標**:
- `rollout/ep_rew_mean` - 平均 episode 獎勵
- `train/learning_rate` - 當前學習率
- `train/loss` - 訓練損失
- `time/fps` - 訓練速度（frames per second）

### 日誌文件

續訓日誌保存在:
```
logs/sb3_deeplob/PPO_1/
├── events.out.tfevents.xxx  # TensorBoard 事件文件
└── ...
```

---

## 總結

### 續訓功能優勢

1. ✅ **靈活性**: 支持分階段訓練和中斷恢復
2. ✅ **可控性**: 可以精確控制訓練過程
3. ✅ **效率**: 避免重複訓練浪費資源
4. ✅ **安全性**: 自動驗證檢查點有效性

### 快速參考命令

```bash
# 從頭訓練
python scripts/train_sb3_deeplob.py --timesteps 1000000

# 續訓（步數累加）
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/best_model.zip \
    --timesteps 500000

# 續訓（重置步數）
python scripts/train_sb3_deeplob.py \
    --resume checkpoints/sb3/ppo_deeplob/best_model.zip \
    --reset-timesteps \
    --timesteps 500000

# 評估模型
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model
```

---

## 相關文檔

- [CLAUDE.md](../CLAUDE.md) - 專案總覽
- [SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md) - SB3 實施報告
- [configs/sb3_deeplob_config.yaml](../configs/sb3_deeplob_config.yaml) - 完整配置文件

---

**最後更新**: 2025-10-26
**文檔版本**: v1.0
