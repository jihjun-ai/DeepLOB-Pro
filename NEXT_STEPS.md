# DeepLOB-Pro 下一步執行計劃

**日期**: 2025-10-25
**當前狀態**: DeepLOB 訓練完成 (Val Acc 50.24%), 準備進入 RL 階段
**目標**: 實現高頻交易策略 (Sharpe Ratio > 2.0)

---

## 執行路線圖（優先順序）

### ✅ 已完成

1. DeepLOB 訓練 (V5 Exp-4)
   - Val Acc: 50.24%
   - Val F1: 0.4929
   - 檢查點: `checkpoints/v5/deeplob_v5_best.pth`

2. 50% 天花板診斷
   - 確認: 標籤定義問題（持平類 43%）
   - 結論: 當前 DeepLOB 可能已夠用於 RL

3. 配置更新
   - train_v5.yaml → V5 Exp-5 (batch 160 微調)
   - 調參歷史文檔完整記錄

---

## 📋 立即執行步驟（按順序）

### 步驟 1: 快速診斷（30 分鐘）⭐⭐⭐⭐⭐

**目的**: 確認基線、驗證診斷

```bash
# 啟動環境
conda activate deeplob-pro

# 執行診斷
python scripts/diagnose_label_quality.py

# 預期輸出:
# - 標籤分布: 持平 43% ✓
# - 簡單基線: Logistic Regression ≈ 48-50%
# - 診斷報告: results/label_diagnosis/diagnosis_report.md

# 檢查結果
cat results/label_diagnosis/diagnosis_report.md
```

**判斷標準**:
- 若簡單基線 < 48%: DeepLOB 有明顯優勢 → 繼續
- 若簡單基線 ≈ 50%: 確認非容量問題 → 繼續 RL

---

### 步驟 2: 創建 DeepLOB 分桶評估腳本（1 小時）⭐⭐⭐⭐⭐

**目的**: 檢查 DeepLOB 在不同信心區間的準確率

**創建腳本**: `scripts/evaluate_deeplob_by_confidence.py`

```python
"""
DeepLOB 分桶評估
檢查不同信心區間的準確率
"""
import torch
import numpy as np
from pathlib import Path

# 載入模型
checkpoint = torch.load('checkpoints/v5/deeplob_v5_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 載入驗證集
val_data = np.load('data/processed_v7/npz/stock_embedding_val.npz')

# 推理
with torch.no_grad():
    logits = model(X_val)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    confidence = torch.max(probs, dim=1)[0]

# 分桶評估
bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(bins)-1):
    mask = (confidence >= bins[i]) & (confidence < bins[i+1])
    acc = accuracy_score(y_val[mask], preds[mask])
    print(f"信心 {bins[i]:.1f}-{bins[i+1]:.1f}: "
          f"{mask.sum():6d} 樣本, 準確率 {acc*100:.2f}%")
```

**執行**:
```bash
python scripts/evaluate_deeplob_by_confidence.py
```

**預期結果**:
```
信心區間     樣本數    準確率
0.9-1.0      8,000    75-85%  ⭐ 關鍵!
0.8-0.9     16,000    65-75%
0.7-0.8     25,000    60-65%
0.6-0.7     35,000    55-60%
0.5-0.6     78,000    45-50%
```

**判斷標準**:
- ✅ 若高信心區 (>0.8) 準確率 > 70%: 可用於 RL
- ⚠️ 若高信心區 < 65%: 需要先改進 DeepLOB

---

### 步驟 3A: 若高信心區 OK → 直接開始 RL 訓練（4-8 小時）⭐⭐⭐⭐⭐

**目的**: 訓練 PPO Agent 學習交易策略

```bash
# 確認 SB3 環境配置
python scripts/verify_env.py

# 開始 RL 訓練（完整版，1M steps）
python scripts/train_sb3_deeplob.py \
    --timesteps 1000000 \
    --deeplob-checkpoint checkpoints/v5/deeplob_v5_best.pth

# 或快速測試（10K steps，10 分鐘）
python scripts/train_sb3_deeplob.py \
    --timesteps 10000 \
    --test

# 監控訓練
tensorboard --logdir logs/sb3_deeplob/
```

**監控指標**:
- Episode Reward (目標: 持續上升)
- Episode Length
- Policy Loss
- Value Loss

**預期訓練時間**:
- 10K steps (測試): 10 分鐘
- 100K steps: 1 小時
- 1M steps (完整): 4-8 小時

---

### 步驟 3B: 若高信心區不佳 → 改進 DeepLOB（1-2 天）

**選項 1: 收緊持平定義（最快）**

1. 檢查當前標籤生成邏輯
   ```bash
   # 查看預處理腳本中的標籤定義
   grep -n "持平\|stationary" scripts/preprocess_single_day.py
   ```

2. 調整閾值
   ```python
   # 當前 (推測): threshold = 0.05%
   # 改為: threshold = 0.02%

   # 或動態閾值:
   threshold = 0.5 * rolling_std
   ```

3. 重新生成數據
   ```bash
   python scripts/extract_tw_stock_data_v7.py \
       --preprocessed-dir ./data/preprocessed_v5 \
       --output-dir ./data/processed_v8 \
       --config ./configs/config_v8_tight_stationary.yaml
   ```

4. 重新訓練 DeepLOB
   ```bash
   python scripts/train_deeplob_v5.py \
       --config configs/train_v8.yaml
   ```

**選項 2: 增加模型容量（較慢）**

修改 `configs/train_v5.yaml`:
```yaml
# 增加容量
conv_filters: 48  # 32 → 48
lstm_hidden: 64   # 48 → 64

# 適中正則化
dropout: 0.72     # 0.78 → 0.72
weight_decay: 0.002  # 0.0029 → 0.002
batch_size: 256   # 160 → 256
lr: 1.2e-6        # 7.3e-7 → 1.2e-6
```

---

### 步驟 4: 評估 RL 策略（30 分鐘）⭐⭐⭐⭐⭐

**執行評估**:
```bash
python scripts/evaluate_sb3.py \
    --model checkpoints/sb3/ppo_deeplob/best_model \
    --n_episodes 20 \
    --save_report

# 檢查報告
cat results/rl_evaluation_report.json
```

**關鍵指標**:
1. **收益指標**
   - 總收益 (目標: > 0)
   - Sharpe Ratio (目標: > 2.0) ⭐⭐⭐⭐⭐
   - 最大回撤 (目標: < 10%)

2. **交易統計**
   - 勝率 (目標: > 55%)
   - 交易次數 (合理範圍)
   - 平均持倉時間

3. **風險指標**
   - 波動率
   - 最大連續虧損

**判斷標準**:
- ✅ Sharpe > 2.0: 成功! → 優化超參數
- ⚠️ Sharpe 1.5-2.0: 還可以 → 調整獎勵函數
- ❌ Sharpe < 1.5: 需要改進 DeepLOB 或環境設計

---

### 步驟 5: 根據結果決定下一步

#### 情境 A: RL 策略表現優秀 (Sharpe > 2.0) ✅

**下一步**:
1. 超參數優化（PPO 學習率、Gamma 等）
2. 獎勵函數微調
3. 增加訓練時間 (2M-5M steps)
4. 回測系統整合
5. 模型壓縮與部署

#### 情境 B: RL 策略表現中等 (Sharpe 1.5-2.0) ⚠️

**下一步**:
1. 調整獎勵函數權重
2. 嘗試不同 RL 算法 (A2C, SAC)
3. 增加環境複雜度（手續費、滑點等）
4. 分析失敗案例

#### 情境 C: RL 策略表現不佳 (Sharpe < 1.5) ❌

**下一步**:
1. 回到步驟 3B 改進 DeepLOB
2. 重新設計環境（獎勵函數、狀態空間）
3. 檢查數據質量
4. 考慮更複雜的特徵工程

---

## 🎯 推薦執行順序（今天）

### 優先級 1（必做）:
```bash
# 1. 診斷腳本 (30 分鐘)
python scripts/diagnose_label_quality.py

# 2. 檢查當前檢查點
ls -lh checkpoints/v5/deeplob_v5_best.pth
```

### 優先級 2（推薦）:
```bash
# 3. 創建並執行分桶評估 (1 小時)
# → 需要創建 scripts/evaluate_deeplob_by_confidence.py
python scripts/evaluate_deeplob_by_confidence.py
```

### 優先級 3（核心）:
```bash
# 4. 根據分桶評估結果決定:
#    - 若高信心區 > 70%: 直接 RL 訓練
#    - 若高信心區 < 65%: 先改進 DeepLOB

# 選項 A: 直接 RL (快速測試 10K steps)
python scripts/train_sb3_deeplob.py --timesteps 10000 --test

# 選項 B: 改進 DeepLOB
# (需要先分析標籤生成邏輯)
```

---

## 📊 成功標準

### DeepLOB 階段 ✅
- [x] Val Acc > 50% (已達成 50.24%)
- [x] 診斷完成
- [ ] 高信心區 (>0.8) 準確率 > 70%

### RL 階段 ⏳
- [ ] 環境驗證通過
- [ ] PPO 訓練收斂
- [ ] Sharpe Ratio > 2.0
- [ ] 勝率 > 55%
- [ ] 最大回撤 < 10%

### 部署階段 ⏳
- [ ] 模型壓縮
- [ ] 推理優化
- [ ] 回測系統整合
- [ ] 實盤測試

---

## 📁 重要文件位置

```
DeepLOB-Pro/
├── checkpoints/v5/
│   └── deeplob_v5_best.pth         ⭐ 當前最佳模型
├── data/processed_v7/npz/          ⭐ 訓練數據
├── logs/deeplob_v5_exp4/           ⭐ 訓練日誌
├── configs/
│   └── train_v5.yaml               ⭐ 訓練配置
├── scripts/
│   ├── diagnose_label_quality.py   ⭐ 診斷腳本
│   ├── train_sb3_deeplob.py        ⭐ RL 訓練
│   └── evaluate_sb3.py             ⭐ RL 評估
└── docs/
    └── 20251025-deeplob調參歷史.md ⭐ 完整記錄
```

---

## ❓ 常見問題

### Q1: 為什麼不先提升 DeepLOB 到 60%+?
**A**: 因為:
1. 持平類 43% 導致的 50% 是標籤問題，非容量問題
2. RL 只需高信心區域準確，不需要整體 60%
3. 直接測試 RL 可能已經成功

### Q2: 如何判斷是否需要改進 DeepLOB?
**A**: 看兩個指標:
1. 分桶評估: 高信心區 (>0.8) 準確率 < 65%
2. RL 策略: Sharpe Ratio < 1.5

### Q3: V5 Exp-5 還需要訓練嗎?
**A**:
- 優先級低，batch 160 微調最多提升 0.2-0.4%
- 建議先執行 RL 訓練
- 若 RL 不佳再回來訓練

### Q4: 訓練 RL 需要多少時間?
**A**:
- 快速測試 (10K steps): 10 分鐘
- 完整訓練 (1M steps): 4-8 小時 (RTX 5090)
- 建議先跑 10K 測試流程

---

## 🚀 今天的目標

**最低目標** (2-3 小時):
1. ✅ 執行診斷腳本
2. ✅ 創建分桶評估腳本
3. ✅ 執行分桶評估

**理想目標** (4-6 小時):
4. ✅ RL 快速測試 (10K steps)
5. ✅ 檢查訓練曲線
6. ✅ 決定下一步方向

**延伸目標** (8+ 小時):
7. ✅ RL 完整訓練 (1M steps)
8. ✅ 策略評估
9. ✅ 開始優化

---

**準備好了嗎？從步驟 1 開始！** 🎯
