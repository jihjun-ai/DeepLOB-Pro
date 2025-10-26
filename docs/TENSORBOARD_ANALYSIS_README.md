# TensorBoard 自動分析系統 - 快速開始

## 📋 系統概述

這是一個**自動化的 TensorBoard 日誌分析系統**，可以：

1. ✅ **自動提取**所有關鍵訓練指標
2. ✅ **客觀分析**訓練狀況（無主觀判斷）
3. ✅ **診斷問題**並提供具體建議
4. ✅ **生成報告**（JSON + Markdown 雙格式）
5. ✅ **支持 AI 分析**（提供結構化數據）

---

## 🚀 30 秒快速開始

```bash
# 1. 訓練完成後，立即分析
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/analysis

# 2. 查看報告
cat results/analysis.md

# 3. 將 JSON 提供給 AI 分析（推薦）
# 複製 results/analysis.json 的內容
# 貼上到 Claude/GPT-4，使用提示詞模板
```

---

## 📚 完整文檔

| 文檔 | 內容 | 適用場景 |
|------|------|----------|
| **[TENSORBOARD_ANALYSIS_GUIDE.md](TENSORBOARD_ANALYSIS_GUIDE.md)** | 完整使用指南 | 詳細了解所有功能 |
| **[AI_ANALYSIS_PROMPT_TEMPLATE.md](AI_ANALYSIS_PROMPT_TEMPLATE.md)** | AI 分析提示詞模板 | 讓 AI 幫您分析報告 |
| 本文檔 (README) | 快速開始 | 5 分鐘上手 |

---

## 🎯 核心功能

### 功能 1：自動提取指標

**提取 15+ 個關鍵指標**:
- 訓練進度（獎勵、episode 長度）
- 性能指標（損失、解釋方差）
- 穩定性指標（KL 散度、Clip 比例）
- 自定義指標（Sharpe Ratio 等）

### 功能 2：客觀分析

**量化分析**:
- 趨勢分析（線性回歸，R²）
- 收斂檢測（自動判斷）
- 穩定性評估（統計指標）
- 健康度評分（0-100 分）

### 功能 3：問題診斷

**自動檢測**:
- ❌ 獎勵下降或不變
- ❌ KL 散度過高（訓練不穩定）
- ❌ 解釋方差過低（價值函數不佳）
- ❌ NaN 數值（訓練失敗）

### 功能 4：優化建議

**智能建議**:
- 獎勵函數調整（pnl_scale, cost_penalty）
- PPO 超參數調整（learning_rate, clip_range）
- 網絡架構調整（lstm_hidden_size）
- 訓練策略調整（訓練時長、探索率）

---

## 📊 輸出格式

### JSON 格式（給 AI 分析）

```json
{
  "metadata": {
    "total_steps": 100000,
    "duration_hours": 1.25
  },
  "training_progress": {
    "episode_reward": {
      "improvement": 580.23,
      "trend": {
        "slope": 0.058,
        "direction": "increasing"
      }
    }
  },
  "diagnostic": {
    "health_score": 85.0,
    "issues": [],
    "recommendations": [...]
  }
}
```

### Markdown 格式（人類閱讀）

```markdown
# TensorBoard 訓練分析報告

## 📊 訓練概況
- 總訓練步數: 100,000
- 健康度評分: 85/100

## 📈 訓練進度
- 獎勵提升: +580.23
- 趨勢: 持續上升 ✅

## 🚀 優化建議
1. 延長訓練到 500K steps
2. 微調獎勵權重
```

---

## 🔧 使用場景

### 場景 1：訓練完成後檢查

```bash
# 快速檢查訓練狀況
python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1

# 問自己：
# - 健康度評分如何？（> 80 為良好）
# - 獎勵是否上升？
# - 有無嚴重問題？
```

### 場景 2：實驗對比

```bash
# 對比多個實驗
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/ \
    --compare \
    --output results/comparison.json

# 問自己：
# - 哪個配置最好？
# - 為什麼它最好？
# - 下一步如何改進？
```

### 場景 3：AI 深度分析（推薦）⭐⭐⭐⭐⭐

```bash
# 1. 生成結構化報告
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/analysis.json

# 2. 複製 JSON 內容

# 3. 使用提示詞模板（AI_ANALYSIS_PROMPT_TEMPLATE.md）
#    提供給 Claude/GPT-4

# 4. 獲得深入的洞察和建議
```

---

## 💡 最佳實踐

### 實踐 1：每次訓練後都分析

```bash
# 在訓練腳本末尾添加
python scripts/train_sb3_deeplob.py --timesteps 100000
python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1
```

### 實踐 2：保存所有分析報告

```bash
# 使用時間戳命名
timestamp=$(date +%Y%m%d_%H%M%S)
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/analysis_$timestamp
```

### 實踐 3：結合 AI 分析

```bash
# 1. 生成報告
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/for_ai.json

# 2. 使用 AI_ANALYSIS_PROMPT_TEMPLATE.md 中的模板
# 3. 根據 AI 建議調整配置
# 4. 重新訓練
```

---

## 🎓 學習路徑

### 新手（5 分鐘）

1. ✅ 閱讀本文檔（快速開始）
2. ✅ 運行測試腳本：`scripts\test_tensorboard_analysis.bat`
3. ✅ 查看生成的 Markdown 報告

### 進階（30 分鐘）

1. ✅ 閱讀 [TENSORBOARD_ANALYSIS_GUIDE.md](TENSORBOARD_ANALYSIS_GUIDE.md)
2. ✅ 了解所有指標含義
3. ✅ 學習問題診斷方法

### 高級（1 小時）

1. ✅ 閱讀 [AI_ANALYSIS_PROMPT_TEMPLATE.md](AI_ANALYSIS_PROMPT_TEMPLATE.md)
2. ✅ 使用 AI 分析您的訓練結果
3. ✅ 根據建議進行實驗

---

## 🛠️ 快速測試

### Windows 用戶

```bash
# 運行測試腳本
scripts\test_tensorboard_analysis.bat
```

### Linux/Mac 用戶

```bash
# 找到最新日誌
latest_log=$(ls -t logs/sb3_deeplob/ | head -1)

# 運行分析
python scripts/analyze_tensorboard.py \
    --logdir "logs/sb3_deeplob/$latest_log" \
    --output results/test_analysis \
    --format both \
    --verbose
```

---

## 📈 示例輸出

### 終端輸出（摘要）

```
======================================================================
TensorBoard 訓練分析報告
======================================================================

## 📊 訓練概況

- 總訓練步數: 100,000
- 訓練時長: 1.25 小時
- 訓練速度: 22,222 steps/秒

## 📈 訓練進度

### Episode 獎勵

- 初始值: -123.45
- 最終值: 456.78
- 總提升: 580.23
- 趨勢: increasing (R² = 0.892) ✅

## 🏥 診斷報告

### 健康度評分: 85/100 ✅

### 💡 建議

- 建議至少訓練 500K+ steps

## 📝 結論

✅ 訓練狀況良好，可以繼續當前配置。
```

### JSON 輸出（給 AI）

完整的結構化數據，包含：
- 所有原始數值
- 趨勢分析結果
- 問題診斷
- 優化建議

---

## ❓ 常見問題

### Q1: 分析需要多長時間？

**A**: 通常 5-10 秒，取決於日誌大小。

### Q2: 可以分析正在訓練的模型嗎？

**A**: 可以！TensorBoard 會實時寫入數據，分析會獲取當前最新數據。

### Q3: JSON 報告太大怎麼辦？

**A**:
- 只提供關鍵部分給 AI（metadata, diagnostic, recommendations）
- 使用 `jq` 工具過濾：`cat analysis.json | jq '.diagnostic'`

### Q4: 如何比較多個實驗？

**A**: 使用 `--compare` 選項：
```bash
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/ \
    --compare \
    --output results/comparison.json
```

---

## 🔗 相關資源

### 內部文檔

- [TRAIN_SB3_DEEPLOB_GUIDE.md](TRAIN_SB3_DEEPLOB_GUIDE.md) - 訓練調教完整指南
- [REWARD_CALCULATION_GUIDE.md](REWARD_CALCULATION_GUIDE.md) - 獎勵計算說明
- [DEEPLOB_PPO_INTEGRATION.md](DEEPLOB_PPO_INTEGRATION.md) - 架構說明

### 外部資源

- [TensorBoard 官方文檔](https://www.tensorflow.org/tensorboard)
- [Stable-Baselines3 文檔](https://stable-baselines3.readthedocs.io/)

---

## 🎯 下一步

1. ✅ **運行測試**: `scripts\test_tensorboard_analysis.bat`
2. ✅ **分析您的訓練**: `python scripts/analyze_tensorboard.py --logdir [您的日誌目錄]`
3. ✅ **使用 AI 分析**: 參考 [AI_ANALYSIS_PROMPT_TEMPLATE.md](AI_ANALYSIS_PROMPT_TEMPLATE.md)
4. ✅ **調整配置**: 根據建議修改 `configs/sb3_deeplob_config.yaml`
5. ✅ **重新訓練**: `python scripts/train_sb3_deeplob.py`

---

**版本**: v1.0
**最後更新**: 2025-10-26
**維護者**: SB3-DeepLOB Team

**反饋與建議**: 請在專案 Issue 中提出

---

## 總結

**這個工具幫您解決的問題**:

❌ **之前**: 手動打開 TensorBoard → 逐個查看指標 → 主觀判斷 → 不確定下一步

✅ **現在**: 一鍵分析 → 客觀報告 → AI 深度洞察 → 明確的行動計劃

**核心價值**: 讓數據說話，讓 AI 幫您分析！
