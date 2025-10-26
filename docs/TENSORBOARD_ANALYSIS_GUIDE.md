# TensorBoard 自動分析工具使用指南

## 文件信息

- **建立日期**: 2025-10-26
- **版本**: v1.0
- **核心腳本**: `scripts/analyze_tensorboard.py`
- **目的**: 自動分析 TensorBoard 訓練日誌，生成客觀的結構化報告

---

## 一、為什麼需要自動分析？

### 人工分析的問題

**❌ 主觀性**:
- 不同人對同一圖表的解讀可能不同
- 容易忽略細微但重要的趨勢
- 難以量化"好"或"壞"

**❌ 效率低**:
- 需要逐個打開 TensorBoard 查看
- 需要手動記錄數值
- 難以比較多個實驗

**❌ 不完整**:
- 可能遺漏某些指標
- 難以發現隱藏的問題
- 缺乏系統性診斷

### 自動分析的優勢

**✅ 客觀性**:
- 基於數據的量化分析
- 統一的評估標準
- 可重複的分析流程

**✅ 高效性**:
- 一鍵生成完整報告
- 自動提取所有指標
- 支持批量分析和對比

**✅ 完整性**:
- 涵蓋所有關鍵指標
- 自動診斷潛在問題
- 提供優化建議

---

## 二、快速開始

### 2.1 安裝依賴

```bash
# 確保已安裝 tensorboard
pip install tensorboard numpy
```

### 2.2 基礎用法

```bash
# 分析單個訓練日誌
python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1

# 保存為 JSON 和 Markdown 報告
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/analysis \
    --format both
```

**輸出**:
- `results/analysis.json` - 結構化數據（可供 AI 分析）
- `results/analysis.md` - 人類可讀報告

### 2.3 第一次運行

```bash
# 假設您剛完成訓練
python scripts/train_sb3_deeplob.py --test

# 分析訓練結果
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/first_analysis
```

**預期輸出**:
```
📂 載入日誌: logs/sb3_deeplob/PPO_1
✅ 找到 15 個指標

🔍 分析數據...

✅ JSON 報告已保存: results/first_analysis.json
✅ Markdown 報告已保存: results/first_analysis.md

======================================================================
TensorBoard 訓練分析報告
======================================================================
...
```

---

## 三、分析報告結構

### 3.1 JSON 報告結構

```json
{
  "metadata": {
    "total_steps": 10000,
    "num_metrics": 15,
    "duration_hours": 0.5,
    "steps_per_second": 5555.6
  },
  "training_progress": {
    "episode_reward": {
      "initial": -123.45,
      "final": 456.78,
      "improvement": 580.23,
      "trend": {
        "slope": 0.058,
        "direction": "increasing",
        "r_squared": 0.89
      }
    }
  },
  "performance_metrics": {
    "total_loss": {
      "initial": 45.67,
      "final": 12.34,
      "converged": true
    },
    "explained_variance": {
      "final": 0.85,
      "is_good": true
    }
  },
  "stability_metrics": {
    "kl_divergence": {
      "mean": 0.0123,
      "is_stable": true
    }
  },
  "diagnostic": {
    "health_score": 85.0,
    "issues": [],
    "warnings": [],
    "suggestions": []
  },
  "recommendations": [
    {
      "category": "訓練時長",
      "priority": "low",
      "suggestion": "建議至少訓練 500K+ steps"
    }
  ]
}
```

### 3.2 關鍵指標說明

#### Metadata（元數據）

| 字段 | 說明 | 理想值 |
|------|------|--------|
| `total_steps` | 總訓練步數 | > 500K |
| `duration_hours` | 訓練時長（小時） | - |
| `steps_per_second` | 訓練速度 | > 2000 |

#### Training Progress（訓練進度）

**episode_reward**:
```json
{
  "initial": -100,        // 初始獎勵
  "final": 500,           // 最終獎勵
  "improvement": 600,     // 總提升
  "trend": {
    "slope": 0.06,        // 斜率（正值=上升）
    "direction": "increasing",  // 方向
    "r_squared": 0.85     // 擬合度（越接近1越好）
  }
}
```

**評估標準**:
- ✅ `improvement > 0`: 有進步
- ✅ `trend.direction == "increasing"`: 持續上升
- ✅ `trend.r_squared > 0.7`: 穩定上升趨勢

#### Performance Metrics（性能指標）

**total_loss**:
```json
{
  "initial": 50.0,
  "final": 10.0,
  "converged": true     // 是否收斂
}
```

**explained_variance**:
```json
{
  "final": 0.82,
  "mean": 0.78,
  "is_good": true       // > 0.7 為良好
}
```

#### Stability Metrics（穩定性指標）

**kl_divergence**:
```json
{
  "mean": 0.015,
  "max": 0.025,
  "is_stable": true     // < 0.02 為穩定
}
```

**reward_stability**:
```json
{
  "recent_mean": 500.0,
  "recent_std": 50.0,
  "coefficient_of_variation": 0.1  // 變異係數（越小越穩定）
}
```

#### Diagnostic（診斷）

**health_score**: 0-100 分
- 90-100: 優秀 ⭐⭐⭐⭐⭐
- 80-89: 良好 ⭐⭐⭐⭐
- 60-79: 尚可 ⭐⭐⭐
- < 60: 有問題 ⚠️

**issues**: 嚴重問題列表
```json
[
  {
    "type": "獎勵下降",
    "severity": "high",
    "message": "平均獎勵呈下降趨勢"
  }
]
```

**warnings**: 警告列表
```json
[
  {
    "type": "解釋方差低",
    "severity": "medium",
    "message": "解釋方差 0.45 < 0.5"
  }
]
```

---

## 四、使用場景

### 場景 1：訓練完成後的快速檢查

**目的**: 快速了解訓練狀況

```bash
# 訓練完成
python scripts/train_sb3_deeplob.py --timesteps 100000

# 立即分析
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --format markdown

# 查看報告
cat results/tensorboard_analysis.md
```

**關注點**:
- ✅ 健康度評分 > 80？
- ✅ 獎勵是否上升？
- ✅ 有無嚴重問題？

---

### 場景 2：實驗對比（多個配置）

**目的**: 找出最佳配置

```bash
# 訓練多個實驗
python scripts/train_sb3_deeplob.py --config configs/exp1_aggressive.yaml
python scripts/train_sb3_deeplob.py --config configs/exp2_balanced.yaml
python scripts/train_sb3_deeplob.py --config configs/exp3_conservative.yaml

# 對比分析
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/ \
    --compare \
    --output results/experiment_comparison.json
```

**輸出範例**:
```
實驗排名
======================================================================
1. PPO_2 (exp2_balanced)
   分數: 585.20
   最終獎勵: 578.45
   健康度: 90/100

2. PPO_3 (exp3_conservative)
   分數: 523.10
   最終獎勵: 512.34
   健康度: 85/100

3. PPO_1 (exp1_aggressive)
   分數: 445.67
   最終獎勵: 456.78
   健康度: 65/100
```

---

### 場景 3：訓練過程中的監控

**目的**: 實時了解訓練狀況，及時調整

```bash
# 在另一個終端持續監控（每 10 分鐘分析一次）
while true; do
    python scripts/analyze_tensorboard.py \
        --logdir logs/sb3_deeplob/PPO_1 \
        --format json \
        --output results/monitoring.json

    # 檢查健康度
    health=$(cat results/monitoring.json | jq '.diagnostic.health_score')
    echo "當前健康度: $health"

    if (( $(echo "$health < 60" | bc -l) )); then
        echo "⚠️ 健康度過低，建議停止訓練並檢查！"
        break
    fi

    sleep 600  # 等待 10 分鐘
done
```

---

### 場景 4：給 AI 分析（推薦）⭐⭐⭐⭐⭐

**目的**: 獲得更深入的洞察和建議

```bash
# 1. 生成結構化報告
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --output results/analysis.json \
    --format both

# 2. 將 JSON 報告提供給 AI（Claude、GPT-4 等）
```

**提示詞範例**:
```
我剛完成了一個強化學習訓練，以下是自動分析的結果：

[貼上 analysis.json 內容]

請幫我分析：
1. 訓練狀況如何？有哪些問題？
2. 為什麼獎勵提升緩慢？
3. 應該如何調整超參數？
4. 下一步應該做什麼？
```

**AI 會根據數據提供**:
- ✅ 客觀的性能評估
- ✅ 問題根因分析
- ✅ 具體的優化建議
- ✅ 下一步行動計劃

---

## 五、診斷問題示例

### 問題 1: 獎勵不上升

**報告顯示**:
```json
{
  "training_progress": {
    "episode_reward": {
      "improvement": 5.2,
      "trend": {
        "slope": 0.0001,
        "direction": "increasing"
      }
    }
  },
  "diagnostic": {
    "warnings": [
      {
        "type": "獎勵不變",
        "message": "平均獎勵幾乎不變，可能陷入局部最優"
      }
    ]
  },
  "recommendations": [
    {
      "category": "獎勵優化",
      "suggestion": "獎勵提升緩慢，可嘗試調整 pnl_scale 或 cost_penalty"
    }
  ]
}
```

**解決方案**:
```yaml
# 增加 PnL 權重
env_config.reward.pnl_scale: 1.5  # 從 1.0 增加

# 降低成本懲罰
env_config.reward.cost_penalty: 0.5  # 從 1.0 降低
```

---

### 問題 2: 訓練不穩定

**報告顯示**:
```json
{
  "stability_metrics": {
    "kl_divergence": {
      "mean": 0.035,
      "is_stable": false
    }
  },
  "diagnostic": {
    "issues": [
      {
        "type": "KL散度過高",
        "severity": "high",
        "message": "KL散度平均值 0.035 > 0.02，訓練可能不穩定"
      }
    ],
    "suggestions": [
      "降低學習率或減小 clip_range"
    ]
  }
}
```

**解決方案**:
```yaml
# 降低學習率
ppo.learning_rate: 0.0001  # 從 0.0003 降低

# 或減小 clip_range
ppo.clip_range: 0.1  # 從 0.2 降低
```

---

### 問題 3: 價值函數擬合不佳

**報告顯示**:
```json
{
  "performance_metrics": {
    "explained_variance": {
      "mean": 0.42,
      "is_good": false
    }
  },
  "diagnostic": {
    "warnings": [
      {
        "type": "解釋方差低",
        "message": "解釋方差 0.42 < 0.5，價值函數擬合不佳"
      }
    ],
    "suggestions": [
      "增加網絡容量或調整 vf_coef"
    ]
  }
}
```

**解決方案**:
```yaml
# 增加 LSTM 隱藏層大小
ppo.policy_kwargs.lstm_hidden_size: 384  # 從 256 增加

# 或調整價值函數係數
ppo.vf_coef: 0.8  # 從 0.5 增加
```

---

## 六、進階用法

### 6.1 自定義分析腳本

您可以基於 JSON 報告編寫自己的分析腳本：

```python
# scripts/custom_analysis.py

import json

# 載入分析報告
with open('results/analysis.json') as f:
    data = json.load(f)

# 自定義評估
reward_improvement = data['training_progress']['episode_reward']['improvement']
health_score = data['diagnostic']['health_score']

if reward_improvement > 100 and health_score > 80:
    print("✅ 訓練成功！可以進入下一階段")
elif health_score < 60:
    print("❌ 訓練失敗，需要調整配置")
else:
    print("⚠️ 訓練尚可，建議繼續優化")
```

### 6.2 整合到訓練流程

```python
# 在 train_sb3_deeplob.py 末尾添加

import subprocess

# 訓練完成後自動分析
subprocess.run([
    "python", "scripts/analyze_tensorboard.py",
    "--logdir", "logs/sb3_deeplob/PPO_1",
    "--output", "results/auto_analysis.json"
])

# 載入結果
with open('results/auto_analysis.json') as f:
    analysis = json.load(f)

# 根據結果決定下一步
if analysis['diagnostic']['health_score'] < 70:
    print("⚠️ 訓練質量不佳，建議檢查配置")
```

### 6.3 批量分析歷史實驗

```bash
# 分析所有歷史實驗
for dir in logs/sb3_deeplob/PPO_*; do
    exp_name=$(basename "$dir")
    echo "分析 $exp_name"

    python scripts/analyze_tensorboard.py \
        --logdir "$dir" \
        --output "results/history/$exp_name.json" \
        --format json
done

# 生成匯總報告
python scripts/summarize_experiments.py results/history/
```

---

## 七、常見問題

### Q1: 找不到 TensorBoard 日誌？

**錯誤**:
```
⚠️ 在 logs/sb3_deeplob/ 中未找到 TensorBoard 日誌
```

**解決**:
```bash
# 檢查日誌目錄
ls -la logs/sb3_deeplob/

# 確認是否有 events.out.tfevents 文件
find logs/sb3_deeplob/ -name "events.out.tfevents*"

# 如果沒有，檢查訓練是否正確配置了 tensorboard_log
```

---

### Q2: 如何只分析特定指標？

修改腳本，過濾不需要的指標：

```python
# 在 load_data() 方法中
tags = ea.Tags()['scalars']

# 只分析特定指標
filter_tags = ['rollout/ep_rew_mean', 'train/loss', 'train/approx_kl']
tags = [t for t in tags if t in filter_tags]
```

---

### Q3: 分析速度慢？

**優化方法**:

```python
# 在 EventAccumulator 中限制數據點數量
ea = event_accumulator.EventAccumulator(
    str(self.logdir),
    size_guidance={
        event_accumulator.SCALARS: 1000,  # 只載入最近 1000 個數據點
    }
)
```

---

## 八、輸出示例

### Markdown 報告示例

```markdown
# TensorBoard 訓練分析報告

**生成時間**: 2025-10-26T16:30:00

---

## 📊 訓練概況

- **總訓練步數**: 100,000
- **訓練時長**: 1.25 小時
- **訓練速度**: 22,222 steps/秒
- **指標數量**: 15

## 📈 訓練進度

### Episode 獎勵

- **初始值**: -123.45
- **最終值**: 456.78
- **最大值**: 512.34
- **平均值**: 234.56 ± 120.45
- **總提升**: 580.23
- **趨勢**: increasing (R² = 0.892)

## 🎯 性能指標

### 總損失

- **初始值**: 45.6789
- **最終值**: 12.3456
- **趨勢**: decreasing
- **已收斂**: ✅ 是

### 解釋方差

- **最終值**: 0.8234
- **平均值**: 0.7891
- **評估**: ✅ 良好 (>0.7)

## 🔒 穩定性分析

### KL 散度

- **平均值**: 0.015234
- **最大值**: 0.023456
- **穩定性**: ✅ 穩定 (<0.02)

## 🏥 診斷報告

### 健康度評分: 85/100

### 💡 建議

- 增加 LSTM 隱藏層大小或調整網絡架構

## 🚀 優化建議

### 🟢 低優先級

- **訓練時長**: 當前訓練步數較少 (100,000)，建議至少訓練 500K+ steps

## 📝 結論

✅ **訓練狀況良好**，可以繼續當前配置。
```

---

## 九、總結

### 核心價值

**自動化分析工具的三大優勢**:

1. ✅ **客觀性**: 基於數據，消除主觀判斷
2. ✅ **高效性**: 一鍵生成，節省時間
3. ✅ **完整性**: 全面診斷，不遺漏問題

### 推薦工作流

```
訓練完成
  ↓
自動分析（analyze_tensorboard.py）
  ↓
查看報告（JSON + Markdown）
  ↓
提供給 AI 分析（獲得深入建議）
  ↓
調整配置
  ↓
重新訓練
```

### 下一步

1. ✅ 使用此工具分析您的訓練結果
2. ✅ 將 JSON 報告提供給 AI（Claude/GPT-4）
3. ✅ 根據建議調整超參數
4. ✅ 開始下一輪訓練

---

**文件版本**: v1.0
**最後更新**: 2025-10-26
**相關文檔**:
- [TRAIN_SB3_DEEPLOB_GUIDE.md](TRAIN_SB3_DEEPLOB_GUIDE.md) - 訓練調教指南
- [DEEPLOB_PPO_INTEGRATION.md](DEEPLOB_PPO_INTEGRATION.md) - 架構說明
