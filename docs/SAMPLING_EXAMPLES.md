# 時間序列採樣功能說明

## 概述

為了讓 AI 能更好地理解訓練曲線的變化趨勢，同時保持數據量精簡，我們提供了兩種採樣方法：

1. **分段採樣（Segment Sampling）**: 將時間序列均勻分成 N 段，保留整體趨勢
2. **轉折點採樣（Inflection Sampling）**: 檢測曲線的關鍵轉折點，保留重要變化

---

## 使用方法

### 基礎用法

```bash
# 使用預設（兩種採樣都包含，20 段）
python scripts/analyze_tensorboard.py --logdir logs/sb3_deeplob/PPO_1

# 只使用分段採樣（10 段）
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling segment \
    --segments 10

# 只使用轉折點採樣
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling inflection \
    --inflection-sensitivity 0.1

# 兩種都用（自訂參數）
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling both \
    --segments 30 \
    --inflection-sensitivity 0.05
```

---

## 分段採樣（Segment Sampling）

### 原理

將整個訓練過程均勻分成 N 段，每段計算：
- 平均值（代表該段的典型值）
- 最小值和最大值（該段的波動範圍）
- 標準差（該段的穩定性）

### 範例輸出

假設訓練了 10,000 步，分成 20 段：

```json
{
  "episode_reward": {
    "timeseries": {
      "segment_samples": [
        {
          "step": 250,        // 第 1 段中點（0-500 步）
          "value": -120.5,    // 該段平均獎勵
          "min": -150.2,      // 該段最小值
          "max": -90.3,       // 該段最大值
          "std": 15.6         // 該段標準差
        },
        {
          "step": 750,        // 第 2 段中點（500-1000 步）
          "value": -100.2,
          "min": -130.5,
          "max": -70.1,
          "std": 12.3
        },
        // ... 共 20 個數據點
        {
          "step": 9750,       // 第 20 段中點（9500-10000 步）
          "value": 456.8,
          "min": 400.2,
          "max": 500.3,
          "std": 25.4
        }
      ]
    }
  }
}
```

### 視覺化理解

```
原始數據（10,000 個點）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

分段採樣（20 個點）:
▪️     ▪️     ▪️     ▪️  ...  ▪️
│      │      │      │        │
段1    段2    段3    段4      段20

每個點代表一段的平均值和波動範圍
```

### 優勢

✅ **數據量小**: 10,000 點 → 20 點（減少 99.8%）
✅ **保留趨勢**: 整體上升/下降/震盪趨勢清晰可見
✅ **波動信息**: 通過 min/max/std 了解每段的穩定性

### 適用場景

- 查看訓練的整體趨勢
- 對比不同階段的穩定性
- AI 快速理解訓練進展

---

## 轉折點採樣（Inflection Sampling）

### 原理

使用滑動窗口檢測曲線斜率的變化，當斜率變化超過閾值時，記錄為轉折點。

轉折點類型：
- **start**: 起點
- **valley**: 谷底（向上轉折）
- **peak**: 峰頂（向下轉折）
- **change**: 一般轉折
- **end**: 終點

### 範例輸出

```json
{
  "episode_reward": {
    "timeseries": {
      "inflection_samples": [
        {
          "step": 0,
          "value": -120.5,
          "type": "start"
        },
        {
          "step": 1250,
          "value": -95.3,
          "type": "valley",        // 谷底（開始上升）
          "slope_before": -0.05,   // 之前下降
          "slope_after": 0.08      // 之後上升
        },
        {
          "step": 3500,
          "value": 150.2,
          "type": "peak",          // 峰頂（開始下降）
          "slope_before": 0.12,
          "slope_after": -0.03
        },
        {
          "step": 5200,
          "value": 80.5,
          "type": "valley",
          "slope_before": -0.04,
          "slope_after": 0.15
        },
        {
          "step": 10000,
          "value": 456.8,
          "type": "end"
        }
      ]
    }
  }
}
```

### 視覺化理解

```
原始曲線:
       ╭─峰頂(peak)
      ╱               ╱
 谷底╱               ╱
    ╲               ╱谷底
     ╲─────────────╯

轉折點採樣:
●起點    ●谷底    ●峰頂    ●谷底    ●終點
└─────────────────────────────────┘
  只保留關鍵的轉折位置
```

### 優勢

✅ **數據量極小**: 只保留關鍵轉折點（通常 < 10 個）
✅ **保留關鍵信息**: 所有重要的變化都被捕捉
✅ **易於理解**: 清晰顯示訓練的階段性變化

### 適用場景

- 診斷訓練問題（何時開始下降？）
- 理解訓練階段（何時收斂？）
- 識別異常波動

---

## 參數調整指南

### `--segments` (分段數)

**預設**: 20

**建議**:
- 快速查看: 10 段
- 標準分析: 20 段
- 詳細分析: 50 段

**範例**:
```bash
# 10 段（粗略趨勢）
--segments 10

# 50 段（細緻趨勢）
--segments 50
```

### `--inflection-sensitivity` (轉折點敏感度)

**預設**: 0.1

**說明**:
- 數值越小，越敏感（捕捉更多轉折點）
- 數值越大，越不敏感（只捕捉顯著轉折）

**建議**:
- 高敏感度: 0.05（捕捉細微變化）
- 標準敏感度: 0.1（預設）
- 低敏感度: 0.2（只捕捉大轉折）

**範例**:
```bash
# 高敏感度（捕捉更多轉折點）
--inflection-sensitivity 0.05

# 低敏感度（只捕捉顯著轉折）
--inflection-sensitivity 0.2
```

---

## 完整示例

### 示例 1: 診斷訓練問題

```bash
# 使用轉折點採樣，查看何時開始下降
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling inflection \
    --inflection-sensitivity 0.1 \
    --output results/diagnosis.json
```

**輸出**:
```json
{
  "episode_reward": {
    "timeseries": {
      "inflection_samples": [
        {"step": 0, "value": -120.5, "type": "start"},
        {"step": 2500, "value": 200.3, "type": "peak"},      // ⚠️ 峰頂
        {"step": 3000, "value": 150.2, "type": "valley"},    // 開始下降！
        {"step": 10000, "value": -50.2, "type": "end"}       // 最終下降
      ]
    }
  }
}
```

**分析**: 在 2500 步達到峰頂後開始下降，需要檢查配置。

---

### 示例 2: 對比實驗

```bash
# 實驗 1: 激進策略
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling segment \
    --segments 20 \
    --output results/exp1.json

# 實驗 2: 保守策略
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_2 \
    --sampling segment \
    --segments 20 \
    --output results/exp2.json
```

**對比分析**:
```python
import json

# 載入兩個實驗
with open('results/exp1.json') as f:
    exp1 = json.load(f)
with open('results/exp2.json') as f:
    exp2 = json.load(f)

# 對比最後 5 段的平均獎勵
exp1_final = exp1['training_progress']['episode_reward']['timeseries']['segment_samples'][-5:]
exp2_final = exp2['training_progress']['episode_reward']['timeseries']['segment_samples'][-5:]

exp1_avg = sum(s['value'] for s in exp1_final) / 5
exp2_avg = sum(s['value'] for s in exp2_final) / 5

print(f"實驗1 最終平均獎勵: {exp1_avg:.2f}")
print(f"實驗2 最終平均獎勵: {exp2_avg:.2f}")
```

---

### 示例 3: AI 深度分析（推薦）

```bash
# 使用兩種採樣，獲得完整信息
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling both \
    --segments 20 \
    --inflection-sensitivity 0.1 \
    --output results/for_ai.json
```

**提示詞**:
```
我的強化學習訓練結果如下，請分析訓練曲線的變化趨勢：

[貼上 JSON 內容]

重點關注：
1. 分段採樣顯示的整體趨勢如何？
2. 轉折點在哪些位置？為什麼會出現這些轉折？
3. 訓練是否穩定？（看每段的 std）
4. 最後階段是否收斂？
5. 有哪些異常波動？

請給出詳細分析和優化建議。
```

---

## 輸出數據結構

### 完整結構

```json
{
  "training_progress": {
    "episode_reward": {
      "initial": -120.5,
      "final": 456.8,
      "improvement": 577.3,
      "trend": {
        "slope": 0.058,
        "direction": "increasing"
      },
      "timeseries": {
        "segment_samples": [
          // 20 個分段採樣點
        ],
        "inflection_samples": [
          // 關鍵轉折點
        ]
      }
    }
  },
  "performance_metrics": {
    "total_loss": {
      "initial": 45.6,
      "final": 12.3,
      "timeseries": {
        "segment_samples": [...],
        "inflection_samples": [...]
      }
    }
  }
}
```

---

## 數據量對比

### 原始數據 vs 採樣數據

假設訓練 100 萬步，TensorBoard 記錄 5000 個數據點：

| 方法 | 數據點數 | 壓縮比 | 文件大小（估計）|
|------|---------|--------|----------------|
| 原始完整數據 | 5000 | - | ~500 KB |
| 分段採樣（20 段）| 20 | 99.6% | ~2 KB |
| 轉折點採樣 | 5-15 | 99.7%+ | ~1 KB |
| 兩種都用 | 25-35 | 99.3%+ | ~3 KB |

**結論**: 採樣後數據量減少 **99%+**，但保留了所有關鍵信息！

---

## 最佳實踐

### 推薦配置

**標準分析**（平衡）:
```bash
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling both \
    --segments 20 \
    --inflection-sensitivity 0.1
```

**快速查看**（極簡）:
```bash
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling segment \
    --segments 10
```

**深度診斷**（詳細）:
```bash
python scripts/analyze_tensorboard.py \
    --logdir logs/sb3_deeplob/PPO_1 \
    --sampling both \
    --segments 50 \
    --inflection-sensitivity 0.05
```

---

## 總結

### 何時使用分段採樣？

✅ 需要了解整體趨勢
✅ 對比不同階段的穩定性
✅ 給 AI 分析訓練進展

### 何時使用轉折點採樣？

✅ 診斷訓練問題
✅ 找出關鍵變化時刻
✅ 理解訓練階段

### 何時使用兩者？

✅ 給 AI 進行深度分析（推薦）
✅ 撰寫詳細的訓練報告
✅ 需要完整的訓練洞察

---

**版本**: v1.0
**最後更新**: 2025-10-26
**相關文檔**:
- [TENSORBOARD_ANALYSIS_GUIDE.md](TENSORBOARD_ANALYSIS_GUIDE.md) - 完整使用指南
- [AI_ANALYSIS_PROMPT_TEMPLATE.md](AI_ANALYSIS_PROMPT_TEMPLATE.md) - AI 分析模板
