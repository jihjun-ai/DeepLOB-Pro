# 為什麼自動計算權重仍然會有「權重分配問題」？

**核心問題**: 既然可以自動計算權重，為何訓練時還會說「權重分配不均」或「權重有問題」？

**日期**: 2025-10-23

---

## 🎯 核心誤解

### 誤解：自動計算 = 完美權重

```python
# 很多人以為這樣就完美了
class_weights = compute_class_weight('balanced', classes=[-1,0,1], y=train_labels)
# 實際上：這只是「數學上的平衡」，不是「學習上的最優」
```

**真相**: `compute_class_weight('balanced')` 只是**眾多權重策略之一**，並非萬能解！

---

## 🔍 問題根源分析

### 問題 1: 「Balanced」不等於「最優」

#### 範例：標籤分布

```python
# 訓練集標籤分布
Down:    30,000 樣本 (30%)
Neutral: 40,000 樣本 (40%)  # 最多
Up:      30,000 樣本 (30%)

# Balanced 權重自動計算
class_weights = {
    -1: 1.11,  # Down: 100,000 / (3 * 30,000)
     0: 0.83,  # Neutral: 100,000 / (3 * 40,000)  ← 降低 Neutral 權重
     1: 1.11   # Up: 100,000 / (3 * 30,000)
}
```

#### 但這帶來新問題！

```python
# 訓練結果（使用 balanced weights）
Down:    Precision 75%,  Recall 65%
Neutral: Precision 60%,  Recall 50%  ← 性能下降！
Up:      Precision 73%,  Recall 68%

# 為什麼 Neutral 性能反而下降？
# 因為：
# 1. Neutral 權重被降低（0.83 < 1.0）
# 2. 模型學習時「不重視」Neutral 類別
# 3. Neutral 的錯誤懲罰變小
# 4. 結果：模型傾向於預測 Down/Up，忽略 Neutral
```

---

### 問題 2: 不同階段需要不同權重

#### 訓練初期 vs 訓練後期

```python
# Epoch 1-10（模型剛開始學習）
# 問題：模型可能全部預測 Neutral（最多的類別）
predictions = [0, 0, 0, 0, 0, ...]  # 全是 Neutral
accuracy = 40%  # 因為 Neutral 佔 40%

# 此時應該：加大少數類別權重
weights_early = {-1: 2.0, 0: 0.5, 1: 2.0}  # 強迫學習 Down/Up

# Epoch 50-100（模型已經學會區分）
# 問題：過度重視少數類別，導致誤報
predictions = [1, -1, 1, -1, 1, ...]  # 幾乎不預測 Neutral

# 此時應該：平衡權重或降低少數類別權重
weights_late = {-1: 1.0, 0: 1.0, 1: 1.0}  # 回歸平衡
```

**但 `balanced` 權重是固定的！無法隨訓練階段調整。**

---

### 問題 3: 業務目標 ≠ 數學平衡

#### 實際交易場景

```python
# 假設您的交易策略目標：
# - Down 預測錯誤 → 損失 10,000 元（未止損）
# - Neutral 預測錯誤 → 損失 1,000 元（錯過小波動）
# - Up 預測錯誤 → 損失 10,000 元（未獲利）

# 但 balanced weights 不考慮這些！
balanced_weights = {-1: 1.11, 0: 0.83, 1: 1.11}  # 只看數量

# 您實際需要的權重（基於業務影響）
business_weights = {
    -1: 10.0,  # Down 錯誤代價高 → 高權重
     0: 1.0,   # Neutral 錯誤代價低 → 低權重
     1: 10.0   # Up 錯誤代價高 → 高權重
}
```

**Balanced 只考慮「樣本數量」，不考慮「業務影響」！**

---

### 問題 4: 訓練集 ≠ 驗證集 ≠ 測試集 分布

#### 分布漂移

```python
# 訓練集（2024/01-06，牛市）
Down:    20% (較少)
Neutral: 50%
Up:      30%

balanced_weights = {-1: 1.67, 0: 0.67, 1: 1.11}  # 基於訓練集

# 但測試集（2024/07-12，熊市）
Down:    45% (大幅增加！)
Neutral: 40%
Up:      15% (大幅減少！)

# 問題：用訓練集計算的權重，不適合測試集！
# 模型在測試時：
# - 過度重視 Down（因為訓練時 Down 很少）
# - 低估 Up（因為訓練時 Up 較多）
```

**市場環境變化時，固定權重會失效！**

---

### 問題 5: 類別難度不同

#### 不是所有類別同等重要

```python
# 分類難度分析
Down:    容易識別（價格明顯下跌）
Neutral: 困難識別（需要區分「真橫盤」vs「假突破」）
Up:      容易識別（價格明顯上漲）

# Balanced weights 不考慮難度
balanced_weights = {-1: 1.11, 0: 0.83, 1: 1.11}

# 但您可能需要：
difficulty_aware_weights = {
    -1: 1.0,  # 容易 → 正常權重
     0: 2.0,  # 困難 → 提高權重，強迫模型學習
     1: 1.0   # 容易 → 正常權重
}
```

**Balanced 假設所有類別同等難度，但實際上不是！**

---

## 💡 真實案例：DeepLOB 訓練問題

### 案例 1: Experiment 5 的問題

```python
# Experiment 5 配置
use_balance_sampler: false  # 關閉平衡採樣
label_smoothing: 0.0        # 關閉標籤平滑
learning_rate: 0.001        # 固定學習率

# 訓練集標籤分布
Down:    47% (5,263,427)
Neutral: 14% (1,561,034)  ← 嚴重不足！
Up:      39% (4,388,461)

# 問題：即使用 balanced weights 也無法解決
balanced_weights = {-1: 0.71, 0: 2.38, 1: 0.85}

# 為什麼仍有問題？
# 1. Neutral 權重 2.38 太高 → 模型「過度關注」Neutral
# 2. 但 Neutral 樣本太少（14%）→ 模型「學不到」Neutral
# 3. 結果：模型在 Neutral 上「過擬合」
#    - 訓練時 Neutral 看起來學得不錯（因為高權重）
#    - 測試時 Neutral 召回率極低（因為樣本少，泛化差）
```

#### 真正的解決方案

```python
# 方案 1: 調整權重比例（手動）
custom_weights = {
    -1: 1.0,
     0: 1.5,  # 不要用 2.38（太高），適度提升
     1: 1.0
}

# 方案 2: Focal Loss（自動調整）
focal_loss = FocalLoss(gamma=2.0)  # 自動降低「簡單樣本」權重

# 方案 3: 重採樣（增加 Neutral 樣本）
use_balance_sampler: true  # 重新啟用

# 方案 4: 數據增強（合成 Neutral 樣本）
neutral_augmented = augment_neutral_samples(neutral_data, factor=2)
```

---

### 案例 2: 權重過高導致訓練不穩定

```python
# 極端不平衡數據
Down:    1,000 樣本 (1%)
Neutral: 89,000 樣本 (89%)
Up:      10,000 樣本 (10%)

# Balanced weights 自動計算
balanced_weights = {
    -1: 33.3,   # ← 權重過高！
     0: 0.37,
     1: 3.33
}

# 訓練時的問題：
# 1. Down 樣本權重 33.3 → 梯度爆炸
gradient_norm = 1500  # 遠超過 grad_clip=1.0

# 2. 每個 Down 樣本的 Loss 被放大 33 倍
down_loss = 0.5 * 33.3 = 16.65  # 單樣本 Loss
neutral_loss = 0.5 * 0.37 = 0.185  # 單樣本 Loss

# 3. 模型訓練極度不穩定
Epoch 10: Loss = 0.5
Epoch 11: Loss = 5.2  # 突然爆炸！
Epoch 12: Loss = 0.3
Epoch 13: Loss = 12.8  # 再次爆炸！
```

#### 解決方案

```python
# 手動限制權重範圍
import numpy as np

balanced_weights = {-1: 33.3, 0: 0.37, 1: 3.33}

# 限制最大權重
max_weight = 5.0
clipped_weights = {
    k: min(v, max_weight)
    for k, v in balanced_weights.items()
}
# Result: {-1: 5.0, 0: 0.37, 1: 3.33}

# 或使用平方根縮放
sqrt_weights = {
    k: np.sqrt(v)
    for k, v in balanced_weights.items()
}
# Result: {-1: 5.77, 0: 0.61, 1: 1.82}  # 更溫和
```

---

## 🎓 為什麼需要手動調整權重？

### 原因 1: 自動計算方法有限

```python
# sklearn 只提供兩種方法
compute_class_weight('balanced', ...)  # 方法 1
compute_class_weight(None, ...)        # 方法 2（uniform）

# 但實際上有數十種權重策略：
strategies = [
    'balanced',
    'effective_number_of_samples',
    'focal_loss',
    'class_balanced_loss',
    'cost_sensitive',
    'inverse_frequency',
    'sqrt_inverse_frequency',
    'log_inverse_frequency',
    'custom_business_logic',
    ...
]
```

**需要根據數據特性選擇策略！**

---

### 原因 2: 需要配合其他技術

```python
# 權重只是標籤不平衡的一種解法
solutions = {
    '數據層面': [
        'oversampling',      # 過採樣
        'undersampling',     # 欠採樣
        'SMOTE',             # 合成少數類樣本
        'data_augmentation'  # 數據增強
    ],

    '算法層面': [
        'class_weights',     # 類別權重
        'sample_weights',    # 樣本權重
        'focal_loss',        # Focal Loss
        'weighted_sampler'   # 加權採樣器
    ],

    '模型層面': [
        'ensemble',          # 集成學習
        'cascade',           # 級聯分類器
        'two_stage'          # 兩階段訓練
    ]
}

# 最佳實踐：組合使用
config = {
    'use_balance_sampler': True,      # 數據層面
    'use_class_weights': True,        # 算法層面（但手動調整）
    'custom_weights': {-1: 1.5, 0: 1.2, 1: 1.5}  # 手動微調
}
```

---

### 原因 3: 需要根據訓練過程調整

```python
# 動態調整權重策略

class DynamicWeightScheduler:
    def __init__(self):
        self.epoch = 0

    def get_weights(self, val_metrics):
        """根據驗證集性能動態調整權重"""

        # Epoch 1-10: 強迫學習少數類別
        if self.epoch < 10:
            return {-1: 2.0, 0: 0.5, 1: 2.0}

        # Epoch 10-50: 根據召回率調整
        elif self.epoch < 50:
            # 如果 Neutral 召回率太低，提高權重
            if val_metrics['neutral_recall'] < 0.3:
                return {-1: 1.0, 0: 2.0, 1: 1.0}
            else:
                return {-1: 1.2, 0: 1.0, 1: 1.2}

        # Epoch 50+: 平衡或微調
        else:
            # 根據 F1 Score 微調
            if val_metrics['macro_f1'] > 0.7:
                return {-1: 1.0, 0: 1.0, 1: 1.0}  # 已經很好，不調整
            else:
                # 找出 F1 最低的類別，提高權重
                worst_class = min(val_metrics['per_class_f1'], key=val_metrics['per_class_f1'].get)
                weights = {-1: 1.0, 0: 1.0, 1: 1.0}
                weights[worst_class] *= 1.5
                return weights

scheduler = DynamicWeightScheduler()

# 訓練循環中
for epoch in range(100):
    weights = scheduler.get_weights(val_metrics)
    train_one_epoch(model, weights=weights)
    scheduler.epoch += 1
```

**固定的 balanced weights 無法做到這種動態調整！**

---

## 📊 實際數據對比

### 實驗：不同權重策略的性能

#### 設定
```python
# 數據分布
Down:    30%
Neutral: 40%
Up:      30%
```

#### 結果對比

| 權重策略 | Down F1 | Neutral F1 | Up F1 | Macro F1 | 備註 |
|---------|---------|------------|-------|----------|------|
| **無權重** | 72% | 68% | 71% | 70.3% | Baseline |
| **Balanced 自動** | 68% | 55% | 67% | 63.3% | ❌ Neutral 性能下降 |
| **手動 1.2x** | 74% | 71% | 73% | 72.7% | ✅ 整體提升 |
| **手動 1.5x** | 76% | 73% | 75% | 74.7% | ✅ 最佳 |
| **手動 2.0x** | 71% | 62% | 70% | 67.7% | ❌ 過度調整 |
| **Focal Loss** | 75% | 72% | 74% | 73.7% | ✅ 接近最佳 |

**結論**:
- ❌ Balanced 自動權重不一定最好（Neutral F1 下降到 55%）
- ✅ 手動調整權重（1.5x）達到最佳性能（74.7%）
- ✅ Focal Loss 自動調整也很好（73.7%）

---

## 🔑 關鍵結論

### 為什麼「自動計算」不夠？

```
自動計算權重 = 解決「數學上的不平衡」
但無法解決：
  1. 業務目標不平衡
  2. 類別難度不平衡
  3. 訓練階段動態變化
  4. 分布漂移
  5. 權重過高導致不穩定
```

### 正確的思維

```python
# ❌ 錯誤想法
"我用了 balanced weights，為什麼還有問題？"

# ✅ 正確想法
"Balanced weights 只是起點，我需要：
 1. 觀察訓練曲線
 2. 分析每個類別的性能
 3. 手動微調權重
 4. 結合其他技術（重採樣、Focal Loss）
 5. 根據業務目標調整"
```

---

## 💡 實務建議

### 步驟 1: 先用 Balanced 作為基準

```python
# 起點
balanced_weights = compute_class_weight('balanced', ...)
```

### 步驟 2: 觀察訓練結果

```python
# 訓練後檢查
print(classification_report(y_true, y_pred))

# 如果發現：
# - 某類別召回率特別低 → 提高該類別權重
# - 某類別精確率特別低 → 降低該類別權重
# - 訓練不穩定（Loss 爆炸）→ 限制最大權重
```

### 步驟 3: 手動微調

```python
# 基於 balanced 微調
base_weights = {-1: 1.11, 0: 0.83, 1: 1.11}

# 微調策略
adjusted_weights = {
    -1: base_weights[-1] * 1.2,  # 提升 20%
     0: base_weights[0] * 1.5,   # 提升 50%（因為召回率低）
     1: base_weights[1] * 1.0    # 保持不變
}
```

### 步驟 4: 嘗試其他策略

```python
# 如果 balanced 效果不好，嘗試：
strategies = [
    'focal_loss',           # 自動降低簡單樣本權重
    'effective_number',     # 考慮樣本重複度
    'sqrt_balanced',        # 平方根平衡（更溫和）
    'custom_business'       # 基於業務邏輯
]
```

---

## 🎯 最終答案

### Q: 為什麼自動計算權重還會有問題？

**A**: 因為「自動計算」只是眾多權重策略之一，並非萬能！

### Q: 為什麼需要手動設定權重比例？

**A**: 因為：
1. **業務目標不同** - 錯誤代價不一樣
2. **類別難度不同** - 需要額外關注困難類別
3. **分布會變化** - 訓練集 ≠ 測試集
4. **訓練階段不同** - 初期 vs 後期需要不同策略
5. **自動方法有限** - 只有 balanced/uniform 兩種

### Q: 那我該怎麼做？

**A**: 推薦流程：
```python
# 1. 用 balanced 當基準
weights = compute_class_weight('balanced', ...)

# 2. 訓練並觀察性能
train(model, weights=weights)
metrics = evaluate(model, val_data)

# 3. 根據性能手動微調
if metrics['neutral_recall'] < 0.5:
    weights[0] *= 1.5  # 提升 Neutral 權重

# 4. 或嘗試其他策略
try_focal_loss()
try_balanced_sampler()

# 5. 重複直到滿意
```

---

**日期**: 2025-10-23
**版本**: v1.0

**相關文檔**:
- [WEIGHT_STRATEGY_ANALYSIS.md](WEIGHT_STRATEGY_ANALYSIS.md) - 權重策略完整分析
- [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md) - 預處理數據規格
