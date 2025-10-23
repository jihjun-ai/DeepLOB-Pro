# 權重計算策略分析

**問題**: 應該在預處理階段保存權重，還是在訓練時動態計算？

**日期**: 2025-10-23

---

## 方案對比

### 方案 A：預處理時計算並保存權重 ⚖️

#### 實作方式

```python
# 在 preprocess_single_day.py 的 compute_label_preview() 中

def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    """
    計算樣本權重（基於標籤分布）

    Args:
        labels: 標籤陣列 (-1, 0, 1, NaN)

    Returns:
        sample_weights: 樣本權重陣列（與 labels 等長）
    """
    # 過濾 NaN
    valid_labels = labels[~np.isnan(labels)]

    # 計算類別計數
    unique, counts = np.unique(valid_labels, return_counts=True)
    total = len(valid_labels)
    n_classes = len(unique)

    # 計算 class weights (balanced)
    class_weights = {}
    for label, count in zip(unique, counts):
        class_weights[int(label)] = total / (n_classes * count)

    # 分配給每個樣本
    sample_weights = np.full_like(labels, np.nan, dtype=np.float32)
    for label, weight in class_weights.items():
        mask = (labels == label)
        sample_weights[mask] = weight

    return sample_weights


# 在 save_preprocessed_npz() 中保存
def save_preprocessed_npz(..., labels=None):
    # 如果有標籤，計算權重
    sample_weights = None
    if labels is not None:
        sample_weights = compute_sample_weights(labels)

    # 保存
    save_data = {
        'features': features,
        'mids': mids,
        'labels': labels,
        'sample_weights': sample_weights,  # 🆕 新增
        ...
    }
    np.savez_compressed(npz_path, **save_data)
```

#### NPZ 文件結構（新增後）

```python
data = np.load('2330.npz')
data.keys()
# ['features', 'mids', 'labels', 'sample_weights', 'metadata', ...]

# 使用範例
labels = data['labels']          # (15957,) [-1, 0, 1, NaN]
sample_weights = data['sample_weights']  # (15957,) [1.11, 0.83, 1.11, NaN]
```

---

### ✅ 方案 A 的優點

#### 1. 一次計算，多次使用
```python
# 預處理時計算一次
sample_weights = compute_sample_weights(labels)  # 只計算一次

# 訓練時直接使用（不同實驗都用同一份）
experiment_1: loss = criterion(pred, target, weight=sample_weights)
experiment_2: loss = criterion(pred, target, weight=sample_weights)
experiment_3: loss = criterion(pred, target, weight=sample_weights)
```
**優勢**: 避免重複計算，節省時間（雖然權重計算很快）

---

#### 2. 確保實驗一致性
```python
# 方案 A: 所有實驗使用相同權重（從 NPZ 讀取）
weights_exp1 = data['sample_weights']  # [1.11, 0.83, 1.11, ...]
weights_exp2 = data['sample_weights']  # [1.11, 0.83, 1.11, ...] ✅ 完全相同

# 方案 B: 每次實驗可能略有不同（如果訓練集有變化）
weights_exp1 = compute_class_weight(train_set_1)  # [1.10, 0.84, 1.12]
weights_exp2 = compute_class_weight(train_set_2)  # [1.12, 0.82, 1.11] ⚠️ 略有差異
```
**優勢**: 實驗可重現性更高

---

#### 3. 便於視覺化和檢查
```python
# 在 Label Viewer 中可以顯示權重分布
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=sample_weights[~np.isnan(sample_weights)],
    name='Sample Weights'
))
fig.update_layout(title='樣本權重分布')
```
**優勢**: 可以在訓練前檢查權重是否合理

---

#### 4. 支援複雜權重策略
```python
# 可以保存多種權重策略
def compute_multiple_weight_strategies(labels):
    return {
        'balanced': compute_balanced_weights(labels),
        'focal': compute_focal_weights(labels, gamma=2.0),
        'effective_num': compute_effective_num_weights(labels, beta=0.999),
        'custom': compute_custom_weights(labels)
    }

# 保存到 metadata
metadata['weight_strategies'] = {
    'balanced': {...},
    'focal': {...},
    'effective_num': {...}
}
```
**優勢**: 可以預先計算多種策略，訓練時選擇

---

#### 5. 減少訓練腳本複雜度
```python
# 方案 A: 訓練腳本簡單
weights = torch.FloatTensor(data['sample_weights'])
loss = criterion(pred, target, weight=weights)

# 方案 B: 訓練腳本需要計算邏輯
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[-1,0,1], y=train_labels)
sample_weights = np.array([class_weights[label+1] for label in train_labels])
weights = torch.FloatTensor(sample_weights)
```
**優勢**: 訓練代碼更簡潔

---

### ❌ 方案 A 的缺點

#### 1. 增加文件大小
```python
# 檔案大小比較（單一股票單日）
無權重 NPZ:  250 KB
有權重 NPZ:  300 KB  (+20%)

# 全部 195 檔股票單日
無權重:  50 MB
有權重:  60 MB  (+10 MB)

# 全年數據（250 交易日）
無權重:  12.5 GB
有權重:  15.0 GB  (+2.5 GB)
```
**影響**: 磁碟空間增加約 20%

---

#### 2. 靈活性降低
```python
# 如果想嘗試不同權重策略，需要：

# 方案 A: 重新預處理（耗時）
python preprocess_single_day.py --weight-strategy focal  # 30 分鐘
python preprocess_single_day.py --weight-strategy effective_num  # 30 分鐘

# 方案 B: 修改訓練腳本（即時）
train.py --weight-strategy focal  # 0 秒
train.py --weight-strategy effective_num  # 0 秒
```
**影響**: 調參靈活性降低

---

#### 3. 可能過時或不一致
```python
# 情境：後續調整了標籤生成邏輯

# 方案 A: 權重可能與標籤不匹配
labels = regenerate_labels(mids, new_config)  # 標籤改變
weights = data['sample_weights']  # ⚠️ 還是舊的權重！

# 方案 B: 動態計算，自動適應
labels = regenerate_labels(mids, new_config)
weights = compute_weights(labels)  # ✅ 自動更新
```
**影響**: 維護成本增加

---

#### 4. 權重計算本身很快
```python
import time
import numpy as np

labels = np.random.choice([-1, 0, 1], size=1_000_000)

# 計算權重
start = time.time()
weights = compute_sample_weights(labels)
elapsed = time.time() - start

print(f"計算 100 萬樣本權重: {elapsed:.3f} 秒")
# 輸出: 計算 100 萬樣本權重: 0.002 秒
```
**影響**: 預先計算的時間優勢微乎其微

---

#### 5. 不符合主流做法
```python
# PyTorch 標準做法（訓練時計算）
from torch.utils.data import WeightedRandomSampler

class_weights = compute_class_weight('balanced', ...)
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# TensorFlow 標準做法（訓練時計算）
class_weight = {0: 1.5, 1: 0.5, 2: 3.0}
model.fit(X, y, class_weight=class_weight)
```
**影響**: 與主流框架慣例不一致

---

## 方案 B：訓練時動態計算權重（當前做法）

### 實作方式

```python
# 在 train_deeplob_generic.py 中

from sklearn.utils.class_weight import compute_class_weight
import torch

# 訓練前一次性計算
train_labels = train_dataset.labels  # 從訓練集獲取標籤

# 計算 class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.array([-1, 0, 1]),
    y=train_labels
)

# 轉為 sample weights
sample_weights = np.array([class_weights[label + 1] for label in train_labels])

# 用於 Loss Function
criterion = nn.CrossEntropyLoss(
    weight=torch.FloatTensor(class_weights)
)

# 或用於 Sampler
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, sampler=sampler)
```

---

### ✅ 方案 B 的優點

#### 1. 靈活性極高
```python
# 可以輕鬆切換不同權重策略

# Balanced weights
weights = compute_class_weight('balanced', classes=[-1,0,1], y=labels)

# Focal loss（動態調整 gamma）
focal_loss = FocalLoss(gamma=2.0)

# Effective number of samples
weights = compute_effective_num_weights(labels, beta=0.999)

# 自定義權重
weights = {-1: 2.0, 0: 1.0, 1: 1.5}
```

---

#### 2. 文件大小不增加
```
NPZ 大小: 250 KB (不變)
磁碟空間節省: 2.5 GB (全年數據)
```

---

#### 3. 計算速度極快
```python
# 即使是 100 萬樣本
compute_class_weight(...)  # < 0.01 秒
```

---

#### 4. 自動適應數據變化
```python
# 如果訓練集改變，權重自動更新
train_set_v1 = load_data(config_v1)
weights_v1 = compute_weights(train_set_v1)  # 自動適應 v1

train_set_v2 = load_data(config_v2)
weights_v2 = compute_weights(train_set_v2)  # 自動適應 v2
```

---

#### 5. 符合主流框架慣例
```python
# PyTorch 官方範例
class_weights = compute_class_weight(...)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))

# TensorFlow 官方範例
class_weight = {0: 1.0, 1: 2.0}
model.fit(X, y, class_weight=class_weight)
```

---

### ❌ 方案 B 的缺點

#### 1. 每次訓練都要計算
```python
# 每次運行訓練都需要這段代碼
weights = compute_class_weight(...)  # 雖然只需 0.01 秒
```
**影響**: 微乎其微（< 訓練總時間的 0.001%）

---

#### 2. 可能不同實驗不一致
```python
# 如果訓練集略有變化
exp1_weights = compute_weights(train_set_1)  # [1.10, 0.84, 1.12]
exp2_weights = compute_weights(train_set_2)  # [1.12, 0.82, 1.11]
```
**影響**: 但這通常是**正確行為**（權重應該反映實際訓練集）

---

## 決策樹 🌳

```
是否在預處理時保存權重？
│
├─ 是，如果：
│   ├─ 權重計算非常複雜（> 1 分鐘）
│   ├─ 需要多種預計算策略供選擇
│   ├─ 實驗一致性極度重要
│   └─ 磁碟空間充足（+20%）
│
└─ 否，如果：（推薦 ✅）
    ├─ 權重計算快速（< 1 秒）
    ├─ 需要靈活調整權重策略
    ├─ 磁碟空間有限
    └─ 遵循主流框架慣例
```

---

## 推薦方案 🎯

### 建議：方案 B（訓練時動態計算）✅

**理由**：

1. ⚡ **權重計算極快**（< 0.01 秒），預先計算優勢極小
2. 🔧 **靈活性關鍵**：調參時經常需要嘗試不同權重策略
3. 💾 **節省空間**：全年數據可節省 2.5 GB
4. 🎓 **標準做法**：PyTorch/TensorFlow 都是訓練時計算
5. 🔄 **自動適應**：訓練集變化時權重自動更新

---

## 特殊情況：何時考慮方案 A？

### 情境 1：超大規模數據集
```python
# 如果數據量極大，權重計算可能較慢
n_samples = 100_000_000  # 1 億樣本
compute_weights(...)  # 可能需要幾分鐘
```
**此時**: 預先計算可節省時間

---

### 情境 2：複雜自定義權重
```python
# 如果權重基於外部複雜計算
def compute_complex_weights(labels, market_data, sentiment_scores):
    # 複雜計算邏輯（可能需要幾分鐘）
    weights = []
    for i, label in enumerate(labels):
        market_factor = analyze_market(market_data[i])
        sentiment_factor = analyze_sentiment(sentiment_scores[i])
        weight = custom_formula(label, market_factor, sentiment_factor)
        weights.append(weight)
    return np.array(weights)
```
**此時**: 預先計算避免重複耗時

---

### 情境 3：需要多種策略對比
```python
# 如果想同時嘗試多種權重策略
metadata['weight_strategies'] = {
    'balanced': {...},
    'focal_gamma_1': {...},
    'focal_gamma_2': {...},
    'effective_num_beta_0.9': {...},
    'effective_num_beta_0.99': {...},
    'custom_v1': {...},
    'custom_v2': {...}
}
```
**此時**: 預先計算可在訓練時快速切換

---

## 實作建議 💡

### 如果選擇方案 A（預先保存）

**最佳實踐**：
1. **保存多種策略**，而非單一策略
2. **在 metadata 中記錄**權重計算方法
3. **提供降級方案**（如果 NPZ 中無權重，自動計算）
4. **定期驗證**權重與標籤的一致性

**範例實作**：
```python
# preprocess_single_day.py
weight_strategies = {
    'balanced': compute_balanced_weights(labels),
    'focal': compute_focal_weights(labels, gamma=2.0),
    'none': np.ones_like(labels)  # 不使用權重
}

metadata['weight_strategies'] = {
    name: {
        'class_weights': {...},
        'method': '...',
        'params': {...}
    }
    for name, weights in weight_strategies.items()
}

# 只保存一種到 NPZ（節省空間），其他存 metadata
np.savez_compressed(
    npz_path,
    sample_weights=weight_strategies['balanced'],  # 預設使用 balanced
    ...
)
```

---

### 如果選擇方案 B（訓練時計算）

**最佳實踐**：
1. **封裝權重計算函數**
2. **記錄使用的權重策略**
3. **提供多種策略選項**

**範例實作**：
```python
# utils/weight_utils.py
def get_sample_weights(labels, strategy='balanced', **kwargs):
    """
    計算樣本權重

    Args:
        labels: 標籤陣列
        strategy: 'balanced', 'focal', 'effective_num', 'custom'
        **kwargs: 策略參數
    """
    if strategy == 'balanced':
        return compute_balanced_weights(labels)
    elif strategy == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return compute_focal_weights(labels, gamma)
    elif strategy == 'effective_num':
        beta = kwargs.get('beta', 0.999)
        return compute_effective_num_weights(labels, beta)
    elif strategy == 'custom':
        return kwargs.get('weights')
    else:
        return np.ones_like(labels)

# train_deeplob_generic.py
weights = get_sample_weights(
    train_labels,
    strategy=config.weight_strategy,  # 從配置讀取
    gamma=config.focal_gamma
)
```

---

## 結論表格

| 評估維度 | 方案 A（預先保存） | 方案 B（訓練時計算） | 推薦 |
|---------|-------------------|---------------------|------|
| **計算開銷** | 預處理時一次 | 訓練時 < 0.01s | - |
| **靈活性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ B |
| **文件大小** | +20% | 無增加 | ✅ B |
| **實驗一致性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | - |
| **維護成本** | 較高 | 較低 | ✅ B |
| **標準做法** | ❌ | ✅ | ✅ B |
| **適應性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ B |
| **複雜度** | 較高 | 較低 | ✅ B |

---

## 最終推薦 🏆

**推薦方案 B：訓練時動態計算權重**

**核心理由**：
1. 權重計算速度極快（< 0.01 秒），預先計算無明顯優勢
2. 調參靈活性至關重要，預先保存會限制實驗
3. 符合 PyTorch/TensorFlow 主流做法
4. 節省磁碟空間（全年數據可節省 2.5 GB）

**適用場景**：✅ 99% 的情況

---

**例外情況**：只有在以下**同時滿足**時才考慮方案 A：
1. 權重計算非常複雜（> 1 分鐘）
2. 需要多次重複訓練（> 100 次）
3. 磁碟空間充足

---

**日期**: 2025-10-23
**版本**: v1.0
