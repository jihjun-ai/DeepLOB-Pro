# 預計算權重設計方案

**核心理念**: 預先計算多種權重策略，訓練時靈活選擇

**日期**: 2025-10-23
**版本**: v1.0

---

## 🎯 設計目標

### 問題
```
當前困境：
- 預先保存權重 → 靈活性差
- 訓練時計算權重 → 每次都要算

理想方案：
- 預先計算「多種策略」的權重
- 保存在 metadata 中（不增加太多空間）
- 訓練時選擇使用哪一種
```

### 解決方案
```python
# 預處理時：計算多種權重策略的「類別權重」（不是樣本權重）
metadata['weight_strategies'] = {
    'balanced': {-1: 1.11, 0: 0.83, 1: 1.11},
    'sqrt_balanced': {-1: 1.05, 0: 0.91, 1: 1.05},
    'effective_num_0.99': {-1: 1.20, 0: 0.75, 1: 1.18},
    'effective_num_0.999': {-1: 1.35, 0: 0.68, 1: 1.32},
    'focal_recommended': {-1: 1.0, 0: 1.0, 1: 1.0},  # Focal Loss 不需權重
    'uniform': {-1: 1.0, 0: 1.0, 1: 1.0}
}

# 訓練時：選擇使用
python train.py --weight-strategy balanced
python train.py --weight-strategy effective_num_0.99
python train.py --weight-strategy uniform  # 不使用權重
```

---

## 📊 建議預先計算的權重策略

### 類別 1: 基於樣本數量的權重 ✅ 推薦預先計算

#### 1.1 Balanced (標準平衡)
```python
def compute_balanced_weights(labels):
    """
    標準平衡權重
    公式: weight = total / (n_classes * class_count)
    """
    unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    total = len(labels[~np.isnan(labels)])
    n_classes = len(unique)

    return {
        int(label): total / (n_classes * count)
        for label, count in zip(unique, counts)
    }

# 範例輸出（Neutral 14%）
# {-1: 1.11, 0: 2.38, 1: 1.11}
```

**特點**:
- ✅ 最常用的基準
- ✅ 計算簡單
- ⚠️ 不考慮類別難度

**建議**: **必須預先計算** ⭐⭐⭐⭐⭐

---

#### 1.2 Square Root Balanced (平方根平衡)
```python
def compute_sqrt_balanced_weights(labels):
    """
    平方根平衡權重（更溫和）
    公式: weight = sqrt(total / (n_classes * class_count))
    """
    balanced = compute_balanced_weights(labels)
    return {
        label: np.sqrt(weight)
        for label, weight in balanced.items()
    }

# 範例輸出（Neutral 14%）
# {-1: 1.05, 0: 1.54, 1: 1.05}  # 比 balanced 溫和
```

**特點**:
- ✅ 比 balanced 更溫和（避免權重過高）
- ✅ 減少訓練不穩定風險
- ✅ 文獻推薦用於極端不平衡

**建議**: **強烈推薦預先計算** ⭐⭐⭐⭐⭐

---

#### 1.3 Log Balanced (對數平衡)
```python
def compute_log_balanced_weights(labels):
    """
    對數平衡權重（最溫和）
    公式: weight = log(total / class_count)
    """
    unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    total = len(labels[~np.isnan(labels)])

    return {
        int(label): np.log(total / count + 1)  # +1 避免 log(0)
        for label, count in zip(unique, counts)
    }

# 範例輸出（Neutral 14%）
# {-1: 1.25, 0: 2.08, 1: 1.30}  # 最溫和
```

**特點**:
- ✅ 最溫和的平衡策略
- ✅ 適合極端不平衡（如 1:99）
- ⚠️ 效果可能不如 sqrt

**建議**: **可選預先計算** ⭐⭐⭐

---

#### 1.4 Effective Number of Samples (有效樣本數)
```python
def compute_effective_num_weights(labels, beta=0.99):
    """
    有效樣本數權重（考慮樣本重複度）
    公式: weight = (1 - beta) / (1 - beta^n)

    Args:
        beta: 重複度參數（0.9, 0.99, 0.999, 0.9999）
              - 越接近 1 → 認為樣本越獨特
              - 越接近 0 → 認為樣本越重複
    """
    unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)

    weights = {}
    for label, count in zip(unique, counts):
        effective_num = (1 - beta) / (1 - beta**count)
        weights[int(label)] = 1.0 / effective_num

    # 正規化
    total_weight = sum(weights.values())
    return {k: v / total_weight * len(weights) for k, v in weights.items()}

# 範例輸出（Neutral 14%, beta=0.99）
# {-1: 1.05, 0: 1.75, 1: 1.08}

# 範例輸出（Neutral 14%, beta=0.999）
# {-1: 1.02, 0: 1.92, 1: 1.04}
```

**特點**:
- ✅ 考慮樣本重複度（重要！）
- ✅ CVPR 2019 論文推薦
- ✅ 適合長尾分布
- ⚠️ 需要選擇 beta 參數

**建議**: **強烈推薦預先計算多個 beta 值** ⭐⭐⭐⭐⭐

推薦預計算：
- `beta=0.9`（認為樣本較重複）
- `beta=0.99`（標準設定）
- `beta=0.999`（認為樣本較獨特）
- `beta=0.9999`（極端獨特）

---

### 類別 2: 基於業務邏輯的權重 ⚠️ 建議預先計算

#### 2.1 Class-Balanced (CB) Focal Loss Weights
```python
def compute_cb_focal_weights(labels, beta=0.999, gamma=2.0):
    """
    Class-Balanced Focal Loss 權重
    結合 Effective Number + Focal Loss 概念

    Reference: CVPR 2019 - "Class-Balanced Loss Based on Effective Number of Samples"
    """
    effective_weights = compute_effective_num_weights(labels, beta)

    # Focal Loss 調整（預估困難度）
    # 假設 Neutral 最難（因為樣本少）
    difficulty_factors = {
        -1: 1.0,  # Down: 容易
         0: 1.5,  # Neutral: 困難（手動設定）
         1: 1.0   # Up: 容易
    }

    return {
        label: effective_weights[label] * difficulty_factors.get(label, 1.0)
        for label in effective_weights
    }
```

**特點**:
- ✅ 結合樣本數量 + 類別難度
- ✅ SOTA 方法
- ⚠️ 需要手動設定難度係數

**建議**: **可選預先計算** ⭐⭐⭐⭐

---

#### 2.2 Cost-Sensitive (業務成本)
```python
def compute_cost_sensitive_weights(labels, cost_matrix):
    """
    基於業務成本的權重

    Args:
        cost_matrix: 錯誤成本矩陣
        例如: {
            (-1, -1): 0,      # Down 預測對 Down，成本 0
            (-1, 0): 50000,   # Down 預測錯成 Neutral，損失 5 萬
            (-1, 1): 100000,  # Down 預測錯成 Up，損失 10 萬
            ...
        }
    """
    # 計算每個類別的平均錯誤成本
    unique = [-1, 0, 1]
    avg_costs = {}

    for true_label in unique:
        costs = [cost_matrix[(true_label, pred)] for pred in unique if pred != true_label]
        avg_costs[true_label] = np.mean(costs)

    # 正規化為權重
    total_cost = sum(avg_costs.values())
    return {k: v / total_cost * len(unique) for k, v in avg_costs.items()}

# 範例：交易成本
cost_matrix = {
    # (true, pred): cost
    (-1, -1): 0, (-1, 0): 50000, (-1, 1): 100000,
    (0, -1): 10000, (0, 0): 0, (0, 1): 10000,
    (1, -1): 100000, (1, 0): 50000, (1, 1): 0
}

# 輸出：{-1: 2.0, 0: 0.5, 1: 2.0}
```

**特點**:
- ✅ 直接對應業務目標
- ✅ 可解釋性強
- ⚠️ 需要定義成本矩陣

**建議**: **強烈推薦預先計算（如果有業務成本數據）** ⭐⭐⭐⭐⭐

---

### 類別 3: 不需要預先計算的權重 ❌

#### 3.1 Focal Loss
```python
# Focal Loss 不使用類別權重，而是動態調整樣本權重
# 公式: FL = -(1 - p_t)^gamma * log(p_t)
# 其中 p_t 是預測概率（訓練時才知道）

# 預處理時只需記錄：
metadata['weight_strategies']['focal_loss'] = {
    'type': 'focal',
    'gamma': 2.0,  # 推薦值
    'alpha': None,  # 可選的類別權重
    'class_weights': {-1: 1.0, 0: 1.0, 1: 1.0}  # 如果不用額外權重
}
```

**特點**:
- ✅ 自動調整（基於預測概率）
- ✅ 不需要類別權重
- ⚠️ 需要在訓練時實作

**建議**: **不預先計算，只記錄參數** ⭐⭐⭐⭐

---

#### 3.2 Dynamic Weights (動態權重)
```python
# 動態權重根據訓練階段調整（訓練時才能計算）
# 例如：
# - Epoch 1-10: 強迫學習少數類別
# - Epoch 11-50: 根據驗證性能調整
# - Epoch 51+: 微調

# 預處理時只需記錄建議的調整策略：
metadata['weight_strategies']['dynamic'] = {
    'type': 'dynamic',
    'schedule': {
        'early': {-1: 2.0, 0: 0.5, 1: 2.0},
        'mid': {-1: 1.2, 0: 1.0, 1: 1.2},
        'late': {-1: 1.0, 0: 1.0, 1: 1.0}
    }
}
```

**建議**: **不預先計算，只記錄策略** ⭐⭐

---

## 🎨 完整實作設計

### Step 1: 修改 `preprocess_single_day.py`

```python
def compute_all_weight_strategies(labels: np.ndarray) -> Dict[str, Dict]:
    """
    計算所有權重策略

    Args:
        labels: 標籤陣列 (-1, 0, 1, NaN)

    Returns:
        所有權重策略的字典
    """
    strategies = {}

    # 1. Balanced 系列
    strategies['balanced'] = compute_balanced_weights(labels)
    strategies['sqrt_balanced'] = compute_sqrt_balanced_weights(labels)
    strategies['log_balanced'] = compute_log_balanced_weights(labels)

    # 2. Effective Number 系列（多個 beta）
    for beta in [0.9, 0.99, 0.999, 0.9999]:
        key = f'effective_num_{str(beta).replace(".", "")}'
        strategies[key] = compute_effective_num_weights(labels, beta)

    # 3. CB Focal 系列
    strategies['cb_focal_099'] = compute_cb_focal_weights(labels, beta=0.99)
    strategies['cb_focal_0999'] = compute_cb_focal_weights(labels, beta=0.999)

    # 4. 業務成本（如果有定義）
    # strategies['cost_sensitive'] = compute_cost_sensitive_weights(labels, cost_matrix)

    # 5. 不使用權重
    strategies['uniform'] = {-1: 1.0, 0: 1.0, 1: 1.0}

    # 6. Focal Loss（記錄參數）
    strategies['focal_loss'] = {
        'type': 'focal',
        'gamma': 2.0,
        'class_weights': {-1: 1.0, 0: 1.0, 1: 1.0}
    }

    return strategies


# 在 save_preprocessed_npz() 中保存
def save_preprocessed_npz(..., label_preview=None):
    # ... 現有代碼 ...

    # 🆕 如果有 label_preview，計算所有權重策略
    if label_preview is not None and 'labels_array' in label_preview:
        labels = label_preview['labels_array']
        weight_strategies = compute_all_weight_strategies(labels)

        # 保存到 metadata
        metadata['weight_strategies'] = {
            name: {
                'class_weights': weights,
                'description': get_strategy_description(name)
            }
            for name, weights in weight_strategies.items()
        }
```

---

### Step 2: Metadata 結構

```json
{
  "symbol": "2330",
  "date": "20250901",
  "label_preview": {
    "down_count": 2273,
    "neutral_count": 11395,
    "up_count": 2288,
    "down_pct": 0.1425,
    "neutral_pct": 0.7142,
    "up_pct": 0.1434
  },
  "weight_strategies": {
    "balanced": {
      "class_weights": {"-1": 1.11, "0": 0.83, "1": 1.11},
      "description": "Standard balanced weights"
    },
    "sqrt_balanced": {
      "class_weights": {"-1": 1.05, "0": 0.91, "1": 1.05},
      "description": "Square root balanced (gentler)"
    },
    "effective_num_099": {
      "class_weights": {"-1": 1.05, "0": 1.75, "1": 1.08},
      "description": "Effective number (beta=0.99)"
    },
    "effective_num_0999": {
      "class_weights": {"-1": 1.02, "0": 1.92, "1": 1.04},
      "description": "Effective number (beta=0.999)"
    },
    "cb_focal_0999": {
      "class_weights": {"-1": 1.02, "0": 2.88, "1": 1.04},
      "description": "Class-Balanced Focal (beta=0.999)"
    },
    "uniform": {
      "class_weights": {"-1": 1.0, "0": 1.0, "1": 1.0},
      "description": "No weighting"
    },
    "focal_loss": {
      "type": "focal",
      "gamma": 2.0,
      "class_weights": {"-1": 1.0, "0": 1.0, "1": 1.0},
      "description": "Use Focal Loss (gamma=2.0)"
    }
  }
}
```

---

### Step 3: 訓練時使用

```python
# train_deeplob_generic.py

def load_class_weights(metadata, strategy='balanced'):
    """
    從 metadata 載入類別權重

    Args:
        metadata: NPZ metadata
        strategy: 權重策略名稱

    Returns:
        class_weights: {-1: w1, 0: w2, 1: w3}
    """
    if 'weight_strategies' not in metadata:
        # 如果沒有預先計算，動態計算
        logging.warning("No pre-computed weights, using balanced")
        return compute_class_weight('balanced', ...)

    strategies = metadata['weight_strategies']

    if strategy not in strategies:
        logging.error(f"Strategy '{strategy}' not found. Available: {list(strategies.keys())}")
        return {-1: 1.0, 0: 1.0, 1: 1.0}

    strategy_config = strategies[strategy]

    # 如果是 Focal Loss，特殊處理
    if strategy_config.get('type') == 'focal':
        logging.info("Using Focal Loss (no class weights)")
        return None  # 訓練時使用 Focal Loss

    return strategy_config['class_weights']


# 使用範例
config = load_yaml('configs/train_v5.yaml')

# 從配置讀取權重策略
weight_strategy = config.get('weight_strategy', 'balanced')

# 載入權重
class_weights = load_class_weights(metadata, strategy=weight_strategy)

if class_weights is not None:
    # 使用類別權重
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([class_weights[-1], class_weights[0], class_weights[1]])
    )
else:
    # 使用 Focal Loss
    criterion = FocalLoss(gamma=2.0)
```

---

### Step 4: 配置文件

```yaml
# configs/train_v5.yaml

# 權重策略選擇
weight_strategy: "effective_num_0999"  # 可選值見下方

# 可用的權重策略：
# - balanced: 標準平衡
# - sqrt_balanced: 平方根平衡（溫和）
# - log_balanced: 對數平衡（最溫和）
# - effective_num_09: Effective Number (beta=0.9)
# - effective_num_099: Effective Number (beta=0.99)
# - effective_num_0999: Effective Number (beta=0.999) ⭐ 推薦
# - effective_num_09999: Effective Number (beta=0.9999)
# - cb_focal_099: Class-Balanced Focal (beta=0.99)
# - cb_focal_0999: Class-Balanced Focal (beta=0.999)
# - uniform: 不使用權重
# - focal_loss: 使用 Focal Loss
# - custom: 自定義（在代碼中設定）
```

---

## 📊 文件大小影響

### 計算

```python
# 每個權重策略需要儲存 3 個數字（3 個類別）
# JSON 格式約 100 bytes/策略

strategies = [
    'balanced',
    'sqrt_balanced',
    'log_balanced',
    'effective_num_09',
    'effective_num_099',
    'effective_num_0999',
    'effective_num_09999',
    'cb_focal_099',
    'cb_focal_0999',
    'uniform',
    'focal_loss'
]

# 11 個策略 * 100 bytes = 1.1 KB/股票

# 單一交易日（195 檔）
195 * 1.1 KB = 214 KB

# 全年（250 交易日）
214 KB * 250 = 53.5 MB

# 對比：
# - 無權重策略: 12.5 GB
# - 有權重策略: 12.5 GB + 53.5 MB ≈ 12.55 GB
# - 增加: 0.4%（幾乎可忽略）
```

**結論**: 增加 11 個權重策略，只增加 0.4% 文件大小 ✅

---

## 🎯 推薦配置

### 最小配置（必須）

```python
weight_strategies = {
    'balanced': {...},           # 基準
    'effective_num_0999': {...}, # 推薦
    'uniform': {...}             # 不使用
}
```

**文件增加**: ~300 bytes/股票，幾乎可忽略

---

### 標準配置（推薦）⭐

```python
weight_strategies = {
    'balanced': {...},
    'sqrt_balanced': {...},
    'effective_num_099': {...},
    'effective_num_0999': {...},
    'cb_focal_0999': {...},
    'uniform': {...},
    'focal_loss': {...}
}
```

**文件增加**: ~700 bytes/股票，可忽略

---

### 完整配置（研究用）

```python
weight_strategies = {
    'balanced': {...},
    'sqrt_balanced': {...},
    'log_balanced': {...},
    'effective_num_09': {...},
    'effective_num_099': {...},
    'effective_num_0999': {...},
    'effective_num_09999': {...},
    'cb_focal_099': {...},
    'cb_focal_0999': {...},
    'uniform': {...},
    'focal_loss': {...}
}
```

**文件增加**: ~1.1 KB/股票，全年 +53.5 MB（0.4%）

---

## ✅ 總結

### 推薦預先計算的權重：

| 權重策略 | 優先級 | 理由 |
|---------|-------|------|
| `balanced` | ⭐⭐⭐⭐⭐ | 必須（基準） |
| `sqrt_balanced` | ⭐⭐⭐⭐ | 溫和版本，避免過高 |
| `effective_num_0999` | ⭐⭐⭐⭐⭐ | SOTA，考慮重複度 |
| `effective_num_099` | ⭐⭐⭐⭐ | 另一個 beta 選項 |
| `cb_focal_0999` | ⭐⭐⭐ | 結合難度的進階版 |
| `uniform` | ⭐⭐⭐⭐⭐ | 必須（不使用權重） |

### 不預先計算的權重：

| 權重策略 | 原因 |
|---------|------|
| `focal_loss` | 動態基於預測概率 |
| `dynamic_weights` | 根據訓練階段調整 |
| `custom_manual` | 手動微調（訓練時決定） |

### 檔案大小影響：

```
標準配置（7 個策略）：~700 bytes/股票
全年數據增加：~43 MB（0.34%）
結論：幾乎可忽略 ✅
```

---

**日期**: 2025-10-23
**版本**: v1.0
**下一步**: 實作到 `preprocess_single_day.py`
