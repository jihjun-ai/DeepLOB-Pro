# 獎懲計算方式說明 (Long Only 版本)

## 文件信息

- **建立日期**: 2025-10-26
- **版本**: v3.0 (Long Only + 持倉激勵)
- **更新日期**: 2025-10-27 (新增持倉激勵)
- **核心模組**: `src/envs/reward_shaper.py`
- **環境模組**: `src/envs/tw_lob_trading_env.py`

---

## 一、總體架構

強化學習訓練的核心是**獎勵函數設計**。我們的系統使用**多組件獎勵函數**，將獎勵分解為3個主要部分：

```
總獎勵 = PnL獎勵 - 交易成本懲罰 + 持倉激勵
```

**v3.0 更新**（2025-10-27）:
- ✅ 新增**持倉激勵**組件（鼓勵交易）
- ❌ 移除庫存懲罰（避免雙重懲罰）
- ❌ 移除風險懲罰（簡化獎勵函數）

### ⭐ Long Only 策略說明 (v2.0 更新)

**動作空間**:
- `action=0`: Hold/Sell（持有或平倉）
- `action=1`: Buy（買入）

**倉位範圍**: `[0, max_position]`（不允許負倉位，即不做空）

**優勢**:
- 符合台股現貨市場特性（一般散戶無法做空）
- 簡化策略空間，加速訓練
- 降低風險（避免做空風險無限）
- 更符合實際交易場景

### 設計原則

1. **平衡短期盈利與長期穩定性**
   - 不僅獎勵盈利，也懲罰風險
   - 避免過度激進的策略

2. **考慮實際交易成本**
   - 台股手續費：0.04275%（3折後）
   - 台股交易稅：0.15%（當沖）
   - 避免過度交易

3. **鼓勵風險管理**
   - 懲罰長時間持倉
   - 懲罰高波動時持倉

4. **可調整的權重系統**
   - 所有組件權重可配置
   - 支持不同風格策略（激進/保守/平衡）

---

## 二、獎勵組件詳解 (PPO_11 版本: 3組件)

### 組件1: PnL獎勵（盈虧）⭐⭐⭐⭐⭐

**目的**: 這是最核心的獎勵，直接衡量交易盈利能力

**計算公式**:
```
PnL獎勵 = (已實現盈虧 + 增量未實現盈虧) × pnl_scale
```

#### 1.1 已實現盈虧（Realized PnL）

**定義**: 當倉位從非零變為零時產生的真實盈虧

**計算方式**:
```python
if prev_position != 0 and new_position == 0:  # 平倉
    realized_pnl = prev_position × (平倉價 - 進場價)
```

**範例 (Long Only)**:
```
做多盈利:
  進場: position=+1, entry_price=100
  平倉: position=0, exit_price=103
  realized_pnl = +1 × (103 - 100) = +3.0 ✅ 盈利

做多虧損:
  進場: position=+1, entry_price=100
  平倉: position=0, exit_price=98
  realized_pnl = +1 × (98 - 100) = -2.0 ❌ 虧損

做多持平:
  進場: position=+1, entry_price=100
  平倉: position=0, exit_price=100
  realized_pnl = +1 × (100 - 100) = 0.0 ➖ 持平

注意: Long Only 策略中，position 永遠 >= 0（不做空）
```

#### 1.2 增量未實現盈虧（Incremental Unrealized PnL）

**定義**: 持倉期間，每一步的價格變化帶來的浮動盈虧增量

**重要修正（2025-10-13）**:
- ❌ **舊版錯誤**: 計算整個持倉的浮動盈虧 `position × (current_price - entry_price)`
- ✅ **新版正確**: 只計算當前步的價格變化 `position × (current_price - prev_price)`

**為什麼要修正？**
- 舊版會導致獎勵二次方累積（持倉越久獎勵越大）
- 導致訓練出異常高回報的策略
- 實際是錯誤的獎勵計算

**計算方式**:
```python
if prev_position != 0:  # 有持倉
    price_change = current_price - prev_price  # ⭐ 只計算當前步變化
    incremental_pnl = prev_position × price_change
```

**完整範例（做多持倉3步）**:
```
Step 1: 買入 @ 100
  prev: position=0, prev_price=100
  new:  position=1, current_price=100
  pnl = 0  # 剛買入，無盈虧

Step 2: 價格上漲到 101
  prev: position=1, prev_price=100
  new:  position=1, current_price=101
  incremental_pnl = 1 × (101 - 100) = +1.0 ✅

Step 3: 價格上漲到 102
  prev: position=1, prev_price=101  # ⭐ 使用前一步價格
  new:  position=1, current_price=102
  incremental_pnl = 1 × (102 - 101) = +1.0 ✅
  ❌ 舊版錯誤: 1 × (102 - 100) = +2.0

Step 4: 平倉 @ 103
  prev: position=1, entry_price=100, prev_price=102
  new:  position=0, current_price=103
  realized_pnl = 1 × (103 - 100) = +3.0

總累積獎勵 = 0 + 1.0 + 1.0 + 3.0 = 5.0 ✅
❌ 舊版錯誤: 0 + 1.0 + 2.0 + 3.0 = 6.0
```

#### 1.3 PnL權重配置

**參數**: `pnl_scale`（預設 1.0）

**作用**: 控制盈虧在總獎勵中的比重

**配置建議**:
```yaml
# 激進策略（追求高收益）
pnl_scale: 2.0

# 平衡策略（預設）
pnl_scale: 1.0

# 保守策略（注重風險）
pnl_scale: 0.5
```

---

### 組件2: 交易成本懲罰 ⭐⭐⭐⭐

**目的**: 懲罰交易成本，避免過度交易

**計算公式**:
```
交易成本懲罰 = -transaction_cost × cost_penalty_weight
```

**成本來源**:
1. **券商手續費**（買賣雙邊）
   - 法定上限: 0.1425%
   - 折扣後: 0.04275%（3折）
   - 最低費用: 20 TWD

2. **證券交易稅**（僅賣出）
   - 當沖: 0.15%
   - 一般: 0.3%

3. **滑點成本**（可選）
   - 預設: 0.01%

**台股實際計算（重要 ⭐⭐⭐）**:
```python
# 報價是每股，交易單位是張（1張=1000股）
trade_value = price_per_share × 1000 × lots

# 買入成本
commission = max(trade_value × 0.0004275, 20)  # 3折手續費
buy_cost = commission

# 賣出成本
commission = max(trade_value × 0.0004275, 20)
tax = trade_value × 0.0015  # 當沖稅率
sell_cost = commission + tax

# 往返總成本率
roundtrip_rate = (buy_cost + sell_cost) / trade_value
# ≈ 0.2355% (vs 舊版錯誤 0.1%)
```

**範例（1張@100元）**:
```
交易價值 = 100 TWD/股 × 1000 股/張 × 1 張 = 100,000 TWD

買入成本:
  手續費 = 100,000 × 0.04275% = 42.75 TWD

賣出成本:
  手續費 = 100,000 × 0.04275% = 42.75 TWD
  交易稅 = 100,000 × 0.15% = 150.00 TWD
  小計 = 192.75 TWD

往返總成本 = 42.75 + 192.75 = 235.50 TWD (0.2355%)
```

**權重配置**:
```yaml
# 減少交易頻率（高懲罰）
cost_penalty: 2.0

# 平衡（預設）
cost_penalty: 1.0

# 允許頻繁交易（低懲罰）
cost_penalty: 0.5
```

---

### 組件3: 持倉激勵 ⭐⭐⭐ (新增於 PPO_11)

**目的**: 鼓勵模型參與交易，避免學到「永遠不動」策略

**問題背景**:
- PPO_9 測試發現模型 **100% Hold**，從不交易
- Episode 獎勵 = 0.00 看似不錯，實際是「不動」策略
- 原因: `獎勵 = PnL - 成本`，不交易 → 無成本 → 獎勵 = 0，模型學到「最安全」

**解決方案**:
- 給予持倉小額獎勵，鼓勵參與市場
- 獎勵設計要**適中**：太高會過度交易，太低無效

**計算公式**:
```python
if new_position > 0:
    holding_bonus = 0.05  # 每步持倉獎勵 0.05
else:
    holding_bonus = 0.0   # 空倉無獎勵
```

**數值說明**:
- `0.05` 是經驗值，相當於持倉 500 步獲得 25 元獎勵
- 需要盈利 > 25 元才能抵消交易成本（約 0.24%）
- 防止模型只為獎勵而持倉，仍需盈利

**範例**:
```
情境1: 空倉（模型不交易）
  position = 0
  holding_bonus = 0.0
  total_reward = 0.0 - 0.0 + 0.0 = 0.0

情境2: 持倉但價格持平
  position = 1, PnL = 0.0, cost = 0.1
  holding_bonus = 0.05
  total_reward = 0.0 - 0.1 + 0.05 = -0.05 (輕微虧損，鼓勵尋找盈利機會)

情境3: 持倉且小幅盈利
  position = 1, PnL = 0.2, cost = 0.1
  holding_bonus = 0.05
  total_reward = 0.2 - 0.1 + 0.05 = 0.15 (盈利增強)

情境4: 持倉但虧損
  position = 1, PnL = -0.3, cost = 0.1
  holding_bonus = 0.05
  total_reward = -0.3 - 0.1 + 0.05 = -0.35 (虧損仍然虧損，不會因激勵而持續虧損倉位)
```

**預期效果**:
- ✅ 模型開始嘗試交易（Buy 動作比例 > 5%）
- ✅ 實際交易次數增加（> 5 次/Episode）
- ✅ 學習何時進場、何時離場
- ⚠️ 需監控是否過度交易（Buy 比例不應 > 40%）

---

### ~~組件4: 庫存懲罰~~ ❌ 已移除（PPO_6）

**移除原因**:

1. **避免重複懲罰**
   - 虧損時賣出，PnL 已經反映虧損（負獎勵）
   - 再加上庫存懲罰會導致雙重懲罰
   - 不符合實際交易邏輯

2. **讓策略自由決定持倉時間**
   - 移除人為設計的懲罰項
   - 讓模型從數據學習最優持倉時間
   - 有些機會需要長時間持倉才能獲利

3. **簡化獎勵函數**
   - 減少調參複雜度
   - 更直觀的獎勵信號

**舊版計算方式**（僅供參考）:
```python
# ❌ 已移除
# if position != 0:
#     inventory = position × (current_price - entry_price)
#     inventory_penalty = -|inventory| × inventory_penalty_weight
# else:
#     inventory_penalty = 0.0
```

**PPO_6 新配置**:
```yaml
inventory_penalty: 0.0  # 不再懲罰庫存
```

---

### ~~組件4: 風險懲罰~~ ❌ 已移除（PPO_6）

**移除原因**:

1. **簡化獎勵函數**
   - 移除人為設計的複雜懲罰項
   - 讓 PnL 主導學習方向

2. **PnL 已包含風險**
   - 高波動時持倉，價格下跌會導致虧損（負 PnL）
   - 不需要額外懲罰風險

3. **減少調參複雜度**
   - 少一個超參數
   - 更容易找到最優配置

**舊版計算方式**（僅供參考）:
```python
# ❌ 已移除
# risk_penalty = -|position| × volatility × risk_penalty_weight
```

**PPO_6 新配置**:
```yaml
risk_penalty: 0.0  # 不再懲罰風險
```

---

## 三、完整計算流程

### 3.1 step() 方法執行流程

```python
# 1. 獲取當前價格
current_price = self.prices[self.current_data_idx]

# 2. 保存前一狀態
prev_state = {
    'position': self.position,
    'entry_price': self.entry_price,
    'prev_action': self.prev_action,
    'prev_price': current_price  # ⭐ 用於增量PnL計算
}

# 3. 執行交易動作（更新倉位）
if action != prev_position + 1:
    # 計算交易成本
    position_change = abs(action - (prev_position + 1))
    transaction_cost = position_change × current_price × transaction_cost_rate

    # 更新倉位
    if action == 0:  # Sell
        self.position = max(-max_position, position - 1)
    elif action == 2:  # Buy
        self.position = min(max_position, position + 1)

# 4. 更新庫存
if self.position != 0:
    self.inventory = position × (current_price - entry_price)
else:
    self.inventory = 0.0

# 5. 推進到下一步
self.current_step += 1
next_price = self.prices[self.current_data_idx + 1]

# 6. 構建新狀態
new_state = {
    'position': self.position,
    'current_price': next_price,
    'inventory': self.inventory,
    'volatility': 0.01  # 當前使用固定值
}

# 7. 計算獎勵
reward, reward_info = self.reward_shaper.calculate_reward(
    prev_state=prev_state,
    action=action,
    new_state=new_state,
    transaction_cost=transaction_cost
)
```

### 3.2 獎勵計算流程

```python
def calculate_reward(prev_state, action, new_state, transaction_cost):
    """計算多組件獎勵"""

    # 組件1: PnL獎勵
    pnl = _calculate_pnl(prev_state, new_state)
    pnl_reward = pnl × pnl_scale

    # 組件2: 交易成本懲罰
    cost_penalty = -transaction_cost × cost_penalty_weight

    # 組件3: 庫存懲罰
    inventory = new_state['inventory']
    inventory_penalty = -abs(inventory) × inventory_penalty_weight

    # 組件4: 風險懲罰
    position = new_state['position']
    volatility = new_state['volatility']
    risk_penalty = -abs(position) × volatility × risk_penalty_weight

    # 總獎勵
    total_reward = pnl_reward + cost_penalty + inventory_penalty + risk_penalty

    return total_reward, {
        'pnl': pnl_reward,
        'transaction_cost': cost_penalty,
        'inventory_penalty': inventory_penalty,
        'risk_penalty': risk_penalty
    }
```

---

## 四、配置範例與策略類型

### 4.1 激進策略（追求高收益）

**特點**: 高PnL權重，低成本懲罰，允許頻繁交易和長時間持倉

```yaml
reward:
  pnl_scale: 2.0            # 強調盈利 ⬆️
  cost_penalty: 0.5         # 允許頻繁交易 ⬇️
  inventory_penalty: 0.001  # 允許長時間持倉 ⬇️
  risk_penalty: 0.001       # 忽略風險 ⬇️
```

**適用場景**:
- 趨勢明確的市場
- 高勝率策略
- 高波動市場

**風險**:
- 可能過度交易
- 風險管理不足
- 大幅回撤

---

### 4.2 保守策略（注重風險管理）⭐ 推薦

**特點**: 平衡PnL和風險，高成本懲罰，強調風險控制

```yaml
reward:
  pnl_scale: 1.0            # 正常盈利重視 ✅
  cost_penalty: 2.0         # 減少交易 ⬆️
  inventory_penalty: 0.05   # 快速平倉 ⬆️
  risk_penalty: 0.02        # 強調風險管理 ⬆️
```

**適用場景**:
- 震盪市場
- 實盤交易
- 長期穩定性優先

**優勢**:
- 低回撤
- 風險可控
- 適合實盤

---

### 4.3 平衡策略（預設配置）

**特點**: 均衡各個組件，適合初始訓練

```yaml
reward:
  pnl_scale: 1.0            # 預設 ✅
  cost_penalty: 1.0         # 預設 ✅
  inventory_penalty: 0.01   # 預設 ✅
  risk_penalty: 0.005       # 預設 ✅
```

**適用場景**:
- 初始訓練
- 策略探索階段
- 不確定市場風格

---

### 4.4 高頻交易策略

**特點**: 極快進出，低庫存，關注小幅價格變動

```yaml
reward:
  pnl_scale: 1.0            # 正常盈利 ✅
  cost_penalty: 0.3         # 極低成本懲罰（允許頻繁交易）⬇️⬇️
  inventory_penalty: 0.1    # 極高庫存懲罰（極快平倉）⬆️⬆️
  risk_penalty: 0.01        # 中等風險懲罰 ✅
```

**適用場景**:
- 高流動性市場
- 低延遲環境
- 小幅價格波動捕捉

**要求**:
- 低交易成本（折扣優惠）
- 高頻數據
- 快速執行

---

## 五、當前實際配置

### 5.1 sb3_deeplob_config.yaml 配置

```yaml
env_config:
  reward:
    pnl_scale: 1.0
    cost_penalty: 1.0
    inventory_penalty: 0.01
    risk_penalty: 0.005

  transaction_cost:
    shares_per_lot: 1000
    commission:
      base_rate: 0.001425
      discount: 0.3
      min_fee: 20.0
    securities_tax:
      rate: 0.0015
    slippage:
      enabled: false
      rate: 0.0001
```

### 5.2 實際獎勵數值範例

假設場景：買1張@100元，持倉1步後賣出@101元

```
Step 1: 買入
  action = 2 (Buy)
  transaction_cost = 42.75 TWD = 0.0004275 (相對於100元)

  獎勵計算:
    pnl_reward = 0 (剛買入)
    cost_penalty = -0.0004275 × 1.0 = -0.0004275
    inventory_penalty = 0 (剛買入)
    risk_penalty = -1 × 0.01 × 0.005 = -0.00005
    total_reward = -0.0004775 ❌ (買入有成本)

Step 2: 價格上漲到101，持有
  action = 1 (Hold)
  transaction_cost = 0

  獎勵計算:
    pnl_reward = 1 × (101 - 100) × 1.0 = 1.0 ✅
    cost_penalty = 0
    inventory_penalty = -|1.0| × 0.01 = -0.01
    risk_penalty = -1 × 0.01 × 0.005 = -0.00005
    total_reward = 0.98995 ✅ (盈利)

Step 3: 賣出平倉@101
  action = 0 (Sell)
  transaction_cost = 192.75 TWD = 0.0019275

  獎勵計算:
    pnl_reward = 1 × (101 - 100) × 1.0 = 1.0 ✅ (已實現)
    cost_penalty = -0.0019275 × 1.0 = -0.0019275
    inventory_penalty = 0 (已平倉)
    risk_penalty = 0 (已平倉)
    total_reward = 0.9980725 ✅ (平倉盈利)

總累積獎勵 = -0.0004775 + 0.98995 + 0.9980725 = 1.987545

實際盈利 = 100 (價格變動) - 235.5 (交易成本) = -135.5 TWD？
錯誤！應該是：1元/股 × 1000股 - 2.355元 = 997.645元盈利
```

**注意**: 上述獎勵是標準化數值，不等於實際TWD盈利

---

## 六、自適應獎勵（進階功能）

### 6.1 AdaptiveRewardShaper

**目的**: 訓練過程中動態調整獎勵權重，實現課程學習

**核心思想**:
- 訓練初期: 高PnL權重，鼓勵探索盈利機會
- 訓練後期: 降低PnL權重，強化風險管理

**衰減公式**:
```
pnl_scale(t) = max(min_scale, initial_scale × decay^t)
```

**配置範例**:
```yaml
reward:
  pnl_scale: 2.0          # 初始權重
  min_pnl_scale: 0.5      # 最小權重
  scale_decay: 0.9999     # 衰減率
```

**權重變化軌跡**:
```
Episode 0:    pnl_scale = 2.0   (初始)
Episode 100:  pnl_scale ≈ 1.98  (衰減 1%)
Episode 1000: pnl_scale ≈ 1.81  (衰減 10%)
Episode 5000: pnl_scale ≈ 1.21  (衰減 40%)
Episode 10000: pnl_scale ≈ 0.74 (衰減 63%)
穩定值:       pnl_scale = 0.5   (最小值)
```

**使用方式**:
```python
# 創建自適應獎勵塑形器
from src.envs.reward_shaper import AdaptiveRewardShaper

shaper = AdaptiveRewardShaper(config)

# 訓練循環
for episode in range(10000):
    # ... 運行episode ...

    # 更新權重
    shaper.update_scales(episode, metrics={'avg_reward': avg_reward})
```

**適用場景**:
- 長期訓練（百萬步以上）
- 需要逐步引導策略
- 避免過度激進

---

## 七、調試與監控

### 7.1 獎勵組件監控

在訓練過程中，應記錄各組件的數值：

```python
reward, components = reward_shaper.calculate_reward(...)

# 記錄到TensorBoard
writer.add_scalar('reward/total', reward, step)
writer.add_scalar('reward/pnl', components['pnl'], step)
writer.add_scalar('reward/cost', components['transaction_cost'], step)
writer.add_scalar('reward/inventory', components['inventory_penalty'], step)
writer.add_scalar('reward/risk', components['risk_penalty'], step)
```

### 7.2 異常檢測

**正常獎勵範圍**:
- 總獎勵: [-10, 10] 每步
- PnL: [-5, 5] 每步
- 成本: [-0.01, 0] 每步
- 庫存懲罰: [-0.1, 0] 每步
- 風險懲罰: [-0.001, 0] 每步

**異常情況**:
- ⚠️ 總獎勵 > 100: PnL計算可能有誤（檢查是否二次方累積）
- ⚠️ 成本 < -1: 交易成本率設置錯誤
- ⚠️ 獎勵全為正: 懲罰項失效
- ⚠️ 獎勵全為負: PnL權重過低或市場不利

### 7.3 超參數調優建議

**Step 1: 基礎驗證**（預設配置）
```yaml
pnl_scale: 1.0
cost_penalty: 1.0
inventory_penalty: 0.01
risk_penalty: 0.005
```
- 訓練10K steps
- 檢查是否學會基本交易（買低賣高）

**Step 2: 調整PnL權重**
```yaml
# 試驗1: 激進
pnl_scale: 2.0

# 試驗2: 保守
pnl_scale: 0.5
```
- 各訓練50K steps
- 比較Sharpe Ratio

**Step 3: 調整成本懲罰**
```yaml
# 試驗1: 減少交易
cost_penalty: 2.0

# 試驗2: 增加交易
cost_penalty: 0.5
```
- 各訓練50K steps
- 檢查交易頻率和盈利

**Step 4: 精細調整**
- 根據Step 2-3結果組合
- 訓練100K+ steps
- 驗證穩定性

---

## 八、常見問題

### Q1: 為什麼獎勵為負數？

**A**: 獎勵為負數表示當前步產生了淨損失（虧損或成本大於盈利）

**可能原因**:
- 交易虧損（買高賣低）
- 交易成本過高
- 長時間持倉（庫存懲罰）

**解決方法**:
- 降低成本懲罰權重（允許更多交易）
- 增加PnL權重（強調盈利）
- 調整策略（更好的進出時機）

---

### Q2: 智能體為什麼不交易（只Hold）？

**A**: 成本懲罰過高，導致交易的預期獎勵低於持有

**解決方法**:
```yaml
# 降低成本懲罰
cost_penalty: 0.5

# 提高PnL權重
pnl_scale: 2.0

# 降低庫存懲罰
inventory_penalty: 0.001
```

---

### Q3: 智能體為什麼過度交易？

**A**: PnL權重過高或成本懲罰過低

**解決方法**:
```yaml
# 提高成本懲罰
cost_penalty: 2.0

# 降低PnL權重
pnl_scale: 0.5

# 提高庫存懲罰（如果頻繁進出）
inventory_penalty: 0.05
```

---

### Q4: 訓練回報異常高（如每步獎勵>100）？

**A**: 可能是增量PnL計算錯誤（二次方累積）

**檢查**:
```python
# reward_shaper.py 的 _calculate_pnl() 方法
# 確認使用 prev_price 而非 entry_price

# ✅ 正確
incremental_pnl = prev_position × (current_price - prev_price)

# ❌ 錯誤
incremental_pnl = prev_position × (current_price - entry_price)
```

---

### Q5: 如何平衡盈利和風險？

**A**: 使用保守策略配置

```yaml
reward:
  pnl_scale: 1.0          # 正常重視盈利
  cost_penalty: 1.5       # 中高成本懲罰
  inventory_penalty: 0.02 # 中高庫存懲罰
  risk_penalty: 0.01      # 中高風險懲罰
```

**驗證指標**:
- Sharpe Ratio > 2.0
- 最大回撤 < 10%
- 勝率 > 55%

---

## 九、總結

### 核心要點

1. **多組件設計**: 4個獎勵組件（PnL + 3個懲罰）
2. **增量PnL**: 修正後的計算避免二次方累積
3. **台股成本**: 考慮實際手續費和交易稅（往返0.2355%）
4. **可調整權重**: 支持激進/保守/平衡策略
5. **課程學習**: 自適應獎勵支持長期訓練

### 推薦配置（實盤）

```yaml
reward:
  pnl_scale: 1.0
  cost_penalty: 1.5
  inventory_penalty: 0.02
  risk_penalty: 0.01

transaction_cost:
  shares_per_lot: 1000
  commission:
    base_rate: 0.001425
    discount: 0.3  # 根據實際券商折扣調整
  securities_tax:
    rate: 0.0015   # 當沖稅率
```

### 下一步

1. 使用預設配置訓練10K steps
2. 檢查TensorBoard監控各組件
3. 根據監控結果調整權重
4. 進行多組超參數實驗
5. 選擇最佳Sharpe Ratio配置
6. 進行長期訓練（100K+ steps）
7. 回測驗證策略穩定性

---

**文件版本**: v1.0
**最後更新**: 2025-10-26
**作者**: SB3-DeepLOB Team
**相關文件**:
- [src/envs/reward_shaper.py](../src/envs/reward_shaper.py)
- [src/envs/tw_lob_trading_env.py](../src/envs/tw_lob_trading_env.py)
- [configs/sb3_deeplob_config.yaml](../configs/sb3_deeplob_config.yaml)
- [docs/TAIWAN_STOCK_TRANSACTION_COSTS.md](TAIWAN_STOCK_TRANSACTION_COSTS.md)
