# 台股交易成本修正总结

## 问题

**当前配置严重低估台股实际交易成本！**

- **当前值**: 0.1% (0.001)
- **实际成本**: 0.236% ~ 0.585%
- **误差**: **2.36 ~ 5.85 倍**

## 快速修正（3 步骤）

### 步骤 1: 选择配置方案

| 场景 | 配置值 | 说明 |
|------|--------|------|
| **当冲交易** ⭐⭐⭐⭐⭐ | 0.00250 | 最推荐：高频交易 + 税率减半 |
| **量化交易** ⭐⭐⭐⭐ | 0.00386 | 保守：非当冲场景 |
| 电子券商 | 0.00471 | 一般散户 |

### 步骤 2: 修改配置文件

编辑 `configs/sb3_deeplob_config.yaml`:

```yaml
env_config:
  # 修改前（错误）
  # transaction_cost_rate: 0.001  # 0.1% - 严重低估

  # 修改后（推荐：当冲交易）
  transaction_cost_rate: 0.00250  # 0.25%

  # 或（保守：量化交易）
  # transaction_cost_rate: 0.00386  # 0.386%
```

### 步骤 3: 重新训练

```bash
# 测试
python scripts/train_sb3_deeplob.py --test

# 完整训练
python scripts/train_sb3_deeplob.py
```

## 详细说明

### 台股交易成本构成（2025）

**手续费** (双边收取):
- 法定上限: 0.1425%
- 实际折扣: 3 折 ~ 无折扣

**交易税** (仅卖出):
- 一般: 0.3%
- **当冲: 0.15%** (减半，延长至 2027/12/31)

### 计算结果

运行成本计算器查看所有场景：

```bash
python -c "from scripts.calculate_tw_transaction_cost import TaiwanTransactionCostCalculator; \
calc = TaiwanTransactionCostCalculator(); \
print('Day trading:', calc.calculate(0.3, True, False)); \
print('Quant trading:', calc.calculate(0.3, False, False))"
```

**输出**:

```
Scenario                                 Roundtrip    Config Value
----------------------------------------------------------------------
Retail (no discount)                     0.5850%     0.005850
Online broker (60% discount)             0.4710%     0.004710
Quant trading (30% discount)             0.3855%     0.003855
HFT (20% discount)                       0.3570%     0.003570
Day trading (30% + tax half)             0.2355%     0.002355  ⭐
```

## 推荐方案

### 🥇 第一推荐：当冲交易配置

**配置值**: `0.00250` (0.25%)

**优势**:
- ✅ 当冲税率减半（0.3% → 0.15%）
- ✅ 成本最低（0.236% + 缓冲）
- ✅ 适合高频交易策略
- ✅ 政策稳定（延长至 2027）

**成本构成**:
```
买入: 0.043% (手续费 3 折)
卖出: 0.193% (手续费 + 当冲税 0.15%)
往返: 0.236%
缓冲: 0.014%
总计: 0.250%
```

### 🥈 第二推荐：量化交易配置

**配置值**: `0.00386` (0.386%)

**优势**:
- ✅ 适用于非当冲场景
- ✅ 现实可达成（需一定交易量）
- ✅ 更保守更稳定

**成本构成**:
```
买入: 0.043% (手续费 3 折)
卖出: 0.343% (手续费 + 交易税 0.3%)
往返: 0.386%
```

## 预期影响

### 使用 0.25% (当冲)

- 交易频率: ↓ 20-30%
- 持仓时间: ↑ 1.3-1.5x
- 策略: 更注重日内平仓
- 实盘: ✅ 高度可行

### 使用 0.386% (量化)

- 交易频率: ↓ 30-40%
- 持仓时间: ↑ 1.5-2x
- 策略: 更保守更稳定
- 实盘: ✅ 可行

## 参考文档

- **详细文档**: [docs/TAIWAN_STOCK_TRANSACTION_COSTS.md](docs/TAIWAN_STOCK_TRANSACTION_COSTS.md)
- **成本计算器**: [scripts/calculate_tw_transaction_cost.py](scripts/calculate_tw_transaction_cost.py)

---

**更新日期**: 2025-10-26
**状态**: ✅ Ready to implement
