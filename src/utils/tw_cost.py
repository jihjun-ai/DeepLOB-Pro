"""台股交易成本计算器（最简化版）

简单明了的台股交易成本计算：
1. 券商手续费：买卖双边，折扣相同
2. 证券交易税：仅卖出
3. 滑点：可选

作者: SB3-DeepLOB Team
日期: 2025-10-26
版本: v3.0 (最简化)
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TaiwanCost:
    """台股交易成本计算器"""

    def __init__(self, config: Dict):
        """初始化

        config 结构:
            shares_per_lot: 1000      # 每张股数
            commission:
                base_rate: 0.001425
                discount: 0.3
                min_fee: 20.0
            securities_tax:
                rate: 0.0015
            slippage:
                enabled: false
                rate: 0.0001
        """
        # 台股交易单位：每张 = 1000 股
        self.shares_per_lot = config.get('shares_per_lot', 1000)

        # 手续费
        comm = config.get('commission', {})
        base = comm.get('base_rate', 0.001425)
        disc = comm.get('discount', 0.3)
        self.commission_rate = base * disc
        self.min_fee = comm.get('min_fee', 20.0)

        # 证券交易税（仅卖出）
        tax = config.get('securities_tax', {})
        self.tax_rate = tax.get('rate', 0.0015)

        # 滑点（可选）
        slip = config.get('slippage', {})
        self.slippage_enabled = slip.get('enabled', False)
        self.slippage_rate = slip.get('rate', 0.0001)

        # 统计
        self.total_commission = 0.0
        self.total_tax = 0.0
        self.total_cost = 0.0
        self.trade_count = 0

        self._log_config()

    def _log_config(self):
        """记录配置"""
        logger.info("=" * 60)
        logger.info("台股交易成本配置")
        logger.info("=" * 60)
        logger.info(f"交易单位: 1 张 = {self.shares_per_lot} 股")
        logger.info(f"手续费率: {self.commission_rate * 100:.4f}% (买卖双边)")
        logger.info(f"最低手续费: {self.min_fee} 元")
        logger.info(f"证券交易税: {self.tax_rate * 100:.2f}% (仅卖出)")
        if self.slippage_enabled:
            logger.info(f"滑点: {self.slippage_rate * 100:.4f}%")

        # 预期成本
        buy = self.get_buy_cost_rate()
        sell = self.get_sell_cost_rate()
        logger.info(f"\n预期成本率:")
        logger.info(f"  买入: {buy * 100:.4f}%")
        logger.info(f"  卖出: {sell * 100:.4f}%")
        logger.info(f"  往返: {(buy + sell) * 100:.4f}%")

        # 实际成本示例（报价 100 元/股，1 张）
        example_price = 100.0
        example_lots = 1.0
        example_value = example_price * self.shares_per_lot * example_lots
        buy_cost = max(example_value * self.commission_rate, self.min_fee)
        sell_cost = max(example_value * self.commission_rate, self.min_fee) + example_value * self.tax_rate

        logger.info(f"\n成本示例（报价 {example_price} 元/股，买卖 {example_lots:.0f} 张）:")
        logger.info(f"  交易价值: {example_value:,.0f} 元")
        logger.info(f"  买入成本: {buy_cost:,.2f} 元")
        logger.info(f"  卖出成本: {sell_cost:,.2f} 元")
        logger.info(f"  往返成本: {buy_cost + sell_cost:,.2f} 元")
        logger.info("=" * 60)

    def calculate(self, action: int, price: float, quantity: float) -> Tuple[float, Dict]:
        """计算交易成本

        参数:
            action: 0=Hold, 1=Buy, 2=Sell
            price: 价格（每股报价）
            quantity: 数量（张数）

        返回:
            (total_cost, breakdown)

        重要：
            - price 是每股报价（如 100 元/股）
            - quantity 是张数（如 1 张）
            - 交易价值 = price × shares_per_lot × quantity
            - 例：100 元/股 × 1000 股/张 × 1 张 = 100,000 元
        """
        if action == 0:  # Hold
            return 0.0, {'commission': 0, 'tax': 0, 'total': 0, 'value': 0}

        # 计算交易价值（报价 × 每张股数 × 张数）
        value = price * self.shares_per_lot * quantity

        # 手续费（买卖都收）
        commission = max(value * self.commission_rate, self.min_fee)

        # 交易税（仅卖出）
        tax = value * self.tax_rate if action == 2 else 0.0

        # 滑点
        slippage = value * self.slippage_rate if self.slippage_enabled else 0.0

        total = commission + tax + slippage

        # 统计
        self.total_commission += commission
        self.total_tax += tax
        self.total_cost += total
        self.trade_count += 1

        return total, {
            'commission': commission,
            'tax': tax,
            'slippage': slippage,
            'total': total,
            'value': value,
            'price_per_share': price,
            'lots': quantity,
            'shares': quantity * self.shares_per_lot,
        }

    def get_buy_cost_rate(self) -> float:
        """买入成本率"""
        rate = self.commission_rate
        if self.slippage_enabled:
            rate += self.slippage_rate
        return rate

    def get_sell_cost_rate(self) -> float:
        """卖出成本率"""
        rate = self.commission_rate + self.tax_rate
        if self.slippage_enabled:
            rate += self.slippage_rate
        return rate

    def get_roundtrip_rate(self) -> float:
        """往返成本率"""
        return self.get_buy_cost_rate() + self.get_sell_cost_rate()

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'trades': self.trade_count,
            'total_commission': self.total_commission,
            'total_tax': self.total_tax,
            'total_cost': self.total_cost,
        }

    def reset(self):
        """重置统计"""
        self.total_commission = 0.0
        self.total_tax = 0.0
        self.total_cost = 0.0
        self.trade_count = 0
