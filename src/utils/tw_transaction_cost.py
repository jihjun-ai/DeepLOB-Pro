"""台股交易成本计算器（简化版）

根据实际台股交易规则计算成本：
1. 券商手续费：买卖双边收取，折扣相同
2. 证券交易税：仅卖出收取，支持当冲减半
3. 滑点成本：可选

作者: SB3-DeepLOB Team
日期: 2025-10-26
版本: v2.0 (简化版)
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TaiwanTransactionCost:
    """台股交易成本计算器

    特点：
    - 简单明了，符合实际交易规则
    - 买卖手续费折扣相同
    - 支持当冲税率减半
    """

    def __init__(self, config: Dict):
        """初始化成本计算器

        参数:
            config: 交易成本配置字典
                commission (dict): 手续费配置
                    - base_rate: 法定上限 (0.001425)
                    - discount: 折扣 (0.1-1.0)
                    - min_fee: 最低手续费 (台币)
                transaction_tax (dict): 交易税配置
                    - normal_rate: 一般税率 (0.003)
                    - daytrading_rate: 当冲税率 (0.0015)
                    - use_daytrading: 是否使用当冲优惠
                slippage (dict): 滑点配置 (可选)
                    - enabled: 是否启用
                    - rate: 滑点率
        """
        self.config = config

        # 券商手续费
        commission_cfg = config.get('commission', {})
        self.commission_base = commission_cfg.get('base_rate', 0.001425)
        self.commission_discount = commission_cfg.get('discount', 0.3)
        self.commission_min_fee = commission_cfg.get('min_fee', 20.0)

        # 实际手续费率（买卖相同）
        self.commission_rate = self.commission_base * self.commission_discount

        # 证券交易税（仅卖出）
        tax_cfg = config.get('transaction_tax', {})
        self.tax_normal = tax_cfg.get('normal_rate', 0.003)
        self.tax_daytrading = tax_cfg.get('daytrading_rate', 0.0015)
        self.use_daytrading = tax_cfg.get('use_daytrading', True)

        # 滑点（可选）
        slippage_cfg = config.get('slippage', {})
        self.slippage_enabled = slippage_cfg.get('enabled', False)
        self.slippage_rate = slippage_cfg.get('rate', 0.0001)

        # 成本统计
        self.total_commission = 0.0
        self.total_tax = 0.0
        self.total_slippage = 0.0
        self.trade_count = 0

        self._log_config()

    def _log_config(self):
        """记录配置信息"""
        logger.info("=" * 60)
        logger.info("台股交易成本配置")
        logger.info("=" * 60)
        logger.info(f"券商手续费:")
        logger.info(f"  基准费率: {self.commission_base * 100:.4f}%")
        logger.info(f"  折扣: {int(self.commission_discount * 10)} 折")
        logger.info(f"  实际费率: {self.commission_rate * 100:.6f}%")
        logger.info(f"  最低手续费: {self.commission_min_fee} 元")
        logger.info(f"  应用: 买入 + 卖出 (相同)")

        logger.info(f"证券交易税:")
        logger.info(f"  一般税率: {self.tax_normal * 100:.2f}%")
        logger.info(f"  当冲税率: {self.tax_daytrading * 100:.2f}%")
        logger.info(f"  使用当冲优惠: {self.use_daytrading}")
        logger.info(f"  应用: 仅卖出")

        if self.slippage_enabled:
            logger.info(f"滑点成本:")
            logger.info(f"  滑点率: {self.slippage_rate * 100:.4f}%")
            logger.info(f"  已启用")
        else:
            logger.info(f"滑点成本: 未启用")

        # 计算并显示预期成本
        buy_cost = self.get_buy_cost_rate()
        sell_cost_normal = self.get_sell_cost_rate(is_daytrading=False)
        sell_cost_daytrading = self.get_sell_cost_rate(is_daytrading=True)

        logger.info(f"\n预期成本:")
        logger.info(f"  买入成本率: {buy_cost * 100:.4f}%")
        logger.info(f"  卖出成本率 (一般): {sell_cost_normal * 100:.4f}%")
        logger.info(f"  卖出成本率 (当冲): {sell_cost_daytrading * 100:.4f}%")
        logger.info(f"  往返成本 (一般): {(buy_cost + sell_cost_normal) * 100:.4f}%")
        logger.info(f"  往返成本 (当冲): {(buy_cost + sell_cost_daytrading) * 100:.4f}%")
        logger.info("=" * 60)

    def calculate_cost(
        self,
        action: int,
        price: float,
        quantity: float,
        is_daytrading: bool = None
    ) -> Tuple[float, Dict[str, float]]:
        """计算交易成本

        参数:
            action: 动作 (0=Hold, 1=Buy, 2=Sell)
            price: 价格
            quantity: 数量（正数）
            is_daytrading: 是否当冲（None时使用配置的默认值）

        返回:
            (total_cost, breakdown)
            - total_cost: 总成本（台币）
            - breakdown: 成本明细
        """
        if action == 0:  # Hold
            return 0.0, self._empty_breakdown()

        if is_daytrading is None:
            is_daytrading = self.use_daytrading

        trade_value = price * quantity

        # 1. 券商手续费（买卖都收）
        commission = max(
            trade_value * self.commission_rate,
            self.commission_min_fee
        )

        # 2. 证券交易税（仅卖出）
        tax = 0.0
        if action == 2:  # Sell
            tax_rate = self.tax_daytrading if is_daytrading else self.tax_normal
            tax = trade_value * tax_rate

        # 3. 滑点（可选）
        slippage = 0.0
        if self.slippage_enabled:
            slippage = trade_value * self.slippage_rate

        # 总成本
        total = commission + tax + slippage

        # 成本明细
        breakdown = {
            'commission': commission,
            'tax': tax,
            'slippage': slippage,
            'total': total,
            'trade_value': trade_value,
            'commission_rate': commission / trade_value if trade_value > 0 else 0,
            'tax_rate': tax / trade_value if trade_value > 0 else 0,
            'total_rate': total / trade_value if trade_value > 0 else 0,
            'is_daytrading': is_daytrading,
        }

        # 更新统计
        self.total_commission += commission
        self.total_tax += tax
        self.total_slippage += slippage
        self.trade_count += 1

        return total, breakdown

    def get_buy_cost_rate(self) -> float:
        """获取买入成本率

        返回:
            买入成本率（手续费 + 滑点）
        """
        rate = self.commission_rate
        if self.slippage_enabled:
            rate += self.slippage_rate
        return rate

    def get_sell_cost_rate(self, is_daytrading: bool = None) -> float:
        """获取卖出成本率

        参数:
            is_daytrading: 是否当冲（None时使用配置）

        返回:
            卖出成本率（手续费 + 交易税 + 滑点）
        """
        if is_daytrading is None:
            is_daytrading = self.use_daytrading

        # 手续费
        rate = self.commission_rate

        # 交易税
        tax_rate = self.tax_daytrading if is_daytrading else self.tax_normal
        rate += tax_rate

        # 滑点
        if self.slippage_enabled:
            rate += self.slippage_rate

        return rate

    def get_roundtrip_cost_rate(self, is_daytrading: bool = None) -> float:
        """获取往返成本率（买入 + 卖出）

        参数:
            is_daytrading: 是否当冲（None时使用配置）

        返回:
            往返总成本率
        """
        buy_rate = self.get_buy_cost_rate()
        sell_rate = self.get_sell_cost_rate(is_daytrading)
        return buy_rate + sell_rate

    def get_statistics(self) -> Dict[str, float]:
        """获取成本统计"""
        total_cost = self.total_commission + self.total_tax + self.total_slippage

        return {
            'trade_count': self.trade_count,
            'total_commission': self.total_commission,
            'total_tax': self.total_tax,
            'total_slippage': self.total_slippage,
            'total_cost': total_cost,
            'avg_commission': self.total_commission / self.trade_count if self.trade_count > 0 else 0,
            'avg_tax': self.total_tax / self.trade_count if self.trade_count > 0 else 0,
            'avg_cost': total_cost / self.trade_count if self.trade_count > 0 else 0,
        }

    def reset_statistics(self):
        """重置统计"""
        self.total_commission = 0.0
        self.total_tax = 0.0
        self.total_slippage = 0.0
        self.trade_count = 0

    def _empty_breakdown(self) -> Dict[str, float]:
        """空成本明细（Hold）"""
        return {
            'commission': 0.0,
            'tax': 0.0,
            'slippage': 0.0,
            'total': 0.0,
            'trade_value': 0.0,
            'commission_rate': 0.0,
            'tax_rate': 0.0,
            'total_rate': 0.0,
            'is_daytrading': False,
        }

    def print_summary(self):
        """打印成本汇总"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("交易成本汇总")
        print("=" * 60)
        print(f"总交易次数: {stats['trade_count']}")
        print(f"总手续费: {stats['total_commission']:,.2f} 元")
        print(f"总交易税: {stats['total_tax']:,.2f} 元")
        if self.slippage_enabled:
            print(f"总滑点: {stats['total_slippage']:,.2f} 元")
        print(f"总成本: {stats['total_cost']:,.2f} 元")
        print(f"\n平均单笔:")
        print(f"  手续费: {stats['avg_commission']:,.2f} 元")
        print(f"  交易税: {stats['avg_tax']:,.2f} 元")
        print(f"  总成本: {stats['avg_cost']:,.2f} 元")
        print("=" * 60)
