"""台股交易成本计算器

此模块实现详细的台股交易成本计算，包括：
- 券商手续费（买卖双边，支持折扣）
- 证券交易税（仅卖出，支持当冲优惠）
- 滑点成本（可选）
- 市场冲击成本（可选）

作者: SB3-DeepLOB Team
日期: 2025-10-26
版本: v1.0
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TransactionCostCalculator:
    """台股交易成本计算器

    支持两种模式：
    1. 详细模式：分别计算手续费、交易税、滑点等
    2. 简化模式：使用单一成本率（向后兼容）
    """

    def __init__(self, config: Dict):
        """初始化成本计算器

        参数:
            config: 交易成本配置字典，包含以下键值:
                use_detailed_model (bool): 是否使用详细成本模型
                commission (dict): 券商手续费配置
                securities_transaction_tax (dict): 证券交易税配置
                other_costs (dict): 其他成本配置
                simple_rate (float): 简化模式的成本率
        """
        self.config = config
        self.use_detailed = config.get('use_detailed_model', True)

        if self.use_detailed:
            self._init_detailed_model()
        else:
            self._init_simple_model()

        # 记录成本统计
        self.total_commission = 0.0
        self.total_tax = 0.0
        self.total_slippage = 0.0
        self.total_cost = 0.0

        logger.info(f"交易成本计算器已初始化 (详细模式: {self.use_detailed})")
        if self.use_detailed:
            self._log_detailed_config()

    def _init_detailed_model(self):
        """初始化详细成本模型"""
        # 券商手续费配置
        commission_config = self.config.get('commission', {})
        self.commission_rate_max = commission_config.get('rate_max', 0.001425)
        self.commission_discount = commission_config.get('discount', 1.0)
        self.commission_min_charge = commission_config.get('min_charge', 20.0)
        self.commission_apply_to_buy = commission_config.get('apply_to_buy', True)
        self.commission_apply_to_sell = commission_config.get('apply_to_sell', True)

        # 实际手续费率
        self.commission_rate = self.commission_rate_max * self.commission_discount

        # 证券交易税配置
        stt_config = self.config.get('securities_transaction_tax', {})
        self.stt_stock_normal = stt_config.get('stock_rate_normal', 0.003)
        self.stt_stock_daytrading = stt_config.get('stock_rate_daytrading', 0.0015)
        self.stt_etf_normal = stt_config.get('etf_rate_normal', 0.001)
        self.stt_etf_daytrading = stt_config.get('etf_rate_daytrading', 0.0005)
        self.stt_apply_to_buy = stt_config.get('apply_to_buy', False)
        self.stt_apply_to_sell = stt_config.get('apply_to_sell', True)
        self.stt_security_type = stt_config.get('security_type', 'stock')
        self.stt_enable_daytrading = stt_config.get('enable_daytrading_discount', True)

        # 其他成本配置
        other_costs = self.config.get('other_costs', {})
        self.slippage_rate = other_costs.get('slippage_rate', 0.0001)
        self.market_impact = other_costs.get('market_impact', 0.0)
        self.enable_slippage = other_costs.get('enable_slippage', False)

    def _init_simple_model(self):
        """初始化简化成本模型"""
        self.simple_rate = self.config.get('simple_rate', 0.0025)
        logger.info(f"使用简化成本模型: {self.simple_rate * 100:.4f}%")

    def _log_detailed_config(self):
        """记录详细配置信息"""
        logger.info("详细成本配置:")
        logger.info(f"  券商手续费:")
        logger.info(f"    - 法定上限: {self.commission_rate_max * 100:.4f}%")
        logger.info(f"    - 折扣: {self.commission_discount * 10:.0f} 折")
        logger.info(f"    - 实际费率: {self.commission_rate * 100:.6f}%")
        logger.info(f"    - 最低手续费: {self.commission_min_charge} 元")
        logger.info(f"    - 买入收费: {self.commission_apply_to_buy}")
        logger.info(f"    - 卖出收费: {self.commission_apply_to_sell}")

        logger.info(f"  证券交易税:")
        logger.info(f"    - 证券类型: {self.stt_security_type}")
        logger.info(f"    - 一般税率: {self.stt_stock_normal * 100:.2f}%")
        logger.info(f"    - 当冲税率: {self.stt_stock_daytrading * 100:.2f}%")
        logger.info(f"    - 当冲优惠: {self.stt_enable_daytrading}")
        logger.info(f"    - 买入收税: {self.stt_apply_to_buy}")
        logger.info(f"    - 卖出收税: {self.stt_apply_to_sell}")

        logger.info(f"  其他成本:")
        logger.info(f"    - 滑点费率: {self.slippage_rate * 100:.4f}%")
        logger.info(f"    - 启用滑点: {self.enable_slippage}")

    def calculate_cost(
        self,
        action: int,
        price: float,
        quantity: float,
        is_daytrading: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """计算交易成本

        参数:
            action: 交易动作 (0=Hold, 1=Buy, 2=Sell)
            price: 交易价格
            quantity: 交易数量（正数）
            is_daytrading: 是否为当冲交易

        返回:
            (total_cost, cost_breakdown)
            - total_cost: 总成本（台币）
            - cost_breakdown: 成本明细字典
        """
        if action == 0:  # Hold - 无交易成本
            return 0.0, self._empty_breakdown()

        if self.use_detailed:
            return self._calculate_detailed_cost(action, price, quantity, is_daytrading)
        else:
            return self._calculate_simple_cost(action, price, quantity)

    def _calculate_detailed_cost(
        self,
        action: int,
        price: float,
        quantity: float,
        is_daytrading: bool
    ) -> Tuple[float, Dict[str, float]]:
        """计算详细成本"""
        trade_value = price * quantity

        # 1. 券商手续费
        commission = 0.0
        if (action == 1 and self.commission_apply_to_buy) or \
           (action == 2 and self.commission_apply_to_sell):
            commission = max(
                trade_value * self.commission_rate,
                self.commission_min_charge
            )

        # 2. 证券交易税
        tax = 0.0
        if (action == 1 and self.stt_apply_to_buy) or \
           (action == 2 and self.stt_apply_to_sell):
            # 确定税率
            if self.stt_security_type == 'stock':
                tax_rate = self.stt_stock_daytrading if (is_daytrading and self.stt_enable_daytrading) \
                          else self.stt_stock_normal
            elif self.stt_security_type == 'etf':
                tax_rate = self.stt_etf_daytrading if (is_daytrading and self.stt_enable_daytrading) \
                          else self.stt_etf_normal
            else:
                tax_rate = self.stt_stock_normal

            tax = trade_value * tax_rate

        # 3. 滑点成本
        slippage = 0.0
        if self.enable_slippage:
            slippage = trade_value * self.slippage_rate

        # 4. 市场冲击成本（暂不启用）
        market_impact = 0.0

        # 总成本
        total_cost = commission + tax + slippage + market_impact

        # 成本明细
        breakdown = {
            'commission': commission,
            'tax': tax,
            'slippage': slippage,
            'market_impact': market_impact,
            'total': total_cost,
            'trade_value': trade_value,
            'commission_rate': commission / trade_value if trade_value > 0 else 0,
            'tax_rate': tax / trade_value if trade_value > 0 else 0,
            'total_rate': total_cost / trade_value if trade_value > 0 else 0,
        }

        # 更新统计
        self.total_commission += commission
        self.total_tax += tax
        self.total_slippage += slippage
        self.total_cost += total_cost

        return total_cost, breakdown

    def _calculate_simple_cost(
        self,
        action: int,
        price: float,
        quantity: float
    ) -> Tuple[float, Dict[str, float]]:
        """计算简化成本"""
        trade_value = price * quantity
        total_cost = trade_value * self.simple_rate

        breakdown = {
            'commission': 0.0,
            'tax': 0.0,
            'slippage': 0.0,
            'market_impact': 0.0,
            'total': total_cost,
            'trade_value': trade_value,
            'commission_rate': 0.0,
            'tax_rate': 0.0,
            'total_rate': self.simple_rate,
        }

        self.total_cost += total_cost

        return total_cost, breakdown

    def _empty_breakdown(self) -> Dict[str, float]:
        """空成本明细（Hold 动作）"""
        return {
            'commission': 0.0,
            'tax': 0.0,
            'slippage': 0.0,
            'market_impact': 0.0,
            'total': 0.0,
            'trade_value': 0.0,
            'commission_rate': 0.0,
            'tax_rate': 0.0,
            'total_rate': 0.0,
        }

    def get_statistics(self) -> Dict[str, float]:
        """获取成本统计信息"""
        return {
            'total_commission': self.total_commission,
            'total_tax': self.total_tax,
            'total_slippage': self.total_slippage,
            'total_cost': self.total_cost,
        }

    def reset_statistics(self):
        """重置成本统计"""
        self.total_commission = 0.0
        self.total_tax = 0.0
        self.total_slippage = 0.0
        self.total_cost = 0.0

    def get_buy_cost_rate(self, is_daytrading: bool = False) -> float:
        """获取买入成本率

        参数:
            is_daytrading: 是否为当冲交易

        返回:
            买入成本率（小数形式）
        """
        if not self.use_detailed:
            return self.simple_rate

        cost_rate = 0.0

        # 手续费
        if self.commission_apply_to_buy:
            cost_rate += self.commission_rate

        # 交易税（买入通常不收税）
        if self.stt_apply_to_buy:
            if self.stt_security_type == 'stock':
                tax_rate = self.stt_stock_daytrading if (is_daytrading and self.stt_enable_daytrading) \
                          else self.stt_stock_normal
            else:
                tax_rate = self.stt_stock_normal
            cost_rate += tax_rate

        # 滑点
        if self.enable_slippage:
            cost_rate += self.slippage_rate

        return cost_rate

    def get_sell_cost_rate(self, is_daytrading: bool = False) -> float:
        """获取卖出成本率

        参数:
            is_daytrading: 是否为当冲交易

        返回:
            卖出成本率（小数形式）
        """
        if not self.use_detailed:
            return self.simple_rate

        cost_rate = 0.0

        # 手续费
        if self.commission_apply_to_sell:
            cost_rate += self.commission_rate

        # 交易税（卖出收税）
        if self.stt_apply_to_sell:
            if self.stt_security_type == 'stock':
                tax_rate = self.stt_stock_daytrading if (is_daytrading and self.stt_enable_daytrading) \
                          else self.stt_stock_normal
            elif self.stt_security_type == 'etf':
                tax_rate = self.stt_etf_daytrading if (is_daytrading and self.stt_enable_daytrading) \
                          else self.stt_etf_normal
            else:
                tax_rate = self.stt_stock_normal
            cost_rate += tax_rate

        # 滑点
        if self.enable_slippage:
            cost_rate += self.slippage_rate

        return cost_rate

    def get_roundtrip_cost_rate(self, is_daytrading: bool = False) -> float:
        """获取往返成本率（买入 + 卖出）

        参数:
            is_daytrading: 是否为当冲交易

        返回:
            往返成本率（小数形式）
        """
        buy_rate = self.get_buy_cost_rate(is_daytrading)
        sell_rate = self.get_sell_cost_rate(is_daytrading)
        return buy_rate + sell_rate
