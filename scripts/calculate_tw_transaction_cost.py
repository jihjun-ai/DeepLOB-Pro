"""台股交易成本计算器

此脚本用于计算台股的实际交易成本，支持多种场景：
- 一般散户
- 电子券商
- 量化交易
- 当冲交易

使用范例:
    # 计算一般散户成本
    python scripts/calculate_tw_transaction_cost.py --scenario retail

    # 计算当冲交易成本
    python scripts/calculate_tw_transaction_cost.py --scenario daytrading

    # 自定义手续费折扣
    python scripts/calculate_tw_transaction_cost.py --commission-discount 0.3 --daytrading

作者: SB3-DeepLOB Team
日期: 2025-10-26
版本: v1.0
"""

import argparse
from typing import Dict, Tuple


class TaiwanTransactionCostCalculator:
    """台股交易成本计算器"""

    # 法定费率
    STT_RATE_NORMAL = 0.003      # 一般证券交易税 0.3%
    STT_RATE_DAYTRADING = 0.0015  # 当冲交易税 0.15% (减半)
    COMMISSION_MAX = 0.001425    # 法定最高手续费 0.1425%

    def __init__(self):
        pass

    def calculate(
        self,
        commission_discount: float = 1.0,
        is_daytrading: bool = False,
        verbose: bool = True
    ) -> Dict[str, float]:
        """计算交易成本

        参数:
            commission_discount: 手续费折扣 (0.1 ~ 1.0)
                - 1.0 = 无折扣 (0.1425%)
                - 0.6 = 6 折 (0.0855%)
                - 0.3 = 3 折 (0.0428%)
                - 0.2 = 2 折 (0.0285%)
            is_daytrading: 是否为当冲交易
            verbose: 是否显示详细信息

        返回:
            成本字典，包含:
                - buy_cost: 买入成本率
                - sell_cost: 卖出成本率
                - roundtrip_cost: 往返总成本率
        """
        # 计算实际手续费率
        actual_commission = self.COMMISSION_MAX * commission_discount

        # 买入成本 = 手续费
        buy_cost = actual_commission

        # 卖出成本 = 手续费 + 交易税
        stt_rate = self.STT_RATE_DAYTRADING if is_daytrading else self.STT_RATE_NORMAL
        sell_cost = actual_commission + stt_rate

        # 往返总成本
        roundtrip_cost = buy_cost + sell_cost

        if verbose:
            self._print_details(
                commission_discount=commission_discount,
                actual_commission=actual_commission,
                stt_rate=stt_rate,
                is_daytrading=is_daytrading,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                roundtrip_cost=roundtrip_cost
            )

        return {
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'roundtrip_cost': roundtrip_cost
        }

    def _print_details(
        self,
        commission_discount: float,
        actual_commission: float,
        stt_rate: float,
        is_daytrading: bool,
        buy_cost: float,
        sell_cost: float,
        roundtrip_cost: float
    ):
        """打印详细信息"""
        print("=" * 60)
        print("台股交易成本计算器")
        print("=" * 60)

        # 场景信息
        scenario = "当冲交易" if is_daytrading else "一般交易"
        print(f"\n场景: {scenario}")

        # 手续费信息
        discount_pct = int(commission_discount * 10)
        if discount_pct == 10:
            discount_desc = "无折扣"
        else:
            discount_desc = f"{discount_pct} 折"

        print(f"\n手续费:")
        print(f"  法定上限: {self.COMMISSION_MAX * 100:.4f}%")
        print(f"  折扣: {discount_desc}")
        print(f"  实际费率: {actual_commission * 100:.4f}%")

        # 交易税信息
        print(f"\n证券交易税:")
        if is_daytrading:
            print(f"  一般税率: {self.STT_RATE_NORMAL * 100:.2f}%")
            print(f"  当冲税率: {stt_rate * 100:.2f}% (减半)")
        else:
            print(f"  税率: {stt_rate * 100:.2f}%")

        # 成本计算
        print(f"\n成本计算:")
        print(f"  买入成本 = {buy_cost * 100:.4f}% (手续费)")
        print(f"  卖出成本 = {sell_cost * 100:.4f}% (手续费 + 交易税)")
        print(f"  往返总成本 = {roundtrip_cost * 100:.4f}%")

        # 配置建议
        print(f"\n配置建议:")
        print(f"  transaction_cost_rate: {roundtrip_cost:.6f}  # {roundtrip_cost * 100:.4f}%")

        # 买卖分开配置（可选）
        print(f"\n买卖分开配置（可选）:")
        print(f"  buy_cost_rate: {buy_cost:.6f}   # {buy_cost * 100:.4f}%")
        print(f"  sell_cost_rate: {sell_cost:.6f}  # {sell_cost * 100:.4f}%")

        print("\n" + "=" * 60)


def calculate_preset_scenarios():
    """计算预设场景"""
    calculator = TaiwanTransactionCostCalculator()

    scenarios = [
        ("一般散户 (无折扣)", 1.0, False),
        ("电子券商 (6 折)", 0.6, False),
        ("量化交易 (3 折)", 0.3, False),
        ("高频交易 (2 折)", 0.2, False),
        ("当冲交易 (3 折 + 税率减半)", 0.3, True),
    ]

    results = []
    for name, discount, is_daytrading in scenarios:
        print(f"\n{'=' * 60}")
        print(f"场景: {name}")
        print('=' * 60)

        result = calculator.calculate(
            commission_discount=discount,
            is_daytrading=is_daytrading,
            verbose=False
        )

        results.append({
            'name': name,
            'discount': discount,
            'is_daytrading': is_daytrading,
            **result
        })

        print(f"  买入成本: {result['buy_cost'] * 100:.4f}%")
        print(f"  卖出成本: {result['sell_cost'] * 100:.4f}%")
        print(f"  往返总成本: {result['roundtrip_cost'] * 100:.4f}%")
        print(f"  配置值: transaction_cost_rate: {result['roundtrip_cost']:.6f}")

    # 汇总表格
    print("\n" + "=" * 60)
    print("汇总表格")
    print("=" * 60)
    print(f"{'场景':<30} {'往返成本':<12} {'配置值'}")
    print("-" * 60)

    for r in results:
        daytrading_mark = " (当冲)" if r['is_daytrading'] else ""
        print(f"{r['name']:<30} {r['roundtrip_cost'] * 100:>6.4f}%     {r['roundtrip_cost']:.6f}")

    print("\n推荐配置:")
    print("  高频/当冲策略: 0.00250 (0.25%)")
    print("  量化交易策略: 0.00386 (0.386%)")
    print("  一般电子券商: 0.00471 (0.471%)")


def main():
    parser = argparse.ArgumentParser(
        description='台股交易成本计算器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预设场景:
  retail        - 一般散户 (无折扣)
  online        - 电子券商 (6 折)
  quant         - 量化交易 (3 折)
  hft           - 高频交易 (2 折)
  daytrading    - 当冲交易 (3 折 + 税率减半)
  all           - 显示所有场景 (默认)

使用范例:
  # 显示所有场景
  python scripts/calculate_tw_transaction_cost.py

  # 计算特定场景
  python scripts/calculate_tw_transaction_cost.py --scenario daytrading

  # 自定义折扣
  python scripts/calculate_tw_transaction_cost.py --commission-discount 0.3

  # 自定义折扣 + 当冲
  python scripts/calculate_tw_transaction_cost.py --commission-discount 0.3 --daytrading
        """
    )

    parser.add_argument(
        '--scenario',
        type=str,
        choices=['retail', 'online', 'quant', 'hft', 'daytrading', 'all'],
        default='all',
        help='预设场景'
    )

    parser.add_argument(
        '--commission-discount',
        type=float,
        default=None,
        help='手续费折扣 (0.1 ~ 1.0)，例如 0.3 表示 3 折'
    )

    parser.add_argument(
        '--daytrading',
        action='store_true',
        help='是否为当冲交易（税率减半）'
    )

    args = parser.parse_args()

    calculator = TaiwanTransactionCostCalculator()

    # 自定义场景
    if args.commission_discount is not None:
        calculator.calculate(
            commission_discount=args.commission_discount,
            is_daytrading=args.daytrading,
            verbose=True
        )
        return

    # 预设场景
    scenario_map = {
        'retail': (1.0, False, "一般散户"),
        'online': (0.6, False, "电子券商"),
        'quant': (0.3, False, "量化交易"),
        'hft': (0.2, False, "高频交易"),
        'daytrading': (0.3, True, "当冲交易"),
    }

    if args.scenario == 'all':
        calculate_preset_scenarios()
    else:
        discount, is_daytrading, name = scenario_map[args.scenario]
        print(f"\n场景: {name}")
        calculator.calculate(
            commission_discount=discount,
            is_daytrading=is_daytrading,
            verbose=True
        )


if __name__ == "__main__":
    main()
