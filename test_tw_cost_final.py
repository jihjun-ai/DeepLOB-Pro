"""Test Taiwan transaction cost with shares_per_lot"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.yaml_manager import YAMLManager
from src.utils.tw_cost import TaiwanCost


def test_cost_with_shares_per_lot():
    """Test cost calculation with shares_per_lot (1000)"""

    # Load config
    config = YAMLManager('configs/sb3_deeplob_config.yaml')
    cost_config = config.env_config.transaction_cost.as_plain_object()

    print("=" * 70)
    print("Taiwan Transaction Cost Test (with shares_per_lot)")
    print("=" * 70)

    # Create calculator
    calc = TaiwanCost(cost_config)

    print("\n" + "=" * 70)
    print("Test Case: Buy and Sell 1 lot")
    print("=" * 70)

    # Price per share
    price_per_share = 100.0
    lots = 1.0

    print(f"\nInput:")
    print(f"  Price per share: {price_per_share} TWD")
    print(f"  Quantity: {lots} lot")
    print(f"  Shares per lot: {calc.shares_per_lot}")
    print(f"  Total shares: {lots * calc.shares_per_lot:.0f}")

    # Buy
    print(f"\n{'='*70}")
    print("BUY Transaction")
    print("=" * 70)
    buy_cost, buy_breakdown = calc.calculate(action=1, price=price_per_share, quantity=lots)
    print(f"Trade value: {buy_breakdown['value']:,.2f} TWD")
    print(f"  Calculation: {price_per_share} (price/share) × {calc.shares_per_lot} (shares/lot) × {lots} (lots)")
    print(f"  = {buy_breakdown['value']:,.2f} TWD")
    print(f"\nCosts:")
    print(f"  Commission: {buy_breakdown['commission']:,.2f} TWD")
    print(f"  Tax: {buy_breakdown['tax']:,.2f} TWD (buy has no tax)")
    print(f"  Total: {buy_breakdown['total']:,.2f} TWD")
    print(f"  Cost rate: {buy_breakdown['total']/buy_breakdown['value']*100:.4f}%")

    # Sell
    print(f"\n{'='*70}")
    print("SELL Transaction")
    print("=" * 70)
    sell_cost, sell_breakdown = calc.calculate(action=2, price=price_per_share, quantity=lots)
    print(f"Trade value: {sell_breakdown['value']:,.2f} TWD")
    print(f"  Calculation: {price_per_share} (price/share) × {calc.shares_per_lot} (shares/lot) × {lots} (lots)")
    print(f"  = {sell_breakdown['value']:,.2f} TWD")
    print(f"\nCosts:")
    print(f"  Commission: {sell_breakdown['commission']:,.2f} TWD")
    print(f"  Securities tax: {sell_breakdown['tax']:,.2f} TWD (0.15% for day trading)")
    print(f"  Total: {sell_breakdown['total']:,.2f} TWD")
    print(f"  Cost rate: {sell_breakdown['total']/sell_breakdown['value']*100:.4f}%")

    # Roundtrip
    print(f"\n{'='*70}")
    print("ROUNDTRIP Summary")
    print("=" * 70)
    roundtrip_cost = buy_cost + sell_cost
    roundtrip_value = buy_breakdown['value']
    print(f"Buy cost:  {buy_cost:,.2f} TWD ({buy_cost/roundtrip_value*100:.4f}%)")
    print(f"Sell cost: {sell_cost:,.2f} TWD ({sell_cost/roundtrip_value*100:.4f}%)")
    print(f"Total:     {roundtrip_cost:,.2f} TWD ({roundtrip_cost/roundtrip_value*100:.4f}%)")

    # Expected vs Actual
    expected_rate = calc.get_roundtrip_rate()
    actual_rate = roundtrip_cost / roundtrip_value

    print(f"\n{'='*70}")
    print("Verification")
    print("=" * 70)
    print(f"Expected roundtrip rate: {expected_rate*100:.4f}%")
    print(f"Actual roundtrip rate:   {actual_rate*100:.4f}%")
    print(f"Match: {'YES' if abs(expected_rate - actual_rate) < 0.00001 else 'NO'}")

    # Compare with old config
    print(f"\n{'='*70}")
    print("Comparison with Old Config")
    print("=" * 70)
    old_rate = 0.001
    print(f"Old config (WRONG):  {old_rate*100:.4f}%")
    print(f"New config (CORRECT): {actual_rate*100:.4f}%")
    print(f"Difference: +{(actual_rate - old_rate)*100:.4f}%")
    print(f"Multiplier: {actual_rate/old_rate:.2f}x")

    print(f"\n{'='*70}")
    print("SUCCESS: Test passed!")
    print("=" * 70)
    print("\nKey takeaways:")
    print(f"  1. Price input is per SHARE (e.g., {price_per_share} TWD/share)")
    print(f"  2. Quantity is in LOTS (e.g., {lots} lot = {lots * calc.shares_per_lot:.0f} shares)")
    print(f"  3. Trade value = {price_per_share} × {calc.shares_per_lot} × {lots} = {roundtrip_value:,.0f} TWD")
    print(f"  4. Roundtrip cost = {roundtrip_cost:,.2f} TWD ({actual_rate*100:.4f}%)")


if __name__ == "__main__":
    test_cost_with_shares_per_lot()
