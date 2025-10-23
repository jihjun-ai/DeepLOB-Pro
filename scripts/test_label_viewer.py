#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label Viewer 測試腳本

功能：測試 Label Viewer 的核心功能是否正常
作者：DeepLOB-Pro Team
日期：2025-10-23
"""

import sys
import os
from pathlib import Path

# 設定輸出編碼為 UTF-8
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json


def test_preprocessed_loader():
    """測試 preprocessed_loader 模組"""
    print("="*70)
    print("測試 1: preprocessed_loader 模組")
    print("="*70)

    try:
        from label_viewer.utils.preprocessed_loader import (
            load_preprocessed_stock,
            scan_preprocessed_directory,
            get_label_preview_stats
        )
        print("[OK] 成功導入 preprocessed_loader 模組")

        # 測試掃描目錄
        test_dir = "data/preprocessed_swing"
        if Path(test_dir).exists():
            print(f"\n測試掃描目錄: {test_dir}")
            dir_info = scan_preprocessed_directory(test_dir)
            print(f"  找到 {len(dir_info['dates'])} 個交易日")

            if len(dir_info['dates']) > 0:
                test_date = dir_info['dates'][0]
                print(f"  第一個日期: {test_date}")
                print(f"  該日股票數: {len(dir_info['stocks_by_date'][test_date])}")

                # 測試載入單一股票
                test_stocks = dir_info['stocks_by_date'][test_date][:2]
                if len(test_stocks) > 0:
                    test_symbol = test_stocks[0]
                    test_npz = f"{test_dir}/daily/{test_date}/{test_symbol}.npz"

                    print(f"\n測試載入股票: {test_symbol}")
                    stock_data = load_preprocessed_stock(test_npz)
                    print(f"  Features shape: {stock_data['features'].shape}")
                    print(f"  Mids shape: {stock_data['mids'].shape}")
                    print(f"  Has labels: {'labels' in stock_data and stock_data['labels'] is not None}")

                    metadata = stock_data.get('metadata', {})
                    if 'label_preview' in metadata and metadata['label_preview']:
                        lp = metadata['label_preview']
                        print(f"  Label preview:")
                        print(f"    Total: {lp['total_labels']:,}")
                        print(f"    Down: {lp['down_pct']:.1%}")
                        print(f"    Neutral: {lp['neutral_pct']:.1%}")
                        print(f"    Up: {lp['up_pct']:.1%}")
                        print(f"    Method: {lp.get('labeling_method', 'N/A')}")
        else:
            print(f"[WARN] 測試目錄不存在: {test_dir}")
            print("   請先執行 preprocess_single_day.py 生成數據")

        return True

    except Exception as e:
        print(f"[FAIL] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_preview_panel():
    """測試 label_preview_panel 組件"""
    print("\n" + "="*70)
    print("測試 2: label_preview_panel 組件")
    print("="*70)

    try:
        from label_viewer.components.label_preview_panel import (
            create_label_preview_bar,
            create_metadata_table,
            create_overall_label_stats_bar
        )
        print("[OK] 成功導入 label_preview_panel 組件")

        # 測試數據
        label_preview = {
            'total_labels': 14278,
            'down_count': 4512,
            'neutral_count': 2145,
            'up_count': 7621,
            'down_pct': 0.316,
            'neutral_pct': 0.150,
            'up_pct': 0.534
        }

        print("\n測試創建標籤預覽柱狀圖...")
        fig = create_label_preview_bar(label_preview)
        print(f"  [OK] 圖表創建成功，高度: {fig.layout.height}px")

        print("\n測試創建元數據表格...")
        metadata = {
            'symbol': '2330',
            'date': '20250901',
            'n_points': 14400,
            'range_pct': 0.0456,
            'return_pct': 0.0123,
            'pass_filter': True
        }
        fig = create_metadata_table(metadata)
        print(f"  ✅ 表格創建成功，高度: {fig.layout.height}px")

        print("\n測試創建整體標籤統計柱狀圖...")
        overall_dist = {
            'down_pct': 0.313,
            'neutral_pct': 0.150,
            'up_pct': 0.537,
            'total_labels': 3012456
        }
        fig = create_overall_label_stats_bar(overall_dist, 211)
        print(f"  [OK] 圖表創建成功，高度: {fig.layout.height}px")

        return True

    except Exception as e:
        print(f"[FAIL] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_structure():
    """測試應用程式結構"""
    print("\n" + "="*70)
    print("測試 3: 應用程式結構")
    print("="*70)

    try:
        # 檢查必要文件
        required_files = [
            "label_viewer/app_preprocessed.py",
            "label_viewer/utils/preprocessed_loader.py",
            "label_viewer/components/label_preview_panel.py",
            "scripts/run_label_viewer.bat",
            "docs/LABEL_VIEWER_GUIDE.md"
        ]

        all_exist = True
        for file_path in required_files:
            full_path = project_root / file_path
            exists = full_path.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {file_path}")
            if not exists:
                all_exist = False

        if all_exist:
            print("\n✅ 所有必要文件都存在")
        else:
            print("\n⚠️  部分文件缺失")

        return all_exist

    except Exception as e:
        print(f"[FAIL] 測試失敗: {e}")
        return False


def test_dependencies():
    """測試依賴套件"""
    print("\n" + "="*70)
    print("測試 4: 依賴套件")
    print("="*70)

    required_packages = {
        'dash': 'Dash Web 框架',
        'plotly': 'Plotly 視覺化',
        'numpy': 'NumPy 數值運算',
        'pandas': 'Pandas 數據處理'
    }

    all_installed = True
    for package, desc in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:15s} - {desc}")
        except ImportError:
            print(f"  ❌ {package:15s} - {desc} (未安裝)")
            all_installed = False

    if not all_installed:
        print("\n⚠️  部分套件未安裝，請執行:")
        print("    conda activate deeplob-pro")
        print("    pip install dash plotly pandas numpy")

    return all_installed


def main():
    """主測試函數"""
    print("\n")
    print("="*70)
    print("Label Viewer 測試腳本")
    print("="*70)
    print()

    # 執行所有測試
    results = {
        "依賴套件": test_dependencies(),
        "應用結構": test_app_structure(),
        "數據載入器": test_preprocessed_loader(),
        "視覺化組件": test_label_preview_panel()
    }

    # 統計結果
    print("\n" + "="*70)
    print("測試結果摘要")
    print("="*70)

    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name:20s}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print("\n" + "="*70)
    print(f"總測試數: {total_tests}")
    print(f"通過數: {passed_tests}")
    print(f"失敗數: {total_tests - passed_tests}")
    print("="*70)

    if passed_tests == total_tests:
        print("\n🎉 所有測試通過！Label Viewer 已就緒")
        print("\n啟動方式:")
        print("  方法 1: 執行 scripts\\run_label_viewer.bat")
        print("  方法 2: python label_viewer/app_preprocessed.py")
        print("\n瀏覽器訪問: http://localhost:8051")
        return 0
    else:
        print("\n⚠️  部分測試失敗，請檢查錯誤訊息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
