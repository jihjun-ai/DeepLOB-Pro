#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_volatility_stats.py - 測試震盪統計功能
=============================================================================
【用途】快速測試震盪統計功能，不需要完整執行數據處理流程

【使用方式】
python scripts/test_volatility_stats.py --input-dir ./data/temp --output-dir ./data/volatility_test

【輸出】
1. volatility_stats.csv - 完整震盪數據
2. volatility_summary.json - 統計摘要
3. 控制台報告
=============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(description="測試震盪統計功能")
    parser.add_argument(
        "--input-dir",
        default="./data/temp",
        help="原始數據目錄"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/volatility_test",
        help="輸出目錄"
    )
    parser.add_argument(
        "--config",
        default="./configs/config_pro_v5.yaml",
        help="配置文件路徑"
    )

    args = parser.parse_args()

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("="*60)
    logging.info("震盪統計測試開始")
    logging.info("="*60)
    logging.info(f"輸入目錄: {args.input_dir}")
    logging.info(f"輸出目錄: {args.output_dir}")
    logging.info(f"配置文件: {args.config}")
    logging.info("="*60 + "\n")

    # 導入並執行主程式
    from extract_tw_stock_data_v5 import main as extract_main

    # 修改 sys.argv 來傳遞參數
    original_argv = sys.argv.copy()
    sys.argv = [
        'extract_tw_stock_data_v5.py',
        '--input-dir', args.input_dir,
        '--output-dir', args.output_dir,
        '--config', args.config,
        '--make-npz'
    ]

    try:
        exit_code = extract_main()

        if exit_code == 0:
            logging.info("\n" + "="*60)
            logging.info("✅ 測試成功！")
            logging.info("="*60)
            logging.info("請檢查以下檔案:")
            logging.info(f"  1. {os.path.join(args.output_dir, 'volatility_stats.csv')}")
            logging.info(f"  2. {os.path.join(args.output_dir, 'volatility_summary.json')}")
            logging.info("="*60 + "\n")
        else:
            logging.error("測試失敗！")

    except Exception as e:
        logging.error(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.argv = original_argv

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
