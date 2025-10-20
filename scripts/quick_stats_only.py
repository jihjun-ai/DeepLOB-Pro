#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quick_stats_only.py - 快速震盪統計工具（不生成訓練數據）
=============================================================================
【用途】快速產生震盪統計報告，跳過耗時的 Triple-Barrier 和訓練數據生成

【優點】
- ⚡ 速度快：只需 5-10% 的時間（相比完整流程）
- 📊 完整報告：CSV + JSON + 控制台報告
- 💾 節省空間：不生成大型 NPZ 檔案

【使用場景】
- 初步分析數據波動特性
- 決定是否需要篩選低波動股票
- 快速驗證數據品質

【使用方式】
# 基本用法
python scripts/quick_stats_only.py

# 指定輸入/輸出目錄
python scripts/quick_stats_only.py \
    --input-dir ./data/temp \
    --output-dir ./data/volatility_stats

【輸出檔案】
- volatility_stats.csv      # 完整震盪數據
- volatility_summary.json   # 統計摘要

【對比完整流程】
完整流程（--make-npz）：
  - 時間：30-60 分鐘
  - 輸出：NPZ + 震盪統計

快速模式（--stats-only）：
  - 時間：3-5 分鐘 ⚡
  - 輸出：震盪統計

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
    parser = argparse.ArgumentParser(
        description="快速震盪統計工具（不生成訓練數據）"
    )
    parser.add_argument(
        "--input-dir",
        default="./data/temp",
        help="原始數據目錄（包含 .txt 檔案）"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/volatility_stats",
        help="輸出目錄（震盪統計報告）"
    )
    parser.add_argument(
        "--config",
        default="./configs/config_pro_v5.yaml",
        help="配置文件路徑（僅用於數據清洗參數）"
    )

    args = parser.parse_args()

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("="*60)
    logging.info("⚡ 快速震盪統計模式")
    logging.info("="*60)
    logging.info(f"輸入目錄: {args.input_dir}")
    logging.info(f"輸出目錄: {args.output_dir}")
    logging.info(f"配置文件: {args.config}")
    logging.info("="*60)
    logging.info("提示：此模式跳過訓練數據生成，僅產生震盪統計報告")
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
        '--stats-only'  # 關鍵參數：只產生統計
    ]

    try:
        exit_code = extract_main()

        if exit_code == 0:
            logging.info("\n" + "="*60)
            logging.info("✅ 震盪統計完成！")
            logging.info("="*60)
            logging.info("輸出檔案:")
            logging.info(f"  1. {os.path.join(args.output_dir, 'volatility_stats.csv')}")
            logging.info(f"     → 完整震盪數據（可用 Excel 開啟）")
            logging.info(f"  2. {os.path.join(args.output_dir, 'volatility_summary.json')}")
            logging.info(f"     → 統計摘要（包含閾值建議）")
            logging.info("="*60)
            logging.info("\n下一步:")
            logging.info("  1. 查看統計報告，決定是否需要篩選")
            logging.info("  2. 如需生成訓練數據，請執行:")
            logging.info(f"     python scripts/extract_tw_stock_data_v5.py \\")
            logging.info(f"         --input-dir {args.input_dir} \\")
            logging.info(f"         --output-dir ./data/processed_v5")
            logging.info("="*60 + "\n")
        else:
            logging.error("統計過程中發生錯誤！")

    except Exception as e:
        logging.error(f"執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.argv = original_argv

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
