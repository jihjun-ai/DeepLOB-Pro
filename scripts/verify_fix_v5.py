#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_fix_v5.py - 快速驗證跨日污染修復

用途：
1. 使用少量數據快速測試修復是否有效
2. 檢查 Class 1（持平）是否出現
3. 比較修復前後的標籤分布差異

使用方式：
python scripts/verify_fix_v5.py --input-dir ./data/temp --output-dir ./data/test_fix
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    """解析命令列參數"""
    p = argparse.ArgumentParser(
        "verify_fix_v5",
        description="快速驗證跨日污染修復"
    )
    p.add_argument(
        "--input-dir",
        default="./data/temp",
        type=str,
        help="原始數據目錄"
    )
    p.add_argument(
        "--output-dir",
        default="./data/test_fix",
        type=str,
        help="輸出目錄"
    )
    p.add_argument(
        "--max-files",
        default=5,
        type=int,
        help="最多處理幾個檔案（快速測試用）"
    )
    return p.parse_args()


def main():
    """主程式"""
    args = parse_args()

    logging.info("="*60)
    logging.info("跨日污染修復驗證腳本")
    logging.info("="*60)
    logging.info(f"輸入目錄: {args.input_dir}")
    logging.info(f"輸出目錄: {args.output_dir}")
    logging.info(f"最多處理: {args.max_files} 個檔案")
    logging.info("="*60)

    # 檢查輸入目錄
    if not os.path.exists(args.input_dir):
        logging.error(f"輸入目錄不存在: {args.input_dir}")
        return 1

    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 導入主程式（這會使用修復後的代碼）
    try:
        from scripts.extract_tw_stock_data_v5 import main as extract_main
    except ImportError as e:
        logging.error(f"無法導入 extract_tw_stock_data_v5: {e}")
        logging.error("請確保專案結構正確")
        return 1

    # 修改 sys.argv 以傳遞參數給主程式
    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            'extract_tw_stock_data_v5.py',
            '--input-dir', args.input_dir,
            '--output-dir', args.output_dir,
            '--config', './configs/config_pro_v5_ml_optimal.yaml',
            '--make-npz'
        ]

        logging.info("\n開始執行數據抽取（使用修復後的代碼）...")
        logging.info("配置：config_pro_v5_ml_optimal.yaml (respect_day_boundary=true)")

        result = extract_main()

        if result == 0:
            logging.info("\n" + "="*60)
            logging.info("✅ 驗證完成！")
            logging.info("="*60)
            logging.info("請檢查輸出報告：")
            logging.info(f"  1. 標籤分布統計: {args.output_dir}/npz/normalization_meta.json")
            logging.info(f"  2. 震盪統計: {args.output_dir}/volatility_summary.json")
            logging.info("\n預期結果：")
            logging.info("  - Class 1 (持平) 比例應在 35-45%")
            logging.info("  - 觸發原因中 'time' 應佔一定比例")
            logging.info("  - 無跨日污染警告")
            logging.info("="*60)
        else:
            logging.error("驗證失敗，請檢查錯誤訊息")

        return result

    finally:
        # 恢復原始 argv
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
