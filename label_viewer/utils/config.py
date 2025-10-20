# -*- coding: utf-8 -*-
"""
config.py - 配置管理模組
========================================================
定義全局配置、路徑、顏色等常數
"""

import os
from pathlib import Path

# ============================================================
# 路徑配置
# ============================================================

# 自動檢測主專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent  # DeepLOB-Pro/

# 預設數據目錄
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed_v5" / "npz"

# 支援多個數據源（用於數據集切換）
DATA_SOURCES = {
    "v5": PROJECT_ROOT / "data" / "processed_v5" / "npz",
    "v5_balanced": PROJECT_ROOT / "data" / "processed_v5_balanced" / "npz",
    "v5-44.61": PROJECT_ROOT / "data" / "processed_v5-44.61" / "npz",
}

# ============================================================
# 標籤配置
# ============================================================

# 標籤名稱映射 {標籤值: 顯示名稱}
LABEL_NAMES = {
    0: '下跌',
    1: '持平',
    2: '上漲'
}

# 標籤顏色映射 {標籤值: 顏色代碼}
LABEL_COLORS = {
    0: '#e74c3c',  # 紅色（下跌）
    1: '#95a5a6',  # 灰色（持平）
    2: '#2ecc71'   # 綠色（上漲）
}

# ============================================================
# UI 配置
# ============================================================

# Dash 服務器配置
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8050
DEBUG_MODE = True

# 圖表配置
CHART_HEIGHT_MAIN = 500      # 主圖表高度（px）
CHART_HEIGHT_SECONDARY = 300  # 次級圖表高度（px）
CHART_TEMPLATE = 'plotly_white'  # 圖表主題

# 顯示配置
MAX_STOCKS_IN_DROPDOWN = 50  # 下拉選單最多顯示股票數
DEFAULT_TIME_RANGE_MAX = 1000  # 預設時間範圍最大值
DOWNSAMPLE_THRESHOLD = 5000  # 超過此點數進行降採樣

# ============================================================
# 數據處理配置
# ============================================================

# LRU 快取大小
CACHE_SIZE = 3  # 快取最多 3 個數據集

# Z-Score 反向轉換索引
# 根據 extract_tw_stock_data_v5.py 的特徵定義：
# feat = bids_p (0-4) + asks_p (5-9) + bids_q (10-14) + asks_q (15-19)
BID1_INDEX = 0  # 第 1 檔買價
ASK1_INDEX = 5  # 第 1 檔賣價

# ============================================================
# 應用程式資訊
# ============================================================

APP_TITLE = '訓練數據標籤互動查看器 - DeepLOB-Pro v2.0'
APP_VERSION = '2.0.0'
APP_DESCRIPTION = '互動式 Web 工具，用於查看訓練數據標籤正確性'
