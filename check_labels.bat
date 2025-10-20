@echo off
REM 訓練數據標籤視覺化檢查批次腳本
REM 用於快速檢查 extract_tw_stock_data_v5.py 產生的訓練數據標籤正確性

echo ============================================================
echo 訓練數據標籤視覺化工具
echo ============================================================
echo.

REM 檢查參數
if "%1"=="" (
    echo 使用方式:
    echo   check_labels.bat [data_dir] [split] [n_stocks]
    echo.
    echo 範例:
    echo   check_labels.bat data/processed_v5/npz train 5
    echo   check_labels.bat data/processed_v5_balanced/npz val 3
    echo.
    echo 預設使用: data/processed_v5/npz train 3
    echo.
    set DATA_DIR=data/processed_v5/npz
    set SPLIT=train
    set N_STOCKS=3
) else (
    set DATA_DIR=%1
    set SPLIT=%2
    set N_STOCKS=%3
)

REM 設定預設值
if "%SPLIT%"=="" set SPLIT=train
if "%N_STOCKS%"=="" set N_STOCKS=3

echo 配置:
echo   數據目錄: %DATA_DIR%
echo   數據集:   %SPLIT%
echo   股票數:   %N_STOCKS%
echo.
echo 開始檢查...
echo ============================================================
echo.

REM 啟動 conda 環境並執行腳本
conda activate deeplob-pro
python scripts/visualize_training_labels.py ^
    --data-dir ./%DATA_DIR% ^
    --split %SPLIT% ^
    --n-stocks %N_STOCKS% ^
    --max-points 500

echo.
echo ============================================================
echo 完成！圖表已保存至 results/label_visualization/%SPLIT%
echo ============================================================
echo.
echo 請檢查以下文件:
echo   1. overall_statistics.png - 整體統計圖
echo   2. stock_*.png - 個股詳細圖
echo.

pause
