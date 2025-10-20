@echo off
REM ============================================================================
REM quick_stats.bat - 快速震盪統計（Windows 批次腳本）
REM ============================================================================
REM 用途：一鍵執行震盪統計分析（不生成訓練數據）
REM
REM 使用方式：
REM   雙擊此檔案，或在命令列執行：quick_stats.bat
REM
REM 輸出：
REM   - data/volatility_stats/volatility_stats.csv
REM   - data/volatility_stats/volatility_summary.json
REM ============================================================================

chcp 65001 > nul
echo.
echo ============================================================
echo 快速震盪統計模式
echo ============================================================
echo.

REM 激活 conda 環境
call conda activate deeplob-pro
if errorlevel 1 (
    echo 錯誤：無法激活 conda 環境 'deeplob-pro'
    echo 請先執行：conda activate deeplob-pro
    pause
    exit /b 1
)

REM 執行快速統計
python scripts\quick_stats_only.py ^
    --input-dir .\data\temp ^
    --output-dir .\data\volatility_stats

REM 顯示結果
if errorlevel 1 (
    echo.
    echo ============================================================
    echo 執行失敗！請檢查錯誤訊息
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo 執行成功！
    echo ============================================================
    echo.
    echo 請查看以下檔案：
    echo   - data\volatility_stats\volatility_stats.csv
    echo   - data\volatility_stats\volatility_summary.json
    echo ============================================================
)

echo.
pause
