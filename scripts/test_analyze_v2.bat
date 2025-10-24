@echo off
REM 測試新版 analyze_label_distribution.py 的三種模式

echo ========================================
echo 測試智能標籤分布分析工具 v2.0
echo ========================================
echo.

REM 檢查 conda 環境
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [錯誤] 找不到 conda，請先安裝 Anaconda/Miniconda
    pause
    exit /b 1
)

REM 激活環境
echo [1/3] 激活 deeplob-pro 環境...
call conda activate deeplob-pro
if %ERRORLEVEL% NEQ 0 (
    echo [錯誤] 無法激活 deeplob-pro 環境
    pause
    exit /b 1
)
echo.

REM 測試 1: 基礎分析模式
echo ========================================
echo 測試 1: 基礎分析模式 (analyze)
echo ========================================
echo 分析所有預處理數據的標籤分布...
echo.
python scripts\analyze_label_distribution.py ^
    --preprocessed-dir data\preprocessed_v5 ^
    --mode analyze
echo.
echo 按任意鍵繼續測試 2...
pause >nul
echo.

REM 測試 2: 智能推薦模式
echo ========================================
echo 測試 2: 智能推薦模式 (smart_recommend)
echo ========================================
echo 自動生成候選方案並選擇最佳...
echo.
python scripts\analyze_label_distribution.py ^
    --preprocessed-dir data\preprocessed_v5 ^
    --mode smart_recommend ^
    --start-date 20250901 ^
    --target-dist "0.30,0.40,0.30" ^
    --min-samples 50000 ^
    --output results\dataset_selection_auto.json
echo.
echo 按任意鍵繼續測試 3...
pause >nul
echo.

REM 測試 3: 互動模式
echo ========================================
echo 測試 3: 互動模式 (interactive)
echo ========================================
echo 顯示候選方案讓你選擇...
echo.
python scripts\analyze_label_distribution.py ^
    --preprocessed-dir data\preprocessed_v5 ^
    --mode interactive ^
    --start-date 20250901 ^
    --target-dist "0.30,0.40,0.30" ^
    --min-samples 50000
echo.

echo ========================================
echo 測試完成！
echo ========================================
echo.
echo 結果文件：
echo   - results\dataset_selection_auto.json （智能推薦結果）
echo   - dataset_selection_*.json （互動選擇結果，如果有）
echo.
pause
