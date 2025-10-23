@echo off
REM quick_test_label_viewer.bat - Quick test of label viewer with trend_stable
REM =============================================================================
REM
REM 此腳本執行：
REM 1. 預處理一天數據（使用 trend_stable 標籤方法）
REM 2. 啟動 label viewer 查看結果
REM
REM 使用方式：
REM   scripts\quick_test_label_viewer.bat
REM
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo Quick Test: Label Viewer with Trend Stable Labels
echo ============================================================
echo.

REM 配置
set PROJECT_ROOT=%~dp0..
set INPUT_FILE=%PROJECT_ROOT%\data\temp\20240930.txt
set OUTPUT_DIR=%PROJECT_ROOT%\data\preprocessed_v5_test
set CONFIG=%PROJECT_ROOT%\configs\config_pro_v5_ml_optimal.yaml
set CONDA_ENV=deeplob-pro

REM 檢查輸入文件
if not exist "%INPUT_FILE%" (
    echo ERROR: Input file not found: %INPUT_FILE%
    echo.
    echo Please ensure you have a test file at:
    echo   data\temp\20240930.txt
    echo.
    pause
    exit /b 1
)

echo Input file: %INPUT_FILE%
echo Output dir: %OUTPUT_DIR%
echo Config: %CONFIG%
echo.
echo Press any key to continue...
pause >nul
echo.

REM 激活 conda 環境
call conda activate %CONDA_ENV%

REM 步驟 1: 預處理一天數據
echo ============================================================
echo Step 1: Preprocessing (using trend_stable labeling method)
echo ============================================================
echo.

python "%PROJECT_ROOT%\scripts\preprocess_single_day.py" ^
    --input "%INPUT_FILE%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --config "%CONFIG%"

if errorlevel 1 (
    echo.
    echo ERROR: Preprocessing failed!
    pause
    exit /b 1
)

echo.
echo [OK] Preprocessing completed!
echo.

REM 檢查輸出目錄
set DATE_DIR=%OUTPUT_DIR%\daily\20240930
if not exist "%DATE_DIR%" (
    echo ERROR: Output directory not found: %DATE_DIR%
    pause
    exit /b 1
)

REM 統計輸出文件
set npz_count=0
for %%f in ("%DATE_DIR%\*.npz") do set /a npz_count+=1

echo Generated %npz_count% stock NPZ files in:
echo   %DATE_DIR%
echo.

if %npz_count%==0 (
    echo WARNING: No NPZ files generated!
    pause
    exit /b 1
)

REM 顯示前 3 個文件
echo Sample files:
set count=0
for %%f in ("%DATE_DIR%\*.npz") do (
    if !count! LSS 3 (
        echo   - %%~nxf
        set /a count+=1
    )
)
echo.

REM 步驟 2: 啟動 Label Viewer
echo ============================================================
echo Step 2: Launching Label Viewer
echo ============================================================
echo.
echo The label viewer will open in your browser at:
echo   http://localhost:8051
echo.
echo In the web interface:
echo   1. The path is already filled: data/preprocessed_v5_test/daily/20240930
echo   2. Click "載入目錄" button
echo   3. Select a stock from dropdown to view labels
echo   4. Check the labeling_method in metadata (should be "trend_stable")
echo.
echo Press Ctrl+C in this window to stop the server.
echo.
echo Press any key to launch...
pause >nul
echo.

cd "%PROJECT_ROOT%\label_viewer"
python app_preprocessed.py

REM Cleanup (optional)
echo.
echo Label viewer stopped.
echo.

endlocal
exit /b 0
