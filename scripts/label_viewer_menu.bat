@echo off
setlocal enabledelayedexpansion
REM ========================================
REM Label Viewer 統合工具選單
REM ========================================
REM 功能：整合所有 Label Viewer 相關功能
REM 作者：DeepLOB-Pro Team
REM 日期：2025-10-23
REM ========================================

:MENU
cls
echo.
echo ========================================
echo Label Viewer 工具選單
echo ========================================
echo.
echo 請選擇操作：
echo.
echo [1] 啟動 Label Viewer (查看已有數據)
echo [2] 快速測試 (預處理+查看) - 使用 trend_stable
echo [3] 檢查環境與數據
echo [4] 查看使用說明
echo [0] 退出
echo.
echo ========================================
echo.

set /p choice="請輸入選項 (0-4): "

if "%choice%"=="1" goto RUN_VIEWER
if "%choice%"=="2" goto QUICK_TEST
if "%choice%"=="3" goto CHECK_ENV
if "%choice%"=="4" goto SHOW_GUIDE
if "%choice%"=="0" goto END

echo.
echo [錯誤] 無效的選項
timeout /t 2 >nul
goto MENU

REM ========================================
REM 選項 1: 啟動 Label Viewer
REM ========================================
:RUN_VIEWER
cls
echo.
echo ========================================
echo 啟動 Label Viewer
echo ========================================
echo.

REM 檢查可用的數據目錄
set DATA_FOUND=0

if exist "data\preprocessed_swing\daily" (
    set DATA_FOUND=1
    set DEFAULT_PATH=data/preprocessed_swing/daily
    echo [找到] data\preprocessed_swing\daily
)

if exist "data\preprocessed_v5_1hz\daily" (
    set DATA_FOUND=1
    if not defined DEFAULT_PATH set DEFAULT_PATH=data/preprocessed_v5_1hz/daily
    echo [找到] data\preprocessed_v5_1hz\daily
)

if exist "data\preprocessed_v5_test\daily" (
    set DATA_FOUND=1
    if not defined DEFAULT_PATH set DEFAULT_PATH=data/preprocessed_v5_test/daily
    echo [找到] data\preprocessed_v5_test\daily
)

if %DATA_FOUND%==0 (
    echo.
    echo [警告] 沒有找到預處理數據目錄！
    echo.
    echo 請先執行以下操作之一：
    echo   1. 執行 scripts\batch_preprocess.bat (批次預處理所有數據)
    echo   2. 選擇選項 [2] 快速測試 (預處理單天數據)
    echo.
    pause
    goto MENU
)

echo.
echo 預設目錄: %DEFAULT_PATH%
echo.
echo 按任意鍵啟動 Label Viewer...
pause >nul

call "%~dp0run_label_viewer.bat"
goto MENU

REM ========================================
REM 選項 2: 快速測試 (預處理+查看)
REM ========================================
:QUICK_TEST
cls
echo.
echo ========================================
echo 快速測試 (預處理 + Label Viewer)
echo ========================================
echo.
echo 此選項會：
echo   1. 預處理一天的測試數據 (20240930)
echo   2. 使用 trend_stable 標籤方法
echo   3. 自動啟動 Label Viewer 查看結果
echo.

REM 檢查測試文件
set TEST_FILE=data\temp\20240930.txt
if not exist "%TEST_FILE%" (
    echo [錯誤] 測試文件不存在: %TEST_FILE%
    echo.
    echo 請確保有測試數據文件或使用其他選項
    echo.
    pause
    goto MENU
)

echo 測試文件: %TEST_FILE%
echo.
echo 按任意鍵繼續...
pause >nul

call "%~dp0quick_test_label_viewer.bat"
goto MENU

REM ========================================
REM 選項 3: 檢查環境與數據
REM ========================================
:CHECK_ENV
cls
echo.
echo ========================================
echo 環境與數據檢查
echo ========================================
echo.

REM 檢查 Conda 環境
echo [檢查] Conda 環境...
call conda info --envs | findstr "deeplob-pro" >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 deeplob-pro 環境
    echo        請先創建環境
) else (
    echo [OK] deeplob-pro 環境存在
)
echo.

REM 檢查必要文件
echo [檢查] 必要文件...
set FILES_OK=1

if exist "label_viewer\app_preprocessed.py" (
    echo [OK] label_viewer\app_preprocessed.py
) else (
    echo [錯誤] 缺少 app_preprocessed.py
    set FILES_OK=0
)

if exist "label_viewer\utils\preprocessed_loader.py" (
    echo [OK] preprocessed_loader.py
) else (
    echo [錯誤] 缺少 preprocessed_loader.py
    set FILES_OK=0
)

if exist "docs\LABEL_VIEWER_GUIDE.md" (
    echo [OK] LABEL_VIEWER_GUIDE.md
) else (
    echo [錯誤] 缺少使用說明
    set FILES_OK=0
)
echo.

REM 檢查數據目錄
echo [檢查] 數據目錄...
set DATA_DIRS=0

if exist "data\preprocessed_swing\daily" (
    set /a DATA_DIRS+=1
    echo [OK] data\preprocessed_swing\daily

    REM 統計日期數
    set date_count=0
    for /d %%d in ("data\preprocessed_swing\daily\*") do set /a date_count+=1
    echo      找到 !date_count! 個交易日
)

if exist "data\preprocessed_v5_1hz\daily" (
    set /a DATA_DIRS+=1
    echo [OK] data\preprocessed_v5_1hz\daily

    set date_count=0
    for /d %%d in ("data\preprocessed_v5_1hz\daily\*") do set /a date_count+=1
    echo      找到 !date_count! 個交易日
)

if exist "data\preprocessed_v5_test\daily" (
    set /a DATA_DIRS+=1
    echo [OK] data\preprocessed_v5_test\daily

    set date_count=0
    for /d %%d in ("data\preprocessed_v5_test\daily\*") do set /a date_count+=1
    echo      找到 !date_count! 個交易日
)

if %DATA_DIRS%==0 (
    echo [警告] 沒有找到預處理數據
    echo        請執行 batch_preprocess.bat
)
echo.

REM 摘要
echo ========================================
echo 檢查摘要
echo ========================================
if %FILES_OK%==1 (
    echo 必要文件: [OK]
) else (
    echo 必要文件: [錯誤]
)

if %DATA_DIRS% GTR 0 (
    echo 數據目錄: [OK] 找到 %DATA_DIRS% 個目錄
) else (
    echo 數據目錄: [警告] 沒有數據
)
echo.

pause
goto MENU

REM ========================================
REM 選項 4: 查看使用說明
REM ========================================
:SHOW_GUIDE
cls
echo.
echo ========================================
echo Label Viewer 使用說明
echo ========================================
echo.
echo 完整說明文檔位於:
echo   docs\LABEL_VIEWER_GUIDE.md
echo.
echo 快速開始:
echo.
echo 1. 啟動應用
echo    - 選擇選項 [1] 或執行 scripts\run_label_viewer.bat
echo    - 瀏覽器訪問 http://localhost:8051
echo.
echo 2. 載入數據
echo    - 輸入日期目錄路徑 (例如: data/preprocessed_swing/daily/20250901)
echo    - 點擊「載入目錄」按鈕
echo.
echo 3. 查看股票
echo    - 選擇「全部股票」查看整體統計
echo    - 選擇單一股票查看詳細資料
echo.
echo 4. 視覺化選項
echo    - 中間價折線圖 (含標籤疊加)
echo    - 標籤預覽分布 (柱狀圖/圓餅圖)
echo    - 元數據表格
echo.
echo 標籤顏色:
echo   - 紅色 = 下跌 (-1)
echo   - 灰色 = 持平 (0)
echo   - 綠色 = 上漲 (1)
echo.
echo 支持的標籤方法:
echo   - triple_barrier: Triple-Barrier 方法 (高頻交易)
echo   - trend_adaptive: 趨勢標籤 (自適應版)
echo   - trend_stable: 趨勢標籤 (穩定版，推薦)
echo.
echo 按任意鍵返回選單...
pause >nul
goto MENU

REM ========================================
REM 退出
REM ========================================
:END
echo.
echo 再見！
echo.
endlocal
exit /b 0
