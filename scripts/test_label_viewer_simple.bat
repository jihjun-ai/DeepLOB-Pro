@echo off
REM Label Viewer 簡易測試腳本
REM 功能：測試 Label Viewer 是否可以正常啟動

echo.
echo ========================================
echo Label Viewer 測試
echo ========================================
echo.

REM 啟動環境
call conda activate deeplob-pro
if errorlevel 1 (
    echo [錯誤] 無法啟動 conda 環境
    pause
    exit /b 1
)

REM 切換到專案根目錄
cd /d "%~dp0.."

REM 檢查必要文件
echo [檢查] 必要文件...
if not exist "label_viewer\app_preprocessed.py" (
    echo [錯誤] 找不到 label_viewer\app_preprocessed.py
    pause
    exit /b 1
)
echo [OK] label_viewer\app_preprocessed.py

if not exist "label_viewer\utils\preprocessed_loader.py" (
    echo [錯誤] 找不到 preprocessed_loader.py
    pause
    exit /b 1
)
echo [OK] preprocessed_loader.py

if not exist "scripts\run_label_viewer.bat" (
    echo [錯誤] 找不到 run_label_viewer.bat
    pause
    exit /b 1
)
echo [OK] run_label_viewer.bat

REM 檢查數據目錄
echo.
echo [檢查] 數據目錄...
if exist "data\preprocessed_swing\daily" (
    echo [OK] data\preprocessed_swing\daily 存在
    dir /b "data\preprocessed_swing\daily" | find /c /v "" > nul
    if not errorlevel 1 (
        echo [OK] 找到預處理數據
    )
) else if exist "data\preprocessed_v5_1hz\daily" (
    echo [OK] data\preprocessed_v5_1hz\daily 存在
) else (
    echo [警告] 沒有找到預處理數據目錄
    echo        請先執行 batch_preprocess.bat
)

echo.
echo ========================================
echo 測試完成
echo ========================================
echo.
echo 啟動 Label Viewer:
echo   scripts\run_label_viewer.bat
echo.
pause
