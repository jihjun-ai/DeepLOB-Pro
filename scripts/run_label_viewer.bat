@echo off
REM ========================================
REM Label Viewer 啟動腳本
REM ========================================
REM 功能：啟動標籤查看器應用程式
REM 作者：DeepLOB-Pro Team
REM 日期：2025-10-23
REM ========================================

echo.
echo ========================================
echo Label Viewer 啟動中...
echo ========================================
echo.

REM 啟動 Conda 環境
call conda activate deeplob-pro
if errorlevel 1 (
    echo [錯誤] 無法啟動 conda 環境 deeplob-pro
    echo 請確保已安裝並配置 Conda 環境
    pause
    exit /b 1
)

REM 切換到專案根目錄
cd /d "%~dp0.."

REM 檢查應用程式文件
if not exist "label_viewer\app_preprocessed.py" (
    echo [錯誤] 找不到 label_viewer\app_preprocessed.py
    pause
    exit /b 1
)

echo [資訊] 環境：deeplob-pro
echo [資訊] 應用：label_viewer/app_preprocessed.py
echo [資訊] 埠號：8051
echo.
echo 應用啟動後，請在瀏覽器訪問：
echo   http://localhost:8051
echo.
echo 按 Ctrl+C 停止應用
echo ========================================
echo.

REM 啟動應用
python label_viewer\app_preprocessed.py

if errorlevel 1 (
    echo.
    echo [錯誤] 應用啟動失敗
    pause
    exit /b 1
)

pause
