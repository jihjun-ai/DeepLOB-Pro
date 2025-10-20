@echo off
REM Label Viewer 快速啟動腳本 (Windows)
REM 作者：DeepLOB-Pro Team
REM 最後更新：2025-10-20

echo ========================================
echo   Label Viewer 啟動腳本
echo ========================================
echo.

REM 檢查 Python 是否已安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 未檢測到 Python，請先安裝 Python 3.11+
    pause
    exit /b 1
)

echo [資訊] Python 版本:
python --version
echo.

REM 檢查是否在 conda 環境中
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [資訊] 檢測到 Conda，建議使用專案環境:
    echo   conda activate deeplob-pro
    echo.
)

REM 檢查依賴是否已安裝
echo [資訊] 檢查依賴套件...
python -c "import dash" >nul 2>&1
if errorlevel 1 (
    echo [警告] 缺少必要套件，正在安裝...
    pip install -r requirements.txt
    echo.
)

REM 啟動應用
echo [資訊] 啟動 Label Viewer...
echo [資訊] 應用將在瀏覽器中開啟: http://localhost:8050
echo.
echo 按 Ctrl+C 停止服務器
echo ========================================
echo.

python app.py

pause
