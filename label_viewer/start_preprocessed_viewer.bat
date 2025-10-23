@echo off
chcp 65001 >nul
call conda activate deeplob-pro
echo ========================================
echo   Label Viewer - 預處理數據模式
echo ========================================
echo.
echo 功能: 查看 preprocess_single_day.py 產生的數據和標籤預覽
echo.
echo 應用網址: http://localhost:8051
echo 按 Ctrl+C 停止
echo.
echo ========================================
echo.

python app_preprocessed.py

pause
