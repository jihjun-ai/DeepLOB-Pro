@echo off
chcp 65001 >nul
call conda activate deeplob-pro
echo ========================================
echo   Label Viewer - Starting...
echo ========================================
echo.
echo Application URL: http://localhost:8050
echo Press Ctrl+C to stop
echo.
echo ========================================
echo.

python app.py

pause
