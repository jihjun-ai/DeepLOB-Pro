@echo off
REM ============================================================
REM V5 資料健康檢查快速腳本
REM ============================================================

echo ========================================
echo V5 資料健康檢查工具
echo ========================================
echo.

REM 預設檢查 processed_v5
echo [1/3] 檢查 processed_v5...
conda run -n deeplob-pro python scripts/check_data_health_v5.py --data-dir ./data/processed_v5/npz --save-report --verbose

echo.
echo ========================================
echo 檢查完成！請查看報告：
echo   - 控制台輸出
echo   - ./data/processed_v5/npz/health_report.json
echo ========================================
echo.

pause
