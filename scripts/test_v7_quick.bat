@echo off
REM Quick test for extract_tw_stock_data_v7.py
REM Tests with a small subset of data

echo ========================================
echo Extract V7 Quick Test
echo ========================================
echo.

REM Activate environment
call C:\ProgramData\miniconda3\Scripts\activate.bat deeplob-pro

REM Run V7 with test output
python scripts\extract_tw_stock_data_v7.py ^
    --preprocessed-dir data\preprocessed_swing ^
    --output-dir data\processed_v7_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

echo.
echo ========================================
echo Test completed. Check output above for:
echo   - Labels reused count
echo   - Reuse rate percentage
echo   - Expected time savings
echo ========================================

pause
