@echo off
REM 測試完整數據流水線（預處理 + 訓練數據生成）

echo ============================================================
echo 測試完整數據流水線
echo ============================================================
echo.

call conda activate deeplob-pro

echo [步驟 1/2] 預處理單日數據（測試 mids=0 修復）
echo ------------------------------------------------------------
python scripts\preprocess_single_day.py ^
    --input data\temp\20250902.txt ^
    --output-dir data\preprocessed_v5_1hz_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 預處理失敗！
    pause
    exit /b 1
)

echo.
echo ✅ 預處理成功！
echo.

echo [步驟 2/2] 生成訓練數據（測試 NaN 處理）
echo ------------------------------------------------------------
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir data\preprocessed_v5_1hz_test ^
    --output-dir data\processed_v6_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 訓練數據生成失敗！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✅ 完整流水線測試成功！
echo ============================================================
echo.
echo 輸出位置:
echo   - 預處理數據: data\preprocessed_v5_1hz_test\
echo   - 訓練數據: data\processed_v6_test\npz\
echo.

pause
