@echo off
REM 快速測試單一檔案預處理（驗證修改）
REM 完成時間: 2025-10-25

echo ========================================
echo 測試單一檔案預處理（驗證價格和成交量提取）
echo ========================================

REM 激活環境
call conda activate deeplob-pro

REM 測試單一檔案
python scripts\preprocess_single_day.py ^
    --data-file data\raw\tw_stock\20250909.txt ^
    --output-dir data\preprocessed_v5_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

echo.
echo ========================================
echo 測試完成！
echo ========================================
echo.
echo 檢查生成的 NPZ 檔案:
python -c "import numpy as np; import glob; files = glob.glob('data/preprocessed_v5_test/daily/20250909/*.npz'); print(f'生成檔案數: {len(files)}'); data = np.load(files[0], allow_pickle=True) if files else None; print(f'Keys: {data.files if data else None}'); print(f'last_prices 存在: {\"last_prices\" in data if data else False}'); print(f'last_volumes 存在: {\"last_volumes\" in data if data else False}'); print(f'total_volumes 存在: {\"total_volumes\" in data if data else False}')"

pause
