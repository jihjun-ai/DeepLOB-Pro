@echo off
REM 測試修復後的 V6 數據生成流程

call conda activate deeplob-pro

python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir data\preprocessed_v5_1hz ^
    --output-dir data\processed_v6_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

pause
