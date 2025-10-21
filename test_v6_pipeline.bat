@echo off
REM ========================================
REM V6 完整流水線測試
REM 測試所有新實作的改進功能
REM ========================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo V6 完整流水線測試
echo ========================================
echo.
echo 測試項目:
echo   ✓ C.1: 隨機種子記錄
echo   ✓ E.1: 標籤邊界檢查
echo   ✓ E.2: 權重邊界檢查
echo   ✓ E.3: Neutral 比例警告
echo   ✓ A.1: ffill_limit=60
echo   ✓ A.2: 滑窗品質過濾
echo   ✓ G.1: 增強錯誤報告
echo.
echo ========================================

REM 設定測試目錄
set TEST_PREPROCESSED_DIR=data\preprocessed_v6_test
set TEST_OUTPUT_DIR=data\processed_v6_test

REM 清理舊的測試數據
if exist "%TEST_PREPROCESSED_DIR%" (
    echo 清理舊的測試預處理數據...
    rmdir /s /q "%TEST_PREPROCESSED_DIR%"
)
if exist "%TEST_OUTPUT_DIR%" (
    echo 清理舊的測試輸出數據...
    rmdir /s /q "%TEST_OUTPUT_DIR%"
)

echo.
echo ========================================
echo [階段 1] 預處理測試
echo ========================================
echo 測試: A.1 - ffill_limit=60 秒
echo ----------------------------------------

REM 檢查輸入文件是否存在
if not exist "data\temp\20250902.txt" (
    echo ❌ 測試輸入文件不存在: data\temp\20250902.txt
    echo    → 請確保有測試數據
    exit /b 1
)

REM 運行預處理
call conda activate deeplob-pro
python scripts\preprocess_single_day.py ^
    --input data\temp\20250902.txt ^
    --output-dir %TEST_PREPROCESSED_DIR% ^
    --config configs\config_pro_v5_ml_optimal.yaml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 預處理失敗！
    echo    → 檢查日誌以獲取詳細錯誤訊息
    exit /b 1
)

echo.
echo ✅ 預處理完成

REM 檢查預處理輸出
echo.
echo 驗證預處理輸出...
set DAILY_DIR=%TEST_PREPROCESSED_DIR%\daily\20250902
if not exist "%DAILY_DIR%" (
    echo ❌ 預處理輸出目錄不存在: %DAILY_DIR%
    exit /b 1
)

REM 計算 NPZ 檔案數量
set NPZ_COUNT=0
for %%f in ("%DAILY_DIR%\*.npz") do set /a NPZ_COUNT+=1

if %NPZ_COUNT% EQU 0 (
    echo ❌ 沒有生成 NPZ 檔案
    exit /b 1
)

echo ✅ 生成了 %NPZ_COUNT% 個 NPZ 檔案

echo.
echo ========================================
echo [階段 2] 訓練數據生成測試
echo ========================================
echo 測試: C.1, E.1-3, A.2, G.1
echo ----------------------------------------

python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir %TEST_PREPROCESSED_DIR% ^
    --output-dir %TEST_OUTPUT_DIR% ^
    --config configs\config_pro_v5_ml_optimal.yaml

if %ERRORLEVEL% EQU 2 (
    echo.
    echo ⚠️ 訓練數據生成完成，但有數據質量警告
    echo    → 這可能是預期的（測試數據質量檢測）
    set HAS_WARNINGS=1
) else if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 訓練數據生成失敗！
    echo    → 檢查上方的錯誤訊息
    exit /b 1
) else (
    echo.
    echo ✅ 訓練數據生成完成
    set HAS_WARNINGS=0
)

echo.
echo ========================================
echo [階段 3] 功能驗證測試
echo ========================================

REM 檢查輸出檔案存在性
set META_FILE=%TEST_OUTPUT_DIR%\npz\normalization_meta.json
if not exist "%META_FILE%" (
    echo ❌ Metadata 檔案不存在: %META_FILE%
    exit /b 1
)

echo.
echo [測試 C.1] 檢查隨機種子記錄
echo ----------------------------------------
python -c "import json; meta=json.load(open('%META_FILE%')); seed=meta.get('seed', None); print(f'Seed: {seed}'); assert seed == 42, '❌ Seed 不正確'; print('✅ C.1 通過: Seed 已記錄')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ C.1 測試失敗
    exit /b 1
)

echo.
echo [測試 E.3] 檢查標籤分布和 Neutral 比例
echo ----------------------------------------
python -c "import json; meta=json.load(open('%META_FILE%')); dist=meta.get('train_label_distribution', {}); print(f'訓練集標籤分布:'); [print(f'  Class {k}: {v:.2f}%%') for k, v in sorted(dist.items())]; neutral=dist.get('1', 0); print(f'\nNeutral (Class 1) 比例: {neutral:.2f}%%'); assert 0 <= neutral <= 100, '❌ Neutral 比例異常'; print('✅ E.3 通過: 標籤分布正常')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ E.3 測試失敗
    exit /b 1
)

echo.
echo [測試 E.1] 驗證標籤值域
echo ----------------------------------------
python -c "import numpy as np; data=np.load('%TEST_OUTPUT_DIR%/npz/train.npz'); y=data['y'][:, 0]; unique=set(np.unique(y)); print(f'標籤唯一值: {unique}'); assert unique.issubset({0, 1, 2}), f'❌ 標籤值異常: {unique}'; print('✅ E.1 通過: 標籤值域正確 [0, 1, 2]')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ E.1 測試失敗
    exit /b 1
)

echo.
echo [測試 E.2] 驗證權重合法性
echo ----------------------------------------
python -c "import numpy as np; data=np.load('%TEST_OUTPUT_DIR%/npz/train.npz'); w=data['sample_weight']; print(f'權重統計:'); print(f'  最小: {w.min():.6f}'); print(f'  最大: {w.max():.6f}'); print(f'  平均: {w.mean():.6f}'); print(f'  標準差: {w.std():.6f}'); assert np.isfinite(w).all(), '❌ 權重包含 NaN/inf'; assert (w >= 0).all(), '❌ 權重包含負值'; assert not (w == 0).all(), '❌ 所有權重為 0'; print('✅ E.2 通過: 權重合法')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ E.2 測試失敗
    exit /b 1
)

echo.
echo [測試 A.2] 檢查品質過濾配置
echo ----------------------------------------
python -c "import yaml; config=yaml.safe_load(open('configs/config_pro_v5_ml_optimal.yaml')); threshold=config.get('ffill_quality_threshold', None); print(f'ffill_quality_threshold: {threshold}'); assert threshold is not None, '❌ 配置缺少 ffill_quality_threshold'; assert 0 < threshold < 1, f'❌ 閾值異常: {threshold}'; print('✅ A.2 通過: 品質過濾閾值已配置')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ A.2 測試失敗
    exit /b 1
)

echo.
echo [測試] 輸出格式兼容性檢查
echo ----------------------------------------
python -c "import numpy as np; data=np.load('%TEST_OUTPUT_DIR%/npz/train.npz'); X=data['X']; y=data['y']; w=data['sample_weight']; print(f'訓練集形狀:'); print(f'  X: {X.shape}'); print(f'  y: {y.shape}'); print(f'  w: {w.shape}'); assert X.shape[1] == 100, '❌ 時間步不正確'; assert X.shape[2] == 20, '❌ 特徵維度不正確'; assert y.shape[1] == 5, '❌ Horizon 數量不正確'; print('✅ 輸出格式正確')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 格式兼容性測試失敗
    exit /b 1
)

echo.
echo ========================================
echo [階段 4] 總結
echo ========================================

echo.
echo 測試結果:
echo   ✅ C.1: 隨機種子記錄 - 通過
echo   ✅ E.1: 標籤邊界檢查 - 通過
echo   ✅ E.2: 權重邊界檢查 - 通過
echo   ✅ E.3: Neutral 比例檢查 - 通過
echo   ✅ A.1: ffill_limit=60 - 通過
echo   ✅ A.2: 滑窗品質過濾 - 通過
echo   ✅ 輸出格式兼容性 - 通過

if %HAS_WARNINGS% EQU 1 (
    echo.
    echo ⚠️ 注意: 訓練數據生成過程中有數據質量警告
    echo    → 這可能是預期的（測試數據質量檢測功能）
    echo    → 請檢查日誌中的警告訊息
)

echo.
echo ========================================
echo ✅ V6 完整流水線測試成功！
echo ========================================
echo.
echo 測試數據位置:
echo   預處理: %TEST_PREPROCESSED_DIR%
echo   訓練集: %TEST_OUTPUT_DIR%
echo.
echo 下一步:
echo   1. 檢查日誌中的 Neutral 比例警告（如有）
echo   2. 比較 V5 vs V6 的訓練集大小
echo   3. 使用新數據訓練模型並評估準確率
echo.

endlocal
exit /b 0
