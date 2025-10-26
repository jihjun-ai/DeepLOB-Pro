@echo off
REM TensorBoard 分析工具快速測試腳本
REM
REM 此腳本用於測試 analyze_tensorboard.py 是否正常工作

echo ======================================================================
echo TensorBoard 分析工具測試
echo ======================================================================
echo.

REM 檢查是否有訓練日誌
if not exist "logs\sb3_deeplob\" (
    echo [錯誤] 找不到訓練日誌目錄: logs\sb3_deeplob\
    echo.
    echo 請先運行訓練:
    echo   python scripts\train_sb3_deeplob.py --test
    echo.
    pause
    exit /b 1
)

REM 找到最新的訓練日誌
for /f "delims=" %%i in ('dir /b /ad /o-d "logs\sb3_deeplob\PPO_*" 2^>nul') do (
    set "latest_log=%%i"
    goto :found
)

:found
if not defined latest_log (
    echo [錯誤] 找不到任何 TensorBoard 日誌
    echo.
    echo 請先運行訓練:
    echo   python scripts\train_sb3_deeplob.py --test
    echo.
    pause
    exit /b 1
)

echo [信息] 找到訓練日誌: %latest_log%
echo.

REM 創建輸出目錄
if not exist "results" mkdir results

REM 運行分析
echo [執行] 分析 TensorBoard 日誌...
echo.

python scripts\analyze_tensorboard.py --logdir "logs\sb3_deeplob\%latest_log%" --output "results\test_analysis" --format both --verbose

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo 測試完成！
    echo ======================================================================
    echo.
    echo 生成的文件:
    echo   - results\test_analysis.json
    echo   - results\test_analysis.md
    echo.
    echo 下一步:
    echo   1. 查看 Markdown 報告: type results\test_analysis.md
    echo   2. 將 JSON 提供給 AI 分析
    echo   3. 根據建議調整配置
    echo.
) else (
    echo.
    echo [錯誤] 分析失敗，請檢查錯誤訊息
    echo.
)

pause
