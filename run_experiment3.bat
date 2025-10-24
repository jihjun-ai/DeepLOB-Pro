@echo off
REM 實驗 3: 中庸之道
REM 日期: 2025-10-24

echo ========================================
echo 實驗 3: 中庸之道
echo ========================================
echo.
echo 實驗總結:
echo   實驗 1: 過擬合 (Train 61%% vs Val 44%%, Gap 17%%)
echo   實驗 2: 欠擬合 (Train 43%% vs Val 44%%, Gap -0.5%%)
echo   實驗 3: 找平衡點！
echo.
echo 配置變更 (相對實驗 2):
echo - 模型容量: LSTM/FC 24 -^> 28 (折衷)
echo - Dropout: 0.8 -^> 0.7 (降低正則化)
echo - Learning Rate: 3e-6 -^> 5e-6 (加快學習)
echo - Weight Decay: 0.01 -^> 0.005 (減半)
echo - Grad Clip: 0.5 -^> 0.6 (稍微放寬)
echo - Epochs: 15 -^> 20 (延長訓練)
echo - Patience: 3 -^> 5 (更寬容)
echo - Min Delta: 0.001 -^> 0.0005 (降低門檻)
echo.
echo 預期目標:
echo - Train Acc: 48-52%%
echo - Val Acc: 45-48%%
echo - Train-Val Gap: 3-5%% (健康範圍)
echo - Class 1 Recall: ^>30%% (改善持平類)
echo.
echo ========================================

REM 啟動訓練
call conda activate deeplob-pro
python scripts/train_deeplob_v5.py --config configs/train_v5.yaml

echo.
echo ========================================
echo 訓練完成！
echo 請將結果更新到: docs\20251024-deeplob調參歷史.md
echo   實驗 3 結果欄位
echo ========================================
pause
