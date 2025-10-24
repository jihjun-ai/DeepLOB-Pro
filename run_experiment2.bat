@echo off
REM 實驗 2: 激進抑制過擬合
REM 日期: 2025-10-24

echo ========================================
echo 實驗 2: 激進抑制過擬合
echo ========================================
echo.
echo 配置變更:
echo - 模型容量: LSTM/FC 32 -^> 24
echo - Dropout: 0.7 -^> 0.8
echo - Learning Rate: 4e-6 -^> 3e-6
echo - Weight Decay: 0.001 -^> 0.01
echo - Grad Clip: 0.8 -^> 0.5
echo - Epochs: 30 -^> 15
echo - Patience: 5 -^> 3
echo.
echo 預期目標:
echo - Val Acc: 47-50%% (提升 2-5%%)
echo - Train-Val Gap: 5-7%% (縮小)
echo - Grad Norm: ^<5.0 (穩定)
echo.
echo ========================================

REM 啟動訓練
call conda activate deeplob-pro
python scripts/train_deeplob_generic.py --config configs/train_v5.yaml

echo.
echo ========================================
echo 訓練完成！
echo 請將結果更新到: docs\20251024-deeplob調參歷史.md
echo ========================================
pause
