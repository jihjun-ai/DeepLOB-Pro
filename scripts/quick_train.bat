@echo off
REM Quick training script for train_sb3_deeplob.py
REM Usage: quick_train.bat [test|full]

echo ============================================================
echo PPO + DeepLOB Quick Training Script
echo ============================================================
echo.

REM Activate conda environment
call conda activate deeplob-pro

REM Check command line argument
if "%1"=="test" (
    echo Mode: TEST ^(10K steps^)
    python scripts\train_sb3_deeplob.py --test
) else if "%1"=="full" (
    echo Mode: FULL ^(1M steps^)
    python scripts\train_sb3_deeplob.py
) else (
    echo Usage: quick_train.bat [test^|full]
    echo.
    echo Options:
    echo   test  - Quick test ^(10K steps, ~10 minutes^)
    echo   full  - Full training ^(1M steps, ~4-8 hours^)
    echo.
    echo Example:
    echo   quick_train.bat test
    echo   quick_train.bat full
    exit /b 1
)

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
