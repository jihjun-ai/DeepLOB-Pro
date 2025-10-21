@echo off
REM test_preprocess.bat - Test single file preprocessing
REM Usage: Run from project root directory D:\Case-New\python\DeepLOB-Pro

setlocal

set PROJECT_ROOT=%~dp0..
set TEST_FILE=%PROJECT_ROOT%\data\temp\20250901.txt
set OUTPUT_DIR=%PROJECT_ROOT%\data\preprocessed_v5_1hz_test
set CONFIG=%PROJECT_ROOT%\configs\config_pro_v5_ml_optimal.yaml
set CONDA_ENV=deeplob-pro

echo ============================================================
echo Test Single File Preprocessing
echo ============================================================
echo Test file: %TEST_FILE%
echo Output dir: %OUTPUT_DIR%
echo Conda env: %CONDA_ENV%
echo.

REM Check file exists
if not exist "%TEST_FILE%" (
    echo ERROR: Test file not found: %TEST_FILE%
    pause
    exit /b 1
)

REM Clean old test results
if exist "%OUTPUT_DIR%" (
    echo Cleaning old test results...
    rmdir /s /q "%OUTPUT_DIR%"
)

echo ============================================================
echo Starting preprocessing...
echo ============================================================
echo.

REM Activate conda and run
call conda activate %CONDA_ENV%

python "%PROJECT_ROOT%\scripts\preprocess_single_day.py" --input "%TEST_FILE%" --output-dir "%OUTPUT_DIR%" --config "%CONFIG%"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo Preprocessing FAILED
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Preprocessing SUCCESS
echo ============================================================
echo.

REM Check output
echo Checking output files...
echo.

if exist "%OUTPUT_DIR%\daily\20250901\summary.json" (
    echo Summary file generated
    type "%OUTPUT_DIR%\daily\20250901\summary.json"
) else (
    echo WARNING: Summary file not found
)

echo.
echo ============================================================
echo Test Complete
echo ============================================================

pause
endlocal
