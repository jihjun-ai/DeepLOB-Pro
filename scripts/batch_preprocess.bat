@echo off
REM batch_preprocess.bat - Batch preprocessing script
REM Usage: Run from project root directory D:\Case-New\python\DeepLOB-Pro

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_ROOT=%~dp0..
set INPUT_DIR=%PROJECT_ROOT%\data\temp
set OUTPUT_DIR=%PROJECT_ROOT%\data\preprocessed_v5_1hz
set CONFIG=%PROJECT_ROOT%\configs\config_pro_v5_ml_optimal.yaml
set CONDA_ENV=deeplob-pro

echo ============================================================
echo Batch Preprocessing Script
echo ============================================================
echo Input dir: %INPUT_DIR%
echo Output dir: %OUTPUT_DIR%
echo Config: %CONFIG%
echo ============================================================
echo.

REM Check input directory
if not exist "%INPUT_DIR%" (
    echo ERROR: Input directory not found: %INPUT_DIR%
    pause
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Count files
set total_files=0
for %%f in ("%INPUT_DIR%\*.txt") do (
    set /a total_files+=1
)

if %total_files%==0 (
    echo ERROR: No .txt files found in %INPUT_DIR%
    pause
    exit /b 1
)

echo Found %total_files% files to process
echo.

REM Activate conda and process files
call conda activate %CONDA_ENV%

set processed_files=0
set failed_files=0

for %%f in ("%INPUT_DIR%\*.txt") do (
    set /a processed_files+=1
    set filename=%%~nxf

    echo [!processed_files!/%total_files%] Processing: !filename!

    python "%PROJECT_ROOT%\scripts\preprocess_single_day.py" --input "%%f" --output-dir "%OUTPUT_DIR%" --config "%CONFIG%"

    if errorlevel 1 (
        echo FAILED: !filename!
        set /a failed_files+=1
    ) else (
        echo SUCCESS: !filename!
    )
    echo.
)

REM Generate report
echo.
echo ============================================================
echo Generating overall report...
echo ============================================================
echo.

python "%PROJECT_ROOT%\scripts\generate_preprocessing_report.py" --preprocessed-dir "%OUTPUT_DIR%"

if errorlevel 1 (
    echo WARNING: Report generation failed
) else (
    echo Report generated successfully
)

REM Summary
set /a success_files=processed_files-failed_files

echo.
echo ============================================================
echo Batch Processing Complete
echo ============================================================
echo Total files: %total_files%
echo Success: %success_files%
echo Failed: %failed_files%
echo ============================================================
echo.

pause
endlocal
exit /b 0
