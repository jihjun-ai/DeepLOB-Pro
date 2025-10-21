@echo off
REM analyze_1hz.bat - Analyze 1Hz aggregation output
REM Usage: Run from project root directory D:\Case-New\python\DeepLOB-Pro

setlocal

set PROJECT_ROOT=%~dp0..
set CONDA_ENV=deeplob-pro

echo ============================================================
echo Analyzing 1Hz Output
echo ============================================================
echo.

call conda activate %CONDA_ENV%

python "%PROJECT_ROOT%\scripts\quick_analyze.py"

echo.
pause
endlocal
