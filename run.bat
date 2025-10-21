@echo off
REM run.bat - Main entry point for all batch scripts
REM Usage:
REM   run.bat test       - Test single file preprocessing
REM   run.bat batch      - Batch process all files
REM   run.bat analyze    - Analyze 1Hz output

setlocal

if "%~1"=="" (
    echo Usage: run.bat [command]
    echo.
    echo Commands:
    echo   test       - Test single file preprocessing
    echo   batch      - Batch process all files
    echo   analyze    - Analyze 1Hz output
    echo.
    pause
    exit /b 1
)

if "%~1"=="test" (
    call scripts\test_preprocess.bat
    goto :end
)

if "%~1"=="batch" (
    call scripts\batch_preprocess.bat
    goto :end
)

if "%~1"=="analyze" (
    call scripts\analyze_1hz.bat
    goto :end
)

echo ERROR: Unknown command: %~1
echo Run 'run.bat' to see available commands
pause
exit /b 1

:end
endlocal
