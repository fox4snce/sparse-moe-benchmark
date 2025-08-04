@echo off
REM Windows batch file equivalent to Makefile

if "%1"=="quick" (
    set PYTHONPATH=%PYTHONPATH%;%CD%
    python 10_core\run.py --config 10_core\configs\moe_3expert_quick.yaml
) else if "%1"=="full" (
    python 10_core\run.py --config 20_extended\configs\moe_full.yaml
) else if "%1"=="sweep" (
    bash 20_extended\sweep.sh
) else if "%1"=="test" (
    pytest 00_sanity\ -v
) else if "%1"=="clean" (
    rmdir /s /q benchmarks\*.csv 2>nul
    rmdir /s /q __pycache__\ 2>nul
    rmdir /s /q *\__pycache__\ 2>nul
    rmdir /s /q .pytest_cache\ 2>nul
) else (
    echo Usage: make.bat [quick^|full^|sweep^|test^|clean]
    echo.
    echo Commands:
    echo   quick  - Run quick benchmark (^<10 min^)
    echo   full   - Run full evaluation suite (overnight^)
    echo   sweep  - Run hyperparameter sweep
    echo   test   - Run sanity tests
    echo   clean  - Clean generated files
) 