@echo off
REM Windows batch file equivalent to Makefile

if "%1"=="quick" (
    set PYTHONPATH=%PYTHONPATH%;%CD%
    echo 🔒 Running bulletproof benchmark with locked GPU settings...
    python run_benchmark.py --model dense120 --lock-gpu --output benchmarks/dense120_results.json
    python run_benchmark.py --model moe --lock-gpu --output benchmarks/moe_results.json
    python run_benchmark.py --model dense300 --lock-gpu --output benchmarks/dense300_results.json
    echo 📊 Generating comparison...
    python 10_core\compare.py
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