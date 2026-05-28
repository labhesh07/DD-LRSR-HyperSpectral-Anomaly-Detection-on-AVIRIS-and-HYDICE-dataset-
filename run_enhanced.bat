@echo off
REM Simple batch script to run the enhanced demo on Windows

echo ========================================
echo Hyperspectral Anomaly Detection - Enhanced
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3 and try again
    pause
    exit /b 1
)

echo Python found. Starting enhanced demo...
echo.

REM Run the enhanced demo
python demo_enhanced.py

echo.
echo ========================================
echo Execution completed!
echo Check the 'results' folder for outputs
echo ========================================
pause

