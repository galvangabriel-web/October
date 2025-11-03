@echo off
REM Racing Data Analysis - Windows Setup Script
REM This script creates a virtual environment and installs all dependencies

echo ========================================
echo Racing Data Analysis Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10+ from:
    echo - Microsoft Store, or
    echo - https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/4] Python found:
python --version
echo.

echo [2/4] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

echo [4/5] Creating data directories...
if not exist data\processed mkdir data\processed
if not exist data\models mkdir data\models
echo Data directories created successfully
echo.

echo [5/5] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the virtual environment in the future:
echo   venv\Scripts\activate
echo.
echo To run data analysis:
echo   python deep_data_analysis.py
echo.
pause
