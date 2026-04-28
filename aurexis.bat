@echo off
setlocal enabledelayedexpansion

REM Aurexis launcher — Windows batch script.
REM Opens Command Prompt and runs the menu. Stays open to show errors.

echo.
echo ================================================================
echo  AUREXIS — offline measurement substrate launcher
echo ================================================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
echo [1/3] Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Python is not on PATH.
    echo.
    echo Install Python 3.10+ from https://python.org
    echo Make SURE to check "Add Python to PATH" during install.
    echo.
    echo After installing, close this window and try again.
    echo.
    pause
    exit /b 1
)
python --version
echo  OK - Python found.
echo.

REM Check and install deps
echo [2/3] Checking Python libraries...
python -c "import requests; import kymatio; import sklearn; import skimage; import joblib; import datasets; import cv2; print('  OK - all libraries present')" 2>nul
if errorlevel 1 (
    echo  MISSING - installing now (this takes 3-5 minutes, first run only)...
    echo.
    python -m pip install --upgrade pip --quiet
    python -m pip install requests numpy scipy pillow scikit-image scikit-learn kymatio joblib datasets opencv-python
    echo.
    echo  Libraries installed.
    echo.
)
echo.

REM Run the app
echo [3/3] Starting Aurexis menu...
echo.
python cli.py
if errorlevel 1 (
    echo.
    echo [ERROR] The app crashed. Details above ^^.
    echo.
)

echo.
echo ================================================================
echo  Press any key to close...
echo ================================================================
pause >nul
