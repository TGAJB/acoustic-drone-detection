@echo off
REM ============================================================
REM  One-click environment setup for acoustic-drone-detection.
REM  Double-click this file (or run it in a terminal) to create
REM  the virtual environment and install all dependencies.
REM ============================================================

echo.
echo [1/3] Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo.
    echo ERROR: could not create the venv. Is Python installed and on PATH?
    echo Download Python from https://www.python.org/downloads/ and tick
    echo "Add python.exe to PATH" during install, then run this again.
    pause
    exit /b 1
)

echo.
echo [2/3] Activating and upgrading pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: dependency install failed. See messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done. The environment is ready.
echo  To use it in a new terminal, run:  .venv\Scripts\activate
echo ============================================================
pause
