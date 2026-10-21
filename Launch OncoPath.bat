@echo off
setlocal
cd /d "%~dp0"
set "ONCOPATH_VENV=%LOCALAPPDATA%\OncoPath\venv"

if not exist "%ONCOPATH_VENV%\Scripts\python.exe" (
    echo Creating Python environment in the short path "%ONCOPATH_VENV%"...
    python -m venv "%ONCOPATH_VENV%"
    if errorlevel 1 (
        echo Could not create the environment. Install Python 3.10 or newer and try again.
        pause
        exit /b 1
    )
)

call "%ONCOPATH_VENV%\Scripts\activate.bat"
python -c "import fastapi, uvicorn, streamlit, pandas, numpy, networkx, scipy, sklearn, plotly, requests, dotenv" >nul 2>&1
if errorlevel 1 (
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check your internet connection and requirements.txt.
        pause
        exit /b 1
    )
)

start "OncoPath API" cmd /k ""%~dp0run_backend.bat""
timeout /t 3 /nobreak >nul
start "OncoPath Dashboard" cmd /k ""%~dp0run_frontend.bat""
echo OncoPath is starting. Keep both terminal windows open.
