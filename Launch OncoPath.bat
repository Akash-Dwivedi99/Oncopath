@echo off
setlocal
cd /d "%~dp0"

python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Streamlit is not installed in the Python environment used by this launcher.
    echo Install the project dependencies with:
    echo     python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

python -m streamlit run "%~dp0app.py"
if errorlevel 1 (
    echo.
    echo OncoPath stopped with an error. Review the message above.
    pause
)
