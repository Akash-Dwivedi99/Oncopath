@echo off
cd /d "%~dp0"
set "ONCOPATH_VENV=%LOCALAPPDATA%\OncoPath\venv"
call "%ONCOPATH_VENV%\Scripts\activate.bat"
python -m streamlit run frontend\app.py
