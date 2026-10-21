@echo off
cd /d "%~dp0"
set "ONCOPATH_VENV=%LOCALAPPDATA%\OncoPath\venv"
call "%ONCOPATH_VENV%\Scripts\activate.bat"
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
