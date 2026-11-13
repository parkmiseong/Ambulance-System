@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment not found: .venv\Scripts\python.exe
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B CODE\web_dashboard.py
if errorlevel 1 pause
