@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -m venv .venv
  )
  if errorlevel 1 goto :error

  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" app.py
exit /b %errorlevel%

:error
echo.
echo Setup failed. Check that Python 3.10+ is installed.
pause
exit /b 1
