@echo off
setlocal
cd /d "%~dp0"
set APPNAME=WB_QuickFix

if not exist ".buildvenv\Scripts\python.exe" (
  echo Creating build environment...
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .buildvenv
  ) else (
    py -m venv .buildvenv
  )
  if errorlevel 1 goto :error
)

".buildvenv\Scripts\python.exe" -m pip install --upgrade pip
".buildvenv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo.
echo Building EXE with PyInstaller...
".buildvenv\Scripts\python.exe" -m PyInstaller ^
  --clean ^
  --noconfirm ^
  --onedir ^
  --windowed ^
  --name "%APPNAME%" ^
  app.py
if errorlevel 1 goto :error

echo.
echo Build complete: dist\%APPNAME%\%APPNAME%.exe
pause
exit /b 0

:error
echo.
echo Build failed.
pause
exit /b 1
