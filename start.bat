@echo off
title Shorts Runner
echo ========================================
echo    Shorts Runner - Starting...
echo ========================================

:: Start folder helper (hidden in background)
start /B /MIN pythonw "%~dp0folder_helper.py" 2>nul
if errorlevel 1 (
    start /B /MIN python "%~dp0folder_helper.py" 2>nul
)

:: Start Docker
echo Starting Docker containers...
cd /d "%~dp0"
docker compose up -d

:: Open browser
timeout /t 3 /nobreak >nul
start http://localhost:8001

echo ========================================
echo    Ready! http://localhost:8001
echo ========================================
echo.
echo Press any key to stop...
pause >nul

:: Cleanup - stop helper
taskkill /F /FI "WINDOWTITLE eq Folder*" 2>nul
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq pythonw.exe" /FO LIST ^| find "PID:"') do taskkill /PID %%a /F 2>nul
