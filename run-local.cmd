@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-local.ps1" %*
if errorlevel 1 (
    echo Launch failed. Read the error above before closing this window.
    pause
    exit /b 1
)
