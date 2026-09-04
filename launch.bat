@echo off
setlocal
title PixelMemory Platform Launcher

cd /d "%~dp0"

:: Check for virtual environment Python
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

:: Run launcher
"%PYTHON_EXE%" launch.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Launcher exited with error code %ERRORLEVEL%.
    pause
)
