@echo off
setlocal
title PixelMemory Platform Launcher

cd /d "%~dp0"

:: Auto-run setup if virtual environment is missing
if exist ".venv\Scripts\python.exe" goto :have_venv

echo [!] Virtual environment not found in .venv. Running setup...
call setup.bat --no-pause

:have_venv
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) else (
    set "PYTHON_EXE=python"
)

:: Ensure Cargo bin is in PATH
if exist "%USERPROFILE%\.cargo\bin" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

:: Run launcher
"%PYTHON_EXE%" launch.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Launcher exited with error code %ERRORLEVEL%.
    pause
)
