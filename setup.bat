@echo off
setlocal enabledelayedexpansion
title PixelMemory Automated Setup

cd /d "%~dp0"

echo =====================================================================
echo   PixelMemory Platform Setup (Windows)
echo   Local Semantic Photo Search - Privacy Preserving - Zero Cloud
echo =====================================================================
echo.

:: 1. Locate suitable Python (3.10 to 3.12)
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto :python_found
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :python_found
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :python_found
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%i in ('where python') do (
        set "PYTHON_EXE=%%i"
        goto :python_found
    )
)

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py -3.12"
    goto :python_found
)

:python_found
if "%PYTHON_EXE%"=="" (
    echo [!] Python 3.10-3.12 was not detected on your system.
    echo.
    echo Would you like to install Python 3.12 via winget now?
    set /p INSTALL_PY="Install Python 3.12 now? (Y/n): "
    if /i "!INSTALL_PY!" neq "n" (
        echo.
        echo [*] Installing Python 3.12...
        winget install -e --id Python.Python.3.12 --scope user
        echo.
        echo Please restart setup.bat after installation completes.
        pause
        exit /b 1
    ) else (
        echo Please install Python 3.10-3.12 from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

echo [OK] Using Python: %PYTHON_EXE%
echo.

:: 2. Virtual Environment
if exist ".venv\Scripts\python.exe" goto :venv_exists
echo [*] Creating isolated virtual environment in .venv...
%PYTHON_EXE% -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo [!] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.

:venv_exists
set "VENV_PY=.venv\Scripts\python.exe"

:: 3. Python Dependencies
echo.
echo [*] Installing / upgrading Python dependencies (requirements.txt)...
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo [!] Warning: Some dependencies may have had installation issues.
) else (
    echo [OK] All Python dependencies installed successfully.
)

:: 4. Ollama Check
echo.
echo ---------------------------------------------------------------------
echo   Ollama Local AI Check
echo ---------------------------------------------------------------------
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
)
where ollama >nul 2>nul
if %ERRORLEVEL% equ 0 goto :ollama_found

echo [!] Ollama was not detected on this machine.
echo     Ollama powers fast local GPU vision (Moondream2)
echo     and travel story generation (Gemma 4).
echo.
set /p INSTALL_OLLAMA="Install Ollama automatically via winget? (Y/n): "
if /i "!INSTALL_OLLAMA!" equ "n" goto :ollama_skipped

echo [*] Installing Ollama via winget...
winget install -e --id Ollama.Ollama
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
)
echo [*] Starting Ollama background service...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    timeout /t 5 >nul
)
where ollama >nul 2>nul
if %ERRORLEVEL% equ 0 goto :ollama_found

:ollama_skipped
echo [i] You can install Ollama manually anytime from: https://ollama.com/download/windows
goto :desktop_check

:ollama_found
echo [OK] Ollama is ready.
echo [*] Ensuring default models are installed:
echo     - Vision Model: moondream (Moondream2 1.8B)
echo     - Story Model:  gemma4:e2b (Gemma 4 2B)
echo.
echo Checking / pulling moondream...
ollama pull moondream
echo.
echo Checking / pulling gemma4:e2b...
ollama pull gemma4:e2b
echo [OK] Ollama models verified.

:desktop_check
:: 5. Native Desktop (Tauri) Check
echo.
echo ---------------------------------------------------------------------
echo   Native Desktop (Tauri) Check
echo ---------------------------------------------------------------------
if exist "%USERPROFILE%\.cargo\bin" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)
where cargo >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Rust/Cargo detected. Native desktop build enabled.
) else (
    echo [i] Rust/Cargo not found. Desktop mode requires Rust from https://rustup.rs
)

where npm >nul 2>nul
if %ERRORLEVEL% equ 0 (
    if not exist "node_modules" (
        echo [*] Installing Tauri frontend dependencies...
        npm install --silent
    )
    echo [OK] Tauri frontend dependencies ready.
)

echo.
echo =====================================================================
echo   [OK] PixelMemory Setup Complete!
echo =====================================================================
echo.
echo Launch PixelMemory anytime:
echo.
echo   - Native Desktop App:   .\launch.bat --desktop
echo   - Web Browser Mode:     .\launch.bat
echo   - System Diagnostics:   .\launch.bat --status
echo.
if "%1" neq "--no-pause" (
    pause
)
