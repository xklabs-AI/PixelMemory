<#
.SYNOPSIS
    PixelMemory Automated Setup Script for Windows PowerShell
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  PixelMemory Platform Setup (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "  Local Semantic Photo Search · Privacy Preserving · Zero Cloud" -ForegroundColor Gray
Write-Host "=====================================================================`n" -ForegroundColor Cyan

# 1. Locate Python 3.10-3.12
$PythonExe = $null
if (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    $PythonExe = "$ScriptDir\.venv\Scripts\python.exe"
} elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") {
    $PythonExe = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
} elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe") {
    $PythonExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
} else {
    $SysPy = (Get-Command python -ErrorAction SilentlyContinue)
    if ($SysPy) {
        $PythonExe = $SysPy.Source
    }
}

if (-not $PythonExe) {
    Write-Host "[!] Python 3.10-3.12 was not detected." -ForegroundColor Yellow
    $reply = Read-Host "Would you like to install Python 3.12 via winget now? (Y/n)"
    if ($reply -notmatch "^[Nn]") {
        winget install -e --id Python.Python.3.12 --scope currentuser
        Write-Host "Please re-run this setup script once installation finishes." -ForegroundColor Green
        return
    }
}

Write-Host "[OK] Using Python: $PythonExe" -ForegroundColor Green

# 2. Virtual Environment
if (-not (Test-Path "$ScriptDir\.venv\Scripts\python.exe")) {
    Write-Host "`n[*] Creating virtual environment in .venv..." -ForegroundColor Cyan
    & $PythonExe -m venv "$ScriptDir\.venv"
}
$VenvPy = "$ScriptDir\.venv\Scripts\python.exe"

# 3. Dependencies
Write-Host "`n[*] Installing / upgrading Python dependencies (requirements.txt)..." -ForegroundColor Cyan
& $VenvPy -m pip install --upgrade pip --quiet
& $VenvPy -m pip install -r "$ScriptDir\requirements.txt" --quiet
Write-Host "[OK] Python dependencies ready." -ForegroundColor Green

# 4. Ollama Check
Write-Host "`n── Ollama Local AI Check ──────────────────────────────────────────" -ForegroundColor Cyan
$HasOllama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $HasOllama -and (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
    $env:PATH = "$env:LOCALAPPDATA\Programs\Ollama;$env:PATH"
    $HasOllama = Get-Command ollama -ErrorAction SilentlyContinue
}

if (-not $HasOllama) {
    Write-Host "[!] Ollama was not detected." -ForegroundColor Yellow
    $installOllama = Read-Host "Install Ollama automatically via winget? (Y/n)"
    if ($installOllama -notmatch "^[Nn]") {
        winget install -e --id Ollama.Ollama
        Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
        Start-Sleep -Seconds 4
        $HasOllama = Get-Command ollama -ErrorAction SilentlyContinue
    }
}

if ($HasOllama) {
    Write-Host "[OK] Ollama detected. Ensuring models are pulled:" -ForegroundColor Green
    Write-Host "  • Vision model: moondream" -ForegroundColor Gray
    ollama pull moondream
    Write-Host "`n  • Story model:  gemma4:e2b" -ForegroundColor Gray
    ollama pull gemma4:e2b
    Write-Host "[OK] Ollama models ready." -ForegroundColor Green
}

# 5. Desktop build prerequisites
Write-Host "`n── Native Desktop (Tauri) Check ───────────────────────────────────" -ForegroundColor Cyan
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $CargoBin) {
    $env:PATH = "$CargoBin;$env:PATH"
}
$HasCargo = Get-Command cargo -ErrorAction SilentlyContinue
if ($HasCargo) {
    Write-Host "[OK] Rust/Cargo detected. Native desktop build enabled." -ForegroundColor Green
} else {
    Write-Host "[i] Rust/Cargo not found. Desktop mode requires Rust (https://rustup.rs)." -ForegroundColor Yellow
}

Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host "  [OK] Setup Complete!" -ForegroundColor Green
Write-Host "=====================================================================`n" -ForegroundColor Green
Write-Host "Start PixelMemory:"
Write-Host "  .\launch.bat --desktop    # Native desktop app" -ForegroundColor Cyan
Write-Host "  .\launch.bat              # Web browser mode" -ForegroundColor Cyan
Write-Host ""
