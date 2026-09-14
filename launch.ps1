<#
.SYNOPSIS
    PixelMemory Platform Launcher for PowerShell
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[!] Virtual environment (.venv) not found. Running automated setup..." -ForegroundColor Yellow
    & (Join-Path $ScriptDir "setup.ps1")
}
if (-not (Test-Path $PythonExe)) {
    $Py312 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path $Py312) {
        $PythonExe = $Py312
    } else {
        $PythonExe = "python"
    }
}
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $CargoBin) {
    $env:PATH = "$CargoBin;$env:PATH"
}

$ForwardArgs = @()
foreach ($arg in $args) {
    if ($arg -match '^-([a-zA-Z].*)' -and -not ($arg -match '^--')) {
        $ForwardArgs += "--$($matches[1].ToLower())"
    } else {
        $ForwardArgs += $arg
    }
}

& $PythonExe (Join-Path $ScriptDir "launch.py") @ForwardArgs
