<#
.SYNOPSIS
    PixelMemory Platform Launcher for PowerShell
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe (Join-Path $ScriptDir "launch.py") @args
