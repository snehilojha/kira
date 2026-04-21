# setup_local_voice_task.ps1
# Registers Kira Local Voice as a Windows Task Scheduler task.
#
# Usage: Right-click -> "Run with PowerShell"
#
# To check:  Task Scheduler -> Task Scheduler Library -> Kira Local Voice
# To stop:   Stop-ScheduledTask -TaskName "Kira Local Voice"
# To start:  Start-ScheduledTask -TaskName "Kira Local Voice"
# To remove: Unregister-ScheduledTask -TaskName "Kira Local Voice" -Confirm:$false

$ErrorActionPreference = "Stop"

function Pause-BeforeExit {
    Write-Host ""
    Read-Host "Press Enter to close this window"
}

function Test-IsElevated {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsElevated)) {
    Write-Host "Task Scheduler registration requires elevation on this Windows setup."
    Write-Host "Requesting administrator permission..."
    $argumentList = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "`"$PSCommandPath`""
    )
    Start-Process -FilePath "powershell.exe" -ArgumentList $argumentList -Verb RunAs
    exit
}

try {

$TaskName = "Kira Local Voice"
$WorkingDir = Join-Path $PSScriptRoot "..\"
$WorkingDir = (Resolve-Path $WorkingDir).Path

# Auto-detect venv: try .venv first, then venv. Prefer pythonw.exe so no console stays open.
$VenvPath = Join-Path $WorkingDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPath)) {
    $VenvPath = Join-Path $WorkingDir "venv\Scripts\python.exe"
}
if (-not (Test-Path $VenvPath)) {
    Write-Host "ERROR: No virtual environment found at .venv or venv" -ForegroundColor Red
    Write-Host "       Create one with: python -m venv .venv"
    Pause-BeforeExit
    exit 1
}
$PythonExe = $VenvPath
$PythonwExe = $PythonExe -replace "python\.exe$", "pythonw.exe"
if (Test-Path $PythonwExe) {
    $PythonExe = $PythonwExe
}

$LogDir = Join-Path $WorkingDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "local_voice.log"

Write-Host "Registering Task Scheduler entry..."
Write-Host "  Task name:   $TaskName"
Write-Host "  Python:      $PythonExe"
Write-Host "  Command:     -m bot.local_voice"
Write-Host "  Working dir: $WorkingDir"
Write-Host "  Log file:    $LogPath"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Local voice needs the interactive user session for keyboard hotkeys, mic, and speakers.
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Trigger.Delay = "PT15S"

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m bot.local_voice" `
    -WorkingDirectory $WorkingDir

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Kira local voice hotkey listener" `
    -ErrorAction Stop

Write-Host ""
Write-Host "Done. Task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host ""
Write-Host "It starts after you log in, because global hotkeys, microphone, and speakers need your desktop session."
Write-Host "If startup fails, inspect: $LogPath"
}
catch {
    Write-Host ""
    Write-Host "ERROR: Failed to register Kira Local Voice." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
finally {
    Pause-BeforeExit
}
