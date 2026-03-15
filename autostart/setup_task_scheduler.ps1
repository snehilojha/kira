# setup_task_scheduler.ps1
# Registers kira as a Windows Task Scheduler task that starts at boot.
#
# Usage: Right-click → "Run with PowerShell as Administrator"
#
# To check:  Task Scheduler → Task Scheduler Library → Kira
# To stop:   Stop-ScheduledTask -TaskName "Kira"
# To start:  Start-ScheduledTask -TaskName "Kira"
# To remove: Unregister-ScheduledTask -TaskName "Kira" -Confirm:$false

$TaskName = "Kira"
$WorkingDir = Join-Path $PSScriptRoot "..\"
$WorkingDir = (Resolve-Path $WorkingDir).Path

# Auto-detect venv: try .venv first, then venv
$VenvPath = Join-Path $WorkingDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPath)) {
    $VenvPath = Join-Path $WorkingDir "venv\Scripts\python.exe"
}
if (-not (Test-Path $VenvPath)) {
    Write-Host "ERROR: No virtual environment found at .venv or venv" -ForegroundColor Red
    Write-Host "       Create one with: python -m venv .venv"
    exit 1
}
$PythonExe = $VenvPath

Write-Host "Registering Task Scheduler entry..."
Write-Host "  Task name:   $TaskName"
Write-Host "  Python:      $PythonExe"
Write-Host "  Command:     -m bot.main"
Write-Host "  Working dir: $WorkingDir"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Trigger: at system startup, with a 30-second delay to allow network to come up
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Trigger.Delay = "PT30S"

# Action: run python -m bot.main
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m bot.main" `
    -WorkingDirectory $WorkingDir

# Settings: restart on failure up to 3 times with 1 minute delay
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

# Register — runs whether user is logged on or not (S4U = no password stored, works at boot)
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType S4U `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Kira bot - persistent background service"

Write-Host ""
Write-Host "Done. Task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: Set Power Options -> Never sleep (when plugged in) to keep the bot alive."
Write-Host "           Screen off is fine. Sleep/Hibernate kill the network adapter."
