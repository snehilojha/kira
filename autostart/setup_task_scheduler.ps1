# setup_task_scheduler.ps1
# Registers telegram-runner as a Windows Task Scheduler task that starts at boot.
#
# Usage: Right-click → "Run with PowerShell as Administrator"
#
# To check:  Task Scheduler → Task Scheduler Library → TelegramRunner
# To stop:   Stop-ScheduledTask -TaskName "TelegramRunner"
# To start:  Start-ScheduledTask -TaskName "TelegramRunner"
# To remove: Unregister-ScheduledTask -TaskName "TelegramRunner" -Confirm:$false

$TaskName = "TelegramRunner"
$PythonExe = (Get-Command python).Source
$ScriptPath = Join-Path $PSScriptRoot "..\bot\main.py"
$WorkingDir = Join-Path $PSScriptRoot "..\"

# Resolve to absolute paths
$ScriptPath = (Resolve-Path $ScriptPath).Path
$WorkingDir = (Resolve-Path $WorkingDir).Path

Write-Host "Registering Task Scheduler entry..."
Write-Host "  Task name:   $TaskName"
Write-Host "  Python:      $PythonExe"
Write-Host "  Script:      $ScriptPath"
Write-Host "  Working dir: $WorkingDir"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Trigger: at system startup
$Trigger = New-ScheduledTaskTrigger -AtStartup

# Action: run python bot/main.py
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory $WorkingDir

# Settings: restart on failure up to 3 times with 1 minute delay
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

# Register — runs whether user is logged on or not
Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Description "Telegram Runner bot — persistent background service" `
    -RunLevel Highest

Write-Host ""
Write-Host "Done. Task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: Set Power Options -> Never sleep (when plugged in) to keep the bot alive."
Write-Host "           Screen off is fine. Sleep/Hibernate kill the network adapter."
