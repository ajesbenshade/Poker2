# Registers (or with -Remove, deletes) the "Poker2 blueprint auto-resume" task
# for the current user. It runs auto_resume.ps1 at logon and every 15 minutes;
# that script restarts the blueprint from its last checkpoint only if it died
# (reboot or crash), never after a deliberate stop or a finished run.
#
# A logon trigger needs someone to sign in after a reboot (Windows does not sign
# in automatically unless configured to). No admin rights needed.
#
#   powershell -ExecutionPolicy Bypass -File engine\scripts\install_auto_resume.ps1
#   powershell -ExecutionPolicy Bypass -File engine\scripts\install_auto_resume.ps1 -Remove
param([switch]$Remove)
$ErrorActionPreference = 'Stop'
$name = 'Poker2 blueprint auto-resume'

if ($Remove) {
  Unregister-ScheduledTask -TaskName $name -Confirm:$false
  Write-Output "removed scheduled task '$name'"
  return
}

$script = (Resolve-Path (Join-Path $PSScriptRoot 'auto_resume.ps1')).Path
# conhost --headless runs PowerShell without flashing a console window every 15 minutes.
$action = New-ScheduledTaskAction -Execute 'conhost.exe' `
  -Argument "--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$script`""
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$atLogon.Delay = 'PT2M'  # let the desktop settle after signing in
$every15 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
  -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Trigger @($atLogon, $every15) -Settings $settings `
  -Principal $principal -Description 'Restarts the Poker2 blueprint training run from its last checkpoint after a reboot or crash.' -Force | Out-Null
Write-Output "registered scheduled task '$name' (at logon and every 15 minutes)"
