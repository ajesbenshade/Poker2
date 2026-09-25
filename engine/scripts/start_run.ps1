# Starts a training script (engine/scripts/<Script>.sh) inside WSL, detached
# from this terminal.
#
# WSL shuts its VM down shortly after the last wsl.exe client exits, so a
# wsl.exe must stay alive for the whole run. Creating it through WMI puts it
# outside this terminal's process tree, so closing the terminal (or the app
# that opened it) does not end the run. Output goes to ~/poker2-runs/<Script>-stdout.log.
#
# Usage:  powershell -ExecutionPolicy Bypass -File engine\scripts\start_run.ps1 [-Script blueprint|pilot] [--resume]
param(
  [ValidateSet('blueprint', 'pilot')][string]$Script = 'blueprint',
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs = @()
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$drive = $repo.Substring(0, 1).ToLower()
$wslRepo = '/mnt/' + $drive + ($repo.Substring(2) -replace '\\', '/')
$extra = $ExtraArgs -join ' '
$inner = "mkdir -p ~/poker2-runs && bash '$wslRepo/engine/scripts/$Script.sh' $extra >> ~/poker2-runs/$Script-stdout.log 2>&1"
$cmd = "wsl.exe -d Ubuntu-24.04 -- bash -c `"$inner`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
if ($result.ReturnValue -ne 0) { throw "failed to start wsl.exe (WMI code $($result.ReturnValue))" }
Write-Output "$Script started (wsl.exe pid $($result.ProcessId)); follow it with:"
Write-Output "  wsl -d Ubuntu-24.04 -- tail -f ~/poker2-runs/$Script/train.log"
