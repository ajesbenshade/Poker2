# Full blueprint run on native Windows (roadmap days 9-27).
#
# Same settings as blueprint.sh, but runs poker2_train.exe directly on Windows:
# no WSL VM that the host can kill (the first full run died that way on
# 2026-09-25 after 95 minutes, before its first checkpoint), and the whole
# 64 GB of RAM instead of WSL's 30 GB. Checkpoints every 30 minutes.
#
# Blueprint training is CPU-only (24 threads); the GPU is not used.
#
#   powershell -ExecutionPolicy Bypass -File engine\scripts\blueprint.ps1           start a new run
#   powershell -ExecutionPolicy Bypass -File engine\scripts\blueprint.ps1 -Resume   continue from the last checkpoint
#   powershell -ExecutionPolicy Bypass -File engine\scripts\blueprint.ps1 -Stop     finish the epoch, checkpoint, exit
#   Get-Content $env:USERPROFILE\poker2-runs\blueprint\train.log -Wait -Tail 20     follow progress
#
# Build first:  powershell -ExecutionPolicy Bypass -File engine\build.ps1 -Test
param(
  [switch]$Resume,
  [switch]$Stop,
  [string]$Run = "$env:USERPROFILE\poker2-runs\blueprint",
  [string]$Abstraction = "$env:USERPROFILE\poker2-data\abstraction"
)
$ErrorActionPreference = 'Stop'

if ($Stop) {
  New-Item -ItemType File -Force -Path (Join-Path $Run 'STOP') | Out-Null
  Write-Output "STOP requested; the run checkpoints and exits after its current epoch (up to ~10 minutes)."
  return
}

$exe = (Resolve-Path (Join-Path $PSScriptRoot '..\build-win\poker2_train.exe')).Path
if (Get-Process poker2_train -ErrorAction SilentlyContinue) { throw 'poker2_train is already running' }
New-Item -ItemType Directory -Force -Path $Run | Out-Null

$trainArgs = @(
  '--abstraction', "`"$Abstraction`"",
  '--run', "`"$Run`"",
  '--profile', 'blueprint',
  '--hours', '432',
  '--epoch', '50000000',
  '--linear-minutes', '400',
  '--prune-after-minutes', '200',
  '--checkpoint-minutes', '30',
  '--eval-minutes', '60',
  '--lbr-hands', '100000',
  '--h2h-deals', '50000'
)
if ($Resume) { $trainArgs += '--resume' }

# Created through WMI so the process is outside this terminal's process tree
# and keeps running when the terminal (or the app that opened it) closes.
# Below-normal priority keeps the desktop responsive; no console window.
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ PriorityClass = [uint32]16384; ShowWindow = [uint16]0 }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "`"$exe`" $($trainArgs -join ' ')"
  CurrentDirectory = $Run
  ProcessStartupInformation = $startup
}
if ($result.ReturnValue -ne 0) { throw "failed to start poker2_train (WMI code $($result.ReturnValue))" }
Write-Output "blueprint started (pid $($result.ProcessId)); follow it with:"
Write-Output "  Get-Content `"$Run\train.log`" -Wait -Tail 20"
