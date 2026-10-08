# Restarts the blueprint run after a reboot or crash. Run by the scheduled task
# that install_auto_resume.ps1 creates (at logon and every 15 minutes).
#
# Resumes only when all of these hold:
#   - poker2_train is not running;
#   - the run has a checkpoint;
#   - train.log does not end with "exiting normally". A run that finished its
#     time budget or was stopped on purpose (blueprint.ps1 -Stop) ends that way
#     and stays stopped; a run killed by a reboot or crash never writes it.
# Every decision is appended to auto_resume.log in the run directory.
param(
  [string]$Run = "$env:USERPROFILE\poker2-runs\blueprint",
  [switch]$DryRun  # for testing: report the decision, skip the running check, never launch
)
$ErrorActionPreference = 'Stop'
$log = Join-Path $Run 'auto_resume.log'

function Note([string]$message) {
  if (Test-Path $Run) {
    Add-Content -Path $log -Value ("[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $message)
  }
}

if (-not $DryRun -and (Get-Process poker2_train -ErrorAction SilentlyContinue)) { return }  # running (not logged)
if (-not (Test-Path (Join-Path $Run 'checkpoint.bin'))) { Note 'no checkpoint yet; not resuming'; 'no checkpoint'; return }
$last = Get-Content (Join-Path $Run 'train.log') -Tail 1 -ErrorAction SilentlyContinue
if ($last -match 'exiting normally') { 'stay stopped'; return }  # finished or stopped on purpose (not logged)
if ($DryRun) { 'would resume'; return }

Note "poker2_train is not running and the run did not exit normally (last log line: $last); resuming"
try {
  $output = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'blueprint.ps1') -Resume -Run $Run 2>&1
  Note ("resume launched: " + ($output -join ' '))
} catch {
  Note "resume failed: $_"
}
